# -*- coding: utf-8 -*-
"""V1 监督初始化、专家回放、DQN、验证选模训练器。"""

import copy
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from action import ActionSelector, legal_action_mask
from evaluation import evaluate
from memory import Transition
from state import build_state_vector, predict_actions_from_state


class DQNTrainer(object):
    def __init__(self, env, train_data, validation_data, policy_net, target_net, memory,
                 optimizer, device, logger, batch_size=64, gamma=0.95, tau=0.01,
                 eps_start=0.7, eps_end=0.02, eps_decay=10000, n_step=1,
                 min_actions_before_stop=1, use_aux_bce=False, aux_bce_weight=0.0,
                 aux_pos_weight_max=8.0, pretrain_pos_weight_max=18.0,
                 supervised_loss='class_balanced', focal_gamma=2.0,
                 use_per=True, per_beta_start=0.4, per_beta_frames=5000,
                 rare_per_scale=0.0, stop_margin_threshold=None, use_amp=False):
        self.env = env
        self.train_data = [(item[0], item[1]) for item in train_data]
        self.validation_data = [(item[0], item[1]) for item in validation_data]
        self.policy_net = policy_net
        self.target_net = target_net
        self.memory = memory
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.n_step = max(1, int(n_step))
        self.min_actions_before_stop = max(0, int(min_actions_before_stop))
        self.use_aux_bce = bool(use_aux_bce) and float(aux_bce_weight) > 0
        self.aux_bce_weight = float(aux_bce_weight)
        self.pretrain_pos_weight_max = float(pretrain_pos_weight_max)
        self.supervised_loss_name = supervised_loss
        self.focal_gamma = float(focal_gamma)
        self.use_per = bool(use_per)
        self.per_beta_start = float(per_beta_start)
        self.per_beta_frames = max(1, int(per_beta_frames))
        self.rare_per_scale = float(rare_per_scale)
        self.stop_margin_threshold = stop_margin_threshold
        self.use_amp = bool(use_amp) and device.type == 'cuda'
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            self.grad_scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        else:
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.optimization_steps = 0
        self.target_update_count = 0
        self.action_selector = ActionSelector(
            env, policy_net, len(env.action_space), device, eps_start, eps_end, eps_decay,
            min_actions_before_stop=self.min_actions_before_stop,
        )
        for parameter in self.target_net.parameters():
            parameter.requires_grad_(False)
        self.target_net.eval()

        counts = np.ones(len(env.action_space), dtype=np.float32)
        for _, labels in self.train_data:
            for label in labels:
                action = env.label_to_action[label]
                counts[action] += 1
        counts[env.stop_action] = max(len(self.train_data), 1)
        negatives = np.maximum(len(self.train_data) - counts, 1.0)
        pos_weight = np.sqrt(negatives / np.maximum(counts, 1.0))
        pos_weight = np.clip(pos_weight, 1.0, float(aux_pos_weight_max))
        pos_weight[env.stop_action] = 1.0
        self.pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)
        self.logger.info(f'训练精度模式: AMP={self.use_amp}, device={self.device}')

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        if hasattr(torch, 'autocast'):
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        return torch.cuda.amp.autocast(dtype=torch.float16)

    def _optimizer_step(self, loss, optimizer):
        optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
            return
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        optimizer.step()

    def _current_per_beta(self):
        progress = min(1.0, self.optimization_steps / self.per_beta_frames)
        return self.per_beta_start + progress * (1.0 - self.per_beta_start)

    def _soft_update_target(self):
        with torch.no_grad():
            for target, policy in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target.data.mul_(1.0 - self.tau).add_(policy.data, alpha=self.tau)
        self.target_update_count += 1

    def _remaining_target(self, selected_actions, true_actions):
        target = np.zeros(len(self.env.action_space), dtype=np.float32)
        remaining = [action for action in true_actions if action not in selected_actions]
        if remaining:
            target[remaining] = 1.0
        else:
            target[self.env.stop_action] = 1.0
        return target

    def _loss(self, logits, targets, pos_weight):
        # 普通 BCE 不使用类别权重；class-balanced 和 focal 使用训练集导出的 pos_weight。
        effective_pos_weight = None if self.supervised_loss_name == 'bce' else pos_weight
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=effective_pos_weight, reduction='none'
        )
        if self.supervised_loss_name == 'focal':
            probabilities = torch.sigmoid(logits)
            p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
            bce = (1.0 - p_t).pow(self.focal_gamma) * bce
        return bce.mean()

    def pretrain_supervised(self, epochs, learning_rate, num_orders=1):
        if epochs <= 0:
            return
        states, targets = [], []
        for symptoms, label_names in self.train_data:
            true_actions = [self.env.label_to_action[label] for label in label_names]
            orders = [list(true_actions)]
            for _ in range(max(0, int(num_orders) - 1)):
                order = list(true_actions)
                random.shuffle(order)
                if order not in orders:
                    orders.append(order)
            for order in orders:
                selected = []
                for step in range(len(order) + 1):
                    states.append(build_state_vector(self.env, symptoms, selected))
                    targets.append(self._remaining_target(selected, true_actions))
                    if step < len(order):
                        selected.append(order[step])
        state_tensor = torch.tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        target_tensor = torch.tensor(np.asarray(targets), dtype=torch.float32, device=self.device)
        optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, weight_decay=self.optimizer.param_groups[0]['weight_decay'])
        self.logger.info(f'开始监督预训练: epochs={epochs}, orders={num_orders}, loss={self.supervised_loss_name}')
        self.policy_net.train()
        batch_size = min(256, len(state_tensor))
        for epoch in range(int(epochs)):
            order = torch.randperm(len(state_tensor), device=self.device)
            loss_sum = 0.0
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                with self._autocast():
                    logits = self.policy_net(state_tensor[indices])
                    loss = self._loss(logits, target_tensor[indices], self.pos_weight)
                self._optimizer_step(loss, optimizer)
                loss_sum += loss.item() * len(indices)
            self.logger.info(f'监督预训练 epoch {epoch + 1}/{epochs}, loss={loss_sum / len(state_tensor):.6f}')
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def estimate_expert_transitions(self, num_orders=1):
        return sum((len(labels) + 1) * max(1, int(num_orders)) for _, labels in self.train_data)

    def _rare_multiplier(self, action, true_actions):
        if not (0 <= action < self.env.Se_action_num) or action not in set(true_actions):
            return 1.0
        return 1.0 + self.rare_per_scale * max(0.0, self.env.Se_weights.get(action, 1.0) - 1.0)

    def _store_trajectory(self, trajectory, is_expert):
        added = 0
        for start in range(len(trajectory)):
            state, action, _, _, _, aux_target, true_actions = trajectory[start]
            total_reward, discount, next_state, next_selected = 0.0, 1.0, None, []
            for offset in range(self.n_step):
                index = start + offset
                if index >= len(trajectory):
                    break
                _, _, reward, candidate_next, candidate_selected, _, _ = trajectory[index]
                total_reward += discount * reward
                next_state, next_selected = candidate_next, candidate_selected
                discount *= self.gamma
                if candidate_next is None:
                    discount = 0.0
                    break
            action_index = int(action.item())
            multiplier = self._rare_multiplier(action_index, true_actions)
            priority = abs(total_reward) * multiplier + 1.0
            self.memory.push(
                state, action,
                torch.tensor([total_reward], dtype=torch.float32, device=self.device),
                next_state, next_selected,
                torch.tensor([discount], dtype=torch.float32, device=self.device),
                torch.as_tensor(aux_target, dtype=torch.float32, device=self.device).unsqueeze(0),
                multiplier, is_expert, priority=priority,
            )
            added += 1
        return added

    def warmup_replay_with_expert(self, num_orders=1):
        expected = self.estimate_expert_transitions(num_orders)
        if self.memory.capacity < expected:
            raise ValueError(f'Replay capacity={self.memory.capacity} 小于专家轨迹预估={expected}；请增大容量或减少 orders')
        added = 0
        for symptoms, label_names in self.train_data:
            true_actions = [self.env.label_to_action[label] for label in label_names]
            orders = [list(true_actions)]
            for _ in range(max(0, int(num_orders) - 1)):
                order = list(true_actions)
                random.shuffle(order)
                if order not in orders:
                    orders.append(order)
            for order in orders:
                self.env.reset((symptoms, label_names))
                selected, trajectory = [], []
                for action_index in order:
                    state = torch.tensor(build_state_vector(self.env, symptoms, selected), dtype=torch.float32, device=self.device).unsqueeze(0)
                    action = torch.tensor([[action_index]], dtype=torch.long, device=self.device)
                    observation, reward, exhausted = self.env.step(action_index, selected)
                    selected.append(action_index)
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                    trajectory.append((state, action, reward, next_state, selected.copy(), self._remaining_target(selected[:-1], true_actions), true_actions))
                    if exhausted:
                        break
                stop_state = torch.tensor(build_state_vector(self.env, symptoms, selected), dtype=torch.float32, device=self.device).unsqueeze(0)
                stop_action = torch.tensor([[self.env.stop_action]], dtype=torch.long, device=self.device)
                _, stop_reward, _ = self.env.step(self.env.stop_action, selected)
                trajectory.append((stop_state, stop_action, stop_reward, None, selected.copy(), self._remaining_target(selected, true_actions), true_actions))
                added += self._store_trajectory(trajectory, is_expert=True)
        self.logger.info(f'专家轨迹预填充: expected={expected}, added={added}, replay={self.memory.summary()}')

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None
        if self.memory.prioritized:
            transitions, indices, weights = self.memory.sample(self.batch_size, self._current_per_beta())
            sample_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            transitions = self.memory.sample(self.batch_size)
            indices = None
            sample_weights = torch.ones(len(transitions), dtype=torch.float32, device=self.device)
        batch = Transition(*zip(*transitions))
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        discounts = torch.cat(batch.discount)
        with self._autocast():
            q_values = self.policy_net(states).gather(1, actions).squeeze(1)
            # AMP 下网络输出是 FP16，但奖励、折扣和 TD 目标保持 FP32。
            next_values = torch.zeros(len(transitions), dtype=torch.float32, device=self.device)
            valid_indices = [index for index, state in enumerate(batch.next_state) if state is not None]
            if valid_indices:
                next_states = torch.cat([batch.next_state[index] for index in valid_indices])
                next_selected = [batch.next_selected_actions[index] for index in valid_indices]
                with torch.no_grad():
                    online_q = legal_action_mask(self.policy_net(next_states), next_selected, self.env, self.min_actions_before_stop)
                    next_actions = online_q.max(1).indices.view(-1, 1)
                    target_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
                    valid_index_tensor = torch.tensor(valid_indices, dtype=torch.long, device=self.device)
                    next_values[valid_index_tensor] = target_q.float()
            targets = rewards + discounts * next_values
            loss = (nn.functional.smooth_l1_loss(q_values, targets, reduction='none') * sample_weights).mean()
            aux_loss = None
            if self.use_aux_bce:
                aux_targets = torch.cat(batch.aux_target)
                aux_loss = self._loss(self.policy_net(states), aux_targets, self.pos_weight)
                loss = loss + self.aux_bce_weight * aux_loss
        td_errors = targets.detach() - q_values.detach()
        self._optimizer_step(loss, self.optimizer)
        if self.memory.prioritized:
            self.memory.update_priorities(indices, td_errors.abs().detach().cpu().numpy(), batch.rare_multiplier)
        self.optimization_steps += 1
        self._soft_update_target()
        return float(loss.item()), None if aux_loss is None else float(aux_loss.item())

    def _online_trajectory(self, symptoms, label_names):
        self.env.reset((symptoms, label_names))
        true_actions = [self.env.label_to_action[label] for label in label_names]
        selected, trajectory = [], []
        reward_sum = 0.0
        for _ in range(self.env.Se_action_num + 1):
            state = torch.tensor(build_state_vector(self.env, symptoms, selected), dtype=torch.float32, device=self.device).unsqueeze(0)
            action, actor, epsilon = self.action_selector.select_action(state, selected)
            action_index = int(action.item())
            observation, reward, exhausted = self.env.step(action_index, selected)
            reward_sum += reward
            if action_index == self.env.stop_action:
                trajectory.append((state, action, reward, None, selected.copy(), self._remaining_target(selected, true_actions), true_actions))
                break
            next_selected = selected + [action_index]
            next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            trajectory.append((state, action, reward, next_state, next_selected.copy(), self._remaining_target(selected, true_actions), true_actions))
            selected = next_selected
            if exhausted:
                stop_state = next_state
                stop_action = torch.tensor([[self.env.stop_action]], dtype=torch.long, device=self.device)
                _, stop_reward, _ = self.env.step(self.env.stop_action, selected)
                trajectory.append((stop_state, stop_action, stop_reward, None, selected.copy(), self._remaining_target(selected, true_actions), true_actions))
                reward_sum += stop_reward
                break
        return trajectory, reward_sum, len(selected)

    def train_epoch(self):
        self.policy_net.train()
        shuffled = list(self.train_data)
        random.shuffle(shuffled)
        rewards, losses, selected_counts = [], [], []
        for symptoms, labels in shuffled:
            trajectory, reward, selected_count = self._online_trajectory(symptoms, labels)
            self._store_trajectory(trajectory, is_expert=False)
            rewards.append(reward)
            selected_counts.append(selected_count)
            updates = max(1, len(trajectory) // 2)
            for _ in range(updates):
                result = self.optimize_model()
                if result is not None:
                    losses.append(result[0])
        return {
            'avg_reward': float(np.mean(rewards)) if rewards else 0.0,
            'avg_loss': float(np.mean(losses)) if losses else 0.0,
            'avg_selected': float(np.mean(selected_counts)) if selected_counts else 0.0,
            'replay_size': len(self.memory),
        }

    def evaluate(self, data, dataset_name):
        was_training = self.policy_net.training
        self.policy_net.eval()
        metrics = evaluate(self.env, self.policy_net, self.device, data, self.logger, dataset_name,
                           self.stop_margin_threshold, self.min_actions_before_stop)
        if was_training:
            self.policy_net.train()
        return metrics

    def fit(self, num_episodes, patience, min_delta, selection_metric='supported_macro_f1', checkpoint_callback=None):
        best_value, best_state, best_epoch, stalled, history = -float('inf'), None, -1, 0, []
        start = time.perf_counter()
        self.action_selector.reset_steps()
        for epoch in range(int(num_episodes)):
            stats = self.train_epoch()
            validation = self.evaluate(self.validation_data, '验证集')['auto']
            value = float(validation.get(selection_metric, validation['sample_f1']))
            improved = value > best_value + float(min_delta)
            if improved:
                best_value, best_state, best_epoch, stalled = value, copy.deepcopy(self.policy_net.state_dict()), epoch, 0
                if checkpoint_callback is not None:
                    checkpoint_callback('best', epoch, validation)
            else:
                stalled += 1
            history.append({'epoch': epoch, **stats, 'validation': validation, 'selected_metric': value, 'improved': improved})
            progress = f'stalled={stalled}/{patience}' if patience > 0 else 'early_stopping=off'
            self.logger.info(f'epoch={epoch} reward={stats["avg_reward"]:.4f} loss={stats["avg_loss"]:.6f} '
                             f'val_{selection_metric}={value:.4f} best={best_value:.4f} {progress}')
            if checkpoint_callback is not None:
                checkpoint_callback('last', epoch, validation)
            if patience > 0 and stalled >= int(patience):
                self.logger.info(f'验证早停: best_epoch={best_epoch}, {selection_metric}={best_value:.4f}')
                break
        if best_state is not None:
            self.policy_net.load_state_dict(best_state)
            self.target_net.load_state_dict(best_state)
        self.logger.info(f'训练完成: elapsed={time.perf_counter() - start:.2f}s, best_epoch={best_epoch}')
        return {'best_epoch': best_epoch, 'best_value': best_value, 'history': history}

    def training_state(self):
        """返回可写入 ``last.pt`` 的训练器状态。

        replay 中的 transition 已经是 CPU/GPU tensor；torch.save 可以直接序列化它。
        该状态只服务于恢复训练，推理始终使用轻量的 ``best.pt``。
        """
        return {
            'memory': self.memory,
            'action_selector_steps': self.action_selector.steps_done,
            'optimization_steps': self.optimization_steps,
            'target_update_count': self.target_update_count,
            'grad_scaler_state': self.grad_scaler.state_dict() if self.use_amp else None,
        }

    def restore_training_state(self, state):
        """从 ``last.pt`` 恢复回放池及优化计数器。"""
        if not state:
            return
        if state.get('memory') is not None:
            self.memory = state['memory']
        self.action_selector.steps_done = int(state.get('action_selector_steps', 0))
        self.optimization_steps = int(state.get('optimization_steps', 0))
        self.target_update_count = int(state.get('target_update_count', 0))
        if self.use_amp and state.get('grad_scaler_state'):
            self.grad_scaler.load_state_dict(state['grad_scaler_state'])

    def inference_actions(self, symptoms):
        state = build_state_vector(self.env, symptoms)
        actions = predict_actions_from_state(self.env, self.policy_net, self.device, state,
                                             stop_margin_threshold=self.stop_margin_threshold,
                                             min_actions=self.min_actions_before_stop)
        return [self.env.action_space[action] for action in actions]
