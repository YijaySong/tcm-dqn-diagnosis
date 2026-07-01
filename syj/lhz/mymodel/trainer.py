# -*- coding: utf-8 -*-
"""DQN训练流程文件。

封装监督预训练、专家轨迹预填充ReplayMemory、DQN主训练循环、测试集评估入口、
预测接口和模型checkpoint保存逻辑。
"""

import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from action import ActionSelector, mask_selected_actions
from evaluation import evaluate
from memory import Transition
from state import build_state_vector, predict_actions_from_state


class DQNTrainer(object):
    def __init__(
        self, env, training_tcm_data, test_tcm_data, policy_net, target_net, memory,
        optimizer, device, logger, batch_size, gamma, tau, weight_decay,
        eps_start, eps_end, eps_decay, n_actions,
        aux_supervised_weight=0.05, pretrain_all_permutations=1,
        target_update_strategy='soft', target_update_interval=1,
        per_beta_start=0.4, per_beta_frames=5000, priority_clip=10.0,
        optimize_interval=5, n_step=3, aux_supervised_start=0.5,
        aux_pos_weight_max=20.0, pretrain_anchor_weight=1e-4,
        use_curriculum=True, curriculum_stage1=0.3, curriculum_stage2=0.6,
        min_actions_before_stop=1, stop_margin_threshold=None, rare_priority_scale=0.5,
    ):
        self.env = env
        self.training_tcm_data = training_tcm_data
        self.test_tcm_data = test_tcm_data
        self.policy_net = policy_net
        self.target_net = target_net
        self.memory = memory
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.weight_decay = weight_decay
        self.n_actions = n_actions
        self.aux_supervised_weight_end = aux_supervised_weight
        self.aux_supervised_weight_start = aux_supervised_start
        self.pretrain_all_permutations = pretrain_all_permutations
        self.target_update_strategy = target_update_strategy
        self.target_update_interval = max(1, target_update_interval)
        self.per_beta_start = per_beta_start
        self.per_beta_frames = max(1, per_beta_frames)
        self.priority_clip = priority_clip
        self.optimize_interval = max(1, optimize_interval)
        self.n_step = max(1, n_step)
        self.aux_pos_weight_max = aux_pos_weight_max
        self.pretrain_anchor_weight = pretrain_anchor_weight
        self.use_curriculum = bool(use_curriculum)
        self.curriculum_stage1 = curriculum_stage1
        self.curriculum_stage2 = curriculum_stage2
        self.min_actions_before_stop = min_actions_before_stop
        self.stop_margin_threshold = stop_margin_threshold
        self.rare_priority_scale = rare_priority_scale
        self.training_progress = 0.0
        self.optimization_steps = 0
        self.target_update_count = 0
        self.pretrain_anchor = None

        for parameter in self.target_net.parameters():
            parameter.requires_grad_(False)
        self.target_net.eval()

        self.symptom_to_true_actions = defaultdict(list)
        action_counts = np.ones(self.n_actions, dtype=np.float32)
        for symptoms, true_Se_names in self.training_tcm_data:
            key = tuple(sorted(symptoms))
            actions = tuple(sorted(self.env.swapped_action_space[name] for name in true_Se_names))
            if actions not in self.symptom_to_true_actions[key]:
                self.symptom_to_true_actions[key].append(actions)
            for action in actions:
                action_counts[action] += 1.0
        action_counts[self.env.stop_action] = max(len(self.training_tcm_data), 1)
        sample_count = max(len(self.training_tcm_data), 1)
        neg_counts = np.maximum(sample_count - action_counts, 1.0)
        aux_pos_weight = np.sqrt(neg_counts / np.maximum(action_counts, 1.0))
        aux_pos_weight = np.clip(aux_pos_weight, 1.0, self.aux_pos_weight_max)
        aux_pos_weight[self.env.stop_action] = 1.0
        self.aux_pos_weight = torch.tensor(aux_pos_weight, dtype=torch.float32, device=self.device)

        self.action_selector = ActionSelector(
            env, policy_net, n_actions, device, eps_start, eps_end, eps_decay,
            min_actions_before_stop=min_actions_before_stop
        )

    def _current_per_beta(self):
        progress = min(1.0, self.optimization_steps / float(self.per_beta_frames))
        return self.per_beta_start + progress * (1.0 - self.per_beta_start)

    def current_aux_weight(self):
        return self.aux_supervised_weight_start * (1.0 - self.training_progress) + self.aux_supervised_weight_end * self.training_progress

    def _soft_update_target(self):
        with torch.no_grad():
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(policy_param.data, alpha=self.tau)
        self.target_update_count += 1

    def _hard_update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_update_count += 1

    def maybe_update_target_network(self):
        if self.target_update_strategy == 'soft':
            self._soft_update_target()
            return
        if self.target_update_strategy == 'hard':
            if self.optimization_steps % self.target_update_interval == 0:
                self._hard_update_target()
            return
        raise ValueError(f'未知目标网络更新策略: {self.target_update_strategy}')

    def _remaining_target_vector(self, selected_actions, true_actions):
        target = np.zeros(self.n_actions, dtype=np.float32)
        remaining = [action for action in true_actions if action not in selected_actions]
        if remaining:
            target[remaining] = 1.0
        else:
            target[self.env.stop_action] = 1.0
        return target

    def supervised_auxiliary_loss(self, state_batch):
        if self.current_aux_weight() <= 0:
            return None
        states_np = state_batch.detach().cpu().numpy()
        targets = np.zeros((states_np.shape[0], self.n_actions), dtype=np.float32)
        has_target = np.zeros(states_np.shape[0], dtype=bool)
        for row_idx, state_np in enumerate(states_np):
            symptoms = [
                self.env.state_space[idx]
                for idx in range(self.env.symp_len)
                if state_np[idx] > 0.5
            ]
            selected_actions = [
                idx for idx in range(self.env.Se_action_num)
                if state_np[self.env.symp_len + idx] > 0.5
            ]
            true_action_sets = self.symptom_to_true_actions.get(tuple(sorted(symptoms)))
            if not true_action_sets:
                continue
            target_sum = np.zeros(self.n_actions, dtype=np.float32)
            for true_actions in true_action_sets:
                target_sum += self._remaining_target_vector(selected_actions, true_actions)
            targets[row_idx] = target_sum / len(true_action_sets)
            has_target[row_idx] = True
        if not has_target.any():
            return None
        valid_mask = torch.tensor(has_target, dtype=torch.bool, device=self.device)
        target_tensor = torch.tensor(targets[has_target], dtype=torch.float32, device=self.device)
        logits = self.policy_net(state_batch[valid_mask])
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.aux_pos_weight)
        return criterion(logits, target_tensor)

    def capture_pretrain_anchor(self):
        if self.pretrain_anchor_weight <= 0:
            self.pretrain_anchor = None
            return
        self.pretrain_anchor = {
            name: parameter.detach().clone()
            for name, parameter in self.policy_net.named_parameters()
            if parameter.requires_grad
        }

    def pretrained_anchor_loss(self):
        if not self.pretrain_anchor or self.pretrain_anchor_weight <= 0:
            return None
        loss = torch.zeros((), dtype=torch.float32, device=self.device)
        for name, parameter in self.policy_net.named_parameters():
            anchor = self.pretrain_anchor.get(name)
            if anchor is not None:
                loss = loss + (parameter - anchor).pow(2).mean()
        return loss

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None

        if getattr(self.memory, 'prioritized', False):
            transitions, indices, weights_np = self.memory.sample(self.batch_size, beta=self._current_per_beta())
            sample_weights = torch.tensor(weights_np, dtype=torch.float32, device=self.device)
        else:
            transitions = self.memory.sample(self.batch_size)
            indices = None
            sample_weights = torch.ones(len(transitions), dtype=torch.float32, device=self.device)

        actual_batch_size = len(transitions)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_indices = [idx for idx, state in enumerate(batch.next_state) if state is not None]
        non_final_next_states = [batch.next_state[idx] for idx in non_final_indices]

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        discount_batch = torch.cat(batch.discount)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        next_state_values = torch.zeros(actual_batch_size, device=self.device)
        with torch.no_grad():
            if non_final_next_states:
                next_states = torch.cat(non_final_next_states)
                next_invalid_actions = [batch.next_invalid_actions[idx] for idx in non_final_indices]
                next_policy_q = mask_selected_actions(self.policy_net(next_states), next_invalid_actions, self.env)
                next_actions = next_policy_q.max(1).indices.view(-1, 1)
                next_target_q = self.target_net(next_states)
                next_values = next_target_q.gather(1, next_actions).squeeze(1)
                next_state_values[non_final_mask] = next_values
        expected_state_action_values = reward_batch + discount_batch * next_state_values

        td_errors = expected_state_action_values.detach() - state_action_values.detach()
        criterion = nn.SmoothL1Loss(reduction='none')
        td_loss = criterion(state_action_values, expected_state_action_values)
        loss = (td_loss * sample_weights).mean()

        aux_loss = self.supervised_auxiliary_loss(state_batch)
        aux_weight = self.current_aux_weight()
        if aux_loss is not None and aux_weight > 0:
            loss = loss + aux_weight * aux_loss

        anchor_loss = self.pretrained_anchor_loss()
        anchor_weight = self.pretrain_anchor_weight * (1.0 - 0.5 * self.training_progress)
        if anchor_loss is not None and anchor_weight > 0:
            loss = loss + anchor_weight * anchor_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        if getattr(self.memory, 'prioritized', False):
            priorities = td_errors.abs().cpu().numpy()
            if self.priority_clip and self.priority_clip > 0:
                priorities = np.clip(priorities, 0.0, self.priority_clip)
            self.memory.update_priorities(indices, priorities)

        self.optimization_steps += 1
        return loss.item()

    def pretrain_supervised(self, pretrain_epochs, pretrain_lr):
        if pretrain_epochs <= 0:
            return

        self.logger.info(f"开始监督预训练: epochs={pretrain_epochs}, lr={pretrain_lr}")
        states = []
        targets = []

        for symptoms, true_Se_names in self.training_tcm_data:
            true_actions = [self.env.swapped_action_space[name] for name in true_Se_names]
            trajectories = [list(true_actions)]
            for _ in range(max(0, self.pretrain_all_permutations - 1)):
                shuffled = list(true_actions)
                random.shuffle(shuffled)
                trajectories.append(shuffled)

            for trajectory in trajectories:
                selected_actions = []
                for step_idx in range(len(trajectory) + 1):
                    target = np.zeros(self.n_actions, dtype=np.float32)
                    remaining_actions = [action for action in true_actions if action not in selected_actions]
                    if remaining_actions:
                        target[remaining_actions] = 1.0
                    else:
                        target[self.env.stop_action] = 1.0
                    states.append(build_state_vector(self.env, symptoms, selected_actions))
                    targets.append(target)
                    if step_idx < len(trajectory):
                        selected_actions.append(trajectory[step_idx])

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        targets = torch.tensor(np.array(targets), dtype=torch.float32, device=self.device)

        pos_count = targets.sum(dim=0)
        neg_count = targets.shape[0] - pos_count
        pos_weight = (neg_count / torch.clamp(pos_count, min=1.0)).sqrt().clamp(min=1.0, max=self.aux_pos_weight_max)
        pos_weight[self.env.stop_action] = 1.0

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        pretrain_optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=pretrain_lr,
            weight_decay=self.weight_decay, amsgrad=True
        )
        batch_size = min(256, len(states))

        self.policy_net.train()
        self.target_net.eval()
        for epoch in range(pretrain_epochs):
            permutation = torch.randperm(len(states), device=self.device)
            total_loss = 0.0
            for start in range(0, len(states), batch_size):
                indices = permutation[start:start + batch_size]
                batch_states = states[indices]
                batch_targets = targets[indices]
                logits = self.policy_net(batch_states)
                loss = criterion(logits, batch_targets)
                pretrain_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
                pretrain_optimizer.step()
                total_loss += loss.item() * len(indices)
            self.logger.info(f"监督预训练 epoch {epoch + 1}/{pretrain_epochs}, loss={total_loss / len(states):.6f}")

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.capture_pretrain_anchor()

    def _store_n_step_trajectory(self, trajectory):
        added = 0
        for start_idx, transition in enumerate(trajectory):
            state, action, _, _, _, action_idx = transition
            total_reward = 0.0
            discount = 1.0
            bootstrap_discount = 0.0
            next_state = None
            next_invalid_actions = transition[4]
            for offset in range(self.n_step):
                idx = start_idx + offset
                if idx >= len(trajectory):
                    break
                _, _, reward, candidate_next_state, candidate_invalid_actions, _ = trajectory[idx]
                total_reward += discount * reward
                next_state = candidate_next_state
                next_invalid_actions = candidate_invalid_actions
                discount *= self.gamma
                if candidate_next_state is None:
                    bootstrap_discount = 0.0
                    break
                bootstrap_discount = discount

            reward_tensor = torch.tensor([total_reward], dtype=torch.float32, device=self.device)
            discount_tensor = torch.tensor([bootstrap_discount], dtype=torch.float32, device=self.device)
            rare_bonus = 0.0
            if 0 <= action_idx < self.env.Se_action_num:
                rare_bonus = max(0.0, self.env.Se_weights.get(action_idx, 1.0) - 1.0) * self.rare_priority_scale
            priority = abs(total_reward) + 1.0 + rare_bonus
            self.memory.push(state, action, reward_tensor, next_state, next_invalid_actions, discount_tensor, priority=priority)
            added += 1
        return added

    def warmup_replay_with_expert(self):
        self.logger.info(f"开始使用专家轨迹预填充ReplayMemory: n_step={self.n_step}, permutations={max(1, self.pretrain_all_permutations)}")
        added = 0
        for data_piece in self.training_tcm_data:
            symptoms, true_Se_names = data_piece
            true_actions = [self.env.swapped_action_space[name] for name in true_Se_names]
            trajectories = [list(true_actions)]
            for _ in range(max(0, self.pretrain_all_permutations - 1)):
                shuffled = list(true_actions)
                random.shuffle(shuffled)
                trajectories.append(shuffled)

            for expert_actions in trajectories:
                self.env.reset(data_piece)
                selected_actions = []
                transition_buffer = []

                for action_idx in expert_actions:
                    state_np = build_state_vector(self.env, symptoms, selected_actions)
                    state = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action = torch.tensor([[action_idx]], device=self.device, dtype=torch.long)
                    observation, reward, terminated = self.env.step(action_idx, selected_actions)
                    selected_actions.append(action_idx)
                    next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                    transition_buffer.append((state, action, reward, next_state, selected_actions.copy(), action_idx))
                    if terminated:
                        break

                if len(selected_actions) < self.env.Se_action_num:
                    state_np = build_state_vector(self.env, symptoms, selected_actions)
                    state = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action = torch.tensor([[self.env.stop_action]], device=self.device, dtype=torch.long)
                    _, reward, _ = self.env.step(self.env.stop_action, selected_actions)
                    transition_buffer.append((state, action, reward, None, selected_actions.copy(), self.env.stop_action))

                added += self._store_n_step_trajectory(transition_buffer)

        self.logger.info(f"ReplayMemory预填充完成，新增transition数量: {added}, 当前容量: {len(self.memory)}")

    def _curriculum_episode_data(self, episode_idx, num_episodes):
        if not self.use_curriculum or num_episodes <= 1:
            return list(self.training_tcm_data), 'all'
        progress = episode_idx / max(num_episodes - 1, 1)
        if progress < self.curriculum_stage1:
            max_len = 2
            stage = 'len<=2'
        elif progress < self.curriculum_stage2:
            max_len = 4
            stage = 'len<=4'
        else:
            max_len = None
            stage = 'all'
        if max_len is None:
            return list(self.training_tcm_data), stage
        subset = [item for item in self.training_tcm_data if len(item[1]) <= max_len]
        return (subset if subset else list(self.training_tcm_data)), stage

    def train(self, num_episodes):
        self.logger.info(f"总的迭代次数: {num_episodes}")

        self.action_selector.reset_steps()
        st = time.perf_counter()

        for i_episode in range(num_episodes):
            self.training_progress = i_episode / max(num_episodes - 1, 1) if num_episodes > 1 else 1.0
            episode_data, curriculum_stage = self._curriculum_episode_data(i_episode, num_episodes)
            self.logger.info(
                f"第 {i_episode} 次迭代开始... curriculum={curriculum_stage}, "
                f"samples={len(episode_data)}, aux_weight={self.current_aux_weight():.4f}"
            )

            iter_Se = 0
            episode_rewards_list = []
            loss_list = []
            agent_count = 0
            random_count = 0
            stop_count = 0
            selected_count_list = []
            target_updates_before = self.target_update_count

            epi_st = time.perf_counter()
            random.shuffle(episode_data)
            self.policy_net.train()
            self.target_net.eval()

            for data_piece in episode_data:
                state_np = self.env.reset(data_piece)
                state = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)

                sample_rewards = 0.0
                selected_actions = []
                transition_buffer = []
                for _ in range(self.env.Se_action_num + 1):
                    action, actor, _ = self.action_selector.select_action(state, selected_actions)
                    action_idx = action.item()

                    observation, reward, terminated = self.env.step(action_idx, selected_actions)
                    if action_idx == self.env.stop_action:
                        stop_count += 1
                    else:
                        selected_actions.append(action_idx)

                    if actor == "agent":
                        agent_count += 1
                    elif actor == "random":
                        random_count += 1
                    sample_rewards += reward

                    next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                    transition_buffer.append((state, action, reward, next_state, selected_actions.copy(), action_idx))
                    state = next_state

                    if terminated:
                        break

                added = self._store_n_step_trajectory(transition_buffer)
                iter_Se += added
                selected_count_list.append(len(selected_actions))
                while len(self.memory) >= self.batch_size and iter_Se >= self.optimize_interval:
                    loss = self.optimize_model()
                    if loss is not None:
                        loss_list.append(loss)
                        self.maybe_update_target_network()
                    iter_Se -= self.optimize_interval

                episode_rewards_list.append(sample_rewards)

            epi_et = time.perf_counter()

            avg_loss = float(np.mean(loss_list)) if loss_list else 0.0
            avg_reward = float(np.mean(episode_rewards_list)) if episode_rewards_list else 0.0
            avg_selected = float(np.mean(selected_count_list)) if selected_count_list else 0.0
            target_updates = self.target_update_count - target_updates_before
            self.logger.info(
                f"第 {i_episode} 次迭代训练统计: avg_reward={avg_reward:.4f}, "
                f"avg_loss={avg_loss:.6f}, avg_selected={avg_selected:.4f}, "
                f"agent_actions={agent_count}, random_actions={random_count}, "
                f"stop_actions={stop_count}, optimize_steps={self.optimization_steps}, "
                f"target_updates={target_updates}"
            )
            self.logger.info(f"第 {i_episode} 次迭代的训练时间: {epi_et-epi_st:.6f}秒")
            self.logger.info(f"第 {i_episode} 次迭代完成。")

        et = time.perf_counter()
        self.training_progress = 1.0
        self.logger.info(f"训练时间总共: {et-st:.6f}秒")
        self.logger.info("训练完成，测试集评估将在训练结束后单独执行")

    def evaluate(self, eval_data=None, dataset_name="测试集"):
        was_training = self.policy_net.training
        self.policy_net.eval()
        self.target_net.eval()
        metrics = evaluate(
            self.env, self.policy_net, self.action_selector,
            self.device, self.test_tcm_data if eval_data is None else eval_data,
            self.logger, dataset_name=dataset_name,
            stop_margin_threshold=self.stop_margin_threshold,
            min_actions=self.min_actions_before_stop,
        )
        if was_training:
            self.policy_net.train()
            self.target_net.eval()
        return metrics

    def predict_symptoms(self, symptoms_str, max_actions=None, force_top_k=None):
        state_vector = np.zeros(len(self.env.state_space), dtype=np.float32)
        for symptom in symptoms_str.split(','):
            symptom = symptom.strip()
            if symptom in self.env.swapped_state_space:
                state_vector[self.env.swapped_state_space[symptom]] = 1

        selected_actions = predict_actions_from_state(
            self.env, self.policy_net, self.action_selector, self.device,
            state_vector, force_top_k=force_top_k, max_actions=max_actions,
            stop_margin_threshold=self.stop_margin_threshold,
            min_actions=self.min_actions_before_stop,
        )
        return [self.env.action_space[action_idx] for action_idx in selected_actions]


def save_checkpoint(model_dir, policy_net, symptoms, Se, nn_units, nn_units2, dropout,
                    state_vector_len, n_actions, seed, test_ratio, test_metrics,
                    model_type='dqn', reward_config=None, training_config=None):
    checkpoint = {
        'checkpoint_version': 2,
        'model_state_dict': policy_net.state_dict(),
        'symptoms': symptoms,
        'Se': Se,
        'nn_units': nn_units,
        'nn_units2': nn_units2,
        'dropout': dropout,
        'state_vector_len': state_vector_len,
        'n_actions': n_actions,
        'model_type': model_type,
        'seed': seed,
        'test_ratio': test_ratio,
        'test_metrics': test_metrics,
    }
    if reward_config is not None:
        checkpoint['reward_config'] = reward_config
    if training_config is not None:
        checkpoint['training_config'] = training_config
    model_path = os.path.join(model_dir, 'dqn_model.pth')
    torch.save(checkpoint, model_path)
    return model_path
