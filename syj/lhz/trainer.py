# -*- coding: utf-8 -*-
"""DQN训练流程文件。

封装监督预训练、专家轨迹预填充ReplayMemory、DQN主训练循环、测试集评估入口、
预测接口和模型checkpoint保存逻辑。
"""

import os
import random
import time

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
        eps_start, eps_end, eps_decay, n_actions
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
        self.action_selector = ActionSelector(
            env, policy_net, n_actions, device, eps_start, eps_end, eps_decay
        )

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_indices = [idx for idx, state in enumerate(batch.next_state) if state is not None]
        non_final_next_states = [batch.next_state[idx] for idx in non_final_indices]

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            if non_final_next_states:
                next_states = torch.cat(non_final_next_states)
                next_invalid_actions = [batch.next_invalid_actions[idx] for idx in non_final_indices]
                next_policy_q = mask_selected_actions(self.policy_net(next_states), next_invalid_actions, self.env)
                next_actions = next_policy_q.max(1).indices.view(-1, 1)
                next_target_q = self.target_net(next_states)
                next_values = next_target_q.gather(1, next_actions).squeeze(1)
                next_state_values[non_final_mask] = next_values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()
        return loss.item()

    def pretrain_supervised(self, pretrain_epochs, pretrain_lr):
        if pretrain_epochs <= 0:
            return

        self.logger.info(f"开始监督预训练: epochs={pretrain_epochs}, lr={pretrain_lr}")
        states = []
        targets = []

        for symptoms, true_Se_names in self.training_tcm_data:
            true_actions = [self.env.swapped_action_space[name] for name in true_Se_names]

            target = np.zeros(self.n_actions, dtype=np.float32)
            target[true_actions] = 1.0
            states.append(build_state_vector(self.env, symptoms))
            targets.append(target)

            for selected_action in true_actions:
                remaining_actions = [action for action in true_actions if action != selected_action]
                target = np.zeros(self.n_actions, dtype=np.float32)
                target[remaining_actions] = 1.0
                states.append(build_state_vector(self.env, symptoms, [selected_action]))
                targets.append(target)

            target = np.zeros(self.n_actions, dtype=np.float32)
            target[self.env.stop_action] = 1.0
            states.append(build_state_vector(self.env, symptoms, true_actions))
            targets.append(target)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        targets = torch.tensor(np.array(targets), dtype=torch.float32, device=self.device)

        pos_count = targets.sum(dim=0)
        neg_count = targets.shape[0] - pos_count
        pos_weight = (neg_count / torch.clamp(pos_count, min=1.0)).clamp(min=1.0, max=10.0)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        pretrain_optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=pretrain_lr,
            weight_decay=self.weight_decay, amsgrad=True
        )
        batch_size = min(256, len(states))

        self.policy_net.train()
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

    def warmup_replay_with_expert(self):
        self.logger.info("开始使用专家轨迹预填充ReplayMemory")
        added = 0
        for data_piece in self.training_tcm_data:
            self.env.reset(data_piece)
            symptoms, true_Se_names = data_piece
            true_actions = [self.env.swapped_action_space[name] for name in true_Se_names]
            random.shuffle(true_actions)
            selected_actions = []

            for action_idx in true_actions:
                state_np = build_state_vector(self.env, symptoms, selected_actions)
                state = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                action = torch.tensor([[action_idx]], device=self.device, dtype=torch.long)
                observation, reward, terminated = self.env.step(action_idx, selected_actions)
                selected_actions.append(action_idx)
                reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
                next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                self.memory.push(state, action, reward_tensor, next_state, selected_actions.copy())
                added += 1
                if terminated:
                    break

            if len(selected_actions) < self.env.Se_action_num:
                state_np = build_state_vector(self.env, symptoms, selected_actions)
                state = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                action = torch.tensor([[self.env.stop_action]], device=self.device, dtype=torch.long)
                _, reward, _ = self.env.step(self.env.stop_action, selected_actions)
                reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
                self.memory.push(state, action, reward_tensor, None, selected_actions.copy())
                added += 1

        self.logger.info(f"ReplayMemory预填充完成，新增transition数量: {added}, 当前容量: {len(self.memory)}")

    def train(self, num_episodes):
        self.logger.info(f"总的迭代次数: {num_episodes}")

        self.action_selector.reset_steps()
        st = time.perf_counter()

        for i_episode in range(num_episodes):
            self.logger.info(f"第 {i_episode} 次迭代开始...")

            iter_Se = 0
            episode_rewards_list = []
            loss_list = []
            agent_count = 0
            random_count = 0
            stop_count = 0

            epi_st = time.perf_counter()
            random.shuffle(self.training_tcm_data)
            self.policy_net.train()

            for data_piece in self.training_tcm_data:
                state_np = self.env.reset(data_piece)
                state = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)

                sample_rewards = 0.0
                selected_actions = []
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

                    reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)

                    next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                    self.memory.push(state, action, reward_tensor, next_state, selected_actions.copy())
                    state = next_state
                    iter_Se += 1

                    if len(self.memory) >= self.batch_size and iter_Se >= 5:
                        loss = self.optimize_model()
                        if loss is not None:
                            loss_list.append(loss)
                        target_net_state_dict = self.target_net.state_dict()
                        policy_net_state_dict = self.policy_net.state_dict()
                        for key in policy_net_state_dict:
                            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                        self.target_net.load_state_dict(target_net_state_dict)
                        iter_Se = 0

                    if terminated:
                        break

                episode_rewards_list.append(sample_rewards)

            epi_et = time.perf_counter()

            avg_loss = float(np.mean(loss_list)) if loss_list else 0.0
            avg_reward = float(np.mean(episode_rewards_list)) if episode_rewards_list else 0.0
            self.logger.info(f"第 {i_episode} 次迭代训练统计: avg_reward={avg_reward:.4f}, avg_loss={avg_loss:.6f}, agent_actions={agent_count}, random_actions={random_count}, stop_actions={stop_count}")
            self.logger.info(f"第 {i_episode} 次迭代的训练时间: {epi_et-epi_st:.6f}秒")
            self.logger.info(f"第 {i_episode} 次迭代完成。")

        et = time.perf_counter()
        self.logger.info(f"训练时间总共: {et-st:.6f}秒")
        self.logger.info("训练完成，测试集评估将在训练结束后单独执行")

    def evaluate(self, eval_data=None, dataset_name="测试集"):
        return evaluate(
            self.env, self.policy_net, self.action_selector,
            self.device, self.test_tcm_data if eval_data is None else eval_data,
            self.logger, dataset_name=dataset_name
        )

    def predict_symptoms(self, symptoms_str, max_actions=None, force_top_k=None):
        state_vector = np.zeros(len(self.env.state_space), dtype=np.float32)
        for symptom in symptoms_str.split(','):
            symptom = symptom.strip()
            if symptom in self.env.swapped_state_space:
                state_vector[self.env.swapped_state_space[symptom]] = 1

        selected_actions = predict_actions_from_state(
            self.env, self.policy_net, self.action_selector, self.device,
            state_vector, force_top_k=force_top_k, max_actions=max_actions
        )
        return [self.env.action_space[action_idx] for action_idx in selected_actions]


def save_checkpoints(model_dir, policy_net, symptoms, Se, nn_units, nn_units2, dropout,
                     state_vector_len, n_actions, seed, test_ratio, test_metrics):
    checkpoint = {
        'model_state_dict': policy_net.state_dict(),
        'symptoms': symptoms,
        'Se': Se,
        'nn_units': nn_units,
        'nn_units2': nn_units2,
        'dropout': dropout,
        'state_vector_len': state_vector_len,
        'n_actions': n_actions,
        'seed': seed,
        'test_ratio': test_ratio,
        'test_metrics': test_metrics,
    }
    for model_name in ('dqn_model.pth', 'dqn_model_best.pth', 'dqn_model_last.pth'):
        torch.save(checkpoint, os.path.join(model_dir, model_name))
