# -*- coding: utf-8 -*-
"""DQN模型结构文件。

定义用于证候要素推荐的神经网络结构，输入为状态向量，输出为每个候选动作
包括停止动作的Q值。
"""

import torch
import torch.nn as nn


class DQNLegacy(nn.Module):
    def __init__(self, state_vector_len, n_actions, n_units=128, n_units2=64, dropout=0.1):
        super(DQNLegacy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_vector_len, n_units, dtype=torch.float32, device='cpu'),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_units, n_units2, dtype=torch.float32, device='cpu'),
            nn.ReLU(),
            nn.Linear(n_units2, n_actions, dtype=torch.float32, device='cpu')
        )

    def forward(self, x):
        return self.net(x)


class DQN(nn.Module):
    def __init__(self, state_vector_len, n_actions, n_units=128, n_units2=64, dropout=0.1):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_vector_len, n_units, dtype=torch.float32, device='cpu'),
            nn.LayerNorm(n_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_units, n_units2, dtype=torch.float32, device='cpu'),
            nn.LayerNorm(n_units2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_units2, n_actions, dtype=torch.float32, device='cpu')
        )

    def forward(self, x):
        return self.net(x)


class DuelingDQN(nn.Module):
    """Dueling DQN：共享特征层 + 状态价值分支 + 动作优势分支。

    对当前任务更合适的原因是：很多病例的“症状状态价值”和“具体证候要素排序”
    可以分开学习，尤其在动作数较多且多数动作无关时，优势分支更容易学习动作差异。
    """

    def __init__(self, state_vector_len, n_actions, n_units=256, n_units2=128, dropout=0.15):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_vector_len, n_units, dtype=torch.float32, device='cpu'),
            nn.LayerNorm(n_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_units, n_units2, dtype=torch.float32, device='cpu'),
            nn.LayerNorm(n_units2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(n_units2, n_units2, dtype=torch.float32, device='cpu'),
            nn.ReLU(),
            nn.Linear(n_units2, 1, dtype=torch.float32, device='cpu'),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(n_units2, n_units2, dtype=torch.float32, device='cpu'),
            nn.ReLU(),
            nn.Linear(n_units2, n_actions, dtype=torch.float32, device='cpu'),
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def build_q_network(model_type, state_vector_len, n_actions, n_units=128, n_units2=64, dropout=0.1):
    if model_type == 'dqn_legacy':
        return DQNLegacy(state_vector_len, n_actions, n_units, n_units2, dropout)
    if model_type == 'dqn':
        return DQN(state_vector_len, n_actions, n_units, n_units2, dropout)
    if model_type == 'dueling':
        return DuelingDQN(state_vector_len, n_actions, n_units, n_units2, dropout)
    raise ValueError(f'未知Q网络类型: {model_type}')
