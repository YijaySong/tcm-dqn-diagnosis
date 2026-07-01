# -*- coding: utf-8 -*-
"""DQN模型结构文件。

定义用于证候要素推荐的神经网络结构，输入为状态向量，输出为每个候选动作
包括停止动作的Q值。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Factorized NoisyNet线性层，用于状态依赖探索。"""

    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32, device='cpu'))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32, device='cpu'))
        self.mu_bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32, device='cpu'))
        self.sigma_bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32, device='cpu'))
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features, dtype=torch.float32))
        self.register_buffer('epsilon_bias', torch.zeros(out_features, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        self.mu_weight.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_weight.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size, device):
        noise = torch.randn(size, device=device)
        return noise.sign() * noise.abs().sqrt()

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features, self.mu_weight.device)
        epsilon_out = self._scale_noise(self.out_features, self.mu_weight.device)
        self.epsilon_weight.copy_(epsilon_out.outer(epsilon_in))
        self.epsilon_bias.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            epsilon_in = self._scale_noise(self.in_features, x.device)
            epsilon_out = self._scale_noise(self.out_features, x.device)
            epsilon_weight = epsilon_out.outer(epsilon_in)
            weight = self.mu_weight + self.sigma_weight * epsilon_weight
            bias = self.mu_bias + self.sigma_bias * epsilon_out
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(x, weight, bias)


def linear_layer(in_features, out_features, noisy=False):
    if noisy:
        return NoisyLinear(in_features, out_features)
    return nn.Linear(in_features, out_features, dtype=torch.float32, device='cpu')


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


class ResidualBlock(nn.Module):
    """小型残差块，适合稀疏二值症状输入上的稳定表征学习。"""

    def __init__(self, n_units, dropout=0.1, noisy=False):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            linear_layer(n_units, n_units, noisy=noisy),
            nn.LayerNorm(n_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            linear_layer(n_units, n_units, noisy=noisy),
            nn.LayerNorm(n_units),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class DuelingDQN(nn.Module):
    """Dueling DQN：共享特征层 + 状态价值分支 + 动作优势分支。"""

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


class ResidualDuelingDQN(nn.Module):
    """残差Dueling DQN。"""

    def __init__(self, state_vector_len, n_actions, n_units=256, n_units2=128, dropout=0.15):
        super(ResidualDuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_vector_len, n_units, dtype=torch.float32, device='cpu'),
            nn.LayerNorm(n_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResidualBlock(n_units, dropout),
            ResidualBlock(n_units, dropout),
            nn.Linear(n_units, n_units2, dtype=torch.float32, device='cpu'),
            nn.LayerNorm(n_units2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(n_units2, n_units2, dtype=torch.float32, device='cpu'),
            nn.LayerNorm(n_units2),
            nn.ReLU(),
            nn.Linear(n_units2, 1, dtype=torch.float32, device='cpu'),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(n_units2, n_units2, dtype=torch.float32, device='cpu'),
            nn.LayerNorm(n_units2),
            nn.ReLU(),
            nn.Linear(n_units2, n_actions, dtype=torch.float32, device='cpu'),
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class SetAwareDuelingDQN(nn.Module):
    """症状集合与已选证候要素集合分路编码的Dueling DQN。

    状态仍是“症状multi-hot + 已选证候要素multi-hot”，但网络不再把二者简单拼成
    一个普通向量处理，而是分别编码症状证据、已选集合和二者交互，再输出下一步动作Q值。
    """

    def __init__(self, state_vector_len, n_actions, n_units=256, n_units2=128, dropout=0.15, noisy=False):
        super(SetAwareDuelingDQN, self).__init__()
        self.Se_action_num = n_actions - 1
        self.symp_len = state_vector_len - self.Se_action_num
        if self.symp_len <= 0 or self.Se_action_num <= 0:
            raise ValueError('set_dueling需要状态向量由症状块和证候要素块组成')

        self.symptom_encoder = nn.Sequential(
            linear_layer(self.symp_len, n_units, noisy=noisy),
            nn.LayerNorm(n_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResidualBlock(n_units, dropout, noisy=noisy),
        )
        self.selected_encoder = nn.Sequential(
            linear_layer(self.Se_action_num, n_units, noisy=noisy),
            nn.LayerNorm(n_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResidualBlock(n_units, dropout, noisy=noisy),
        )
        self.joint = nn.Sequential(
            linear_layer(n_units * 3, n_units, noisy=noisy),
            nn.LayerNorm(n_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResidualBlock(n_units, dropout, noisy=noisy),
            linear_layer(n_units, n_units2, noisy=noisy),
            nn.LayerNorm(n_units2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.value_stream = nn.Sequential(
            linear_layer(n_units2, n_units2, noisy=noisy),
            nn.LayerNorm(n_units2),
            nn.ReLU(),
            linear_layer(n_units2, 1, noisy=noisy),
        )
        self.advantage_stream = nn.Sequential(
            linear_layer(n_units2, n_units2, noisy=noisy),
            nn.LayerNorm(n_units2),
            nn.ReLU(),
            linear_layer(n_units2, n_actions, noisy=noisy),
        )

    def forward(self, x):
        symptom_state = x[:, :self.symp_len]
        selected_state = x[:, self.symp_len:self.symp_len + self.Se_action_num]
        symptom_features = self.symptom_encoder(symptom_state)
        selected_features = self.selected_encoder(selected_state)
        features = self.joint(torch.cat([
            symptom_features,
            selected_features,
            symptom_features * selected_features,
        ], dim=1))
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
    if model_type == 'dueling_residual':
        return ResidualDuelingDQN(state_vector_len, n_actions, n_units, n_units2, dropout)
    if model_type == 'set_dueling':
        return SetAwareDuelingDQN(state_vector_len, n_actions, n_units, n_units2, dropout, noisy=False)
    if model_type == 'set_dueling_noisy':
        return SetAwareDuelingDQN(state_vector_len, n_actions, n_units, n_units2, dropout, noisy=True)
    raise ValueError(f'未知Q网络类型: {model_type}')
