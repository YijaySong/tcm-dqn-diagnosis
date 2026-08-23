# -*- coding: utf-8 -*-
"""V1 轻量集合感知 Dueling Q 网络。

默认只保留症状/已选标签分路编码和 dueling value-advantage 分支，不启用显式探索。
Noisy 版本只用于探索策略消融和历史结果复现。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))
        bound = 1.0 / math.sqrt(in_features)
        self.mu_weight.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_weight.data.fill_(sigma_init / math.sqrt(in_features))
        self.sigma_bias.data.fill_(sigma_init / math.sqrt(out_features))

    @staticmethod
    def _noise(size, device):
        noise = torch.randn(size, device=device)
        return noise.sign() * noise.abs().sqrt()

    def forward(self, inputs):
        if not self.training:
            return functional.linear(inputs, self.mu_weight, self.mu_bias)
        eps_in = self._noise(self.in_features, inputs.device)
        eps_out = self._noise(self.out_features, inputs.device)
        weight = self.mu_weight + self.sigma_weight * eps_out.outer(eps_in)
        bias = self.mu_bias + self.sigma_bias * eps_out
        return functional.linear(inputs, weight, bias)


def linear_layer(in_features, out_features, noisy=False):
    return NoisyLinear(in_features, out_features) if noisy else nn.Linear(in_features, out_features)


class SetAwareDuelingDQN(nn.Module):
    """症状集合、已选标签集合和逐元素交互的轻量 Dueling 网络。"""

    def __init__(self, state_vector_len, n_actions, n_units=256, n_units2=128, dropout=0.15, noisy=False):
        super().__init__()
        self.label_count = n_actions - 1
        self.symptom_count = state_vector_len - self.label_count
        if self.label_count <= 0 or self.symptom_count <= 0:
            raise ValueError('状态必须由症状块、已选标签块和STOP动作组成')
        self.symptom_encoder = nn.Sequential(
            linear_layer(self.symptom_count, n_units, noisy), nn.LayerNorm(n_units), nn.ReLU(), nn.Dropout(dropout),
        )
        self.selected_encoder = nn.Sequential(
            linear_layer(self.label_count, n_units, noisy), nn.LayerNorm(n_units), nn.ReLU(), nn.Dropout(dropout),
        )
        self.joint = nn.Sequential(
            linear_layer(n_units * 3, n_units2, noisy), nn.LayerNorm(n_units2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.value_stream = nn.Sequential(
            linear_layer(n_units2, n_units2, noisy), nn.ReLU(), linear_layer(n_units2, 1, noisy),
        )
        self.advantage_stream = nn.Sequential(
            linear_layer(n_units2, n_units2, noisy), nn.ReLU(), linear_layer(n_units2, n_actions, noisy),
        )

    def forward(self, inputs):
        symptom = self.symptom_encoder(inputs[:, :self.symptom_count])
        selected = self.selected_encoder(inputs[:, self.symptom_count:self.symptom_count + self.label_count])
        features = self.joint(torch.cat([symptom, selected, symptom * selected], dim=1))
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def build_q_network(model_type, state_vector_len, n_actions, n_units=256, n_units2=128, dropout=0.15):
    if model_type == 'set_dueling':
        return SetAwareDuelingDQN(state_vector_len, n_actions, n_units, n_units2, dropout, noisy=False)
    if model_type == 'set_dueling_noisy':
        return SetAwareDuelingDQN(state_vector_len, n_actions, n_units, n_units2, dropout, noisy=True)
    raise ValueError(f'V1 仅支持 set_dueling 或 set_dueling_noisy，收到: {model_type}')
