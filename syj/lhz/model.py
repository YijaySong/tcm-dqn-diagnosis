# -*- coding: utf-8 -*-
"""DQN模型结构文件。

定义用于证候要素推荐的神经网络结构，输入为状态向量，输出为每个候选动作
包括停止动作的Q值。
"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, state_vector_len, n_actions, n_units=128, n_units2=64, dropout=0.1):
        super(DQN, self).__init__()
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
