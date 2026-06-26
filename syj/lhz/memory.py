# -*- coding: utf-8 -*-
"""经验回放池文件。

定义DQN训练使用的Transition数据结构和ReplayMemory，用于保存、
随机采样训练过程中的状态转移样本。
"""

import random
from collections import deque, namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'next_invalid_actions'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
