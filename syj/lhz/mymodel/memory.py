# -*- coding: utf-8 -*-
"""经验回放池文件。

定义DQN训练使用的Transition数据结构和ReplayMemory，用于保存、
随机采样训练过程中的状态转移样本。支持普通均匀采样和SumTree优先经验回放(PER)。
"""

import random
from collections import deque, namedtuple

import numpy as np


Transition = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_state', 'next_invalid_actions', 'discount')
)


class SumTree(object):
    """PER使用的二叉和树，采样和优先级更新复杂度均为O(log N)。"""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.data = [None] * self.capacity
        self.write = 0
        self.size = 0

    @property
    def total(self):
        return float(self.tree[0])

    def add(self, priority, data):
        tree_idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(tree_idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return tree_idx

    def update(self, tree_idx, priority):
        change = float(priority) - self.tree[tree_idx]
        self.tree[tree_idx] = float(priority)
        parent = (tree_idx - 1) // 2
        while tree_idx != 0:
            self.tree[parent] += change
            tree_idx = parent
            parent = (tree_idx - 1) // 2

    def get(self, mass):
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                data_idx = parent - self.capacity + 1
                return parent, self.tree[parent], self.data[data_idx]
            if mass <= self.tree[left]:
                parent = left
            else:
                mass -= self.tree[left]
                parent = right


class ReplayMemory(object):
    def __init__(self, capacity, prioritized=False, alpha=0.6, priority_epsilon=1e-3):
        self.capacity = int(capacity)
        self.prioritized = prioritized
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.max_priority = 1.0
        if self.prioritized:
            self.tree = SumTree(self.capacity)
            self.memory = None
        else:
            self.memory = deque([], maxlen=self.capacity)
            self.tree = None

    def _scaled_priority(self, priority):
        priority = max(float(priority), self.priority_epsilon)
        return priority ** self.alpha

    def push(self, *args, priority=None):
        transition = Transition(*args)
        if not self.prioritized:
            self.memory.append(transition)
            return
        if priority is None:
            priority = self.max_priority
        self.max_priority = max(self.max_priority, float(priority))
        self.tree.add(self._scaled_priority(priority), transition)

    def sample(self, batch_size, beta=0.4):
        if not self.prioritized:
            return random.sample(self.memory, batch_size)

        transitions = []
        tree_indices = []
        priorities = []
        total_priority = self.tree.total
        segment = total_priority / batch_size
        for idx in range(batch_size):
            left = segment * idx
            right = segment * (idx + 1)
            mass = random.uniform(left, right)
            tree_idx, priority, transition = self.tree.get(mass)
            transitions.append(transition)
            tree_indices.append(tree_idx)
            priorities.append(priority)

        prob = np.asarray(priorities, dtype=np.float64) / max(total_priority, self.priority_epsilon)
        weights = np.power(len(self) * prob, -beta)
        weights = weights / max(weights.max(), self.priority_epsilon)
        return transitions, tree_indices, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        if not self.prioritized or indices is None:
            return
        for tree_idx, priority in zip(indices, priorities):
            priority = max(abs(float(priority)), self.priority_epsilon)
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(int(tree_idx), self._scaled_priority(priority))

    def __len__(self):
        if self.prioritized:
            return self.tree.size
        return len(self.memory)
