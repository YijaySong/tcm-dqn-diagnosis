# -*- coding: utf-8 -*-
"""V1 replay memory：支持普通回放与保留尾部乘子的 PER。"""

import random
from collections import deque, namedtuple

import numpy as np


Transition = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_state', 'next_selected_actions', 'discount',
     'aux_target', 'rare_multiplier', 'is_expert')
)


class SumTree(object):
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.data = [None] * self.capacity
        self.write = 0
        self.size = 0

    @property
    def total(self):
        return float(self.tree[0])

    def add(self, priority, item):
        index = self.write + self.capacity - 1
        self.data[self.write] = item
        self.update(index, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, index, priority):
        delta = float(priority) - self.tree[index]
        self.tree[index] = float(priority)
        while index:
            index = (index - 1) // 2
            self.tree[index] += delta

    def get(self, mass):
        index = 0
        while True:
            left = 2 * index + 1
            if left >= len(self.tree):
                data_index = index - self.capacity + 1
                return index, self.tree[index], self.data[data_index]
            right = left + 1
            if mass <= self.tree[left]:
                index = left
            else:
                mass -= self.tree[left]
                index = right


class ReplayMemory(object):
    def __init__(self, capacity, prioritized=False, alpha=0.6, priority_epsilon=1e-3):
        self.capacity = int(capacity)
        self.prioritized = bool(prioritized)
        self.alpha = float(alpha)
        self.priority_epsilon = float(priority_epsilon)
        self.max_priority = 1.0
        self.expert_inserted = 0
        self.online_inserted = 0
        self.tree = SumTree(self.capacity) if self.prioritized else None
        self.memory = None if self.prioritized else deque([], maxlen=self.capacity)

    def _scaled(self, priority):
        return max(float(priority), self.priority_epsilon) ** self.alpha

    def push(self, *args, priority=None):
        transition = Transition(*args)
        if transition.is_expert:
            self.expert_inserted += 1
        else:
            self.online_inserted += 1
        if not self.prioritized:
            self.memory.append(transition)
            return
        priority = self.max_priority if priority is None else float(priority)
        self.max_priority = max(self.max_priority, priority)
        self.tree.add(self._scaled(priority), transition)

    def sample(self, batch_size, beta=0.4):
        if not self.prioritized:
            return random.sample(self.memory, batch_size)
        total = max(self.tree.total, self.priority_epsilon)
        segment = total / batch_size
        transitions, indices, priorities = [], [], []
        for index in range(batch_size):
            tree_index, priority, transition = self.tree.get(random.uniform(segment * index, segment * (index + 1)))
            transitions.append(transition)
            indices.append(tree_index)
            priorities.append(priority)
        probabilities = np.asarray(priorities, dtype=np.float64) / total
        weights = np.power(len(self) * probabilities, -float(beta))
        weights /= max(weights.max(), self.priority_epsilon)
        return transitions, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors, rare_multipliers=None):
        """每次 TD 更新都保留固定 tail multiplier，而非只在首次写入时生效。"""
        if not self.prioritized or indices is None:
            return
        if rare_multipliers is None:
            rare_multipliers = [1.0] * len(indices)
        for index, error, multiplier in zip(indices, td_errors, rare_multipliers):
            priority = max(abs(float(error)) * max(float(multiplier), 1.0), self.priority_epsilon)
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(int(index), self._scaled(priority))

    def summary(self):
        return {
            'capacity': self.capacity,
            'size': len(self),
            'prioritized': self.prioritized,
            'expert_inserted': self.expert_inserted,
            'online_inserted': self.online_inserted,
        }

    def __len__(self):
        return self.tree.size if self.prioritized else len(self.memory)
