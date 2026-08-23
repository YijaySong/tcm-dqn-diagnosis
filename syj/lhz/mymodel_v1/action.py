# -*- coding: utf-8 -*-
"""V1 统一动作合法性与 epsilon-greedy 策略。"""

import math
import random

import torch

from constants import NEG_INF


def legal_action_mask(q_values, selected_actions_batch, env, min_actions_before_stop=1):
    """统一 mask：训练行为、DDQN bootstrap 和推理必须调用同一函数。"""
    masked = q_values.clone()
    for row_idx, selected_actions in enumerate(selected_actions_batch):
        selected = {action for action in selected_actions if 0 <= action < env.Se_action_num}
        for action in selected:
            masked[row_idx, action] = NEG_INF
        if len(selected) < min_actions_before_stop:
            masked[row_idx, env.stop_action] = NEG_INF
        if len(selected) >= env.Se_action_num:
            masked[row_idx, :env.Se_action_num] = NEG_INF
    return masked


class ActionSelector(object):
    def __init__(self, env, policy_net, n_actions, device, eps_start, eps_end, eps_decay, min_actions_before_stop=1):
        self.env = env
        self.policy_net = policy_net
        self.n_actions = n_actions
        self.device = device
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay = max(1, int(eps_decay))
        self.min_actions_before_stop = max(0, int(min_actions_before_stop))
        self.steps_done = 0

    def reset_steps(self):
        self.steps_done = 0

    def epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.steps_done / self.eps_decay)

    def select_action(self, state, selected_actions):
        epsilon = self.epsilon()
        self.steps_done += 1
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
                masked = legal_action_mask(q_values, [selected_actions], self.env, self.min_actions_before_stop)
                return masked.max(1).indices.view(1, 1), 'agent', epsilon

        selected = set(selected_actions)
        valid = [
            action for action in range(self.n_actions)
            if action not in selected
            and (action != self.env.stop_action or len(selected) >= self.min_actions_before_stop)
        ]
        if not valid:
            valid = [self.env.stop_action]
        action = random.choice(valid)
        return torch.tensor([[action]], device=self.device, dtype=torch.long), 'random', epsilon
