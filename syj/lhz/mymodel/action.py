# -*- coding: utf-8 -*-
"""动作选择文件。

实现已选动作mask、停止动作mask，以及训练时epsilon-greedy动作选择和
评估时贪心动作选择，避免模型重复推荐同一证候要素。
"""

import math
import random

import torch

from constants import NEG_INF


def mask_selected_actions(q_values, selected_actions_batch, env, mask_stop=False):
    masked_q_values = q_values.clone()
    for row_idx, selected_actions in enumerate(selected_actions_batch):
        for action_idx in selected_actions:
            if 0 <= action_idx < env.Se_action_num:
                masked_q_values[row_idx, action_idx] = NEG_INF
    if mask_stop:
        masked_q_values[:, env.stop_action] = NEG_INF
    return masked_q_values


class ActionSelector(object):
    def __init__(self, env, policy_net, n_actions, device, eps_start, eps_end, eps_decay, min_actions_before_stop=1):
        self.env = env
        self.policy_net = policy_net
        self.n_actions = n_actions
        self.device = device
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.min_actions_before_stop = max(0, min_actions_before_stop)
        self.steps_done = 0

    def reset_steps(self):
        self.steps_done = 0

    def select_action(self, state, selected_actions):
        """epsilon-greedy策略选择动作，避免重复；停止动作始终可选。"""
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1. * self.steps_done / self.eps_decay
        )
        self.steps_done += 1

        mask_stop = len(selected_actions) < self.min_actions_before_stop
        if sample > eps_threshold:
            with torch.no_grad():
                output = self.policy_net(state)
                output = mask_selected_actions(output, [selected_actions], self.env, mask_stop=mask_stop)
                action = output.max(1).indices.view(1, 1)
                return action, "agent", True

        valid_actions = [
            action for action in range(self.n_actions)
            if (action != self.env.stop_action or not mask_stop) and (action == self.env.stop_action or action not in selected_actions)
        ]
        action = random.choice(valid_actions)
        return torch.tensor([[action]], device=self.device, dtype=torch.long), "random", False

    def select_action_4eval(self, state, selected_actions, mask_stop=False):
        """评估时选择动作，始终选Q值最大且不重复的合法动作。"""
        with torch.no_grad():
            output = self.policy_net(state)
            output = mask_selected_actions(output, [selected_actions], self.env, mask_stop=mask_stop)
            action = output.max(1).indices.view(1, 1)
            return action
