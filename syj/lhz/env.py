# -*- coding: utf-8 -*-
"""强化学习环境文件。

定义DQN与辨证任务交互的环境，包括状态空间、动作空间、停止动作、
环境重置逻辑，以及选择证候要素后的奖励和终止判断。
"""

import numpy as np

from constants import STOP_ACTION
from metrics import calc_tp_fp_fn, set_f1


class Environment(object):
    """状态空间=刻下症+已选证候要素，动作空间=候选证候要素+停止"""

    def __init__(self, symptoms, Se):
        self.state_space = {}
        self.swapped_state_space = {}
        self.symp_len = len(symptoms)
        for index in symptoms:
            self.state_space[index] = symptoms[index]
            self.swapped_state_space[symptoms[index]] = index
        for index in Se:
            self.state_space[index + self.symp_len] = Se[index]
            self.swapped_state_space[Se[index]] = index + self.symp_len

        self.Se_action_num = len(Se)
        self.stop_action = self.Se_action_num
        self.action_space = dict(Se)
        self.action_space[self.stop_action] = STOP_ACTION
        self.swapped_action_space = {}
        for index in self.action_space:
            self.swapped_action_space[self.action_space[index]] = index
        self.Se_weights = {index: 1.0 for index in range(self.Se_action_num)}

    def set_Se_weights(self, Se_weights):
        self.Se_weights = Se_weights

    def reset(self, data_piece):
        self.piece_symptoms = data_piece[0]
        self.piece_Se = [self.swapped_action_space[Se] for Se in data_piece[1]]
        self.state = np.zeros(len(self.state_space), dtype=int)
        for symp in self.piece_symptoms:
            self.state[self.swapped_state_space[symp]] = 1
        return self.state

    def _terminal_reward(self, selected_Se):
        _, fp, fn = calc_tp_fp_fn(selected_Se, self.piece_Se)
        if fp == 0 and fn == 0:
            return 3.0
        f1 = set_f1(selected_Se, self.piece_Se)
        return 2.0 * f1 - 1.0 - 0.8 * fn - 0.6 * fp

    def step(self, action, selected_actions):
        selected_Se = [item for item in selected_actions if item != self.stop_action]

        if action == self.stop_action:
            reward = self._terminal_reward(selected_Se)
            return None, reward, True

        before_f1 = set_f1(selected_Se, self.piece_Se)
        next_selected_Se = selected_Se + [action]
        after_f1 = set_f1(next_selected_Se, self.piece_Se)
        delta_f1 = after_f1 - before_f1

        if action in self.piece_Se:
            reward = 0.8 * self.Se_weights.get(action, 1.0) + 1.5 * delta_f1
        else:
            reward = -1.2 + 1.5 * delta_f1

        reward -= 0.05
        over_select = max(0, len(next_selected_Se) - len(self.piece_Se))
        reward -= 0.5 * over_select

        self.state[self.symp_len + action] = 1

        terminated = len(next_selected_Se) >= self.Se_action_num
        return None if terminated else self.state, reward, terminated

    def step_for_eval(self, action):
        if action == self.stop_action:
            return None, True
        self.state[self.symp_len + action] = 1
        return self.state, False
