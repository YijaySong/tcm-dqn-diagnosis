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
        self.configure_reward()

    def configure_reward(
        self, true_positive_base=1.0, delta_f1_scale=2.0,
        false_positive_penalty=1.4, step_penalty=0.03,
        over_select_penalty=0.7, stop_exact_reward=4.0,
        stop_f1_scale=3.0, stop_base_penalty=0.5,
        stop_fn_penalty=0.9, stop_fp_penalty=0.8,
    ):
        self.reward_config = {
            'true_positive_base': true_positive_base,
            'delta_f1_scale': delta_f1_scale,
            'false_positive_penalty': false_positive_penalty,
            'step_penalty': step_penalty,
            'over_select_penalty': over_select_penalty,
            'stop_exact_reward': stop_exact_reward,
            'stop_f1_scale': stop_f1_scale,
            'stop_base_penalty': stop_base_penalty,
            'stop_fn_penalty': stop_fn_penalty,
            'stop_fp_penalty': stop_fp_penalty,
        }

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
        cfg = self.reward_config
        if fp == 0 and fn == 0:
            return cfg['stop_exact_reward']
        f1 = set_f1(selected_Se, self.piece_Se)
        return (
            cfg['stop_f1_scale'] * f1
            - cfg['stop_base_penalty']
            - cfg['stop_fn_penalty'] * fn
            - cfg['stop_fp_penalty'] * fp
        )

    def step(self, action, selected_actions):
        selected_Se = [item for item in selected_actions if item != self.stop_action]

        if action == self.stop_action:
            reward = self._terminal_reward(selected_Se)
            return None, reward, True

        before_f1 = set_f1(selected_Se, self.piece_Se)
        next_selected_Se = selected_Se + [action]
        after_f1 = set_f1(next_selected_Se, self.piece_Se)
        delta_f1 = after_f1 - before_f1

        cfg = self.reward_config
        if action in self.piece_Se:
            reward = cfg['true_positive_base'] * self.Se_weights.get(action, 1.0) + cfg['delta_f1_scale'] * delta_f1
        else:
            reward = -cfg['false_positive_penalty'] + cfg['delta_f1_scale'] * delta_f1

        reward -= cfg['step_penalty']
        over_select = max(0, len(next_selected_Se) - len(self.piece_Se))
        reward -= cfg['over_select_penalty'] * over_select

        self.state[self.symp_len + action] = 1

        terminated = len(next_selected_Se) >= self.Se_action_num
        return None if terminated else self.state, reward, terminated

    def step_for_eval(self, action):
        if action == self.stop_action:
            return None, True
        self.state[self.symp_len + action] = 1
        return self.state, False
