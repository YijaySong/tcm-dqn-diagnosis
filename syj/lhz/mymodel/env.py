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
        self, mode='potential_f1', true_positive_base=None, delta_f1_scale=None,
        false_positive_penalty=None, step_penalty=None,
        over_select_penalty=None, stop_exact_reward=None,
        stop_f1_scale=None, stop_base_penalty=None,
        stop_fn_penalty=None, stop_fp_penalty=None,
        shaping_gamma=0.95, potential_scale=None, potential_baseline=None,
        cardinality_penalty=None, terminal_f1_scale=None,
        exact_match_bonus=None, terminal_cardinality_penalty=None,
        tp_weight_scale=None,
    ):
        """配置奖励函数。

        ``legacy`` 保留旧版手工奖励；``potential_f1`` 使用与样本F1对齐的
        势函数塑形奖励，减少大量互相竞争的经验系数。
        """
        if mode not in ('legacy', 'potential_f1'):
            raise ValueError(f'未知奖励模式: {mode}')

        legacy_defaults = {
            'true_positive_base': 1.0,
            'delta_f1_scale': 2.0,
            'false_positive_penalty': 1.4,
            'step_penalty': 0.03,
            'over_select_penalty': 0.7,
            'stop_exact_reward': 4.0,
            'stop_f1_scale': 3.0,
            'stop_base_penalty': 0.5,
            'stop_fn_penalty': 0.9,
            'stop_fp_penalty': 0.8,
            'potential_scale': 1.0,
            'potential_baseline': 0.0,
            'cardinality_penalty': 0.08,
            'terminal_f1_scale': 1.0,
            'exact_match_bonus': 0.5,
            'terminal_cardinality_penalty': 0.05,
            'tp_weight_scale': 0.15,
        }
        potential_defaults = dict(legacy_defaults)
        potential_defaults.update({
            'false_positive_penalty': 0.15,
            'step_penalty': 0.02,
            'over_select_penalty': 0.03,
            'stop_base_penalty': 0.0,
            'stop_fn_penalty': 0.08,
            'stop_fp_penalty': 0.08,
        })
        cfg = dict(legacy_defaults if mode == 'legacy' else potential_defaults)
        overrides = {
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
            'potential_scale': potential_scale,
            'potential_baseline': potential_baseline,
            'cardinality_penalty': cardinality_penalty,
            'terminal_f1_scale': terminal_f1_scale,
            'exact_match_bonus': exact_match_bonus,
            'terminal_cardinality_penalty': terminal_cardinality_penalty,
            'tp_weight_scale': tp_weight_scale,
        }
        cfg.update({key: value for key, value in overrides.items() if value is not None})
        cfg['mode'] = mode
        cfg['shaping_gamma'] = shaping_gamma
        self.reward_config = cfg

    def set_Se_weights(self, Se_weights):
        self.Se_weights = Se_weights

    def reset(self, data_piece):
        self.piece_symptoms = data_piece[0]
        self.piece_Se = [self.swapped_action_space[Se] for Se in data_piece[1]]
        self.state = np.zeros(len(self.state_space), dtype=int)
        for symp in self.piece_symptoms:
            self.state[self.swapped_state_space[symp]] = 1
        return self.state

    def _set_potential(self, selected_Se):
        cfg = self.reward_config
        f1 = set_f1(selected_Se, self.piece_Se)
        true_count = max(len(self.piece_Se), 1)
        cardinality_error = abs(len(selected_Se) - len(self.piece_Se)) / true_count
        return cfg['potential_scale'] * (f1 - cfg['potential_baseline']) - cfg['cardinality_penalty'] * cardinality_error

    def _legacy_terminal_reward(self, selected_Se):
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

    def _terminal_reward(self, selected_Se):
        cfg = self.reward_config
        if cfg['mode'] == 'legacy':
            return self._legacy_terminal_reward(selected_Se)

        _, fp, fn = calc_tp_fp_fn(selected_Se, self.piece_Se)
        f1 = set_f1(selected_Se, self.piece_Se)
        cardinality_error = abs(len(selected_Se) - len(self.piece_Se))
        reward = cfg['terminal_f1_scale'] * f1 - cfg['stop_base_penalty']
        if fp == 0 and fn == 0:
            reward += cfg['exact_match_bonus']
        reward -= cfg['stop_fn_penalty'] * fn
        reward -= cfg['stop_fp_penalty'] * fp
        reward -= cfg['terminal_cardinality_penalty'] * cardinality_error
        return reward

    def _legacy_step_reward(self, action, selected_Se, next_selected_Se):
        before_f1 = set_f1(selected_Se, self.piece_Se)
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
        return reward

    def _potential_step_reward(self, action, selected_Se, next_selected_Se):
        cfg = self.reward_config
        reward = (
            cfg['shaping_gamma'] * self._set_potential(next_selected_Se)
            - self._set_potential(selected_Se)
            - cfg['step_penalty']
        )
        if action in self.piece_Se:
            reward += cfg['tp_weight_scale'] * (self.Se_weights.get(action, 1.0) - 1.0)
        else:
            reward -= cfg['false_positive_penalty']
        over_select = max(0, len(next_selected_Se) - len(self.piece_Se))
        reward -= cfg['over_select_penalty'] * over_select
        return reward

    def step(self, action, selected_actions):
        selected_Se = [item for item in selected_actions if item != self.stop_action]

        if action == self.stop_action:
            reward = self._terminal_reward(selected_Se)
            return None, reward, True

        next_selected_Se = selected_Se + [action]
        if self.reward_config['mode'] == 'legacy':
            reward = self._legacy_step_reward(action, selected_Se, next_selected_Se)
        else:
            reward = self._potential_step_reward(action, selected_Se, next_selected_Se)

        self.state[self.symp_len + action] = 1

        terminated = len(next_selected_Se) >= self.Se_action_num
        return None if terminated else self.state, reward, terminated

    def step_for_eval(self, action):
        if action == self.stop_action:
            return None, True
        self.state[self.symp_len + action] = 1
        return self.state, False
