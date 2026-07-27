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
        self.Se_supports = {index: 0 for index in range(self.Se_action_num)}
        self.pretrain_hints = {}
        self.reward_progress = 0.0
        self.configure_reward()

    def configure_reward(
        self, mode='legacy', true_positive_base=None, delta_f1_scale=None,
        false_positive_penalty=None, step_penalty=None,
        over_select_penalty=None, stop_exact_reward=None,
        stop_f1_scale=None, stop_base_penalty=None,
        stop_fn_penalty=None, stop_fp_penalty=None,
        shaping_gamma=0.95, potential_scale=None, potential_baseline=None,
        cardinality_penalty=None, terminal_f1_scale=None,
        exact_match_bonus=None, terminal_cardinality_penalty=None,
        tp_weight_scale=None, false_positive_penalty_start=None,
        fp_weight_scale=None, rare_tp_bonus=None,
        rare_tp_bonus_start=None, rare_tp_bonus_end=None,
        rare_fn_penalty=None, balanced_beta=None,
        terminal_sample_f1_weight=None, terminal_balanced_f1_weight=None,
        uncertainty_tp_bonus=None, pretrain_miss_tp_bonus=None,
        uncertainty_potential_scale=None, terminal_tail_recall_bonus=None,
        pretrain_miss_fn_penalty=None, exploration_tp_bonus=None,
        f1_log_gain=None, over_select_penalty_power=None,
        terminal_under_select_penalty=None, support_confidence_min=None,
        support_confidence_k=None,
    ):
        """配置奖励函数。

        ``legacy`` 保留旧版手工奖励；``potential_f1`` 使用样本F1势函数塑形；
        ``macro_balanced`` 面向macro_f1，将标签稀有度、假阳性代价和F1塑形
        合并到同一套奖励中；``tail_cost_curiosity`` 加入由预训练不确定性门控的
        尾部标签奖励，只有选中真实标签时才鼓励探索。
        """
        if mode not in ('legacy', 'potential_f1', 'macro_balanced', 'tail_cost_curiosity'):
            raise ValueError(f'未知奖励模式: {mode}')

        legacy_defaults = {
            'true_positive_base': 1.0,
            'delta_f1_scale': 2.0,
            'false_positive_penalty': 1.4,
            'false_positive_penalty_start': None,
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
            'fp_weight_scale': 0.0,
            'rare_tp_bonus': 0.0,
            'rare_tp_bonus_start': None,
            'rare_tp_bonus_end': None,
            'rare_fn_penalty': 0.0,
            'balanced_beta': 1.0,
            'terminal_sample_f1_weight': 1.0,
            'terminal_balanced_f1_weight': 0.0,
            'uncertainty_tp_bonus': 0.0,
            'pretrain_miss_tp_bonus': 0.0,
            'uncertainty_potential_scale': 0.0,
            'terminal_tail_recall_bonus': 0.0,
            'pretrain_miss_fn_penalty': 0.0,
            'exploration_tp_bonus': 0.0,
            'f1_log_gain': 0.0,
            'over_select_penalty_power': 1.0,
            'terminal_under_select_penalty': 0.0,
            'support_confidence_min': 1.0,
            'support_confidence_k': 5.0,
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
        macro_defaults = dict(legacy_defaults)
        macro_defaults.update({
            'false_positive_penalty': 0.28,
            'false_positive_penalty_start': 0.08,
            'step_penalty': 0.01,
            'over_select_penalty': 0.06,
            'stop_base_penalty': 0.0,
            'stop_fn_penalty': 0.18,
            'stop_fp_penalty': 0.22,
            'potential_scale': 1.35,
            'potential_baseline': 0.0,
            'cardinality_penalty': 0.04,
            'terminal_f1_scale': 2.4,
            'exact_match_bonus': 0.8,
            'terminal_cardinality_penalty': 0.08,
            'tp_weight_scale': 0.0,
            'fp_weight_scale': 0.5,
            'rare_tp_bonus': 0.30,
            'rare_tp_bonus_start': 0.45,
            'rare_tp_bonus_end': 0.20,
            'rare_fn_penalty': 0.22,
            'balanced_beta': 1.0,
            'terminal_sample_f1_weight': 0.35,
            'terminal_balanced_f1_weight': 0.65,
        })
        tail_curiosity_defaults = dict(macro_defaults)
        tail_curiosity_defaults.update({
            'false_positive_penalty': 0.36,
            'false_positive_penalty_start': 0.12,
            'step_penalty': 0.008,
            'over_select_penalty': 0.08,
            'stop_fn_penalty': 0.28,
            'stop_fp_penalty': 0.30,
            'potential_scale': 1.60,
            'cardinality_penalty': 0.03,
            'terminal_f1_scale': 2.80,
            'exact_match_bonus': 0.60,
            'terminal_cardinality_penalty': 0.10,
            'fp_weight_scale': 0.35,
            'rare_tp_bonus': 0.40,
            'rare_tp_bonus_start': 0.60,
            'rare_tp_bonus_end': 0.30,
            'rare_fn_penalty': 0.34,
            'balanced_beta': 1.30,
            'terminal_sample_f1_weight': 0.20,
            'terminal_balanced_f1_weight': 0.80,
            'uncertainty_tp_bonus': 0.25,
            'pretrain_miss_tp_bonus': 0.35,
            'uncertainty_potential_scale': 0.18,
            'terminal_tail_recall_bonus': 0.45,
            'pretrain_miss_fn_penalty': 0.24,
            'exploration_tp_bonus': 0.15,  # 探索奖励：对于预训练不确定的真实标签
            'f1_log_gain': 9.0,
            'over_select_penalty_power': 2.0,
            'terminal_under_select_penalty': 0.12,
            'support_confidence_min': 0.35,
            'support_confidence_k': 5.0,
        })
        defaults_by_mode = {
            'legacy': legacy_defaults,
            'potential_f1': potential_defaults,
            'macro_balanced': macro_defaults,
            'tail_cost_curiosity': tail_curiosity_defaults,
        }
        cfg = dict(defaults_by_mode[mode])
        overrides = {
            'true_positive_base': true_positive_base,
            'delta_f1_scale': delta_f1_scale,
            'false_positive_penalty': false_positive_penalty,
            'false_positive_penalty_start': false_positive_penalty_start,
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
            'fp_weight_scale': fp_weight_scale,
            'rare_tp_bonus': rare_tp_bonus,
            'rare_tp_bonus_start': rare_tp_bonus_start,
            'rare_tp_bonus_end': rare_tp_bonus_end,
            'rare_fn_penalty': rare_fn_penalty,
            'balanced_beta': balanced_beta,
            'terminal_sample_f1_weight': terminal_sample_f1_weight,
            'terminal_balanced_f1_weight': terminal_balanced_f1_weight,
            'uncertainty_tp_bonus': uncertainty_tp_bonus,
            'pretrain_miss_tp_bonus': pretrain_miss_tp_bonus,
            'uncertainty_potential_scale': uncertainty_potential_scale,
            'terminal_tail_recall_bonus': terminal_tail_recall_bonus,
            'pretrain_miss_fn_penalty': pretrain_miss_fn_penalty,
            'exploration_tp_bonus': exploration_tp_bonus,
            'f1_log_gain': f1_log_gain,
            'over_select_penalty_power': over_select_penalty_power,
            'terminal_under_select_penalty': terminal_under_select_penalty,
            'support_confidence_min': support_confidence_min,
            'support_confidence_k': support_confidence_k,
        }
        cfg.update({key: value for key, value in overrides.items() if value is not None})
        cfg['mode'] = mode
        cfg['shaping_gamma'] = shaping_gamma
        self.reward_config = cfg

    def set_Se_weights(self, Se_weights):
        self.Se_weights = Se_weights

    def set_Se_supports(self, Se_supports):
        self.Se_supports = Se_supports

    def set_pretrain_hints(self, pretrain_hints):
        self.pretrain_hints = pretrain_hints or {}

    def set_reward_progress(self, progress):
        self.reward_progress = min(1.0, max(0.0, float(progress)))

    def reset(self, data_piece):
        self.piece_symptoms = data_piece[0]
        self.piece_Se = [self.swapped_action_space[Se] for Se in data_piece[1]]
        self.state = np.zeros(len(self.state_space), dtype=int)
        for symp in self.piece_symptoms:
            self.state[self.swapped_state_space[symp]] = 1
        return self.state

    def _symptom_key(self):
        return tuple(sorted(self.piece_symptoms))

    def _progress_value(self, start_value, end_value):
        if start_value is None:
            return end_value
        if end_value is None:
            return start_value
        return start_value * (1.0 - self.reward_progress) + end_value * self.reward_progress

    def _current_false_positive_penalty(self):
        cfg = self.reward_config
        return self._progress_value(cfg.get('false_positive_penalty_start'), cfg['false_positive_penalty'])

    def _current_rare_tp_bonus(self):
        cfg = self.reward_config
        start = cfg.get('rare_tp_bonus_start')
        end = cfg.get('rare_tp_bonus_end')
        if start is None and end is None:
            return cfg['rare_tp_bonus']
        return self._progress_value(start if start is not None else cfg['rare_tp_bonus'], end if end is not None else cfg['rare_tp_bonus'])

    def _support_confidence(self, action_idx):
        cfg = self.reward_config
        support = max(0.0, float(self.Se_supports.get(action_idx, 0)))
        k = max(float(cfg.get('support_confidence_k', 5.0)), 1e-6)
        minimum = min(1.0, max(0.0, float(cfg.get('support_confidence_min', 1.0))))
        return minimum + (1.0 - minimum) * (support / (support + k))

    def _action_weight(self, action_idx):
        raw_weight = float(self.Se_weights.get(action_idx, 1.0))
        return 1.0 + (raw_weight - 1.0) * self._support_confidence(action_idx)

    def _false_positive_weight(self, action_idx):
        cfg = self.reward_config
        return 1.0 + cfg['fp_weight_scale'] * max(0.0, self._action_weight(action_idx) - 1.0)

    def _pretrain_hint(self, action_idx):
        return self.pretrain_hints.get(self._symptom_key(), {}).get(action_idx, {})

    def _pretrain_hint_score(self, action_idx):
        return float(self._pretrain_hint(action_idx).get('score', 0.0))

    def _pretrain_miss_flag(self, action_idx):
        return float(self._pretrain_hint(action_idx).get('miss', 0.0))

    def _curiosity_tp_bonus(self, action_idx):
        cfg = self.reward_config
        if cfg['mode'] != 'tail_cost_curiosity':
            return 0.0
        rarity = 1.0 + max(0.0, self._action_weight(action_idx) - 1.0)
        return (
            cfg['uncertainty_tp_bonus'] * self._pretrain_hint_score(action_idx) * rarity
            + cfg['pretrain_miss_tp_bonus'] * self._pretrain_miss_flag(action_idx) * rarity
        )

    def _weighted_counts(self, selected_Se):
        selected_set = set(selected_Se)
        true_set = set(self.piece_Se)
        tp_set = selected_set & true_set
        fp_set = selected_set - true_set
        fn_set = true_set - selected_set
        tp_weight = sum(self._action_weight(action) for action in tp_set)
        fp_weight = sum(self._false_positive_weight(action) for action in fp_set)
        fn_weight = sum(self._action_weight(action) for action in fn_set)
        true_weight = sum(self._action_weight(action) for action in true_set)
        return {
            'tp_weight': tp_weight,
            'fp_weight': fp_weight,
            'fn_weight': fn_weight,
            'true_weight': true_weight,
        }

    def _balanced_f_score(self, counts):
        beta = max(float(self.reward_config.get('balanced_beta', 1.0)), 1e-6)
        beta2 = beta * beta
        tp_weight = counts['tp_weight']
        denom = (1.0 + beta2) * tp_weight + beta2 * counts['fn_weight'] + counts['fp_weight']
        return (1.0 + beta2) * tp_weight / denom if denom > 0 else 0.0

    def _transform_f_score(self, f_score):
        gain = float(self.reward_config.get('f1_log_gain', 0.0) or 0.0)
        f_score = min(1.0, max(0.0, float(f_score)))
        if gain <= 0.0:
            return f_score
        return float(np.log1p(gain * f_score) / np.log1p(gain))

    def _over_select_cost(self, over_select):
        power = max(float(self.reward_config.get('over_select_penalty_power', 1.0)), 1.0)
        return float(max(0, over_select)) ** power

    def _set_potential(self, selected_Se):
        cfg = self.reward_config
        if cfg['mode'] in ('macro_balanced', 'tail_cost_curiosity'):
            return self._macro_set_potential(selected_Se)
        f1 = self._transform_f_score(set_f1(selected_Se, self.piece_Se))
        true_count = max(len(self.piece_Se), 1)
        cardinality_error = abs(len(selected_Se) - len(self.piece_Se)) / true_count
        return cfg['potential_scale'] * (f1 - cfg['potential_baseline']) - cfg['cardinality_penalty'] * cardinality_error

    def _macro_set_potential(self, selected_Se):
        cfg = self.reward_config
        counts = self._weighted_counts(selected_Se)
        balanced_f1 = self._transform_f_score(self._balanced_f_score(counts))
        true_count = max(len(self.piece_Se), 1)
        true_weight = max(counts['true_weight'], 1.0)
        cardinality_error = abs(len(selected_Se) - len(self.piece_Se)) / true_count
        missing_ratio = counts['fn_weight'] / true_weight
        potential = (
            cfg['potential_scale'] * (balanced_f1 - cfg['potential_baseline'])
            - cfg['cardinality_penalty'] * cardinality_error
            - cfg['rare_fn_penalty'] * missing_ratio
        )
        if cfg['mode'] == 'tail_cost_curiosity':
            selected_true = set(selected_Se) & set(self.piece_Se)
            curiosity_hits = sum(
                self._pretrain_hint_score(action) * self._action_weight(action)
                for action in selected_true
            )
            potential += cfg['uncertainty_potential_scale'] * curiosity_hits / true_weight
        return potential

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

    def _macro_terminal_reward(self, selected_Se):
        cfg = self.reward_config
        _, fp, fn = calc_tp_fp_fn(selected_Se, self.piece_Se)
        sample_f1 = set_f1(selected_Se, self.piece_Se)
        counts = self._weighted_counts(selected_Se)
        balanced_f1 = self._balanced_f_score(counts)
        blend_denom = cfg['terminal_sample_f1_weight'] + cfg['terminal_balanced_f1_weight']
        if blend_denom <= 0:
            blended_f1 = balanced_f1
        else:
            blended_f1 = (
                cfg['terminal_sample_f1_weight'] * sample_f1
                + cfg['terminal_balanced_f1_weight'] * balanced_f1
            ) / blend_denom

        true_weight = max(counts['true_weight'], 1.0)
        cardinality_error = abs(len(selected_Se) - len(self.piece_Se))
        selected_set = set(selected_Se)
        true_set = set(self.piece_Se)
        tp_set = selected_set & true_set
        fn_set = true_set - selected_set
        reward = cfg['terminal_f1_scale'] * blended_f1 - cfg['stop_base_penalty']
        if fp == 0 and fn == 0:
            reward += cfg['exact_match_bonus']
        reward -= cfg['stop_fn_penalty'] * counts['fn_weight'] / true_weight
        reward -= cfg['rare_fn_penalty'] * counts['fn_weight'] / true_weight
        reward -= cfg['terminal_under_select_penalty'] * max(0, len(self.piece_Se) - len(selected_Se))
        reward -= cfg['stop_fp_penalty'] * counts['fp_weight']
        reward -= cfg['terminal_cardinality_penalty'] * cardinality_error
        if cfg['mode'] == 'tail_cost_curiosity':
            tail_total = sum(max(0.0, self._action_weight(action) - 1.0) for action in true_set)
            tail_hit = sum(max(0.0, self._action_weight(action) - 1.0) for action in tp_set)
            if tail_total > 0:
                reward += cfg['terminal_tail_recall_bonus'] * tail_hit / tail_total
            missed_uncertain = sum(
                self._pretrain_hint_score(action) * self._action_weight(action)
                for action in fn_set
            )
            reward -= cfg['pretrain_miss_fn_penalty'] * missed_uncertain / true_weight
        return reward

    def _terminal_reward(self, selected_Se):
        cfg = self.reward_config
        if cfg['mode'] == 'legacy':
            return self._legacy_terminal_reward(selected_Se)
        if cfg['mode'] in ('macro_balanced', 'tail_cost_curiosity'):
            return self._macro_terminal_reward(selected_Se)

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
        reward -= cfg['over_select_penalty'] * self._over_select_cost(over_select)
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
        reward -= cfg['over_select_penalty'] * self._over_select_cost(over_select)
        return reward

    def _macro_step_reward(self, action, selected_Se, next_selected_Se):
        cfg = self.reward_config
        reward = (
            cfg['shaping_gamma'] * self._set_potential(next_selected_Se)
            - self._set_potential(selected_Se)
            - cfg['step_penalty']
        )
        if action in self.piece_Se:
            reward += self._current_rare_tp_bonus() * max(0.0, self._action_weight(action) - 1.0)
            reward += self._curiosity_tp_bonus(action)
            if cfg['mode'] == 'tail_cost_curiosity':
                reward += (
                    cfg.get('exploration_tp_bonus', 0.0)
                    * self._pretrain_hint_score(action)
                    * max(0.0, self._action_weight(action) - 1.0)
                    * (1.0 - self.reward_progress)
                )
        else:
            reward -= self._current_false_positive_penalty() * self._false_positive_weight(action)
        over_select = max(0, len(next_selected_Se) - len(self.piece_Se))
        reward -= cfg['over_select_penalty'] * self._over_select_cost(over_select)
        return reward

    def step(self, action, selected_actions):
        selected_Se = [item for item in selected_actions if item != self.stop_action]

        if action == self.stop_action:
            reward = self._terminal_reward(selected_Se)
            return None, reward, True

        next_selected_Se = selected_Se + [action]
        if self.reward_config['mode'] == 'legacy':
            reward = self._legacy_step_reward(action, selected_Se, next_selected_Se)
        elif self.reward_config['mode'] in ('macro_balanced', 'tail_cost_curiosity'):
            reward = self._macro_step_reward(action, selected_Se, next_selected_Se)
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
