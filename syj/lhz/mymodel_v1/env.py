# -*- coding: utf-8 -*-
"""V1 序贯证候要素环境。

默认奖励只使用训练支持度确定的固定标签权重、加权 F-beta 势函数与 STOP 效用；
消融后默认步成本为0，不包含好奇心、预训练提示、动态 FP/FN 或 cardinality 奖惩。
"""

import numpy as np

from constants import STOP_ACTION


class Environment(object):
    """状态为 [症状 multi-hot | 已选标签 multi-hot]，动作是标签或 STOP。"""

    def __init__(self, symptoms, labels):
        self.symptom_to_index = {name: index for index, name in symptoms.items()}
        self.label_to_action = {name: index for index, name in labels.items()}
        self.symptoms = dict(symptoms)
        self.labels = dict(labels)
        self.symp_len = len(symptoms)
        self.Se_action_num = len(labels)
        self.stop_action = self.Se_action_num
        self.action_space = dict(labels)
        self.action_space[self.stop_action] = STOP_ACTION
        self.state_space = {
            **{index: name for index, name in symptoms.items()},
            **{self.symp_len + index: name for index, name in labels.items()},
        }
        self.Se_weights = {index: 1.0 for index in range(self.Se_action_num)}
        self.Se_supports = {index: 0 for index in range(self.Se_action_num)}
        self.configure_reward()

    def configure_reward(self, beta=1.0, step_cost=0.0, stop_utility_scale=1.0, shaping_gamma=0.95):
        if beta <= 0:
            raise ValueError('reward beta 必须大于0')
        self.reward_config = {
            'mode': 'weighted_fbeta',
            'beta': float(beta),
            'step_cost': float(step_cost),
            'stop_utility_scale': float(stop_utility_scale),
            'shaping_gamma': float(shaping_gamma),
        }

    def set_Se_weights(self, weights):
        self.Se_weights = {int(index): float(value) for index, value in weights.items()}

    def set_Se_supports(self, supports):
        self.Se_supports = {int(index): int(value) for index, value in supports.items()}

    def reset(self, data_piece):
        self.piece_symptoms = list(data_piece[0])
        self.piece_Se = [self.label_to_action[label] for label in data_piece[1] if label in self.label_to_action]
        self.state = np.zeros(len(self.state_space), dtype=np.float32)
        self.last_oov_symptoms = []
        for symptom in self.piece_symptoms:
            index = self.symptom_to_index.get(symptom)
            if index is None:
                self.last_oov_symptoms.append(symptom)
            else:
                self.state[index] = 1.0
        return self.state

    def weighted_fbeta(self, selected_actions):
        selected = set(action for action in selected_actions if 0 <= action < self.Se_action_num)
        truth = set(self.piece_Se)
        beta2 = self.reward_config['beta'] ** 2
        tp = sum(self.Se_weights.get(action, 1.0) for action in selected & truth)
        fp = sum(self.Se_weights.get(action, 1.0) for action in selected - truth)
        fn = sum(self.Se_weights.get(action, 1.0) for action in truth - selected)
        denominator = (1.0 + beta2) * tp + beta2 * fn + fp
        return (1.0 + beta2) * tp / denominator if denominator > 0 else 0.0

    def terminal_reward(self, selected_actions):
        return self.reward_config['stop_utility_scale'] * self.weighted_fbeta(selected_actions)

    def step(self, action, selected_actions):
        selected = [item for item in selected_actions if 0 <= item < self.Se_action_num]
        if action == self.stop_action:
            return None, self.terminal_reward(selected), True
        if not 0 <= action < self.Se_action_num:
            raise ValueError(f'非法动作: {action}')
        if action in selected:
            raise ValueError(f'重复选择动作: {action}')

        next_selected = selected + [action]
        before = self.weighted_fbeta(selected)
        after = self.weighted_fbeta(next_selected)
        reward = self.reward_config['shaping_gamma'] * after - before - self.reward_config['step_cost']
        self.state[self.symp_len + action] = 1.0
        # 所有标签都被选中时，调用者将补充一条显式 STOP transition，以获得一致的终止效用。
        exhausted = len(next_selected) >= self.Se_action_num
        return self.state.copy(), reward, exhausted
