# -*- coding: utf-8 -*-
"""状态处理文件。

负责把刻下症和已选证候要素转换为DQN输入状态向量，并根据当前状态循环调用
模型选择动作，生成预测用的证候要素动作序列。
"""

import numpy as np
import torch

from action import mask_selected_actions


def build_state_vector(env, symptoms, selected_actions=None):
    selected_actions = selected_actions or []
    state = np.zeros(len(env.state_space), dtype=np.float32)
    for symptom in symptoms:
        if symptom in env.swapped_state_space:
            state[env.swapped_state_space[symptom]] = 1
    for action in selected_actions:
        if 0 <= action < env.Se_action_num:
            state[env.symp_len + action] = 1
    return state


def actions_to_multihot(env, actions):
    vector = np.zeros(env.Se_action_num, dtype=int)
    for action in actions:
        if 0 <= action < env.Se_action_num:
            vector[action] = 1
    return vector


def Se_names_to_multihot(env, Se_names):
    vector = np.zeros(env.Se_action_num, dtype=int)
    for Se_name in Se_names:
        action = env.swapped_action_space[Se_name]
        vector[action] = 1
    return vector


def predict_actions_from_state(
    env, policy_net, action_selector, device, state_np, max_actions=None,
    stop_margin_threshold=None, min_actions=0, return_trace=False
):
    policy_net.eval()
    selected_actions = []
    trace = []
    max_actions = env.Se_action_num if max_actions is None else min(max_actions, env.Se_action_num)
    min_actions = max(0, min_actions)

    with torch.no_grad():
        while len(selected_actions) < max_actions:
            state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            mask_stop = len(selected_actions) < min_actions
            q_values = policy_net(state_tensor)
            masked_q = mask_selected_actions(q_values, [selected_actions], env, mask_stop=mask_stop)
            label_q = masked_q[:, :env.Se_action_num]
            best_label_q, best_label_action = label_q.max(1)
            stop_q = q_values[:, env.stop_action]
            stop_margin = stop_q - best_label_q

            if (
                stop_margin_threshold is not None
                and len(selected_actions) >= min_actions
                and stop_margin.item() >= stop_margin_threshold
            ):
                trace.append({
                    'action_index': env.stop_action,
                    'is_stop': True,
                    'q_value': float(stop_q.item()),
                    'best_label_q': float(best_label_q.item()),
                    'best_label_action': int(best_label_action.item()),
                    'stop_margin': float(stop_margin.item()),
                })
                break

            action_idx = masked_q.max(1).indices.item()
            is_stop = action_idx == env.stop_action
            trace.append({
                'action_index': int(action_idx),
                'is_stop': is_stop,
                'q_value': float(q_values[0, action_idx].item()),
                'best_label_q': float(best_label_q.item()),
                'best_label_action': int(best_label_action.item()),
                'stop_margin': float(stop_margin.item()),
            })

            if is_stop:
                break

            selected_actions.append(action_idx)
            state_np[env.symp_len + action_idx] = 1

    if return_trace:
        return selected_actions, trace
    return selected_actions
