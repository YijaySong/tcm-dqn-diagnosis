# -*- coding: utf-8 -*-
"""状态处理文件。

负责把刻下症和已选证候要素转换为DQN输入状态向量，并根据当前状态循环调用
模型选择动作，生成预测用的证候要素动作序列。
"""

import numpy as np
import torch


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


def predict_actions_from_state(env, policy_net, action_selector, device, state_np, force_top_k=None, max_actions=None):
    policy_net.eval()
    selected_actions = []
    max_actions = env.Se_action_num if max_actions is None else min(max_actions, env.Se_action_num)

    with torch.no_grad():
        while len(selected_actions) < max_actions:
            state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            mask_stop = force_top_k is not None and len(selected_actions) < force_top_k
            action = action_selector.select_action_4eval(state_tensor, selected_actions, mask_stop=mask_stop)
            action_idx = action.item()

            if action_idx == env.stop_action:
                break

            selected_actions.append(action_idx)
            state_np[env.symp_len + action_idx] = 1

            if force_top_k is not None and len(selected_actions) >= force_top_k:
                break

    return selected_actions
