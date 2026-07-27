# -*- coding: utf-8 -*-
"""V1 状态编码与自主 STOP 推理。"""

import numpy as np
import torch

from action import legal_action_mask
from constants import NEG_INF


def build_state_vector(env, symptoms, selected_actions=None):
    selected_actions = selected_actions or []
    state = np.zeros(len(env.state_space), dtype=np.float32)
    for symptom in symptoms:
        index = env.symptom_to_index.get(symptom)
        if index is not None:
            state[index] = 1.0
    for action in selected_actions:
        if 0 <= action < env.Se_action_num:
            state[env.symp_len + action] = 1.0
    return state


def actions_to_multihot(env, actions):
    vector = np.zeros(env.Se_action_num, dtype=int)
    for action in actions:
        if 0 <= action < env.Se_action_num:
            vector[action] = 1
    return vector


def Se_names_to_multihot(env, label_names):
    vector = np.zeros(env.Se_action_num, dtype=int)
    for label in label_names:
        action = env.label_to_action.get(label)
        if action is not None:
            vector[action] = 1
    return vector


def predict_actions_from_state(env, policy_net, device, state_np, max_actions=None,
                               stop_margin_threshold=None, min_actions=1, return_trace=False,
                               candidate_topk=0):
    policy_net.eval()
    selected, trace = [], []
    max_actions = env.Se_action_num if max_actions is None else min(int(max_actions), env.Se_action_num)
    with torch.no_grad():
        while len(selected) < max_actions:
            state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = policy_net(state)
            masked = legal_action_mask(q_values, [selected], env, min_actions)
            label_q = masked[:, :env.Se_action_num]
            best_label_q, best_label = label_q.max(1)
            stop_q = q_values[:, env.stop_action]
            stop_margin = float((stop_q - best_label_q).item())

            forced_stop = False
            # 若设置 margin，STOP 只有在明显优于最佳标签时才可被选择。
            if stop_margin_threshold is not None and len(selected) >= min_actions:
                if stop_margin >= float(stop_margin_threshold):
                    forced_stop = True
                else:
                    masked[:, env.stop_action] = NEG_INF

            action = env.stop_action if forced_stop else int(masked.max(1).indices.item())
            is_stop = action == env.stop_action
            candidates = []
            if candidate_topk > 0:
                count = min(int(candidate_topk), masked.shape[1])
                values, indices = masked[0].topk(count)
                for value, index in zip(values.tolist(), indices.tolist()):
                    if value <= -1e8:
                        continue
                    candidates.append({
                        'action_index': int(index),
                        'action_name': env.action_space[int(index)],
                        'q_value': float(value),
                        'is_stop': int(index) == env.stop_action,
                    })
            trace.append({
                'action_index': action,
                'action_name': env.action_space[action],
                'is_stop': is_stop,
                'q_value': float(q_values[0, action].item()),
                'best_label_action': int(best_label.item()),
                'best_label_q': float(best_label_q.item()),
                'stop_margin': stop_margin,
                'candidates': candidates,
            })
            if is_stop:
                break
            selected.append(action)
            state_np[env.symp_len + action] = 1.0

    if return_trace:
        return selected, trace
    return selected
