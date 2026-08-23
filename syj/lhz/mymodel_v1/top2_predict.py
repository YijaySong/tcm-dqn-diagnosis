# -*- coding: utf-8 -*-
"""V1 固定 Top-2 对照预测入口，不代表模型自主停止策略。"""

import argparse

import torch

from action import legal_action_mask
from constants import device
from predict_core import load_recommender, normalize_symptoms


def predict_top2(model, env, symptoms):
    selected = []
    state = env.reset((symptoms, []))
    model.eval()
    with torch.no_grad():
        while len(selected) < min(2, env.Se_action_num):
            tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = legal_action_mask(model(tensor), [selected], env, min_actions_before_stop=999)
            action = int(q_values.max(1).indices.item())
            selected.append(action)
            state[env.symp_len + action] = 1.0
    return [env.action_space[action] for action in selected]


def main():
    parser = argparse.ArgumentParser(description='V1 固定Top-2对照预测')
    parser.add_argument('symptoms', nargs='*')
    parser.add_argument('--model-path', required=True)
    args = parser.parse_args()
    model, env, _ = load_recommender(args.model_path)
    symptoms = normalize_symptoms(','.join(args.symptoms))
    known = [item for item in symptoms if item in env.symptom_to_index]
    print(', '.join(predict_top2(model, env, known)))


if __name__ == '__main__':
    main()
