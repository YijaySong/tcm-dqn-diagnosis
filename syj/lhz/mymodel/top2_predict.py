# -*- coding: utf-8 -*-
"""固定Top-2预测独立入口。

本文件专门保留从旧主流程中移出的固定Top-2预测逻辑。它不参与main.py训练、
测试集评估和常规auto预测；只有单独运行本文件时，才会强制输出Q值最高的2个
证候要素，用于临时对照或人工检查。
"""

import argparse

import torch

from constants import NEG_INF, device
from predict_core import load_recommender, normalize_symptoms

DEFAULT_SYMPTOMS = '胸闷,胸痛,畏寒'


def predict_top2(model, env, symptoms_list):
    model.eval()
    state_np = env.reset(symptoms_list)
    selected_actions = []

    with torch.no_grad():
        while len(selected_actions) < min(2, env.Se_action_num):
            state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = model(state_tensor)
            q_values[0, env.stop_action] = NEG_INF
            for action_idx in selected_actions:
                if 0 <= action_idx < env.Se_action_num:
                    q_values[0, action_idx] = NEG_INF

            action_idx = q_values.max(1).indices.item()
            if action_idx == env.stop_action:
                break

            selected_actions.append(action_idx)
            state_np[env.symp_len + action_idx] = 1

    return [env.action_space[action_idx] for action_idx in selected_actions]


def run_prediction(model, env, symptoms_text):
    symptoms_list = normalize_symptoms(symptoms_text)
    known = [symptom for symptom in symptoms_list if symptom in env.swapped_state_space]
    unknown = [symptom for symptom in symptoms_list if symptom not in env.swapped_state_space]

    if unknown:
        print(f"[警告] 以下症状不在训练集中，将被忽略: {unknown}")
    if not known:
        print("[错误] 没有有效的症状！\n")
        return

    recommendations = predict_top2(model, env, known)
    print(f"输入症状: {', '.join(known)}")
    print(f"固定Top-2推荐证候要素: {', '.join(recommendations) if recommendations else '无'}\n")


def build_parser():
    parser = argparse.ArgumentParser(description='固定Top-2辨证预测独立入口')
    parser.add_argument('symptoms', nargs='*', help='症状文本；多个症状可用逗号分隔，也可用空格分隔')
    parser.add_argument('--example', action='store_true', help='使用内置示例症状运行一次')
    return parser


def main():
    args = build_parser().parse_args()
    model, env, metadata = load_recommender()
    print(f"加载模型: {metadata['model_path']}")
    print("注意：本文件是固定Top-2独立对照入口，不代表模型自主停止输出。")

    if args.example:
        run_prediction(model, env, DEFAULT_SYMPTOMS)
        return

    if args.symptoms:
        run_prediction(model, env, ','.join(args.symptoms))
        return

    print("请输入症状，多个症状用逗号分隔。输入 q 退出。")
    print("示例: 胸闷,胸痛,畏寒\n")

    while True:
        symptoms_text = input("请输入症状: ").strip()
        if symptoms_text.lower() == 'q':
            print("退出。")
            break
        if symptoms_text:
            run_prediction(model, env, symptoms_text)


if __name__ == "__main__":
    main()
