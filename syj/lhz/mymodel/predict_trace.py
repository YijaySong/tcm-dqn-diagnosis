# -*- coding: utf-8 -*-
"""过程版辨证预测入口。

运行本文件后，直接输入一组刻下症，程序会依次输出模型每一步选择的证候要素，
并默认显示每一步Q值排名靠前的3个候选动作。
"""

import argparse

from predict_core import (
    load_recommender,
    normalize_symptoms,
    predict_symptoms_trace,
    print_trace_prediction,
)

CANDIDATE_TOPK = 3
DEFAULT_SYMPTOMS = '胸闷,胸痛,畏寒'


def run_prediction(model, env, symptoms_text, candidate_topk):
    symptoms_list = normalize_symptoms(symptoms_text)
    known = [symptom for symptom in symptoms_list if symptom in env.swapped_state_space]
    unknown = [symptom for symptom in symptoms_list if symptom not in env.swapped_state_space]

    if unknown:
        print(f"[警告] 以下症状不在训练集中，将被忽略: {unknown}")
    if not known:
        print("[错误] 没有有效的症状！\n")
        return

    trace_result = predict_symptoms_trace(
        model, env, known, candidate_topk=candidate_topk
    )
    print_trace_prediction(trace_result, known)
    print()


def build_parser():
    parser = argparse.ArgumentParser(description='过程版LHZ辨证预测入口')
    parser.add_argument('symptoms', nargs='*', help='症状文本；多个症状可用逗号分隔，也可用空格分隔')
    parser.add_argument('--candidate-topk', type=int, default=CANDIDATE_TOPK, help='每步显示Q值排名靠前的候选动作数量')
    parser.add_argument('--example', action='store_true', help='使用内置示例症状运行一次')
    return parser


def main():
    args = build_parser().parse_args()
    model, env, metadata = load_recommender()
    print(f"加载模型: {metadata['model_path']}")
    print(f"本入口会逐步输出辨证过程，并显示每步候选动作排名前{args.candidate_topk}项。")

    if args.example:
        run_prediction(model, env, DEFAULT_SYMPTOMS, args.candidate_topk)
        return

    if args.symptoms:
        run_prediction(model, env, ','.join(args.symptoms), args.candidate_topk)
        return

    print("请输入症状，多个症状用逗号分隔。输入 q 退出。")
    print("示例: 胸闷,胸痛,畏寒\n")

    while True:
        symptoms_text = input("请输入症状: ").strip()
        if symptoms_text.lower() == 'q':
            print("退出。")
            break
        if symptoms_text:
            run_prediction(model, env, symptoms_text, args.candidate_topk)


if __name__ == "__main__":
    main()
