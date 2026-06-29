# -*- coding: utf-8 -*-
"""过程版辨证预测入口。

运行本文件后，直接输入一组刻下症，程序会依次输出模型每一步选择的证候要素，
并默认显示每一步Q值最高的前3个候选动作。
"""

from predict_core import (
    load_recommender,
    normalize_symptoms,
    predict_symptoms_trace,
    print_trace_prediction,
)

CANDIDATE_TOPK = 3


def main():
    model, env, metadata = load_recommender()
    print(f"加载模型: {metadata['model_path']}")
    print("请输入症状，多个症状用逗号分隔。输入 q 退出。")
    print(f"本入口会逐步输出辨证过程，并默认显示每步Top-{CANDIDATE_TOPK}候选动作。")
    print("示例: 胸闷,胸痛,畏寒\n")

    while True:
        symptoms_text = input("请输入症状: ").strip()
        if symptoms_text.lower() == 'q':
            print("退出。")
            break
        if not symptoms_text:
            continue

        symptoms_list = normalize_symptoms(symptoms_text)
        known = [symptom for symptom in symptoms_list if symptom in env.swapped_state_space]
        unknown = [symptom for symptom in symptoms_list if symptom not in env.swapped_state_space]

        if unknown:
            print(f"[警告] 以下症状不在训练集中，将被忽略: {unknown}")
        if not known:
            print("[错误] 没有有效的症状！\n")
            continue

        trace_result = predict_symptoms_trace(
            model, env, known, candidate_topk=CANDIDATE_TOPK
        )
        print_trace_prediction(trace_result, known)
        print()


if __name__ == "__main__":
    main()
