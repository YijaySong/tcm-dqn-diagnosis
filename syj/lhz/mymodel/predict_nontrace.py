# -*- coding: utf-8 -*-
"""简洁版辨证预测入口。

运行本文件后，直接输入一组刻下症，程序输出最终推荐证候要素。
"""

from predict_core import load_recommender, normalize_symptoms, predict_symptoms


def main():
    model, env, metadata = load_recommender()
    print(f"加载模型: {metadata['model_path']}")
    print("请输入症状，多个症状用逗号分隔。输入 q 退出。")
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

        recommendations = predict_symptoms(model, env, known)
        print(f"推荐证候要素: {', '.join(recommendations) if recommendations else '无'}\n")


if __name__ == "__main__":
    main()
