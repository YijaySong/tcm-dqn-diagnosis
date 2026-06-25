# -*- coding: utf-8 -*-
"""加载训练好的DQN模型，对输入的症状进行辨证预测（证候要素推荐）"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

# ---------- 设备 ----------
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

STOP_ACTION = "停止"
NEG_INF = -1e9


# ---------- DQN网络（与训练时完全一致） ----------
class DQN(nn.Module):
    def __init__(self, state_vector_len, n_actions, n_units=128, n_units2=64, dropout=0.1):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_vector_len, n_units, dtype=torch.float32, device='cpu'),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_units, n_units2, dtype=torch.float32, device='cpu'),
            nn.ReLU(),
            nn.Linear(n_units2, n_actions, dtype=torch.float32, device='cpu')
        )

    def forward(self, x):
        return self.net(x)


# ---------- 环境（与训练时一致，用于构建状态空间和动作空间映射） ----------
class Environment(object):
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

    def reset(self, symptoms_list):
        state = np.zeros(len(self.state_space), dtype=np.float32)
        for symp in symptoms_list:
            if symp in self.swapped_state_space:
                state[self.swapped_state_space[symp]] = 1
        return state


# ---------- 数据加载（与训练时一致） ----------
def get_tcm_data(filename):
    """读取数据集，构建症状和证候要素映射"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"未找到数据集文件：{filename}")

    with open(filename, 'r', encoding='utf-8') as file:
        file.readline()  # 跳过标题行
        lines = [line.strip() for line in file if line.strip()]

    symptom_set = set()
    Se_set = set()
    symptom_map = {}
    Se_map = {}
    Se_freq_map = {}

    for line in lines:
        parts = line.split(maxsplit=2)
        if len(parts) != 3 or not parts[1].strip() or not parts[2].strip():
            continue

        symptoms = [s for s in parts[1].split(',') if s]
        Se_list = [s for s in parts[2].split(',') if s]

        for Se in Se_list:
            if Se not in Se_set:
                Se_map[len(Se_set)] = Se
                Se_set.add(Se)
                Se_freq_map[Se] = 1
            else:
                Se_freq_map[Se] += 1

        for symp in symptoms:
            if symp not in symptom_set:
                symptom_map[len(symptom_set)] = symp
                symptom_set.add(symp)

    return symptom_map, Se_map, Se_freq_map


# ---------- 预测核心函数 ----------
def mask_selected_actions(q_values, selected_actions):
    """将已选动作的Q值设为负无穷，避免重复选择"""
    masked = q_values.clone()
    for action_idx in selected_actions:
        if 0 <= action_idx < env.Se_action_num:
            masked[0, action_idx] = NEG_INF
    return masked


def predict_symptoms(model, env, symptoms_list, force_top_k=None, max_actions=None):
    """
    给定症状列表，模型自主选择证候要素，直到选择"停止"或达到上限。

    参数:
        model: 训练好的DQN网络
        env: Environment实例（含症状/证候要素映射）
        symptoms_list: 症状名列表，如 ['胸闷', '胸痛', '畏寒']
        force_top_k: 强制选Top-k个（None则自主停止）
        max_actions: 最大可选动作数

    返回:
        list[str]: 推荐的证候要素名称列表
    """
    model.eval()
    state_np = env.reset(symptoms_list)
    selected_actions = []
    max_actions = env.Se_action_num if max_actions is None else min(max_actions, env.Se_action_num)

    with torch.no_grad():
        while len(selected_actions) < max_actions:
            state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = model(state_tensor)

            # 屏蔽已选动作；若要求Top-k则同时屏蔽停止动作
            mask_stop = force_top_k is not None and len(selected_actions) < force_top_k
            if mask_stop:
                q_values[0, env.stop_action] = NEG_INF
            q_values = mask_selected_actions(q_values, selected_actions)

            action_idx = q_values.max(1).indices.item()

            if action_idx == env.stop_action:
                break

            selected_actions.append(action_idx)
            state_np[env.symp_len + action_idx] = 1

            if force_top_k is not None and len(selected_actions) >= force_top_k:
                break

    return [env.action_space[idx] for idx in selected_actions]


# ========== 主程序 ==========
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DQN辨证预测")
    parser.add_argument("--model", type=str, default="dqn_model_best.pth",
                        help="模型权重文件路径（默认: dqn_model_best.pth）")
    parser.add_argument("--data", type=str, default=None,
                        help="数据集路径（默认: ../../dataset/lhz_data.txt）")
    parser.add_argument("--symptoms", type=str, default=None,
                        help="症状，逗号分隔（如: 胸闷,胸痛,畏寒）")
    parser.add_argument("--topk", type=int, default=0,
                        help="强制输出Top-k个证候要素（0=自主停止）")
    parser.add_argument("--interactive", action="store_true",
                        help="交互模式，逐行输入症状")
    args = parser.parse_args()

    # 1. 加载数据集，构建环境映射
    if args.data is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, '..', '..', 'dataset', 'lhz_data.txt')
    else:
        data_path = args.data

    print(f"加载数据集: {data_path}")
    symptom_map, Se_map, Se_freq_map = get_tcm_data(data_path)
    print(f"  刻下症数: {len(symptom_map)}, 证候要素数: {len(Se_map)}")

    # 2. 构建环境
    env = Environment(symptom_map, Se_map)
    state_vector_len = len(env.state_space)
    n_actions = len(env.action_space)
    print(f"  状态向量长度: {state_vector_len}, 动作数: {n_actions}")

    # 3. 加载模型
    if args.model is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'dqn_model_best.pth')
    else:
        model_path = args.model

    print(f"加载模型: {model_path}")
    model = DQN(state_vector_len, n_actions, 128, 64, 0.1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print("  模型加载成功！")

    # 4. 预测
    if args.interactive:
        print("\n=== 交互预测模式 ===")
        print("输入症状（逗号分隔），输入 q 退出")
        print("可用症状示例: 胸闷, 胸痛, 畏寒, 纳呆, 舌淡, 舌苔白, 细脉, 弱脉\n")
        while True:
            user_input = input("请输入症状: ").strip()
            if user_input.lower() == 'q':
                break
            if not user_input:
                continue

            symptoms_list = [s.strip() for s in user_input.split(',') if s.strip()]

            # 检查哪些症状在训练集中存在
            known = [s for s in symptoms_list if s in env.swapped_state_space]
            unknown = [s for s in symptoms_list if s not in env.swapped_state_space]
            if unknown:
                print(f"  [警告] 以下症状不在训练集中，将被忽略: {unknown}")

            if not known:
                print("  [错误] 没有有效的症状！")
                continue

            # 自主停止预测
            result_auto = predict_symptoms(model, env, known)
            print(f"  自主停止推荐: {', '.join(result_auto) if result_auto else '无'}")

            # Top-2 预测
            result_top2 = predict_symptoms(model, env, known, force_top_k=2, max_actions=2)
            print(f"  Top-2推荐:     {', '.join(result_top2) if result_top2 else '无'}")
            print()

    elif args.symptoms:
        symptoms_list = [s.strip() for s in args.symptoms.split(',') if s.strip()]
        known = [s for s in symptoms_list if s in env.swapped_state_space]
        unknown = [s for s in symptoms_list if s not in env.swapped_state_space]
        if unknown:
            print(f"[警告] 以下症状不在训练集中，将被忽略: {unknown}")

        if not known:
            print("[错误] 没有有效的症状！")
            sys.exit(1)

        if args.topk > 0:
            result = predict_symptoms(model, env, known, force_top_k=args.topk, max_actions=args.topk)
            print(f"Top-{args.topk}推荐证候要素: {', '.join(result) if result else '无'}")
        else:
            result = predict_symptoms(model, env, known)
            print(f"自主停止推荐证候要素: {', '.join(result) if result else '无'}")

            result_top2 = predict_symptoms(model, env, known, force_top_k=2, max_actions=2)
            print(f"Top-2推荐证候要素:     {', '.join(result_top2) if result_top2 else '无'}")

    else:
        # 默认：交互模式，提示用户输入
        print("\n" + "=" * 60)
        print("  DQN 辨证论治 - 证候要素推荐")
        print("=" * 60)
        print("  输入症状（逗号分隔），输入 q 退出")
        print("  可用症状示例: 胸闷, 胸痛, 畏寒, 纳呆, 舌淡, 舌苔白, 细脉\n")

        while True:
            user_input = input("请输入症状: ").strip()
            if user_input.lower() == 'q':
                print("退出。")
                break
            if not user_input:
                continue

            symptoms_list = [s.strip() for s in user_input.split(',') if s.strip()]
            known = [s for s in symptoms_list if s in env.swapped_state_space]
            unknown = [s for s in symptoms_list if s not in env.swapped_state_space]
            if unknown:
                print(f"  [警告] 以下症状不在训练集中，将被忽略: {unknown}")
            if not known:
                print("  [错误] 没有有效的症状！\n")
                continue

            result_auto = predict_symptoms(model, env, known)
            result_top2 = predict_symptoms(model, env, known, force_top_k=2, max_actions=2)
            print(f"  自主停止推荐: {', '.join(result_auto) if result_auto else '无'}")
            print(f"  Top-2推荐:     {', '.join(result_top2) if result_top2 else '无'}")
            print()
