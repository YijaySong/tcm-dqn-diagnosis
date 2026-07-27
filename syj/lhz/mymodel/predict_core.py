# -*- coding: utf-8 -*-
"""预测共享核心函数。

供predict_nontrace.py和predict_trace.py复用，负责加载模型、构建映射、
执行普通预测和逐步轨迹预测。
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STOP_ACTION = "停止"
NEG_INF = -1e9

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32, device='cpu'))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32, device='cpu'))
        self.mu_bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32, device='cpu'))
        self.sigma_bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32, device='cpu'))
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features, dtype=torch.float32))
        self.register_buffer('epsilon_bias', torch.zeros(out_features, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        self.mu_weight.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_weight.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def forward(self, x):
        return F.linear(x, self.mu_weight, self.mu_bias)


def linear_layer(in_features, out_features, noisy=False):
    if noisy:
        return NoisyLinear(in_features, out_features)
    return nn.Linear(in_features, out_features, dtype=torch.float32, device='cpu')


class DQNLegacy(nn.Module):
    def __init__(self, state_vector_len, n_actions, n_units=128, n_units2=64, dropout=0.1):
        super(DQNLegacy, self).__init__()
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


class DQN(nn.Module):
    def __init__(self, state_vector_len, n_actions, n_units=128, n_units2=64, dropout=0.1):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_vector_len, n_units, dtype=torch.float32, device='cpu'),
            nn.LayerNorm(n_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_units, n_units2, dtype=torch.float32, device='cpu'),
            nn.LayerNorm(n_units2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_units2, n_actions, dtype=torch.float32, device='cpu')
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, n_units, dropout=0.1, noisy=False):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            linear_layer(n_units, n_units, noisy=noisy),
            nn.LayerNorm(n_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            linear_layer(n_units, n_units, noisy=noisy),
            nn.LayerNorm(n_units),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class DuelingDQN(nn.Module):
    def __init__(self, state_vector_len, n_actions, n_units=256, n_units2=128, dropout=0.15):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_vector_len, n_units, dtype=torch.float32, device='cpu'),
            nn.LayerNorm(n_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_units, n_units2, dtype=torch.float32, device='cpu'),
            nn.LayerNorm(n_units2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(n_units2, n_units2, dtype=torch.float32, device='cpu'),
            nn.ReLU(),
            nn.Linear(n_units2, 1, dtype=torch.float32, device='cpu'),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(n_units2, n_units2, dtype=torch.float32, device='cpu'),
            nn.ReLU(),
            nn.Linear(n_units2, n_actions, dtype=torch.float32, device='cpu'),
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class ResidualDuelingDQN(nn.Module):
    def __init__(self, state_vector_len, n_actions, n_units=256, n_units2=128, dropout=0.15):
        super(ResidualDuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_vector_len, n_units, dtype=torch.float32, device='cpu'),
            nn.LayerNorm(n_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResidualBlock(n_units, dropout),
            ResidualBlock(n_units, dropout),
            nn.Linear(n_units, n_units2, dtype=torch.float32, device='cpu'),
            nn.LayerNorm(n_units2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(n_units2, n_units2, dtype=torch.float32, device='cpu'),
            nn.LayerNorm(n_units2),
            nn.ReLU(),
            nn.Linear(n_units2, 1, dtype=torch.float32, device='cpu'),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(n_units2, n_units2, dtype=torch.float32, device='cpu'),
            nn.LayerNorm(n_units2),
            nn.ReLU(),
            nn.Linear(n_units2, n_actions, dtype=torch.float32, device='cpu'),
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class SetAwareDuelingDQN(nn.Module):
    def __init__(self, state_vector_len, n_actions, n_units=256, n_units2=128, dropout=0.15, noisy=False):
        super(SetAwareDuelingDQN, self).__init__()
        self.Se_action_num = n_actions - 1
        self.symp_len = state_vector_len - self.Se_action_num
        if self.symp_len <= 0 or self.Se_action_num <= 0:
            raise ValueError('set_dueling需要状态向量由症状块和证候要素块组成')

        self.symptom_encoder = nn.Sequential(
            linear_layer(self.symp_len, n_units, noisy=noisy),
            nn.LayerNorm(n_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResidualBlock(n_units, dropout, noisy=noisy),
        )
        self.selected_encoder = nn.Sequential(
            linear_layer(self.Se_action_num, n_units, noisy=noisy),
            nn.LayerNorm(n_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResidualBlock(n_units, dropout, noisy=noisy),
        )
        self.joint = nn.Sequential(
            linear_layer(n_units * 3, n_units, noisy=noisy),
            nn.LayerNorm(n_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResidualBlock(n_units, dropout, noisy=noisy),
            linear_layer(n_units, n_units2, noisy=noisy),
            nn.LayerNorm(n_units2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.value_stream = nn.Sequential(
            linear_layer(n_units2, n_units2, noisy=noisy),
            nn.LayerNorm(n_units2),
            nn.ReLU(),
            linear_layer(n_units2, 1, noisy=noisy),
        )
        self.advantage_stream = nn.Sequential(
            linear_layer(n_units2, n_units2, noisy=noisy),
            nn.LayerNorm(n_units2),
            nn.ReLU(),
            linear_layer(n_units2, n_actions, noisy=noisy),
        )

    def forward(self, x):
        symptom_state = x[:, :self.symp_len]
        selected_state = x[:, self.symp_len:self.symp_len + self.Se_action_num]
        symptom_features = self.symptom_encoder(symptom_state)
        selected_features = self.selected_encoder(selected_state)
        features = self.joint(torch.cat([
            symptom_features,
            selected_features,
            symptom_features * selected_features,
        ], dim=1))
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def build_q_network(model_type, state_vector_len, n_actions, n_units=128, n_units2=64, dropout=0.1):
    if model_type == 'dqn_legacy':
        return DQNLegacy(state_vector_len, n_actions, n_units, n_units2, dropout)
    if model_type == 'dqn':
        return DQN(state_vector_len, n_actions, n_units, n_units2, dropout)
    if model_type == 'dueling':
        return DuelingDQN(state_vector_len, n_actions, n_units, n_units2, dropout)
    if model_type == 'dueling_residual':
        return ResidualDuelingDQN(state_vector_len, n_actions, n_units, n_units2, dropout)
    if model_type == 'set_dueling':
        return SetAwareDuelingDQN(state_vector_len, n_actions, n_units, n_units2, dropout, noisy=False)
    if model_type == 'set_dueling_noisy':
        return SetAwareDuelingDQN(state_vector_len, n_actions, n_units, n_units2, dropout, noisy=True)
    raise ValueError(f'未知Q网络类型: {model_type}')


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


def project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def default_data_path():
    return os.path.join(project_root(), 'dataset', 'lhz_data.txt')


def get_tcm_data(filename):
    """读取数据集，构建症状和证候要素映射。"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"未找到数据集文件：{filename}")

    with open(filename, 'r', encoding='utf-8') as file:
        file.readline()
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

        symptoms = [s.strip() for s in parts[1].split(',') if s.strip()]
        Se_list = [s.strip() for s in parts[2].split(',') if s.strip()]

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


def resolve_model_path(model_path=None):
    """解析模型路径；默认优先使用syj/lhz目录下保存的模型。"""
    if model_path:
        if os.path.exists(model_path):
            return model_path
        script_relative = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
        if os.path.exists(script_relative):
            return script_relative
        root_relative = os.path.join(project_root(), model_path)
        if os.path.exists(root_relative):
            return root_relative
        return model_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, 'dqn_model.pth'),
        os.path.join(project_root(), 'dqn_model.pth'),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def load_checkpoint(model_path):
    """加载本地可信模型文件，兼容新checkpoint和旧state_dict。"""
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


def load_recommender(model_path=None, data_path=None, nn_units=128, nn_units2=64, dropout=0.1):
    """加载训练好的推荐器，返回(model, env, metadata)。"""
    resolved_model_path = resolve_model_path(model_path)
    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(f"未找到模型文件：{resolved_model_path}")

    checkpoint = load_checkpoint(resolved_model_path)
    metadata = {'model_path': resolved_model_path}

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        symptoms = checkpoint['symptoms']
        Se = checkpoint['Se']
        nn_units = checkpoint.get('nn_units', nn_units)
        nn_units2 = checkpoint.get('nn_units2', nn_units2)
        dropout = checkpoint.get('dropout', dropout)
        model_type = checkpoint.get('model_type', 'dqn_legacy')
        model_state_dict = checkpoint['model_state_dict']
        metadata.update({k: v for k, v in checkpoint.items() if k != 'model_state_dict'})
    else:
        resolved_data_path = data_path or default_data_path()
        symptoms, Se, _ = get_tcm_data(resolved_data_path)
        model_state_dict = checkpoint
        model_type = 'dqn_legacy'
        metadata['data_path'] = resolved_data_path

    env = Environment(symptoms, Se)
    training_config = metadata.get('training_config', {}) if isinstance(metadata, dict) else {}
    env.default_min_actions = int(training_config.get('min_actions_before_stop', 0) or 0)
    env.default_stop_margin_threshold = training_config.get('stop_margin_threshold')
    model = build_q_network(model_type, len(env.state_space), len(env.action_space), nn_units, nn_units2, dropout).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model, env, metadata


def mask_selected_actions(q_values, selected_actions, env):
    """将已选动作的Q值设为负无穷，避免重复选择。"""
    masked = q_values.clone()
    for action_idx in selected_actions:
        if 0 <= action_idx < env.Se_action_num:
            masked[0, action_idx] = NEG_INF
    return masked


def predict_symptoms(model, env, symptoms_list, max_actions=None, stop_margin_threshold=None, min_actions=None):
    """给定症状列表，模型自主选择证候要素，直到选择停止或达到上限。"""
    model.eval()
    state_np = env.reset(symptoms_list)
    selected_actions = []
    max_actions = env.Se_action_num if max_actions is None else min(max_actions, env.Se_action_num)
    if min_actions is None:
        min_actions = int(getattr(env, 'default_min_actions', 0) or 0)
    if stop_margin_threshold is None:
        stop_margin_threshold = getattr(env, 'default_stop_margin_threshold', None)

    with torch.no_grad():
        while len(selected_actions) < max_actions:
            state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = model(state_tensor)

            mask_stop = len(selected_actions) < min_actions
            label_q = q_values[:, :env.Se_action_num].clone()
            for action_idx in selected_actions:
                if 0 <= action_idx < env.Se_action_num:
                    label_q[0, action_idx] = NEG_INF
            best_label_q = label_q.max(1).values
            stop_q = q_values[:, env.stop_action]
            if (
                stop_margin_threshold is not None
                and len(selected_actions) >= min_actions
                and (stop_q - best_label_q).item() >= stop_margin_threshold
            ):
                break
            if mask_stop:
                q_values[0, env.stop_action] = NEG_INF
            q_values = mask_selected_actions(q_values, selected_actions, env)

            action_idx = q_values.max(1).indices.item()
            if action_idx == env.stop_action:
                break

            selected_actions.append(action_idx)
            state_np[env.symp_len + action_idx] = 1

    return [env.action_space[idx] for idx in selected_actions]


def top_candidates_from_q_values(q_values, env, candidate_topk):
    """从已完成mask的Q值中提取候选动作排名。"""
    candidate_topk = max(0, int(candidate_topk or 0))
    if candidate_topk <= 0:
        return []

    row = q_values[0]
    valid_count = int((row > NEG_INF / 2).sum().item())
    if valid_count <= 0:
        return []

    topk = min(candidate_topk, valid_count, len(env.action_space))
    values, indices = torch.topk(row, k=topk)
    candidates = []
    for value, action_idx in zip(values.tolist(), indices.tolist()):
        if value <= NEG_INF / 2:
            continue
        candidates.append({
            'name': env.action_space[action_idx],
            'action_index': int(action_idx),
            'q_value': float(value),
        })
    return candidates


def predict_symptoms_trace(model, env, symptoms_list, max_actions=None, candidate_topk=0, stop_margin_threshold=None, min_actions=None):
    """给定症状列表，返回模型逐步选择证候要素的轨迹。"""
    model.eval()
    state_np = env.reset(symptoms_list)
    selected_actions = []
    steps = []
    max_actions = env.Se_action_num if max_actions is None else min(max_actions, env.Se_action_num)
    if min_actions is None:
        min_actions = int(getattr(env, 'default_min_actions', 0) or 0)
    if stop_margin_threshold is None:
        stop_margin_threshold = getattr(env, 'default_stop_margin_threshold', None)

    with torch.no_grad():
        while len(selected_actions) < max_actions:
            state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = model(state_tensor)

            mask_stop = len(selected_actions) < min_actions
            label_q = q_values[:, :env.Se_action_num].clone()
            for selected_action in selected_actions:
                if 0 <= selected_action < env.Se_action_num:
                    label_q[0, selected_action] = NEG_INF
            best_label_q, best_label_action = label_q.max(1)
            stop_q = q_values[:, env.stop_action]
            threshold_stop = (
                stop_margin_threshold is not None
                and len(selected_actions) >= min_actions
                and (stop_q - best_label_q).item() >= stop_margin_threshold
            )

            if mask_stop:
                q_values[0, env.stop_action] = NEG_INF
            q_values = mask_selected_actions(q_values, selected_actions, env)

            action_idx = env.stop_action if threshold_stop else q_values.max(1).indices.item()
            selected_before = [env.action_space[idx] for idx in selected_actions]
            is_stop = action_idx == env.stop_action
            steps.append({
                'step': len(steps) + 1,
                'selected_before': selected_before,
                'action': env.action_space[action_idx],
                'action_index': int(action_idx),
                'is_stop': is_stop,
                'q_value': float(stop_q.item() if is_stop else q_values[0, action_idx].item()),
                'best_label_q': float(best_label_q.item()),
                'best_label_action': int(best_label_action.item()),
                'stop_margin': float((stop_q - best_label_q).item()),
                'threshold_stop': threshold_stop,
                'top_candidates': top_candidates_from_q_values(q_values, env, candidate_topk),
            })

            if is_stop:
                break

            selected_actions.append(action_idx)
            state_np[env.symp_len + action_idx] = 1

    return {
        'recommendations': [env.action_space[idx] for idx in selected_actions],
        'steps': steps,
    }


def normalize_symptoms(symptoms):
    if isinstance(symptoms, str):
        symptoms = symptoms.replace('，', ',')
        return [s.strip() for s in symptoms.split(',') if s.strip()]
    normalized = []
    for symptom in symptoms:
        normalized.extend(s.strip() for s in str(symptom).replace('，', ',').split(',') if s.strip())
    return normalized


def format_selected(names):
    """格式化证候要素列表，空列表显示为无。"""
    return ', '.join(names) if names else '无'


def print_trace_prediction(trace_result, known_symptoms):
    """打印逐步辨证过程。"""
    print(f"  输入刻下症: {format_selected(known_symptoms)}")
    print("\n  === DQN辨证过程 ===")

    steps = trace_result.get('steps', [])
    if not steps:
        print("  无有效辨证步骤。")
    for step in steps:
        print(f"  第{step['step']}步:")
        print(f"    当前已选证候要素: {format_selected(step['selected_before'])}")
        print(f"    模型选择: {step['action']}")
        candidates = step.get('top_candidates') or []
        if candidates:
            ranking = ', '.join(
                f"{item['name']}({item['q_value']:.4f})" for item in candidates
            )
            print(f"    候选排名: {ranking}")
        print()

    print(f"  最终推荐证候要素: {format_selected(trace_result.get('recommendations', []))}")
