# -*- coding: utf-8 -*-
"""DQN智能辨证：输入刻下症，输出推荐的证候要素"""

import os
import math
import copy
import numpy as np
import random
from collections import namedtuple, deque, Counter

import torch
import torch.nn as nn
import torch.optim as optim

import time
from datetime import datetime
import logging
from argparse import ArgumentParser

try:
    import importlib
    sklearn_metrics = importlib.import_module('sklearn.metrics')
    sklearn_model_selection = importlib.import_module('sklearn.model_selection')
    precision_recall_fscore_support = sklearn_metrics.precision_recall_fscore_support
    hamming_loss = sklearn_metrics.hamming_loss
    train_test_split = sklearn_model_selection.train_test_split
except ImportError:
    precision_recall_fscore_support = None
    hamming_loss = None
    train_test_split = None


def createLogger(logname):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(logname, mode='w+')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s:  %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

STOP_ACTION = "停止"
NEG_INF = -1e9


def calc_tp_fp_fn(selected, true):
    selected_set = set(selected)
    true_set = set(true)
    tp = len(selected_set & true_set)
    fp = len(selected_set - true_set)
    fn = len(true_set - selected_set)
    return tp, fp, fn


def set_precision(selected, true):
    tp, fp, _ = calc_tp_fp_fn(selected, true)
    return tp / (tp + fp) if tp + fp > 0 else 0.0


def set_recall(selected, true):
    tp, _, fn = calc_tp_fp_fn(selected, true)
    return tp / (tp + fn) if tp + fn > 0 else 0.0


def set_f1(selected, true):
    p = set_precision(selected, true)
    r = set_recall(selected, true)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def set_jaccard(selected, true):
    selected_set = set(selected)
    true_set = set(true)
    union = selected_set | true_set
    return len(selected_set & true_set) / len(union) if union else 0.0


class Environment(object):
    """强化学习环境：状态空间=刻下症+已选证候要素，动作空间=候选证候要素+停止"""
    def __init__(self, symptoms, Se):
        # 状态空间：刻下症(0..symp_len-1) + 证候要素(symp_len..)
        # “停止”只作为动作，不写入状态空间。
        self.state_space = {}
        self.swapped_state_space = {}
        self.symp_len = len(symptoms)
        for index in symptoms:
            self.state_space[index] = symptoms[index]
            self.swapped_state_space[symptoms[index]] = index
        for index in Se:
            self.state_space[index + self.symp_len] = Se[index]
            self.swapped_state_space[Se[index]] = index + self.symp_len

        # 动作空间：候选证候要素 + 停止动作
        self.Se_action_num = len(Se)
        self.stop_action = self.Se_action_num
        self.action_space = dict(Se)
        self.action_space[self.stop_action] = STOP_ACTION
        self.swapped_action_space = {}
        for index in self.action_space:
            self.swapped_action_space[self.action_space[index]] = index
        self.Se_weights = {index: 1.0 for index in range(self.Se_action_num)}

    def set_Se_weights(self, Se_weights):
        self.Se_weights = Se_weights

    def reset(self, data_piece):
        self.piece_symptoms = data_piece[0]
        self.piece_Se = [self.swapped_action_space[Se] for Se in data_piece[1]]
        self.state = np.zeros(len(self.state_space), dtype=int)
        for symp in self.piece_symptoms:
            self.state[self.swapped_state_space[symp]] = 1
        return self.state

    def _terminal_reward(self, selected_Se):
        _, fp, fn = calc_tp_fp_fn(selected_Se, self.piece_Se)
        if fp == 0 and fn == 0:
            return 3.0
        f1 = set_f1(selected_Se, self.piece_Se)
        return 2.0 * f1 - 1.0 - 0.8 * fn - 0.6 * fp

    def step(self, action, selected_actions):
        selected_Se = [item for item in selected_actions if item != self.stop_action]

        # 选择“停止”时终止本轮；停止奖励按集合F1、漏选和错选进行连续塑形。
        if action == self.stop_action:
            reward = self._terminal_reward(selected_Se)
            return None, reward, True

        before_f1 = set_f1(selected_Se, self.piece_Se)
        next_selected_Se = selected_Se + [action]
        after_f1 = set_f1(next_selected_Se, self.piece_Se)
        delta_f1 = after_f1 - before_f1

        if action in self.piece_Se:
            reward = 0.8 * self.Se_weights.get(action, 1.0) + 1.5 * delta_f1
        else:
            reward = -1.2 + 1.5 * delta_f1

        # 轻微步长惩罚鼓励尽早停止；超过真实证候要素数后额外惩罚，抑制“全选”。
        reward -= 0.05
        over_select = max(0, len(next_selected_Se) - len(self.piece_Se))
        reward -= 0.5 * over_select

        self.state[self.symp_len + action] = 1

        # 真正的正常终止由“停止”动作决定；若所有非停止动作都已选完，则强制终止。
        terminated = len(next_selected_Se) >= self.Se_action_num
        return None if terminated else self.state, reward, terminated

    def step_for_eval(self, action):
        if action == self.stop_action:
            return None, True
        self.state[self.symp_len + action] = 1
        return self.state, False


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'next_invalid_actions'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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


# 获取中医数据，筛选频率最高的前N个证候要素，构建刻下症和证候要素的映射
# 数据集格式：编号 刻下症 证候要素 证候
# 例：1 胸闷痛,气短 气虚,血瘀 气虚血瘀
def get_tcm_data(filename, max_Se_num=0):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        file.readline()  # 跳过标题行
        for line in file:
            line = line.strip()
            if line:
                data.append(line)

    tuples4gen = []
    symptom_set = set()
    Se_set = set()
    symptom_map = {}
    Se_map = {}
    Se_freq_map = {}
    combo_freq_map = Counter()

    for piece in data:
        parts = piece.split()
        if len(parts) != 4:
            raise ValueError(f"数据列数应为4列，实际为{len(parts)}列：{piece}")

        symptoms = parts[1].split(',')  # 刻下症
        Se_list = parts[2].split(',')   # 证候要素
        combo_freq_map[tuple(sorted(Se_list))] += 1

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
        tuples4gen.append((symptoms, Se_list))

    logger.info(f"全量刻下症数: {len(symptom_map)}, 全量证候要素数: {len(Se_map)}")
    logger.info(f"证候要素频次: {dict(sorted(Se_freq_map.items(), key=lambda x: (-x[1], x[0])))}")
    logger.info(f"证候要素组合数: {len(combo_freq_map)}")

    # 默认保留全部证候要素；只有显式设置max_Se_num且小于标签数时才过滤。
    if max_Se_num and 0 < max_Se_num < len(Se_map):
        sorted_Se = sorted(Se_freq_map.items(), key=lambda x: (-x[1], x[0]))
        top_n_Se = {Se for Se, _ in sorted_Se[:max_Se_num]}
        logger.info(f"选择频率最高的前{max_Se_num}个证候要素: {list(top_n_Se)}")
    else:
        top_n_Se = set(Se_freq_map.keys())
        logger.info("保留全部证候要素")

    filt_tuples4gen = []
    filt_symptom_map = {}
    filt_Se_map = {}
    symptom_set.clear()
    Se_set.clear()

    max_Se_len = 0
    for item in tuples4gen:
        if set(item[1]).issubset(top_n_Se):
            filt_tuples4gen.append(item)
            if len(item[1]) > max_Se_len:
                max_Se_len = len(item[1])

            for Se in item[1]:
                if Se not in Se_set:
                    filt_Se_map[len(Se_set)] = Se
                    Se_set.add(Se)
            for symp in item[0]:
                if symp not in symptom_set:
                    filt_symptom_map[len(symptom_set)] = symp
                    symptom_set.add(symp)

    logger.info(f"过滤后数据量: {len(filt_tuples4gen)}, 刻下症数: {len(filt_symptom_map)}, 证候要素数: {len(filt_Se_map)}")

    return filt_tuples4gen, filt_symptom_map, filt_Se_map, max_Se_len


steps_done = 0


def mask_selected_actions(q_values, selected_actions_batch, mask_stop=False):
    masked_q_values = q_values.clone()
    for row_idx, selected_actions in enumerate(selected_actions_batch):
        for action_idx in selected_actions:
            if 0 <= action_idx < env.Se_action_num:
                masked_q_values[row_idx, action_idx] = NEG_INF
    if mask_stop:
        masked_q_values[:, env.stop_action] = NEG_INF
    return masked_q_values


def select_action(state, selected_actions):
    """epsilon-greedy策略选择动作，避免重复；停止动作始终可选。"""
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            output = policy_net(state)
            output = mask_selected_actions(output, [selected_actions])
            action = output.max(1).indices.view(1, 1)
            return action, "agent", True

    valid_actions = [action for action in range(n_actions) if action == env.stop_action or action not in selected_actions]
    action = random.choice(valid_actions)
    return torch.tensor([[action]], device=device, dtype=torch.long), "random", False


def select_action_4eval(state, selected_actions, mask_stop=False):
    """评估时选择动作，始终选Q值最大且不重复的合法动作。"""
    with torch.no_grad():
        output = policy_net(state)
        output = mask_selected_actions(output, [selected_actions], mask_stop=mask_stop)
        action = output.max(1).indices.view(1, 1)
        return action


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_indices = [idx for idx, state in enumerate(batch.next_state) if state is not None]
    non_final_next_states = [batch.next_state[idx] for idx in non_final_indices]

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Double DQN：policy_net选择下一动作，target_net评估该动作；同时mask已选动作。
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        if non_final_next_states:
            next_states = torch.cat(non_final_next_states)
            next_invalid_actions = [batch.next_invalid_actions[idx] for idx in non_final_indices]
            next_policy_q = mask_selected_actions(policy_net(next_states), next_invalid_actions)
            next_actions = next_policy_q.max(1).indices.view(-1, 1)
            next_target_q = target_net(next_states)
            next_values = next_target_q.gather(1, next_actions).squeeze(1)
            next_state_values[non_final_mask] = next_values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
    optimizer.step()
    return loss.item()


def eval_jaccard(selected, true):
    return set_jaccard(selected, true)


def eval_precision(selected, true):
    return set_precision(selected, true)


def eval_recall(selected, true):
    return set_recall(selected, true)


def eval_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def update_metrics(selected, true, max_j, min_j, avg_j, max_p, min_p, avg_p, max_r, min_r, avg_r, max_f, min_f, avg_f):
    j = eval_jaccard(selected, true)
    p = eval_precision(selected, true)
    r = eval_recall(selected, true)
    f = eval_f1(p, r)

    max_j = max(max_j, j); min_j = min(min_j, j); avg_j += j
    max_p = max(max_p, p); min_p = min(min_p, p); avg_p += p
    max_r = max(max_r, r); min_r = min(min_r, r); avg_r += r
    max_f = max(max_f, f); min_f = min(min_f, f); avg_f += f

    return max_j, min_j, avg_j, max_p, min_p, avg_p, max_r, min_r, avg_r, max_f, min_f, avg_f


def actions_to_multihot(actions):
    vector = np.zeros(env.Se_action_num, dtype=int)
    for action in actions:
        if 0 <= action < env.Se_action_num:
            vector[action] = 1
    return vector


def Se_names_to_multihot(Se_names):
    vector = np.zeros(env.Se_action_num, dtype=int)
    for Se_name in Se_names:
        action = env.swapped_action_space[Se_name]
        vector[action] = 1
    return vector


def predict_actions_from_state(state_np, force_top_k=None, max_actions=None):
    policy_net.eval()
    selected_actions = []
    max_actions = env.Se_action_num if max_actions is None else min(max_actions, env.Se_action_num)

    with torch.no_grad():
        while len(selected_actions) < max_actions:
            state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            mask_stop = force_top_k is not None and len(selected_actions) < force_top_k
            action = select_action_4eval(state_tensor, selected_actions, mask_stop=mask_stop)
            action_idx = action.item()

            if action_idx == env.stop_action:
                break

            selected_actions.append(action_idx)
            state_np[env.symp_len + action_idx] = 1

            if force_top_k is not None and len(selected_actions) >= force_top_k:
                break

    return selected_actions


def evaluate_prediction_set(pred_action_sets, true_name_sets, title):
    n_test = len(true_name_sets)
    y_pred = np.array([actions_to_multihot(actions) for actions in pred_action_sets])
    y_true = np.array([Se_names_to_multihot(names) for names in true_name_sets])

    sample_j = []
    sample_p = []
    sample_r = []
    sample_f = []
    exact_match = []
    selected_count = []
    true_count = []
    cardinality_error = []

    for pred_actions, true_names in zip(pred_action_sets, true_name_sets):
        pred_names = [env.action_space[action] for action in pred_actions]
        p = eval_precision(pred_names, true_names)
        r = eval_recall(pred_names, true_names)
        f = eval_f1(p, r)
        sample_j.append(eval_jaccard(pred_names, true_names))
        sample_p.append(p)
        sample_r.append(r)
        sample_f.append(f)
        exact_match.append(1 if set(pred_names) == set(true_names) else 0)
        selected_count.append(len(pred_names))
        true_count.append(len(true_names))
        cardinality_error.append(abs(len(pred_names) - len(true_names)))

    if precision_recall_fscore_support is not None and hamming_loss is not None:
        micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
        macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        label_p, label_r, label_f, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        hamming = hamming_loss(y_true, y_pred)
    else:
        label_tp = (y_true * y_pred).sum(axis=0)
        label_fp = ((1 - y_true) * y_pred).sum(axis=0)
        label_fn = (y_true * (1 - y_pred)).sum(axis=0)
        label_p = np.divide(label_tp, label_tp + label_fp, out=np.zeros_like(label_tp, dtype=float), where=(label_tp + label_fp) != 0)
        label_r = np.divide(label_tp, label_tp + label_fn, out=np.zeros_like(label_tp, dtype=float), where=(label_tp + label_fn) != 0)
        label_f = np.divide(2 * label_p * label_r, label_p + label_r, out=np.zeros_like(label_p, dtype=float), where=(label_p + label_r) != 0)
        total_tp = label_tp.sum()
        total_fp = label_fp.sum()
        total_fn = label_fn.sum()
        micro_p = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
        micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r > 0 else 0.0
        macro_p = float(np.mean(label_p))
        macro_r = float(np.mean(label_r))
        macro_f = float(np.mean(label_f))
        hamming = float(np.not_equal(y_true, y_pred).mean())

    logger.info(f"**{title}")
    logger.info(f"**样本Jaccard avg:{np.mean(sample_j):.4f} max:{np.max(sample_j):.4f} min:{np.min(sample_j):.4f}")
    logger.info(f"**样本Precision avg:{np.mean(sample_p):.4f}, Recall avg:{np.mean(sample_r):.4f}, F1 avg:{np.mean(sample_f):.4f}")
    logger.info(f"**ExactMatch:{np.mean(exact_match):.4f}, HammingLoss:{hamming:.4f}")
    logger.info(f"**Micro P/R/F1:{micro_p:.4f}/{micro_r:.4f}/{micro_f:.4f}")
    logger.info(f"**Macro P/R/F1:{macro_p:.4f}/{macro_r:.4f}/{macro_f:.4f}")
    logger.info(f"**平均推荐数:{np.mean(selected_count):.4f}, 平均真实数:{np.mean(true_count):.4f}, 平均数量误差:{np.mean(cardinality_error):.4f}")
    for action_idx in range(env.Se_action_num):
        logger.info(f"**标签[{env.action_space[action_idx]}] P/R/F1:{label_p[action_idx]:.4f}/{label_r[action_idx]:.4f}/{label_f[action_idx]:.4f}")
    logger.info("")

    return {
        'sample_jaccard': float(np.mean(sample_j)),
        'sample_f1': float(np.mean(sample_f)),
        'exact_match': float(np.mean(exact_match)),
        'micro_f1': float(micro_f),
        'macro_f1': float(macro_f),
        'avg_selected_count': float(np.mean(selected_count)),
        'avg_cardinality_error': float(np.mean(cardinality_error)),
        'n_test': n_test,
    }


def evaluate():
    logger.info("evaluate...")
    if len(new_test_data) == 0:
        logger.info("测试数据为空，跳过评估")
        return {'sample_f1': 0.0, 'exact_match': 0.0}

    auto_pred_actions = []
    top2_pred_actions = []
    true_name_sets = []

    for data_piece in new_test_data:
        state_np = env.reset(data_piece).copy()
        auto_pred_actions.append(predict_actions_from_state(state_np, force_top_k=None))

        state_np = env.reset(data_piece).copy()
        top2_pred_actions.append(predict_actions_from_state(state_np, force_top_k=min(2, env.Se_action_num), max_actions=2))

        true_name_sets.append(data_piece[1])

    auto_metrics = evaluate_prediction_set(auto_pred_actions, true_name_sets, "模型自主停止")
    top2_metrics = evaluate_prediction_set(top2_pred_actions, true_name_sets, "固定Top-2诊断")
    logger.info(f"**自主停止 vs Top-2 样本F1: {auto_metrics['sample_f1']:.4f} / {top2_metrics['sample_f1']:.4f}")
    return auto_metrics


def build_state_vector(symptoms, selected_actions=None):
    selected_actions = selected_actions or []
    state = np.zeros(len(env.state_space), dtype=np.float32)
    for symptom in symptoms:
        if symptom in env.swapped_state_space:
            state[env.swapped_state_space[symptom]] = 1
    for action in selected_actions:
        if 0 <= action < env.Se_action_num:
            state[env.symp_len + action] = 1
    return state


def pretrain_supervised(pretrain_epochs, pretrain_lr):
    if pretrain_epochs <= 0:
        return

    logger.info(f"开始监督预训练: epochs={pretrain_epochs}, lr={pretrain_lr}")
    states = []
    targets = []

    for symptoms, true_Se_names in training_tcm_data:
        true_actions = [env.swapped_action_space[name] for name in true_Se_names]

        # 初始状态：真实证候要素为正，停止为负。
        target = np.zeros(n_actions, dtype=np.float32)
        target[true_actions] = 1.0
        states.append(build_state_vector(symptoms))
        targets.append(target)

        # 中间状态：剩余真实证候要素为正，停止为负。
        for selected_action in true_actions:
            remaining_actions = [action for action in true_actions if action != selected_action]
            target = np.zeros(n_actions, dtype=np.float32)
            target[remaining_actions] = 1.0
            states.append(build_state_vector(symptoms, [selected_action]))
            targets.append(target)

        # 完成状态：停止为正。
        target = np.zeros(n_actions, dtype=np.float32)
        target[env.stop_action] = 1.0
        states.append(build_state_vector(symptoms, true_actions))
        targets.append(target)

    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    targets = torch.tensor(np.array(targets), dtype=torch.float32, device=device)

    pos_count = targets.sum(dim=0)
    neg_count = targets.shape[0] - pos_count
    pos_weight = (neg_count / torch.clamp(pos_count, min=1.0)).clamp(min=1.0, max=10.0)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    pretrain_optimizer = optim.AdamW(policy_net.parameters(), lr=pretrain_lr, weight_decay=WEIGHT_DECAY, amsgrad=True)
    batch_size = min(256, len(states))

    policy_net.train()
    for epoch in range(pretrain_epochs):
        permutation = torch.randperm(len(states), device=device)
        total_loss = 0.0
        for start in range(0, len(states), batch_size):
            indices = permutation[start:start + batch_size]
            batch_states = states[indices]
            batch_targets = targets[indices]
            logits = policy_net(batch_states)
            loss = criterion(logits, batch_targets)
            pretrain_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
            pretrain_optimizer.step()
            total_loss += loss.item() * len(indices)
        logger.info(f"监督预训练 epoch {epoch + 1}/{pretrain_epochs}, loss={total_loss / len(states):.6f}")

    target_net.load_state_dict(policy_net.state_dict())


def warmup_replay_with_expert():
    logger.info("开始使用专家轨迹预填充ReplayMemory")
    added = 0
    for data_piece in training_tcm_data:
        env.reset(data_piece)
        symptoms, true_Se_names = data_piece
        true_actions = [env.swapped_action_space[name] for name in true_Se_names]
        random.shuffle(true_actions)
        selected_actions = []

        for action_idx in true_actions:
            state_np = build_state_vector(symptoms, selected_actions)
            state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            action = torch.tensor([[action_idx]], device=device, dtype=torch.long)
            observation, reward, terminated = env.step(action_idx, selected_actions)
            selected_actions.append(action_idx)
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
            next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            memory.push(state, action, reward_tensor, next_state, selected_actions.copy())
            added += 1
            if terminated:
                break

        if len(selected_actions) < env.Se_action_num:
            state_np = build_state_vector(symptoms, selected_actions)
            state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            action = torch.tensor([[env.stop_action]], device=device, dtype=torch.long)
            _, reward, _ = env.step(env.stop_action, selected_actions)
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
            memory.push(state, action, reward_tensor, None, selected_actions.copy())
            added += 1

    logger.info(f"ReplayMemory预填充完成，新增transition数量: {added}, 当前容量: {len(memory)}")


def train(num_episodes, patience, min_delta):
    logger.info(f"总的迭代次数: {num_episodes}")

    global steps_done
    steps_done = 0
    best_metric = -1.0
    best_state_dict = copy.deepcopy(policy_net.state_dict())
    best_episode = -1
    no_improve_count = 0
    st = time.perf_counter()

    for i_episode in range(num_episodes):
        logger.info(f"第 {i_episode} 次迭代开始...")

        iter_Se = 0
        episode_rewards_list = []
        loss_list = []
        agent_count = 0
        random_count = 0
        stop_count = 0

        epi_st = time.perf_counter()
        random.shuffle(training_tcm_data)
        policy_net.train()

        # 遍历每个训练数据: (刻下症列表, 证候要素列表)
        for data_piece in training_tcm_data:
            state_np = env.reset(data_piece)
            state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)

            sample_rewards = 0.0
            selected_actions = []
            # 不再按照真实证候要素个数人为停止；停止动作由模型在候选动作中自主选择。
            # 循环上限为全部证候要素动作数+停止动作，防止异常情况下无限循环。
            for _ in range(env.Se_action_num + 1):
                action, actor, _ = select_action(state, selected_actions)
                action_idx = action.item()

                observation, reward, terminated = env.step(action_idx, selected_actions)
                if action_idx == env.stop_action:
                    stop_count += 1
                else:
                    selected_actions.append(action_idx)

                if actor == "agent":
                    agent_count += 1
                elif actor == "random":
                    random_count += 1
                sample_rewards += reward

                reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)

                next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                memory.push(state, action, reward_tensor, next_state, selected_actions.copy())
                state = next_state
                iter_Se += 1

                if len(memory) >= BATCH_SIZE and iter_Se >= 5:
                    loss = optimize_model()
                    if loss is not None:
                        loss_list.append(loss)
                    # 软更新目标网络: θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = target_net.state_dict()
                    policy_net_state_dict = policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
                    target_net.load_state_dict(target_net_state_dict)
                    iter_Se = 0

                if terminated:
                    break

            episode_rewards_list.append(sample_rewards)

        epi_et = time.perf_counter()

        metrics = evaluate()
        avg_loss = float(np.mean(loss_list)) if loss_list else 0.0
        avg_reward = float(np.mean(episode_rewards_list)) if episode_rewards_list else 0.0
        logger.info(f"第 {i_episode} 次迭代训练统计: avg_reward={avg_reward:.4f}, avg_loss={avg_loss:.6f}, agent_actions={agent_count}, random_actions={random_count}, stop_actions={stop_count}")
        logger.info(f"第 {i_episode} 次迭代的训练时间: {epi_et-epi_st:.6f}秒")

        current_metric = metrics.get('sample_f1', 0.0)
        if current_metric > best_metric + min_delta:
            best_metric = current_metric
            best_state_dict = copy.deepcopy(policy_net.state_dict())
            best_episode = i_episode
            no_improve_count = 0
            torch.save(best_state_dict, 'dqn_model_best.pth')
            logger.info(f"保存当前最佳模型: episode={best_episode}, sample_f1={best_metric:.4f}")
        else:
            no_improve_count += 1
            logger.info(f"验证指标未提升: {no_improve_count}/{patience}")

        logger.info(f"第 {i_episode} 次迭代完成。")
        if patience > 0 and no_improve_count >= patience:
            logger.info(f"触发早停: best_episode={best_episode}, best_sample_f1={best_metric:.4f}")
            break

    et = time.perf_counter()
    logger.info(f"训练时间总共: {et-st:.6f}秒")
    logger.info(f"训练完成，最佳episode={best_episode}, 最佳sample_f1={best_metric:.4f}")
    policy_net.load_state_dict(best_state_dict)
    return best_metric


def stratified_split_data(tcm_data, seed, test_ratio=0.1):
    rng = random.Random(seed)
    combo_keys = ['|'.join(sorted(item[1])) for item in tcm_data]
    combo_counts = Counter(combo_keys)

    if train_test_split is not None and combo_counts and min(combo_counts.values()) >= 2:
        train_data, test_data = train_test_split(tcm_data, test_size=test_ratio, random_state=seed, stratify=combo_keys)
        logger.info("使用证候要素组合分层划分训练集/测试集")
        return list(train_data), list(test_data)

    if train_test_split is None:
        logger.warning("当前Python环境未安装/未识别scikit-learn，使用内置分层划分逻辑")
    elif not combo_counts or min(combo_counts.values()) < 2:
        logger.warning("存在样本数不足2的证候要素组合，使用内置随机划分逻辑")

    grouped = {}
    for item, key in zip(tcm_data, combo_keys):
        grouped.setdefault(key, []).append(item)

    train_data = []
    test_data = []
    for items in grouped.values():
        items = list(items)
        rng.shuffle(items)
        n_test = max(1, int(round(len(items) * test_ratio))) if len(items) >= 2 else 0
        n_test = min(n_test, len(items) - 1) if len(items) >= 2 else 0
        test_data.extend(items[:n_test])
        train_data.extend(items[n_test:])

    if not test_data:
        shuffled = list(tcm_data)
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_ratio))
        test_data = shuffled[:n_test]
        train_data = shuffled[n_test:]

    rng.shuffle(train_data)
    rng.shuffle(test_data)
    logger.info("使用内置训练集/测试集划分")
    return train_data, test_data


def compute_Se_weights(training_data):
    freq = Counter()
    for _, Se_names in training_data:
        for Se_name in Se_names:
            freq[env.swapped_action_space[Se_name]] += 1
    total = sum(freq.values())
    weights = {}
    for action_idx in range(env.Se_action_num):
        count = max(freq.get(action_idx, 1), 1)
        weight = math.sqrt(total / (env.Se_action_num * count))
        weights[action_idx] = float(min(max(weight, 0.75), 1.5))
    logger.info(f"证候要素奖励权重: { {env.action_space[k]: round(v, 4) for k, v in weights.items()} }")
    return weights


if __name__ == "__main__":
    parser = ArgumentParser(description="DQN Intelligent Syndrome Differentiation")
    parser.add_argument("-episode", type=int, dest="num_episodes", default=100)
    parser.add_argument("-batch", type=int, dest="batch_size", default=64)
    parser.add_argument("-mem_capacity", type=int, dest="mem_capacity", default=10000)
    parser.add_argument("-gamma", type=float, dest="gamma", default=0.95)
    parser.add_argument("-eps_start", type=float, dest="eps_start", default=0.7)
    parser.add_argument("-eps_end", type=float, dest="eps_end", default=0.02)
    parser.add_argument("-eps_decay", type=int, dest="eps_decay", default=10000)
    parser.add_argument("-tau", type=float, dest="tau", default=0.01)
    parser.add_argument("-lr", type=float, dest="lr", default=0.001)
    parser.add_argument("-weight_decay", type=float, dest="weight_decay", default=1e-4)
    parser.add_argument("-nn_units", type=int, dest="nn_units", default=128)
    parser.add_argument("-nn_units2", type=int, dest="nn_units2", default=64)
    parser.add_argument("-dropout", type=float, dest="dropout", default=0.1)
    parser.add_argument("-seed", type=int, dest="seed", default=9)
    parser.add_argument("-max_se_num", type=int, dest="max_Se_num", default=0)
    parser.add_argument("-use_pretrain", type=int, dest="use_pretrain", default=0)
    parser.add_argument("-pretrain_epochs", type=int, dest="pretrain_epochs", default=0)
    parser.add_argument("-pretrain_lr", type=float, dest="pretrain_lr", default=0.001)
    parser.add_argument("-replay_warmup", type=int, dest="replay_warmup", default=0)
    parser.add_argument("-patience", type=int, dest="patience", default=15)
    parser.add_argument("-min_delta", type=float, dest="min_delta", default=0.001)

    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    MEM_CAPACITY = args.mem_capacity
    GAMMA = args.gamma
    EPS_START = args.eps_start
    EPS_END = args.eps_end
    EPS_DECAY = args.eps_decay
    TAU = args.tau
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    nn_units = args.nn_units
    nn_units2 = args.nn_units2
    dropout = args.dropout
    num_episodes = args.num_episodes

    set_seed(args.seed)

    lr_name = str(LR).replace('0.', '')
    now = datetime.now()
    time_sign = f"{now.year}年{now.month}月{now.day}日{now.hour}时{now.minute}分"

    logger = createLogger(f"dqn_safe_LR{lr_name}_{nn_units}_{nn_units2}units_{time_sign}.log")
    logger.info(f"可用设备:{device}")
    logger.info(f"随机种子:{args.seed}")

    # 加载广安门冠心病辨证数据集（不含主述），并构建环境
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset/广安门冠心病辨证数据集（不含主述）.txt')
    tcm_data, symptoms, Se, max_Se_len = get_tcm_data(data_path, max_Se_num=args.max_Se_num)
    env = Environment(symptoms, Se)

    # 小数据多标签任务：按证候要素组合分层划分，保持训练/测试分布稳定。
    training_tcm_data, new_test_data = stratified_split_data(tcm_data, args.seed, test_ratio=0.1)
    env.set_Se_weights(compute_Se_weights(training_tcm_data))

    logger.info(f"训练数据数量:{len(training_tcm_data)}, 测试数据数量:{len(new_test_data)}")
    logger.info(f"最大真实证候要素数:{max_Se_len}")

    # 构建DQN网络
    state_vector_len = len(env.state_space)
    n_actions = len(env.action_space)
    logger.info(f"状态向量长度:{state_vector_len}, 动作数:{n_actions}")

    policy_net = DQN(state_vector_len, n_actions, nn_units, nn_units2, dropout).to(device)
    target_net = DQN(state_vector_len, n_actions, nn_units, nn_units2, dropout).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    memory = ReplayMemory(MEM_CAPACITY)
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, amsgrad=True)

    logger.info(
        f"**设置**: batch={BATCH_SIZE}, mem={MEM_CAPACITY}, gamma={GAMMA}, eps={EPS_START}->{EPS_END}, "
        f"eps_decay={EPS_DECAY}, tau={TAU}, lr={LR}, weight_decay={WEIGHT_DECAY}, "
        f"hidden={nn_units}/{nn_units2}, dropout={dropout}, pretrain={args.use_pretrain}, warmup={args.replay_warmup}"
    )

    if args.use_pretrain:
        pretrain_supervised(args.pretrain_epochs, args.pretrain_lr)

    if args.replay_warmup:
        warmup_replay_with_expert()

    best_metric = train(num_episodes, args.patience, args.min_delta)

    # 保存模型：best在训练过程中已保存，这里额外保存最终加载的最佳权重和last权重文件名。
    torch.save(policy_net.state_dict(), 'dqn_model.pth')
    torch.save(policy_net.state_dict(), 'dqn_model_last.pth')
    print(f'模型已保存，最佳sample_f1={best_metric:.4f}')

    # 预测：输入刻下症，由模型自主决定推荐几个证候要素以及何时停止。
    # “停止”已加入候选动作；当模型选择“停止”时，立即结束预测并输出当前已选证候要素。
    def predict_symptoms(symptoms_str, max_actions=None, force_top_k=None):
        state_vector = np.zeros(len(env.state_space), dtype=np.float32)
        for symptom in symptoms_str.split(','):
            symptom = symptom.strip()
            if symptom in env.swapped_state_space:
                state_vector[env.swapped_state_space[symptom]] = 1

        selected_actions = predict_actions_from_state(state_vector, force_top_k=force_top_k, max_actions=max_actions)
        return [env.action_space[action_idx] for action_idx in selected_actions]

    symptoms_str = '情绪抑郁时胸闷胀痛加重,形体肥胖,痰多体胖,口黏不爽,头昏多寐,脘腹痞满,舌胖苔厚腻,舌暗红,脉弦滑,纳差,大便黏腻'
    predicted_Se = predict_symptoms(symptoms_str)
    predicted_Se_top2 = predict_symptoms(symptoms_str, force_top_k=2, max_actions=2)
    print(f'推荐证候要素(自主停止): {",".join(predicted_Se) if predicted_Se else "无"}')
    print(f'推荐证候要素(固定Top-2诊断): {",".join(predicted_Se_top2) if predicted_Se_top2 else "无"}')
