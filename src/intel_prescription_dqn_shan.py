# -*- coding: utf-8 -*-
"""DQN智能辨证：输入症状，输出推荐的状态要素（证候要素）"""

import os
import math
import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time
from datetime import datetime
import logging
from argparse import ArgumentParser


def createLogger(logname):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
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


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


class Environment(object):
    """强化学习环境：状态空间=症状+已选状态要素，动作空间=候选状态要素"""
    def __init__(self, symptoms, state_elems):
        # 状态空间：症状(0..symp_len-1) + 状态要素(symp_len..)
        self.state_space = {}
        self.swapped_state_space = {}
        self.symp_len = len(symptoms)
        for index in symptoms:
            self.state_space[index] = symptoms[index]
            self.swapped_state_space[symptoms[index]] = index
        for index in state_elems:
            self.state_space[index + self.symp_len] = state_elems[index]
            self.swapped_state_space[state_elems[index]] = index + self.symp_len

        # 动作空间：可选的15个状态要素
        self.action_space = state_elems
        self.swapped_action_space = {}
        for index in self.action_space:
            self.swapped_action_space[self.action_space[index]] = index

    def reset(self, data_piece):
        self.piece_symptoms = data_piece[0]
        self.piece_state_elems = [self.swapped_action_space[se] for se in data_piece[1]]
        self.state = np.zeros(len(self.state_space), dtype=int)
        for symp in self.piece_symptoms:
            self.state[self.swapped_state_space[symp]] = 1
        return self.state

    def step(self, action, selected_actions):
        reward = 1 if action in self.piece_state_elems else -1
        if len(selected_actions) == len(self.piece_state_elems):
            return None, reward, True
        self.state[self.symp_len + action] = 1
        return self.state, reward, False

    def step_for_eval(self, action):
        self.state[self.symp_len + action] = 1
        return self.state


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


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
    def __init__(self, state_vector_len, n_actions, n_units=10):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_vector_len, n_units, dtype=torch.float32, device='cpu')
        self.layer2 = nn.Linear(n_units, n_actions, dtype=torch.float32, device='cpu')

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)

# 获取中医数据，筛选频率最高的前N个状态要素，构建症状和状态要素的映射
def get_tcm_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        file.readline()  # 跳过标题行
        for line in file:
            data.append(line.strip())

    random.seed(9)
    random.shuffle(data)

    tuples4gen = []
    symptom_set = set()
    state_elem_set = set()
    symptom_map = {}
    state_elem_map = {}
    state_elem_freq_map = {}
    MAX_STATE_ELEM_NUM = 15  # 保留频率最高的前15个状态要素

    for piece in data:
        parts = piece.split()
        state_elems = parts[5].split(',')  # 状态要素
        symptoms = parts[3].split(',')     # 症状

        for se in state_elems:
            if se not in state_elem_set:
                state_elem_map[len(state_elem_set)] = se
                state_elem_set.add(se)
                state_elem_freq_map[se] = 1
            else:
                state_elem_freq_map[se] += 1

        for symp in symptoms:
            if symp not in symptom_set:
                symptom_map[len(symptom_set)] = symp
                symptom_set.add(symp)
        tuples4gen.append((symptoms, state_elems))

    logger.info(f"全量症状数: {len(symptom_map)}, 全量状态要素数: {len(state_elem_map)}")

    # 按频率降序取前N个状态要素
    sorted_state_elems = sorted(state_elem_freq_map.items(), key=lambda x: (-x[1], x[0]))
    top_n_state_elems = {se for se, _ in sorted_state_elems[:MAX_STATE_ELEM_NUM]}
    logger.info(f"选择频率最高的前{MAX_STATE_ELEM_NUM}个状态要素: {list(top_n_state_elems)}")

    # 过滤：只保留状态要素全部属于top_n的数据
    filt_tuples4gen = []
    filt_symptom_map = {}
    filt_state_elem_map = {}
    symptom_set.clear()
    state_elem_set.clear()

    max_se_len = 0
    for data in tuples4gen:
        if set(data[1]).issubset(top_n_state_elems):
            filt_tuples4gen.append(data)
            if len(data[1]) > max_se_len:
                max_se_len = len(data[1])

            for se in data[1]:
                if se not in state_elem_set:
                    filt_state_elem_map[len(state_elem_set)] = se
                    state_elem_set.add(se)
            for symp in data[0]:
                if symp not in symptom_set:
                    filt_symptom_map[len(symptom_set)] = symp
                    symptom_set.add(symp)

    logger.info(f"过滤后数据量: {len(filt_tuples4gen)}, 症状数: {len(filt_symptom_map)}, 状态要素数: {len(filt_state_elem_map)}")

    return filt_tuples4gen, filt_symptom_map, filt_state_elem_map, max_se_len

steps_done = 0

def select_action(state, selected_actions):
    """epsilon-greedy策略选择动作，避免重复"""
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    use_first_qmax = True
    if sample > eps_threshold:
        output = policy_net(state)
        while True:
            with torch.no_grad():
                action = output.max(1).indices.view(1, 1)
                if action.item() not in selected_actions:
                    return action, "agent", use_first_qmax
                use_first_qmax = False
                output[0][action] = -999999
    else:
        while True:
            action = random.randrange(n_actions)
            if action not in selected_actions:
                return torch.tensor([[action]], device=device, dtype=torch.long), "random", False


def select_action_4eval(state, selected_actions):
    """评估时选择动作，始终选Q值最大且不重复的"""
    output = policy_net(state)
    while True:
        with torch.no_grad():
            action = output.max(1).indices.view(1, 1)
            if action.item() not in selected_actions:
                return action
            output[0][action] = -999999


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # V(s_{t+1}) = max_a Q_target(s_{t+1}, a)，终结状态为0
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def eval_jaccard(selected, true):
    correct = set(selected) & set(true)
    union = set(selected) | set(true)
    return float(len(correct)) / len(union)


def eval_precision(selected, true):
    correct = set(selected) & set(true)
    return float(len(correct)) / len(selected)


def eval_recall(selected, true):
    correct = set(selected) & set(true)
    return float(len(correct)) / len(true)


def eval_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def print_metrics(n_selected, max_j, min_j, avg_j, max_p, min_p, avg_p, max_r, min_r, avg_r, max_f, min_f, avg_f):
    n_test = len(new_test_data)
    logger.info(f"**推荐状态要素数: {n_selected}")
    logger.info(f"**Jaccard - max:{max_j:.4f} min:{min_j:.4f} avg:{avg_j/n_test:.4f}")
    logger.info(f"**Precision - max:{max_p:.4f} min:{min_p:.4f} avg:{avg_p/n_test:.4f}")
    logger.info(f"**Recall - max:{max_r:.4f} min:{min_r:.4f} avg:{avg_r/n_test:.4f}")
    logger.info(f"**F1 - max:{max_f:.4f} min:{min_f:.4f} avg:{avg_f/n_test:.4f}\n")


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

def evaluate():
    logger.info("evaluate...")
    piece_to_eval = list(enumerate(new_test_data))
    n_state_elems_list = [2, 6]  # 评估推荐2个和6个状态要素的效果

    # 初始化指标: (max, min, avg) for jaccard, precision, recall, f1
    max5_j, min5_j, avg5_j = 0.0, 1.0, 0.0
    max5_p, min5_p, avg5_p = 0.0, 1.0, 0.0
    max5_r, min5_r, avg5_r = 0.0, 1.0, 0.0
    max5_f, min5_f, avg5_f = 0.0, 1.0, 0.0
    max10_j, min10_j, avg10_j = 0.0, 1.0, 0.0
    max10_p, min10_p, avg10_p = 0.0, 1.0, 0.0
    max10_r, min10_r, avg10_r = 0.0, 1.0, 0.0
    max10_f, min10_f, avg10_f = 0.0, 1.0, 0.0

    for _, data_piece in piece_to_eval:
        state = env.reset(data_piece)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        selected_actions = []
        while len(selected_actions) < n_state_elems_list[1]:
            action = select_action_4eval(state, selected_actions)
            selected_actions.append(action.item())
            observation = env.step_for_eval(action.item())
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        selected5 = [env.action_space[a] for a in selected_actions[:n_state_elems_list[0]]]
        selected10 = [env.action_space[a] for a in selected_actions[:n_state_elems_list[1]]]

        max5_j, min5_j, avg5_j, max5_p, min5_p, avg5_p, max5_r, min5_r, avg5_r, max5_f, min5_f, avg5_f = \
            update_metrics(selected5, data_piece[1], max5_j, min5_j, avg5_j, max5_p, min5_p, avg5_p, max5_r, min5_r, avg5_r, max5_f, min5_f, avg5_f)

        max10_j, min10_j, avg10_j, max10_p, min10_p, avg10_p, max10_r, min10_r, avg10_r, max10_f, min10_f, avg10_f = \
            update_metrics(selected10, data_piece[1], max10_j, min10_j, avg10_j, max10_p, min10_p, avg10_p, max10_r, min10_r, avg10_r, max10_f, min10_f, avg10_f)

    print_metrics(n_state_elems_list[0], max5_j, min5_j, avg5_j, max5_p, min5_p, avg5_p, max5_r, min5_r, avg5_r, max5_f, min5_f, avg5_f)
    print_metrics(n_state_elems_list[1], max10_j, min10_j, avg10_j, max10_p, min10_p, avg10_p, max10_r, min10_r, avg10_r, max10_f, min10_f, avg10_f)
def train(num_episodes):
    logger.info(f"总的迭代次数: {num_episodes}")

    epi_rewards_list = []
    st = time.perf_counter()

    for i_episode in range(num_episodes):
        logger.info(f"第 {i_episode} 次迭代开始...")

        iter_state_elems = 0
        episode_rewards_list = []
        agent_count = 0
        random_count = 0
        first_qmax_count = 0

        epi_st = time.perf_counter()

        # 遍历每个训练数据: (症状列表, 状态要素列表)
        for data_piece in training_tcm_data:
            piece_state_elems = data_piece[1]

            state = env.reset(data_piece)
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            sample_rewards = 0
            selected_actions = []
            for _ in range(len(piece_state_elems)):
                action, actor, use_first_qmax = select_action(state, selected_actions)
                selected_actions.append(action.item())

                observation, reward, terminated = env.step(action.item(), selected_actions)
                if reward > 0:
                    if actor == "agent":
                        agent_count += 1
                    elif actor == "random":
                        random_count += 1
                    if use_first_qmax:
                        first_qmax_count += 1
                    sample_rewards += reward

                reward = torch.tensor([reward], device=device)

                next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                memory.push(state, action, reward, next_state)
                state = next_state
                iter_state_elems += 1

                if terminated:
                    break

            if iter_state_elems >= 20:
                optimize_model()
                # 软更新目标网络: θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
                target_net.load_state_dict(target_net_state_dict)
                iter_state_elems = 0

            episode_rewards_list.append(sample_rewards)

        epi_et = time.perf_counter()

        evaluate()
        logger.info(f"第 {i_episode} 次迭代的训练时间: {epi_et-epi_st:.6f}秒")
        epi_rewards_list.append(sum(episode_rewards_list))
        logger.info(f"第 {i_episode} 次迭代完成。")

    et = time.perf_counter()
    logger.info(f"训练时间总共: {et-st:.6f}秒")
    logger.info('训练完成')


if __name__ == "__main__":
    parser = ArgumentParser(description="DQN Intelligent Syndrome Differentiation")
    parser.add_argument("-episode", type=int, dest="num_episodes", default=50)
    parser.add_argument("-batch", type=int, dest="batch_size", default=64)
    parser.add_argument("-mem_capacity", type=int, dest="mem_capacity", default=1000)
    parser.add_argument("-gamma", type=float, dest="gamma", default=0.99)
    parser.add_argument("-eps_start", type=float, dest="eps_start", default=0.9)
    parser.add_argument("-eps_end", type=float, dest="eps_end", default=0.05)
    parser.add_argument("-eps_decay", type=int, dest="eps_decay", default=2000)
    parser.add_argument("-tau", type=float, dest="tau", default=0.005)
    parser.add_argument("-lr", type=float, dest="lr", default=0.01)
    parser.add_argument("-nn_units", type=int, dest="nn_units", default=2)

    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    MEM_CAPACITY = args.mem_capacity
    GAMMA = args.gamma
    EPS_START = args.eps_start
    EPS_END = args.eps_end
    EPS_DECAY = args.eps_decay
    TAU = args.tau
    LR = args.lr
    nn_units = args.nn_units
    num_episodes = args.num_episodes

    lr_name = str(LR).replace('0.', '')
    now = datetime.now()
    time_sign = f"{now.year}年{now.month}月{now.day}日{now.hour}时{now.minute}分"

    memory = ReplayMemory(MEM_CAPACITY)
    logger = createLogger(f"dqn_LR{lr_name}_{nn_units}units_{time_sign}.log")
    logger.info(f"可用设备:{device}")

    # 加载数据并构建环境
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset/伤寒论诊疗数据集.txt')
    tcm_data, symptoms, state_elems, max_se_len = get_tcm_data(data_path)
    env = Environment(symptoms, state_elems)

    # 划分训练集和测试集：状态要素>=5的作为测试集，其余训练集；测试集中保留10%作为最终测试集
    training_tcm_data = []
    test_tcm_data = []
    for item in tcm_data:
        if len(item[1]) >= 5:
            test_tcm_data.append(item)
        else:
            training_tcm_data.append(item)

    give_back_indices = random.sample(list(range(len(test_tcm_data))), len(test_tcm_data) - int(len(tcm_data) * 0.1))
    new_test_data = []
    for i, item in enumerate(test_tcm_data):
        if i in give_back_indices:
            training_tcm_data.append(item)
        else:
            new_test_data.append(item)

    logger.info(f"训练数据数量:{len(training_tcm_data)}, 测试数据数量:{len(new_test_data)}")

    # 构建DQN网络
    state_vector_len = len(env.state_space)
    n_actions = len(env.action_space)
    logger.info(f"状态向量长度:{state_vector_len}, 动作数:{n_actions}")

    policy_net = DQN(state_vector_len, n_actions, nn_units).to(device)
    target_net = DQN(state_vector_len, n_actions, nn_units).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

    logger.info(f"**设置**: batch={BATCH_SIZE}, mem={MEM_CAPACITY}, gamma={GAMMA}, eps={EPS_START}->{EPS_END}, tau={TAU}, lr={LR}")

    train(num_episodes)

    # 保存模型
    torch.save(policy_net.state_dict(), 'dqn_model.pth')
    print('模型已保存.')

    # 预测：输入症状，输出推荐的状态要素
    def predict_symptoms(symptoms_str):
        state_vector = np.zeros(len(env.state_space))
        for symptom in symptoms_str.split(','):
            if symptom in env.swapped_state_space:
                state_vector[env.swapped_state_space[symptom]] = 1
        q_values = policy_net(torch.FloatTensor(state_vector).unsqueeze(0).to(device))
        action_idx = torch.argmax(q_values).item()
        return env.action_space[action_idx]

    symptoms_str = '怕风,发热,汗出'
    predicted_se = predict_symptoms(symptoms_str)
    print(f'推荐状态要素: {predicted_se}')
