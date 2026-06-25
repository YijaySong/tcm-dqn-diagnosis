# -*- coding: utf-8 -*-

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


# 创建日志记录器
def createLogger(logname):
    logger = logging.getLogger('logger')  # 创建一个名为'logger'的日志记录器
    logger.setLevel(logging.INFO)  # 设置记录器的日志级别为INFO
  
    fh = logging.FileHandler(logname, mode='w+')  # 创建一个文件处理器，将日志信息写入指定文件
    fh.setLevel(logging.INFO)  # 设置文件处理器的日志级别为INFO

    ch = logging.StreamHandler()  # 创建一个流处理器，将日志信息输出到控制台
    ch.setLevel(logging.INFO)  # 设置流处理器的日志级别为INFO
  
    formatter = logging.Formatter('%(asctime)s %(levelname)s:  %(message)s')  # 定义日志的输出格式
    fh.setFormatter(formatter)  # 设置文件处理器的输出格式
    ch.setFormatter(formatter)  # 设置流处理器的输出格式
   
    logger.addHandler(fh)  # 将文件处理器添加到日志记录器
    logger.addHandler(ch)  # 将流处理器添加到日志记录器
  
    return logger  # 返回配置好的日志记录器

# 检查是否有可用的GPU，如果没有则使用CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else  # 如果有CUDA可用，使用CUDA
    "mps" if torch.backends.mps.is_available() else  # 如果有Metal Performance Shaders（MPS）可用，使用MPS
    "cpu"  # 否则使用CPU
)

# 定义疾病/综合症环境类
class Environment(object):
    def __init__(self, symptoms, state_elems, herbs):
        # 构建状态空间
        self.state_space = {}  # 定义状态空间的字典
        self.swapped_state_space = {}  # 定义交换状态空间的字典
        self.symp_len = len(symptoms)  # 症状的数量
        self.symp_se_len = self.symp_len + len(state_elems)  # 症状和状态元素的总数量
        for index in symptoms:
            self.state_space[index] = symptoms[index]  # 将症状添加到状态空间
            self.swapped_state_space[symptoms[index]] = index  # 将症状名称映射到索引
        for index in state_elems:
            self.state_space[index + self.symp_len] = state_elems[index]  # 将状态元素添加到状态空间
            self.swapped_state_space[state_elems[index]] = index + self.symp_len  # 将状态元素名称映射到索引
        for index in herbs:
            self.state_space[index + self.symp_se_len] = herbs[index]  # 将草药添加到状态空间
            self.swapped_state_space[herbs[index]] = index + self.symp_se_len  # 将草药名称映射到索引

        self.action_space = herbs  # 定义动作空间
        self.swapped_action_space = {}  # 定义交换动作空间的字典
        for index in self.action_space:
            self.swapped_action_space[self.action_space[index]] = index  # 将草药映射到索引
    
    def reset(self, data_piece):
        self.piece_symptoms = data_piece[0]  # 当前数据片的症状
        self.piece_state_elems = data_piece[1]  # 当前数据片的状态元素
        self.piece_herbs = []  # 当前数据片的草药索引列表，用于判断奖励
        for herb in data_piece[2]:
            self.piece_herbs.append(self.swapped_action_space[herb])  # 将草药映射到索引并添加到当前草药列表

        # 初始化状态为零一向量
        self.state = np.zeros(len(self.state_space), dtype=int)  # 初始化状态向量为零
        # 填充症状部分
        for symp in self.piece_symptoms:
            symp_index = self.swapped_state_space[symp]  # 获取症状的索引
            self.state[symp_index] = 1  # 将对应状态置为1
        # 填充状态元素部分
        for se in self.piece_state_elems:
            se_index = self.swapped_state_space[se]  # 获取状态元素的索引
            self.state[se_index] = 1  # 将对应状态置为1
        
        return self.state  # 返回初始化后的状态

    def step(self, action, selected_actions):
        terminated = False  # 标记是否终止

        if action in self.piece_herbs:
            reward = 1  # 如果选择的动作（草药）在正确的草药列表中，奖励为1
        else:
            reward = -1  # 否则，奖励为-1
        if len(selected_actions) == len(self.piece_herbs):
            terminated = True  # 如果已经选择的动作数量等于正确的草药数量，标记为终止
            return None, reward, terminated  # 返回None（没有下一个状态），奖励和终止标记
        else:
            self.state[self.symp_se_len + action] = 1  # 更新状态，选择的草药对应的状态置为1
            return self.state, reward, terminated  # 返回更新后的状态，奖励和终止标记
    
    def step_for_eval(self, action):
        self.state[self.symp_se_len + action] = 1  # 更新状态，用于评估
        return self.state  # 返回更新后的状态

# 定义“转移”命名元组，表示环境中的一个转移
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))  # 转移包括状态、动作、奖励和下一个状态

# 定义重放记忆类，用于存储和重用过去的转移
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)  # 创建一个最大长度为capacity的双端队列，用于存储转移

    def push(self, *args):
        """保存一个转移"""
        self.memory.append(Transition(*args))  # 将转移添加到记忆中

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # 随机采样batch_size个转移

    def __len__(self):
        return len(self.memory)  # 返回记忆的长度

# 定义DQN神经网络
class DQN(nn.Module):
    def __init__(self, state_vector_len, n_actions, n_units=10):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_vector_len, n_units)  # 定义第一层，全连接层，输入大小为状态向量长度，输出大小为n_units
        self.layer2 = nn.Linear(n_units, n_actions)  # 定义第二层，全连接层，输入大小为n_units，输出大小为动作数量

    def forward(self, x):
        x = F.relu(self.layer1(x))  # 第一个全连接层后接ReLU激活函数
        return self.layer2(x)  # 输出层

# 获取中医数据，并缩减数据集以包括最多N（<=20）种草药
def get_tcm_data(filename):
    # print("generate tcm data...")
    logger.info("generate tcm data...")  # 记录生成中医数据的日志信息
    data = []  # 定义数据列表
    with open(filename, 'r', encoding='utf-8') as file:
        # 跳过第一行标题行：编号 病种分类 病证名 中医症状 中医证型 状态要素 中药 经方
        file.readline()

        # 遍历文件的每一行
        for line in file:
            line = line.strip()  # 移除行末的换行符
            data.append(line)  # 将行数据添加到数据列表中
    logger.info(f"data总数：{len(data)}")  # 记录数据总数的日志信息

    # 注意：shuffle会导致每次使用不同子集的数据集
    logger.info(f"shuffle tcm data.")  # 记录打乱数据集的日志信息
    random.seed(9)  # 设置随机种子，以便结果可复现
    random.shuffle(data)  # 打乱数据集

    tuples4gen = []  # 定义生成的数据的列表
    # 定义症状集合、状态要素集合和草药集合
    symptom_set = set()
    state_elem_set = set()
    herb_set = set()
    # 定义索引到名称的映射
    symptom_map = {}
    state_elem_map = {}
    herb_map = {}

    # 定义草药到频率的映射
    max_herb_num_used = 0  # 处方中使用草药的最大数量
    min_herb_num_used = 100  # 处方中使用草药的最小数量
    max_herb_num = 15  # 使用频率最高的草药数量
    herb_freq_map = {}  # 草药频率字典

    for piece in data:
        parts = piece.split()  # 将行数据分割成各个部分

        # 获取草药部分
        herbs = parts[6].split(',')  # 将草药字段分割成草药列表
        if len(herbs) > max_herb_num_used:
            max_herb_num_used = len(herbs)  # 更新最大草药数量
        if len(herbs) < min_herb_num_used:
            min_herb_num_used = len(herbs)  # 更新最小草药数量

        for herb in herbs:
            if herb not in herb_set:
                herb_map[len(herb_set)] = herb  # 将新的草药添加到映射中
                herb_set.add(herb)  # 将草药添加到集合中
                herb_freq_map[herb] = 1  # 初始化草药频率为1
            else:
                herb_freq_map[herb] += 1  # 增加草药频率

        # 获取所有症状、状态元素和草药及其索引
        symptoms = parts[3].split(',')  # 将症状字段分割成症状列表
        for symp in symptoms:
            if symp not in symptom_set:
                symptom_map[len(symptom_set)] = symp  # 将新的症状添加到映射中
                symptom_set.add(symp)  # 将症状添加到集合中
        state_elems = parts[5].split(',')  # 将状态要素字段分割成状态要素列表
        for se in state_elems:
            if se not in state_elem_set:
                state_elem_map[len(state_elem_set)] = se  # 将新的状态要素添加到映射中
                state_elem_set.add(se)  # 将状态要素添加到集合中

        tuples4gen.append((symptoms, state_elems, herbs))  # 将症状、状态要素和草药作为元组添加到生成的数据列表中

    logger.info(f"症状集合的元素个数为{len(symptom_map)}") # 记录症状集合大小的日志信息
    logger.info(f"状态要素集合的元素个数为{len(state_elem_map)}") # 记录状态要素集合大小的日志信息
    logger.info(f"中药集合的元素个数为{len(herb_map)}") # 记录草药集合大小的日志信息
    logger.info(f"状态空间的基数为{len(symptom_map)+len(state_elem_map)+len(herb_map)}") # 记录状态空间大小的日志信息

    logger.info(f"全集处方最大长度：{max_herb_num_used}") # 处方最大长度：记录日志信息
    logger.info(f"全集处方最小长度：{min_herb_num_used}") # 处方最小长度：记录日志信息

    # 获取使用频率排名前N的草药。按频率降序排列，名称升序排列。
    sorted_herbs = sorted(herb_freq_map.items(), key=lambda x: (-x[1], x[0]))  # 按频率降序排列草药，并按名称升序排列草药
    herb_freq_list = sorted_herbs[:max_herb_num]  # 获取频率最高的前N种草药
    logger.info(f"使用频率最高的前{max_herb_num}味药物：{herb_freq_list}") # 记录日志信息

    top_n_herbs = set()
    for herb, freq in herb_freq_list:
        top_n_herbs.add(herb)  # 获取排序后的顶级草药

    # 过滤数据集以仅包括前N种草药。
    # 还在此过程中重构使用的内容。
    symptom_set = set()  # 重新初始化症状集合
    state_elem_set = set()  # 重新初始化状态要素集合
    herb_set = set()  # 重新初始化草药集合
    filt_symptom_map = {}  # 重新初始化过滤后的症状映射
    filt_state_elem_map = {}  # 重新初始化过滤后的状态要素映射
    filt_herb_map = {}  # 重新初始化过滤后的草药映射
    filt_tuples4gen = []  # 重新初始化过滤后的生成数据列表

    total_herb_num = 0  # 总草药数量
    max_herb_num_used = 0  # 过滤后的处方最大草药数量
    min_herb_num_used = 100  # 过滤后的处方最小草药数量
    for data in tuples4gen:
        if set(data[2]).issubset(top_n_herbs):  # 如果草药集合是顶级草药集合的子集
            filt_tuples4gen.append(data)  # 将数据添加到过滤后的生成数据列表

            if max_herb_num_used < len(data[2]):
                max_herb_num_used = len(data[2])  # 更新最大草药数量
            if min_herb_num_used > len(data[2]):
                min_herb_num_used = len(data[2])  # 更新最小草药数量

            total_herb_num += len(data[2])  # 更新总草药数量

            for herb in data[2]:
                if herb not in herb_set:
                    filt_herb_map[len(herb_set)] = herb  # 将草药添加到过滤后的草药映射
                    herb_set.add(herb)  # 将草药添加到集合

            for symp in data[0]:
                if symp not in symptom_set:
                    filt_symptom_map[len(symptom_set)] = symp  # 将症状添加到过滤后的症状映射
                    symptom_set.add(symp)  # 将症状添加到集合

            for se in data[1]:
                if se not in state_elem_set:
                    filt_state_elem_map[len(state_elem_set)] = se  # 将状态要素添加到过滤后的状态要素映射
                    state_elem_set.add(se)  # 将状态要素添加到集合

    logger.info(f"新集处方最大长度：{max_herb_num_used}") # 记录新集合处方最大长度的日志信息
    logger.info(f"新集处方最小长度：{min_herb_num_used}") # 记录新集合处方最小长度的日志信息
    logger.info(f"新集处方药物出现总次数：{total_herb_num}") # 记录新集合处方草药出现总次数的日志信息

    logger.info(f"减少后data总数：{len(filt_tuples4gen)}") # 记录减少后数据总数的日志信息
    logger.info(f"症状集合的元素个数：{len(filt_symptom_map)}") # 记录新集合症状集合大小的日志信息
    logger.info(f"状态要素集合的元素个数：{len(filt_state_elem_map)}") # 记录新集合状态要素集合大小的日志信息
    logger.info(f"中药集合的元素个数：{len(filt_herb_map)}") # 记录新集合草药集合大小的日志信息
    logger.info(f"状态空间的基数：{len(filt_symptom_map)+len(filt_state_elem_map)+len(filt_herb_map)}") # 记录新集合状态空间大小的日志信息
    logger.info("generation done.") # 记录生成完成的日志信息

    return filt_tuples4gen, filt_symptom_map, filt_state_elem_map, filt_herb_map, max_herb_num_used  # 返回过滤后的生成数据、症状映射、状态要素映射、草药映射和最大草药数量
######################################################################
# - ``select_action`` - 将根据 epsilon 贪婪策略选择一个动作。简单来说，有时我们会使用模型来选择动作，有时我们会直接随机选择一个。选择随机动作的概率将从 ``EPS_START`` 开始，并以指数形式衰减到 ``EPS_END``。``EPS_DECAY`` 控制衰减率。
#

steps_done = 0  # 已完成的步骤数

def select_action(state, selected_actions):  # 根据 epsilon 贪婪策略选择动作

    global steps_done  # 使用全局变量 steps_done
    sample = random.random()  # 生成一个 [0, 1) 之间的随机数
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)  # 计算当前步的 epsilon 值
    steps_done += 1  # 增加已完成的步骤数

    use_first_qmax = True  # 一个标识，用于指示是否使用第一次的 Q 值最大动作
    if sample > eps_threshold:  # 如果随机数大于 epsilon 值
        output = policy_net(state)  # 使用策略网络预测动作
        while True:
            with torch.no_grad():  # 在不计算梯度的情况下
                action = output.max(1).indices.view(1, 1)  # 获取 Q 值最大的动作
                ##
                # 避免重复动作
                #
                if action.item() not in selected_actions:  # 如果这个动作没有被选择过
                    return action, "agent", use_first_qmax  # 返回动作及其类型
                else:  # 如果动作重复
                    use_first_qmax = False  # 将标识设为 False
                    output[0][action] = -999999  # 将此动作的 Q 值设为非常小的值
    else:
        # 随机选择一个动作
        while True:
            action = random.randrange(n_actions)  # 随机生成一个动作
            if action not in selected_actions:  # 如果这个动作没有被选择过
                action = torch.tensor([[action]], device=device, dtype=torch.long)  # 转换为 tensor
                return action, "random", False  # 返回动作及其类型


def select_action_4eval(state, selected_actions):  # 为评估选择动作

    output = policy_net(state)  # 使用策略网络预测动作
    iter_index = 0  # 迭代索引
    while True:
        with torch.no_grad():  # 在不计算梯度的情况下
            action = output.max(1).indices.view(1, 1)  # 获取 Q 值最大的动作
            ##
            # 避免重复动作
            #
            if action.item() not in selected_actions:  # 如果这个动作没有被选择过
                return action  # 返回动作
            else:
                output[0][action] = -999999  # 将此动作的 Q 值设为非常小的值


def optimize_model():  # 优化模型
    if len(memory) < BATCH_SIZE:  # 如果存储的记忆不够一个批次
        return
    transitions = memory.sample(BATCH_SIZE)  # 从记忆中随机采样一个批次
    # 转置批次（详细解释见 https://stackoverflow.com/a/19343/3343043）。这会把批次数组的转换为每个元素包含一个 batch 的样本
    batch = Transition(*zip(*transitions))

    # 计算非终结状态的掩码，并连接批次元素（终止状态是指模拟结束后的状态）
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)  # 连接状态批次
    action_batch = torch.cat(batch.action)  # 连接动作批次
    reward_batch = torch.cat(batch.reward)  # 连接奖励批次

    # 计算 Q(s_t, a)，模型会计算 Q(s_t)，然后我们选择采取的动作的列。根据 policy_net，为每个批次状态采取这些动作
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 计算 V(s_{t+1}) 对于所有的下一个状态。
    # 使用 "较旧" 的 target_net 计算非终结状态动作的期望值；通过 max(1).values 选择最佳奖励
    # 这将基于掩码合并，这样我们要么有期望的状态值，要么如果状态是终结状态，为 0。
    # 这里可以看到，如果下一个状态是终结状态，对应的 next state 值为 0，导致 yj = rj。
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # 计算期望的 Q 值
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 计算 Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    # 就地梯度裁剪
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


##
# 评估 Jaccard 结果。
#
def eval_jaccard(selected_herbs, true_herbs):

    correct_herbs = set(selected_herbs) & set(true_herbs)  # 计算正确的药草
    union_herbs = set(selected_herbs) | set(true_herbs)  # 计算药草的并集

    return float(len(correct_herbs)) / len(union_herbs)  # 返回 Jaccard 值


##
# 评估 precision 结果。
#
def eval_precision(selected_herbs, true_herbs):

    correct_herbs = set(selected_herbs) & set(true_herbs)  # 计算正确的药草

    return float(len(correct_herbs)) / len(selected_herbs)  # 返回 precision 值


##
# 评估 recall 结果。
#
def eval_recall(selected_herbs, true_herbs):

    correct_herbs = set(selected_herbs) & set(true_herbs)  # 计算正确的药草

    return float(len(correct_herbs)) / len(true_herbs)  # 返回 recall 值


##
# 评估 F1 score 结果。
#
def eval_f1(precision, recall):

    if precision + recall == 0:  # 如果 precision 和 recall 都为 0
        return 0.0

    f1_score = 2 * precision * recall / (precision + recall)  # 计算 F1 score
    return f1_score


def print_metrics(selected_herbs,
                  max_jaccard, min_jaccard, avg_jaccard,
                  max_precision, min_precision, avg_precision,
                  max_recall, min_recall, avg_recall,
                  max_f1, min_f1, avg_f1):
    logger.info(f"**# of herbs to prescribe**:@{len(selected_herbs)}")  # 记录需要开出的药草数量
    logger.info(f"**max jaccard**:{max_jaccard}")  # 记录最大 Jaccard 值
    logger.info(f"**min jaccard**:{min_jaccard}")  # 记录最小 Jaccard 值
    logger.info(f"**avg jaccard**:{avg_jaccard / len(new_test_data)}")  # 记录平均 Jaccard 值
    logger.info(f"**max precision**:{max_precision}")  # 记录最大 precision 值
    logger.info(f"**min precision**:{min_precision}")  # 记录最小 precision 值
    logger.info(f"**avg precision**:{avg_precision / len(new_test_data)}")  # 记录平均 precision 值
    logger.info(f"**max recall**:{max_recall}")  # 记录最大 recall 值
    logger.info(f"**min recall**:{min_recall}")  # 记录最小 recall 值
    logger.info(f"**avg recall**:{avg_recall / len(new_test_data)}")  # 记录平均 recall 值
    logger.info(f"**max F1**:{max_f1}")  # 记录最大 F1 值
    logger.info(f"**min F1**:{min_f1}")  # 记录最小 F1 值
    logger.info(f"**avg F1**:{avg_f1 / len(new_test_data)}\n")  # 记录平均 F1 值


def update_metrics(selected_herbs, true_herbs,
                   max_jaccard, min_jaccard, avg_jaccard,
                   max_precision, min_precision, avg_precision,
                   max_recall, min_recall, avg_recall,
                   max_f1, min_f1, avg_f1):
    jaccard = eval_jaccard(selected_herbs, true_herbs)  # 评估 Jaccard 值
    precision = eval_precision(selected_herbs, true_herbs)  # 评估 precision 值
    recall = eval_recall(selected_herbs, true_herbs)  # 评估 recall 值
    f1_score = eval_f1(precision, recall)  # 评估 F1 score

    if max_jaccard < jaccard:  # 更新最大 Jaccard 值
        max_jaccard = jaccard
    if min_jaccard > jaccard:  # 更新最小 Jaccard 值
        min_jaccard = jaccard
    avg_jaccard += jaccard  # 累加平均 Jaccard 值

    if max_precision < precision:  # 更新最大 precision 值
        max_precision = precision
    if min_precision > precision:  # 更新最小 precision 值
        min_precision = precision
    avg_precision += precision  # 累加平均 precision 值

    if max_recall < recall:  # 更新最大 recall 值
        max_recall = recall
    if min_recall > recall:  # 更新最小 recall 值
        min_recall = recall
    avg_recall += recall  # 累加平均 recall 值

    if max_f1 < f1_score:  # 更新最大 F1 值
        max_f1 = f1_score
    if min_f1 > f1_score:  # 更新最小 F1 值
        min_f1 = f1_score
    avg_f1 += f1_score  # 累加平均 F1 值

    return max_jaccard, min_jaccard, avg_jaccard, \
           max_precision, min_precision, avg_precision, \
           max_recall, min_recall, avg_recall, \
           max_f1, min_f1, avg_f1  # 返回更新后的指标值

def evaluate():  # 评估函数
    logger.info("evaluate...")  # 记录评估开始信息
    piece_index = 0  # 数据块索引

    piece_to_eval = []  # 要评估的数据块列表
    for data in new_test_data:
        piece_to_eval.append((piece_index, new_test_data[piece_index]))  # 添加数据块
        piece_index += 1  # 增加数据块索引

    n_herbs_list = [5, 10]  # 要选择的药草数量列表

    avg5_jaccard = 0.0  # 5个药草的平均 Jaccard 值
    max5_jaccard = 0.0  # 5个药草的最大 Jaccard 值
    min5_jaccard = 1.0  # 5个药草的最小 Jaccard 值
    avg5_precision = 0.0  # 5个药草的平均 precision 值
    max5_precision = 0.0  # 5个药草的最大 precision 值
    min5_precision = 1.0  # 5个药草的最小 precision 值
    avg5_recall = 0.0  # 5个药草的平均 recall 值
    max5_recall = 0.0  # 5个药草的最大 recall 值
    min5_recall = 1.0  # 5个药草的最小 recall 值
    avg5_f1 = 0.0  # 5个药草的平均 F1 值
    max5_f1 = 0.0  # 5个药草的最大 F1 值
    min5_f1 = 1.0  # 5个药草的最小 F1 值

    avg10_jaccard = 0.0  # 10个药草的平均 Jaccard 值
    max10_jaccard = 0.0  # 10个药草的最大 Jaccard 值
    min10_jaccard = 1.0  # 10个药草的最小 Jaccard 值
    avg10_precision = 0.0  # 10个药草的平均 precision 值
    max10_precision = 0.0  # 10个药草的最大 precision 值
    min10_precision = 1.0  # 10个药草的最小 precision 值
    avg10_recall = 0.0  # 10个药草的平均 recall 值
    max10_recall = 0.0  # 10个药草的最大 recall 值
    min10_recall = 1.0  # 10个药草的最小 recall 值
    avg10_f1 = 0.0  # 10个药草的平均 F1 值
    max10_f1 = 0.0  # 10个药草的最大 F1 值
    min10_f1 = 1.0  # 10个药草的最小 F1 值

    # 遍历每个要评估的数据块
    for data_piece in piece_to_eval:
        piece_index = data_piece[0]  # 获取数据块的索引

        state = env.reset(data_piece[1])  # 环境重置
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # 转换为 tensor 并增加一个维度

        selected_actions = []  # 已选择的动作列表
        while True:
            if len(selected_actions) == n_herbs_list[1]:  # 如果已选择的动作达到最大数量
                break
            action = select_action_4eval(state, selected_actions)  # 为评估选择动作

            selected_actions.append(action.item())  # 将选择的动作添加到列表中
            observation = env.step_for_eval(action.item())  # 环境评估下一步
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)  # 转换为 tensor 并增加一个维度
            state = next_state  # 更新状态

        selected_herbs5 = []  # 存储5个已选择的药草
        selected_herbs10 = []  # 存储10个已选择的药草
        for action in selected_actions:
            if len(selected_herbs5) < n_herbs_list[0]:  # 如果已选择的药草数量小于5
                selected_herbs5.append(env.action_space[action])  # 添加到已选择的药草列表中
            if len(selected_herbs10) < n_herbs_list[1]:  # 如果已选择的药草数量小于10
                selected_herbs10.append(env.action_space[action])  # 添加到已选择的药草列表中

        max5_jaccard, min5_jaccard, avg5_jaccard, \
        max5_precision, min5_precision, avg5_precision, \
        max5_recall, min5_recall, avg5_recall, \
        max5_f1, min5_f1, avg5_f1 = \
            update_metrics(selected_herbs5, data_piece[1][2],
                           max5_jaccard, min5_jaccard, avg5_jaccard,
                           max5_precision, min5_precision, avg5_precision,
                           max5_recall, min5_recall, avg5_recall,
                           max5_f1, min5_f1, avg5_f1)  # 更新5个已选择药草的指标值

        max10_jaccard, min10_jaccard, avg10_jaccard, \
        max10_precision, min10_precision, avg10_precision, \
        max10_recall, min10_recall, avg10_recall, \
        max10_f1, min10_f1, avg10_f1 = \
            update_metrics(selected_herbs10, data_piece[1][2],
                           max10_jaccard, min10_jaccard, avg10_jaccard,
                           max10_precision, min10_precision, avg10_precision,
                           max10_recall, min10_recall, avg10_recall,
                           max10_f1, min10_f1, avg10_f1)  # 更新10个已选择药草的指标值

    # 打印5个已选择药草的指标结果
    print_metrics(selected_herbs5, max5_jaccard, min5_jaccard, avg5_jaccard,
                  max5_precision, min5_precision, avg5_precision,
                  max5_recall, min5_recall, avg5_recall,
                  max5_f1, min5_f1, avg5_f1)

    # 打印10个已选择药草的指标结果
    print_metrics(selected_herbs10, max10_jaccard, min10_jaccard, avg10_jaccard,
                  max10_precision, min10_precision, avg10_precision,
                  max10_recall, min10_recall, avg10_recall,
                  max10_f1, min10_f1, avg10_f1)
##
# 主体部分。
# 进行智能体和环境之间的交互。
# 智能体在这个过程中不断学习。
#
def train(n_episodes):  # 训练函数，参数为迭代次数
    if torch.cuda.is_available() or torch.backends.mps.is_available():  # 判断是否有 GPU 或 M1 加速器
        num_episodes = n_episodes  # 如果有，则使用传入的迭代次数
    else:
        num_episodes = 50  # 如果没有，则使用默认的50次迭代

    logger.info(f"总的迭代次数: {num_episodes}")  # 记录总的迭代次数

    epi_rewards_list = []  # 用于记录每个迭代的奖励列表

    st = time.perf_counter()  # 获取开始时间
    for i_episode in range(num_episodes):  # 遍历每个迭代
        logger.info(f"第 {i_episode} 次迭代开始...")  # 记录当前迭代次数

        iter_herbs = 0  # 初始化药草迭代次数
        pres_rewards_list = []  # 初始化处方奖励列表

        agent_count = 0  # 智能体选择动作的次数计数
        random_count = 0  # 随机选择动作的次数计数
        first_qmax_count = 0  # 第一次使用最大 Q 值的次数计数

        epi_st = time.perf_counter()  # 获取当前迭代的开始时间

        ##
        # "玩" 有标签的处方。
        # 每个数据项: (症状, 状态元素, 药草)
        #
        for data_piece in training_tcm_data:  # 遍历每个训练数据
            piece_herbs = data_piece[2]  # 获取药草列表（作为迭代次数）

            state = env.reset(data_piece)  # 环境重置
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # 状态转换为 tensor，并增加一个维度

            pres_rewards = 0  # 初始化处方奖励
            selected_actions = []  # 初始化已选择动作列表
            for t in range(len(piece_herbs)):  # 遍历每个药草
                
                action, actor, use_first_qmax = select_action(state, selected_actions)  # 选择一个动作
                selected_actions.append(action.item())  # 将选择的动作添加到列表中

                # 执行动作，观察环境并获取奖励
                observation, reward, terminated = env.step(action.item(), selected_actions)  # 获取新的观察值、奖励和是否终止的标记
                if reward > 0:  # 如果奖励大于 0
                    if actor == "agent":  # 如果动作是智能体选择的
                        agent_count += 1  # 增加智能体动作计数
                    elif actor == "random":  # 如果动作是随机选择的
                        random_count += 1  # 增加随机动作计数
                    if use_first_qmax:  # 如果使用了第一次的最大 Q 值
                        first_qmax_count += 1  # 增加首次最大 Q 值计数
                    pres_rewards += reward  # 累加处方奖励

                reward = torch.tensor([reward], device=device)  # 奖励转换为 tensor

                if terminated:  # 如果处方完成
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)  # 新状态转换为 tensor

                # 存储转换到记忆中
                memory.push(state, action, reward, next_state)

                # 移动到下一个状态
                state = next_state

                iter_herbs += 1  # 增加药草迭代次数

                if terminated:  # 如果处方完成
                    break
            
            if iter_herbs >= 20:  # 如果药草迭代次数大于等于20

                optimize_model()  # 优化模型

                # 软更新目标网络的权重
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()  # 获取目标网络的状态字典
                policy_net_state_dict = policy_net.state_dict()  # 获取策略网络的状态字典
                for key in policy_net_state_dict:  # 遍历所有参数
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)  # 更新目标网络的参数
                target_net.load_state_dict(target_net_state_dict)  # 加载更新后的状态字典到目标网络

                iter_herbs = 0  # 重置药草迭代计数

            pres_rewards_list.append(pres_rewards)  # 将当前处方奖励添加到列表中
        epi_et = time.perf_counter()  # 获取当前迭代的结束时间

        evaluate()  # 评估模型
        logger.info(f"第 {i_episode} 次迭代的训练时间: {epi_et-epi_st:.6f}秒")  # 记录当前迭代的训练时间

        logger.info(f"处方奖励列表:{pres_rewards_list}")  # 记录处方奖励列表
        epi_rewards_list.append(sum(pres_rewards_list))  # 记录当前迭代的总奖励
        logger.info(f"迭代奖励列表:{epi_rewards_list}")  # 记录所有迭代的奖励列表

        logger.info(f"第 {i_episode} 次迭代完成。")  # 记录当前迭代完成信息
        
    et = time.perf_counter()  # 获取总结束时间

    logger.info(f"训练时间总共: {et-st:.6f}秒")  # 记录总的训练时间

    logger.info('训练完成')  # 记录训练完成信息


if __name__ == "__main__":  # 主程序入口
   parser = ArgumentParser(description="DQN Intelligent Prescription Generation")  # 创建命令行参数解析器
   parser.add_argument("-episode", type=int, dest="num_episodes", help="the number of episodes")  # 添加命令行参数：迭代次数
   parser.add_argument("-batch", type=int, dest="batch_size", help="the batch size")  # 添加命令行参数：批次大小
   parser.add_argument("-mem_capacity", type=int, dest="mem_capacity", help="the replay memory capacity")  # 添加命令行参数：重放记忆容量
   parser.add_argument("-gamma", type=float, dest="gamma", help="the gamma")  # 添加命令行参数：gamma
   parser.add_argument("-eps_start", type=float, dest="eps_start", help="the eps_start")  # 添加命令行参数：epsilon 起始值
   parser.add_argument("-eps_end", type=float, dest="eps_end", help="the eps_end")  # 添加命令行参数：epsilon 终止值
   parser.add_argument("-eps_decay", type=int, dest="eps_decay", help="the eps_decay")  # 添加命令行参数：epsilon 衰减值
   parser.add_argument("-tau", type=float, dest="tau", help="the tau")  # 添加命令行参数：tau
   parser.add_argument("-lr", type=float, dest="lr", help="the learning rate")  # 添加命令行参数：学习率
   parser.add_argument("-nn_units", type=int, dest="nn_units", help="the number of NN units one hidden layer")  # 添加命令行参数：隐藏层神经网络单元数量

   args = parser.parse_args()  # 解析命令行参数
   BATCH_SIZE = args.batch_size  # 获取批次大小
   MEM_CAPACITY = args.mem_capacity  # 获取重放记忆容量
   GAMMA = args.gamma  # 获取 gamma 值
   EPS_START = args.eps_start  # 获取 epsilon 起始值
   EPS_END = args.eps_end  # 获取 epsilon 终止值
   EPS_DECAY = args.eps_decay  # 获取 epsilon 衰减值
   TAU = args.tau  # 获取 tau 值
   LR = args.lr  # 获取学习率
   nn_units = args.nn_units  # 获取隐藏层神经网络单元数量
   
   args.num_episodes = 100
   BATCH_SIZE = 64 # 获取批次大小
   MEM_CAPACITY = 1000  # 获取重放记忆容量
   GAMMA = 0.99  # 获取 gamma 值
   EPS_START = 0.9  # 获取 epsilon 起始值
   EPS_END = 0.05  # 获取 epsilon 终止值
   EPS_DECAY = 2000  # 获取 epsilon 衰减值
   TAU = 0.005  # 获取 tau 值
   LR = 0.01  # 获取学习率
   nn_units = 2  # 获取隐藏层神经网络单元数量
   
   lr_name = str(LR)  # 将学习率转换为字符串
   if '.' in lr_name and len(lr_name.split('.')) > 1:  #获取小数部分
    lr_name = lr_name.split('.')[1]

   now = datetime.now()  # 获取当前时间
   time_sign = f"{now.year}年{now.month}月{now.day}日{now.hour}时{now.minute}分"  # 格式化时间为字符串

   memory = ReplayMemory(MEM_CAPACITY)  # 创建重放记忆对象

   logger = createLogger(f"dqn_LR{lr_name}_{nn_units}units_{time_sign}.log")  # 创建日志记录对象
   logger.info(f"可用设备:{device}")  # 记录可用设备

    ##
    # 获取数据并尝试构建状态和动作空间
    #
   tcm_data, symptoms, state_elems, herbs, max_pres_len = get_tcm_data('D:\DaiMa\8.13\dataset\伤寒论诊疗数据集药方.txt')  # 获取数据
   env = Environment(symptoms, state_elems, herbs)  # 创建环境对象

    ##
    # 将数据集分为训练集和测试集
    #
   training_tcm_data = []  # 初始化训练数据列表
   test_tcm_data = []  # 初始化测试数据列表

   for i in range(len(tcm_data)):  # 遍历每个数据项
       if len(tcm_data[i][2]) >= 6:  # 如果数据项中的药草数量大于等于6
           test_tcm_data.append(tcm_data[i])  # 添加到测试数据列表
       else:
           training_tcm_data.append(tcm_data[i])  # 添加到训练数据列表

   give_back_indices = random.sample(list(range(len(test_tcm_data))), len(test_tcm_data) - int(len(tcm_data) * 0.1))  # 随机重新分配10%的数据范围
   new_test_data = []  # 初始化新的测试数据列表
   for i in range(len(test_tcm_data)):  # 遍历每个测试数据
       if i in give_back_indices:  # 如果索引在重新分配的范围
           training_tcm_data.append(test_tcm_data[i])  # 将测试数据添加到训练数据列表中
       else:
           new_test_data.append(test_tcm_data[i])  # 将测试数据添加到新的测试数据列表中

   logger.info(f"训练数据数量:{len(training_tcm_data)}")  # 记录训练数据的数量
   logger.info(f"测试数据数量:{len(new_test_data)}")  # 记录测试数据的数量

   state_vector_len = len(env.state_space)  # 获取状态向量长度
   logger.info(f"状态向量长度:{state_vector_len}")  # 记录状态向量长度

   n_actions = len(env.action_space)  # 获取动作数
   logger.info(f"动作数: {n_actions}")  # 记录动作数

   policy_net = DQN(state_vector_len, n_actions, nn_units).to(device)  # 创建策略网络
   target_net = DQN(state_vector_len, n_actions, nn_units).to(device)  # 创建目标网络
   target_net.load_state_dict(policy_net.state_dict())  # 将策略网络的参数复制到目标网络

   logger.info(f"隐藏层使用的神经网络单元数量: {nn_units}")  # 记录隐藏层使用的神经网络单元数量
   logger.info(f"DQN 模型结构: {policy_net}\n")  # 记录 DQN 模型结构
   for name, param in policy_net.named_parameters():  # 遍历所有参数
       print(f"层: {name} | 大小: {param.size()}\n")  # 打印每一层的名称和大小

   optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)  # 创建优化器

   logger.info(f"**设置**")  # 记录设置
   logger.info(f"批次大小: {BATCH_SIZE}")  # 记录批次大小
   logger.info(f"重放记忆容量: {MEM_CAPACITY}")  # 记录重放记忆容量
   logger.info(f"gamma: {GAMMA}")  # 记录 gamma
   logger.info(f"epsilon 起始值: {EPS_START}")  # 记录 epsilon 起始值
   logger.info(f"epsilon 终止值: {EPS_END}")  # 记录 epsilon 终止值
   logger.info(f"epsilon 衰减值: {EPS_DECAY}")  # 记录 epsilon 衰减值
   logger.info(f"tau: {TAU}")  # 记录 tau
   logger.info(f"学习率: {LR}")  # 记录学习率

   train(args.num_episodes)  # 开始训练，传入迭代次数