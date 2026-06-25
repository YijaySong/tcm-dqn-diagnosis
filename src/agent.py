import numpy as np  # 导入NumPy，用于数值计算
import random  # 导入随机模块，用于随机数生成
from collections import deque  # 导入双端队列，用于存储经验回放
import torch  # 导入PyTorch库，用于深度学习
import torch.nn as nn  # 导入PyTorch中的神经网络模块
import torch.optim as optim  # 导入PyTorch中的优化器模块

class DQN(nn.Module):  # 定义深度Q网络模型，继承自nn.Module
    def __init__(self, state_size, action_size):  # 初始化函数，定义网络结构
        super(DQN, self).__init__()  # 调用父类的初始化方法
        self.fc1 = nn.Linear(state_size, 24)  # 定义第一层全连接层，将状态大小映射到24个神经元
        self.fc2 = nn.Linear(24, 24)  # 定义第二层全连接层，24个神经元到24个神经元
        self.fc3 = nn.Linear(24, action_size)  # 定义第三层全连接层，24个神经元映射到动作大小

    def forward(self, x): # 前向传播函数
        x = torch.relu(self.fc1(x))  # 对第一层的输出应用ReLU激活函数
        x = torch.relu(self.fc2(x))  # 对第二层的输出应用ReLU激活函数
        return self.fc3(x)  # 输出动作值，不应用激活函数

class DQNAgent:   # 定义DQN智能体
    def __init__(self, state_size, action_size):  # 初始化函数
        self.state_size = state_size  # 状态空间的大小
        self.action_size = action_size  # 动作空间的大小
        self.memory = deque(maxlen=2000)  # 经验回放的存储器，最大存储2000条经验
        self.gamma = 0.95    # discount rate 折扣因子，用于计算未来奖励的权重
        self.epsilon = 1.0   # exploration rate 探索率，初始为1.0，表示完全随机选择动作
        self.epsilon_min = 0.01  # 探索率的最小值
        self.epsilon_decay = 0.995  # 探索率的衰减率
        self.learning_rate = 0.001  # 学习率
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置训练设备，优先使用GPU
        self.model = DQN(state_size, action_size).to(self.device)  # 初始化DQN模型并将其移至设备
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # 使用Adam优化器
        self.criterion = nn.MSELoss()  # 使用均方误差作为损失函数

    def remember(self, state, action, reward, next_state, done):  # 存储经验
        self.memory.append((state, action, reward, next_state, done))  # 将五元组(state, action, reward, next_state, done)添加到记忆中

    def act(self, state):  # 选择动作
        if np.random.rand() <= self.epsilon:  # 随机数小于探索率时，随机选择动作
            return random.randrange(self.action_size)  # 随机选择一个动作
        state = torch.FloatTensor(state).to(self.device)  # 将状态转换为张量并移至设备
        act_values = self.model(state)  # 通过网络前向传播计算动作值
        return np.argmax(act_values.cpu().data.numpy())  # 返回动作值最大的动作索引

    def replay(self, batch_size):  # 经验回放
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))  # 从记忆中随机抽取一个小批次
        loss_total = 0
        for state, action, reward, next_state, done in minibatch:  # 遍历小批次中的每个样本
            state = torch.FloatTensor(state).to(self.device)  # 将状态转换为张量并移至设备
            next_state = torch.FloatTensor(next_state).to(self.device)  # 将下一个状态转换为张量并移至设备
            target = reward  # 初始目标值为即时奖励
            if not done:  # 如果当前状态不是终止状态
                target = reward + self.gamma * torch.max(self.model(next_state).detach())  # 更新目标值为即时奖励加上折扣后的最大未来奖励
            target_f = self.model(state)  # 计算当前状态的预测动作值
            target_f[0][action] = target  # 更新目标动作的值
            loss = self.criterion(target_f, self.model(state))  # 计算损失
            self.optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新网络权重
            loss_total += loss.item() 
        if self.epsilon > self.epsilon_min:  # 逐渐降低探索率
            self.epsilon *= self.epsilon_decay  # 以衰减率降低探索率

        return loss_total / len(minibatch)  # Return average loss