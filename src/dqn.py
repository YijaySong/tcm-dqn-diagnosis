import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from sklearn.feature_selection import mutual_info_classif
# 读取数据
# data = pd.read_csv('D:\DaiMa\8.13\dataset\伤寒论诊疗数据集.txt', sep='\s+')

# # 删除指定列
# data = data.drop(columns=['编号', '病种分类', '病证名', '中医证型'])
# print(data)
# # 计算“状态要素”列中的标签数，并新增“标签数”列
# data['标签数'] = data['状态要素'].apply(lambda x: len(x.split(',')))

# # 保存预处理数据
# data.to_csv('D:\DaiMa\8.13\dataset\preprocessed_data.txt', sep=' ', index=False)

df = pd.read_csv('D:\DaiMa\8.13\dataset\preprocessed_data.txt', sep=' ', header=0)
df['状态要素'] = df['状态要素'].apply(lambda x: x.split(','))
# df['中医症状 '] = df['中医症状 '].apply(lambda x: x.split(','))
# 提取所有可能的状态要素
all_labels = set()
for features in df['状态要素']:
    all_labels.update(features)
all_labels = list(all_labels)

print(all_labels)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def calculate_mutual_information(current_state, new_feature):
    current_state.append(new_feature)
    # 编码状态为数值
    state_dict = {elem: idx for idx, elem in enumerate(current_state)}
    encoded_state = [state_dict[elem] for elem in current_state]
    mi = mutual_info_classif([encoded_state], [0])
    current_state.pop()
    return mi[0]

# 参数设置
input_dim = len(all_labels)
output_dim = len(all_labels)
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
buffer_size = 10000
batch_size = 64
update_target_freq = 1000

# 初始化
buffer = ReplayBuffer(buffer_size)
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
criterion = nn.MSELoss()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# 训练循环
for episode in range(1000):
    state_index = random.randint(0, len(df) - 1)  # 随机选择一个患者的中医症状进行训练
    state = df.iloc[state_index]['中医症状'].split(',')
    label_count = df.iloc[state_index]['标签数']
    done = False
    step = 0
    state_action_features = []

    while not done:
        state_vector = np.zeros(input_dim)
        for s in state:
            if s in all_labels:
                state_vector[all_labels.index(s)] = 1

        if np.random.rand() <= epsilon:
            action_index = np.random.choice(range(output_dim))
        else:
            q_values = policy_net(torch.FloatTensor(state_vector).unsqueeze(0))
            action_index = torch.argmax(q_values).item()

        action_feature = all_labels[action_index]
        reward = calculate_mutual_information(state, action_feature)
        if reward > 0:
            reward = 1
            state.append(action_feature)
        else:
            reward = -1

        next_state_vector = np.zeros(input_dim)
        for s in state:
            if s in all_labels:
                next_state_vector[all_labels.index(s)] = 1

        if len(state) >= label_count:
            done = True

        buffer.add((state_vector, action_index, reward, next_state_vector, done))
        
        if len(buffer) > batch_size:
            batch = buffer.sample(batch_size)
            state, action, reward, next_state, done = zip(*batch)
            state = torch.FloatTensor(state)
            action = torch.LongTensor(action).unsqueeze(1)
            reward = torch.FloatTensor(reward).unsqueeze(1)
            next_state = torch.FloatTensor(next_state)
            done = torch.FloatTensor(done).unsqueeze(1)
            
            q_values = policy_net(state).gather(1, action)
            next_q_values = target_net(next_state).max(1)[0].detach().unsqueeze(1)
            expected_q_values = reward + (gamma * next_q_values * (1 - done))
            
            loss = criterion(q_values, expected_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step % update_target_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        step += 1
        
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f'Episode {episode + 1} completed.')

# 保存模型
torch.save(policy_net.state_dict(), 'dqn_model.pth')
print('模型已保存.')

# # 预测函数
# def predict_symptoms(symptoms):
#     state_vector = np.zeros(input_dim)
#     for symptom in symptoms.split(','):
#         state_vector[all_labels.index(symptom)] = 1

#     q_values = policy_net(torch.FloatTensor(state_vector).unsqueeze(0))
#     recommended_feature = all_labels[torch.argmax(q_values).item()]
#     return recommended_feature

# symptoms = '怕风,发热,汗出'
# predicted_feature = predict_symptoms(symptoms)
# print(f'Recommended feature: {predicted_feature}')