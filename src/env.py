import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TCMEnvironment:  # 定义一个中医环境类
    def __init__(self, features, labels):  # 初始化函数，接受特征和标签作为输入
        # 将输入特征转换为文本格式
        self.features = []
        for feature in features:
            self.features.append(f"中医症状为{feature[0].replace(',', '，')}，中医证型为{feature[1].replace(',', '，')}")
        self.labels = labels # [' '.join(label) for label in labels]
        # print(self.features)
        self.labels = [label.replace(',', '，') for label in labels]
        # 创建TfidfVectorizer实例用于编码特征和标签
        self.feature_vectorizer = TfidfVectorizer(max_features = 1000)
        self.label_vectorizer = TfidfVectorizer(max_features = 1000)

        # 编码特征和标签
        self.encoded_features = self.feature_vectorizer.fit_transform(self.features).toarray()
        self.encoded_labels = self.label_vectorizer.fit_transform(self.labels).toarray()
        # print(self.encoded_features[0])
        print(self.encoded_labels.shape)
        # print(asfsaf)
        self.reset()  # 初始化环境状态

    def reset(self):  # 定义重置环境的方法
        self.current_state = np.zeros(self.encoded_features.shape[1])  # 将当前状态初始化为全零数组
        self.current_label_index = np.random.choice(range(self.encoded_labels.shape[0]))  # 随机选择一个标签索引
        self.current_label = self.labels[self.current_label_index]  # 获取当前标签
        self.target_state = self.encoded_features[self.current_label_index]  # 设置目标状态为与当前标签对应的特征
        return self.current_state  # 返回当前状态

    def step(self, action):  # 定义执行一步操作的方法
        if action < 0 or action >= self.encoded_labels.shape[1]:  # 检查动作是否有效
            raise ValueError(f"Invalid action: {action}. Action must be between 0 and {self.encoded_labels.shape[1] - 1}")  # 如果动作无效，抛出异常

        # 将动作解释为预测的标签编码，假设使用one-hot编码
        predicted_label_encoding = np.zeros(self.encoded_labels.shape[1])
        predicted_label_encoding[action] = 1

        # 检查预测的标签是否与当前标签匹配
        if np.array_equal(predicted_label_encoding, self.encoded_labels[self.current_label_index]):
            reward = 1  # 给与正向奖励
            done = True  # 标记为完成状态
        else:
            reward = -0.1  # 给与负向奖励
            done = False  # 标记为未完成状态
            # 更新当前状态，使其朝目标状态移动
            diff = self.target_state - self.current_state  # 计算当前状态与目标状态的差异
            self.current_state += np.sign(diff) * 0.1  # 小步向目标状态移动

        return self.current_state, reward, done  # 返回更新后的当前状态、奖励值和完成标记

    def decode_state(self, state):  # 定义解码状态的方法，将编码状态转换回原始值
        decoded_features = self.feature_vectorizer.inverse_transform([state])
        return decoded_features[0] if decoded_features else []

    def decode_label(self, label_index):  # 定义解码标签的方法，将编码标签转换回原始标签
        decoded_label = self.label_vectorizer.inverse_transform(np.eye(1, self.encoded_labels.shape[1], label_index))
        return decoded_label[0][0] if decoded_label else ""