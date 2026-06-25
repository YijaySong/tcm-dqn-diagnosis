import numpy as np # 导入NumPy库，用于数值计算
import matplotlib.pyplot as plt
import torch
from env import TCMEnvironment # 从自定义的env模块中导入TCMEnvironment类
from data import load_and_preprocess_data, create_datasets # 从自定义的data模块中导入数据加载和预处理函数
from agent import DQNAgent # 从自定义的agent模块中导入DQNAgent类

def train_dqn_for_tcm(train_features, train_labels, episodes=400, batch_size=32): # 定义用于训练中医辨证DQN模型的函数
    env = TCMEnvironment(train_features, train_labels) # 创建中医环境实例
    state_size = env.encoded_features.shape[1]  # Use the shape of encoded features 获取状态空间的维度，即特征的数量
    action_size = len(np.unique(train_labels)) # 获取动作空间的维度，即标签的唯一值数量
    agent = DQNAgent(state_size, action_size) # 创建DQN智能体


    accuracy_list = []
    loss_list = []

    for episode in range(episodes): # 循环执行指定数量的训练集
        state = env.reset() # 重置环境，获取初始状态
        state = np.reshape(state, [1, state_size]) # 重塑状态以符合模型输入形状
        correct_predictions = 0
        
        for time in range(500): # 进行最多500步的环境交互
            action = agent.act(state) # 智能体根据当前状态选择动作
            next_state, reward, done = env.step(action) # 执行动作并接收环境的反馈（下一个状态、奖励和完成标记）
            next_state = np.reshape(next_state, [1, state_size]) # 重塑下一状态以符合模型输入形状
            
            agent.remember(state, action, reward, next_state, done) # 将当前经验存储到智能体的记忆中
            state = next_state # 更新当前状态

            if action == env.current_label_index:
                correct_predictions += 1


            if done:
                accuracy = correct_predictions / (time + 1) 
                accuracy_list.append(accuracy)
                
                loss = agent.replay(batch_size)
                loss_list.append(loss)
                
                print(f"Episode: {episode + 1}/{episodes}, Steps: {time + 1}, "
                      f"Accuracy: {accuracy:.4f}, Loss: {loss:.4f}, "
                      f"Label: {env.decode_label(env.current_label_index)}")
                break
    
    # Plot accuracy and convergence (loss)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(accuracy_list, label="Accuracy")
    plt.title('Accuracy per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss_list, label="Loss")
    plt.title('Convergence (Loss) per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save the trained model
    torch.save(agent.model.state_dict(), 'D:\DaiMa\8.13\src\dqn_model.pth')
    a = 'D:\DaiMa\8.13\src\dqn_model.pth'
    print(f"Model saved to {a}")

    return agent

def load_model(model_path, state_size, action_size):
    agent = DQNAgent(state_size, action_size)
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()
    return agent

def predict_with_model(agent, features, env):
    state = np.reshape(features, [1, env.encoded_features.shape[1]])
    predicted_label_index = agent.act(state)
    return env.decode_label(predicted_label_index)
def enter_symptoms_and_predict(loaded_agent, env):
    print("Enter new symptoms as a comma-separated list of numbers (e.g., '0.5, 0.3, 0.2'):")
    symptoms_input = input("New symptoms: ")
    symptoms = np.array([float(x.strip()) for x in symptoms_input.split(',')])
    predicted_label = predict_with_model(loaded_agent, symptoms, env)
    print(f"Predicted label for entered symptoms: {predicted_label}")

if __name__ == '__main__':
    filepath = 'D:\DaiMa\8.13\dataset\伤寒论诊疗数据集.txt'
    
    features, labels = load_and_preprocess_data(filepath)

    if features is not None and labels is not None:
        train_dataset, test_dataset = create_datasets(features, labels)
        
        train_features = np.array(train_dataset.features)
        train_labels = np.array(train_dataset.labels)
        
        print(f"Shape of train_features: {train_features.shape}")
        print(f"Shape of train_labels: {train_labels.shape}")
        print(f"Number of unique labels: {len(np.unique(train_labels))}")
        
        trained_agent = train_dqn_for_tcm(train_features, train_labels)
        
        env = TCMEnvironment(train_features, train_labels)
        example_features = env.encoded_features[0]
        predicted_label = predict_with_model(trained_agent, example_features, env)
        print(f"Example features: {env.decode_state(example_features)}")
        print(f"Predicted label for example: {predicted_label}")

        # Load model and predict with new data
        model_path = 'D:\DaiMa\8.13\src\dqn_model.pth'
        loaded_agent = load_model(model_path, state_size=train_features.shape[1], action_size=len(np.unique(train_labels)))
        
        # Enter new symptoms and predict
        enter_symptoms_and_predict(loaded_agent, env)
