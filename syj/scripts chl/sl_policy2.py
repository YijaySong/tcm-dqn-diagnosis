from __future__ import division
from __future__ import print_function
'''import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()'''
import tensorflow as tf
# import numpy as np
from itertools import count
import pennylane as qml
from pennylane import numpy as np
import random
# from collections import namedtuple
from functools import reduce
from utils import *
from env import Env
import pickle

import time

# import tensorflow_quantum as tfq
# import gym, cirq, sympy
# from functools import reduce
# from collections import deque, defaultdict
# import matplotlib.pyplot as plt
# from cirq.contrib.svg import SVGCircuit
tf.get_logger().setLevel('ERROR')


import faulthandler
# 在import之后直接添加以下启用代码即可
faulthandler.enable()


#relation = sys.argv[1]
# relation = "concept_athletehomestadium"
dataset = "YAGO3-10"
relation = "wasBornIn"
# episodes = int(sys.argv[2])
dataPath = '../YAGO3-10/'
graphpath = dataPath + relation.replace("/", "@") + '/' + 'graph.txt'
relationPath = dataPath + relation.replace("/", "@") + '/' + 'train_pos'


num_qubits = 16
num_layers = 3


# Transition = namedtuple('Transition',
# 						('state', 'action', 'reward', 'next_state', 'done'))


## Define a FOUR qubit system
dev = qml.device('default.qubit', wires=num_qubits)

# defining a basic block
def layer(W):
    for j in range(num_qubits - 1):
        qml.CNOT(wires=[j, j + 1])
    for i in range(num_qubits):
        qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)

# defining the quantum circuit, in which the number of layers depends on the number of weights taken

@qml.qnode(dev, interface='tf')
def circuit(weights, state=None):
    qml.templates.embeddings.AmplitudeEmbedding(np.array(state), wires=range(num_qubits), pad_with=0.)

    for W in weights:
        layer(W)
    return [qml.expval(qml.PauliZ(ind)) for ind in range(num_qubits)]


output_layer_weights = tf.Variable(tf.random.normal((num_qubits, action_space), mean=0.0, stddev=0.1, dtype=tf.float64), dtype=tf.float64)

def variational_classifier(var_Q_circuit, var_Q_bias , state):
    """The variational classifier."""

    # Change to SoftMax???
    weights = var_Q_circuit
    # print('-------------------')
    # print(type(var_Q_bias))
    # print(type(tf.Variable(circuit(weights, state = state))))
    var_Q_bias = tf.cast(var_Q_bias, tf.float64)
    raw_output = tf.Variable(circuit(weights, state = state)) + var_Q_bias
    # breakpoint()
    # We are approximating Q Value
    # Maybe softmax is no need
    # softMaxOutPut = np.exp(raw_output) / np.exp(raw_output).sum()

    # output_layer_weights = tf.Variable(tf.random.normal((num_qubits, action_space), mean=0.0, stddev=0.1, dtype=tf.float64), dtype=tf.float64)
    # logits = tf.matmul(tf.expand_dims(raw_output, axis=0), output_layer_weights)  # 将原始输出转为 logits
    probabilities = tf.nn.softmax(raw_output[:action_space])  # 应用 Softmax 转换为概率
    # # breakpoint()
    # probabilities = tf.squeeze(probabilities)
    # print('-------------------')
    # print(probabilities)

    return probabilities



#@tf.function
def update(state, action, opt, var_Q_circuit, var_Q_bias):
    state = tf.convert_to_tensor(state)
    action = tf.convert_to_tensor(action)
    #breakpoint()
    # print(state)
    # print(action)


    # action_prob的维度有问题，按理说应该是action的数目
    with tf.GradientTape() as tape:
        action_prob = [variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, state = state[i])for i in range(len(state))]
        # print('------------------------------')
        # print(action_prob)

        # breakpoint()
        p_actions = tf.gather_nd(action_prob, action)
        # breakpoint()
        log_probs = tf.math.log(p_actions)
        loss = tf.math.reduce_sum(-log_probs)

        # loss = tf.abs(circuit4(phi, theta) - 0.5)**2
        # loss = cost(var_Q_circuit, var_Q_bias, batch_sampled, Q_target)

    # gradients = tape.gradient(loss, [phi, theta])
    gradients = tape.gradient(loss, [var_Q_circuit, var_Q_bias])
    # opt.apply_gradients(zip(gradients, [phi, theta]))
    opt.apply_gradients(zip(gradients, [var_Q_circuit, var_Q_bias]))

    return loss

#############################


var_init_circuit = tf.Variable(0.01 * np.random.randn(num_layers, num_qubits, 3).astype(np.float64), dtype=tf.dtypes.float64, trainable=True)
var_init_bias = tf.Variable(np.zeros(num_qubits, dtype=np.float64), dtype=tf.dtypes.float64, trainable=True)
# breakpoint()
var_Q_circuit = var_init_circuit
var_Q_bias = var_init_bias

# var_target_Q_circuit = var_Q_circuit.clone().detach()
# var_target_Q_bias = var_Q_bias.clone().detach()
# opt = tf.keras.optimizers.RMSprop([var_Q_circuit, var_Q_bias], lr=0.01)
opt = tf.keras.optimizers.Adam([var_Q_circuit, var_Q_bias], lr=0.001, amsgrad=True)



def train():
    #policy_nn_model = SupervisedPolicy(learning_rate=0.001)
    with open(relationPath) as f:  # Replace with actual path
        train_data = f.readlines()

    num_samples = len(train_data)
    if num_samples > 500:
        num_samples = 500

    for episode in range(num_samples):
        print(f"Episode {episode}")
        print(f'Training Sample: {train_data[episode % num_samples][:-1]}')
        total_reward = 0

        env = Env(dataPath, train_data[episode % num_samples])  # Replace with actual parameters
        sample = train_data[episode % num_samples].split()

        # breakpoint()
        try:
            good_episodes = teacher(sample[0], sample[1], 5, env, graphpath)  # Replace with actual parameters
        except Exception as e:
            print('Cannot find a path')
            continue

        for item in good_episodes:
            state_batch = []
            action_batch = []
            for t, transition in enumerate(item):
                state_batch.append(transition.state)
                action_batch.append(transition.action)
            state_batch = np.squeeze(state_batch)
            state_batch = np.reshape(state_batch, [-1, state_dim])
            action_batch = np.array(action_batch)
            #action_batch = np.reshape(action_batch, [-1, action_space])
            # print(state_batch)
            # print(action_batch)

            # 创建全零数组
            action_result = np.zeros((len(action_batch), 2), dtype=int)
            # 设置标记
            for num in range(len(action_batch)):
                action_result[num, 1] = action_batch[num]
            # 设置第一列为索引
            action_result[:, 0] = np.arange(len(action_batch))

            # breakpoint()
            # new_shape = (-1, 20)  # -1 自动计算维度
            # state_batch = tf.reshape(state_batch, new_shape)
            # print(action_result)
            # model(state_batch)
            loss = update(state_batch, action_result, opt, var_Q_circuit, var_Q_bias)
            print(f'Loss: {loss.numpy()}')

    # model.save_weights('models/policy_supervised_' + relation)

    # breakpoint()
    model_data = {
        'weights': var_Q_circuit,
        'biases': var_Q_bias,
        # 'circuit': circuit
    }
    with open('models/'+'quantum_model_'+dataset+'_'+relation.replace("/", "@")+'.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print('Model saved')



def test(test_episodes):
    with open(relationPath) as f:  # Replace with actual path
        test_data = f.readlines()

    test_num = len(test_data)

    test_data = test_data[-test_episodes:]
    print(len(test_data))

    success = 0

    with open('models/'+'quantum_model_'+dataset+'_'+relation.replace("/", "@")+'.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    loaded_weights = loaded_model['weights']
    loaded_biases = loaded_model['biases']
    print('Model reloaded')
    for episode in range(len(test_data)):
        print('Test sample %d: %s' % (episode,test_data[episode][:-1]))
        env = Env(dataPath, test_data[episode])
        sample = test_data[episode].split()
        state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]
        for t in count():
            state_vec = env.idx_state(state_idx)
            # action_probs = policy_nn.predict(state_vec)
            action_probs = [variational_classifier(var_Q_circuit = loaded_weights, var_Q_bias = loaded_biases, state = state_vec[i])for i in range(len(state_vec))]

            action_chosen = np.random.choice(np.arange(action_space), p = np.squeeze(action_probs))
            reward, new_state, done = env.interact(state_idx, action_chosen)
            if done or t == max_steps_test:
                if done:
                    print('Success')
                    success += 1
                print('Episode ends\n')
                break
            state_idx = new_state

    print('Success persentage:', success/test_episodes)


if __name__ == "__main__":
    train()
    # test前记得先去服务器下载quantum_model.pkl更新
    test(50)

