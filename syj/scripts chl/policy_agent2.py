from __future__ import division
from __future__ import print_function
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
import collections
from itertools import count
from sklearn.metrics.pairwise import cosine_similarity
import time
import sys
'''import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()'''
import tensorflow as tf
from itertools import count
import pennylane as qml
from pennylane import numpy as np
from collections import namedtuple

from utils import *
from env import Env
import pickle

# 此处设置任务
task = ""
dataset = "YAGO3-10"
relation = "wasBornIn"
dataPath = '../YAGO3-10/'

# episodes = int(sys.argv[2])
graphpath = dataPath + relation.replace("/", "@") + '/' + 'graph.txt'
relationPath = dataPath + relation.replace("/", "@") + '/' + 'train_pos'


num_qubits = 16
num_layers = 3

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
    var_Q_bias = tf.cast(var_Q_bias, tf.float64)
    raw_output = tf.Variable(circuit(weights, state = state)) + var_Q_bias
    probabilities = tf.nn.softmax(raw_output[:action_space])  # 应用 Softmax 转换为概率
    return probabilities

#@tf.function
def update(state, target, action, opt, var_Q_circuit, var_Q_bias):
    state = tf.convert_to_tensor(state)

    # 创建全零数组
    action_result = np.zeros((len(action), 2), dtype=int)
    # 设置标记
    for num in range(len(action)):
        action_result[num, 1] = action[num]
    # 设置第一列为索引
    action_result[:, 0] = np.arange(len(action))
    action = tf.convert_to_tensor(action_result)
    with tf.GradientTape() as tape:
        action_prob = [variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, state = state[i])for i in range(len(state))]

        p_actions = tf.gather_nd(action_prob, action)
        # breakpoint()
        log_probs = tf.math.log(p_actions)
        loss = tf.math.reduce_sum(-log_probs*target)

    gradients = tape.gradient(loss, [var_Q_circuit, var_Q_bias])
    # opt.apply_gradients(zip(gradients, [phi, theta]))
    opt.apply_gradients(zip(gradients, [var_Q_circuit, var_Q_bias]))

    return loss

#############################


def REINFORCE(training_pairs, num_episodes):
    train = training_pairs

    success = 0

    # path_found = set()
    path_found_entity = []
    path_relation_found = []

    for i_episode in range(num_episodes):
        start = time.time()
        print('Episode %d' % i_episode)
        print('Training sample: ', train[i_episode][:-1])

        env = Env(dataPath, train[i_episode])

        sample = train[i_episode].split()
        state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]

        episode = []
        state_batch_negative = []
        action_batch_negative = []
        for t in count():
            state_vec = env.idx_state(state_idx)
            action_probs = [variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, state = state_vec[i])for i in range(len(state_vec))]

            action_chosen = np.random.choice(np.arange(action_space), p = np.squeeze(action_probs))
            reward, new_state, done = env.interact(state_idx, action_chosen)

            if reward == -1: # the action fails for this step
                state_batch_negative.append(state_vec)
                action_batch_negative.append(action_chosen)

            new_state_vec = env.idx_state(new_state)
            episode.append(Transition(state = state_vec, action = action_chosen, next_state = new_state_vec, reward = reward))

            if done or t == max_steps:
                break

            state_idx = new_state

        # Discourage the agent when it choose an invalid step
        if len(state_batch_negative) != 0:
            print('Penalty to invalid steps:', len(state_batch_negative))
            loss = update(np.reshape(state_batch_negative, (-1, state_dim)), -0.05, action_batch_negative, opt, var_Q_circuit, var_Q_bias)

        print('----- FINAL PATH -----')
        print('\t'.join(env.path))
        print('PATH LENGTH', len(env.path))
        print('----- FINAL PATH -----')

        # If the agent success, do one optimization
        if done == 1:
            print('Success')

            path_found_entity.append(path_clean(' -> '.join(env.path)))

            success += 1
            path_length = len(env.path)
            length_reward = 1/path_length
            global_reward = 1

            total_reward = 0.1*global_reward + 0.9*length_reward
            state_batch = []
            action_batch = []
            for t, transition in enumerate(episode):
                if transition.reward == 0:
                    state_batch.append(transition.state)
                    action_batch.append(transition.action)

            update(np.reshape(state_batch,(-1,state_dim)), total_reward, action_batch, opt, var_Q_circuit, var_Q_bias)
        else:
            global_reward = -0.05
            # length_reward = 1/len(env.path)

            state_batch = []
            action_batch = []
            total_reward = global_reward
            for t, transition in enumerate(episode):
                if transition.reward == 0:
                    state_batch.append(transition.state)
                    action_batch.append(transition.action)

            if len(state_batch) != 0:
                update(np.reshape(state_batch,(-1,state_dim)), total_reward, action_batch, opt, var_Q_circuit, var_Q_bias)

            print('Failed, Do one teacher guideline')
            try:
                good_episodes = teacher(sample[0], sample[1], 1, env, graphpath)
                for item in good_episodes:
                    teacher_state_batch = []
                    teacher_action_batch = []
                    total_reward = 0.0*1 + 1*1/len(item)
                    for t, transition in enumerate(item):
                        teacher_state_batch.append(transition.state)
                        teacher_action_batch.append(transition.action)

                    update(np.squeeze(teacher_state_batch), 1, teacher_action_batch, opt, var_Q_circuit, var_Q_bias)
            except Exception as e:
                print('Teacher guideline failed')
        print('Episode time: ', time.time() - start)
        print('\n')
    print('Success percentage:', success/num_episodes)

    for path in path_found_entity:
        rel_ent = path.split(' -> ')
        path_relation = []
        for idx, item in enumerate(rel_ent):
            if idx%2 == 0:
                path_relation.append(item)
        path_relation_found.append(' -> '.join(path_relation))

    relation_path_stats = collections.Counter(path_relation_found).items()
    relation_path_stats = sorted(relation_path_stats, key = lambda x:x[1], reverse=True)

    f = open(dataPath + relation.replace("/", "@") + '/' + 'path_stats.txt', 'w')
    for item in relation_path_stats:
        f.write(item[0]+'\t'+str(item[1])+'\n')
    f.close()
    print('Path stats saved')
    return


var_init_circuit = tf.Variable(0.01 * np.random.randn(num_layers, num_qubits, 3).astype(np.float64), dtype=tf.dtypes.float64, trainable=True)
var_init_bias = tf.Variable(np.zeros(num_qubits, dtype=np.float64), dtype=tf.dtypes.float64, trainable=True)
var_Q_circuit = var_init_circuit
var_Q_bias = var_init_bias

opt = tf.keras.optimizers.Adam([var_Q_circuit, var_Q_bias], lr=0.001, amsgrad=True)


def retrain():
    print('Start retraining')
    with open(relationPath) as f:  # Replace with actual path
        training_pairs = f.readlines()

    with open('models/' + 'quantum_model_' + dataset+'_' + relation.replace("/", "@") +'.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    var_Q_circuit = loaded_model['weights']
    var_Q_bias = loaded_model['biases']
    print('Model reloaded')

    episodes = len(training_pairs)
    if episodes > 300:
        episodes = 300
    REINFORCE(training_pairs, episodes)

    model_data = {
        'weights': var_Q_circuit,
        'biases': var_Q_bias,
    }
    with open('models/' + 'quantum_model_' + dataset+'_' + relation.replace("/", "@") +'_retrained.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print('Retrained model saved')


def test():
    with open(relationPath) as f:  # Replace with actual path
        all_data = f.readlines()

    test_data = all_data
    test_num = len(test_data)
    success = 0
    path_found = []
    path_relation_found = []
    path_set = set()

    with open('models/' + 'quantum_model_' + dataset+'_' + relation.replace("/", "@") + '_retrained.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    var_Q_circuit = loaded_model['weights']
    var_Q_bias = loaded_model['biases']
    print('Model reloaded')

    if test_num > 500:
        test_num = 500

    for episode in range(test_num):
        print('Test sample %d: %s' % (episode,test_data[episode][:-1]))
        env = Env(dataPath, test_data[episode])
        sample = test_data[episode].split()
        state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]

        transitions = []

        for t in count():
            state_vec = env.idx_state(state_idx)
            action_probs = [variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, state = state_vec[i])for i in range(len(state_vec))]
            action_probs = np.squeeze(action_probs)

            action_chosen = np.random.choice(np.arange(action_space), p = action_probs)
            reward, new_state, done = env.interact(state_idx, action_chosen)
            new_state_vec = env.idx_state(new_state)
            transitions.append(Transition(state = state_vec, action = action_chosen, next_state = new_state_vec, reward = reward))

            if done or t == max_steps_test:
                if done:
                    success += 1
                    print("Success\n")
                    path = path_clean(' -> '.join(env.path))
                    path_found.append(path)
                else:
                    print('Episode ends due to step limit\n')
                break
            state_idx = new_state

        if done:
            if len(path_set) != 0:
                path_found_embedding = [env.path_embedding(path.split(' -> ')) for path in path_set]
                curr_path_embedding = env.path_embedding(env.path_relations)
                path_found_embedding = np.reshape(path_found_embedding, (-1,embedding_dim))
                cos_sim = cosine_similarity(path_found_embedding, curr_path_embedding)
                diverse_reward = -np.mean(cos_sim)
                print('diverse_reward', diverse_reward)
                #total_reward = 0.1*global_reward + 0.8*length_reward + 0.1*diverse_reward
                state_batch = []
                action_batch = []
                for t, transition in enumerate(transitions):
                    if transition.reward == 0:
                        state_batch.append(transition.state)
                        action_batch.append(transition.action)

                update(np.reshape(state_batch,(-1,state_dim)), 0.1*diverse_reward, action_batch, opt, var_Q_circuit, var_Q_bias)
            path_set.add(' -> '.join(env.path_relations))


    for path in path_found:
        rel_ent = path.split(' -> ')
        path_relation = []
        for idx, item in enumerate(rel_ent):
            if idx%2 == 0:
                path_relation.append(item)
        path_relation_found.append(' -> '.join(path_relation))

    # path_stats = collections.Counter(path_found).items()
    relation_path_stats = collections.Counter(path_relation_found).items()
    relation_path_stats = sorted(relation_path_stats, key = lambda x:x[1], reverse=True)

    ranking_path = []
    for item in relation_path_stats:
        path = item[0]
        length = len(path.split(' -> '))
        ranking_path.append((path, length))

    ranking_path = sorted(ranking_path, key = lambda x:x[1])
    print('Success persentage:', success/test_num)

    f = open(dataPath + relation.replace("/", "@") + '/' + 'path_to_use.txt', 'w')
    for item in ranking_path:
        f.write(item[0] + '\n')
    f.close()
    print('path to use saved')
    return

if __name__ == "__main__":
	if task == 'test':
		test()
	elif task == 'retrain':
		retrain()
	else:
		retrain()
		test()
	# retrain()	



