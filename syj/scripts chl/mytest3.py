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
from collections import namedtuple
from functools import reduce
from utils import *
from env import Env

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
relation = "concept_athletehomestadium"
# episodes = int(sys.argv[2])
graphpath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'


num_qubits = 4
num_layers = 2


Transition = namedtuple('Transition',
						('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def output_all(self):
		return self.memory

	def __len__(self):
		return len(self.memory)


def decimalToBinaryFixLength(_length, _decimal):
	binNum = bin(int(_decimal))[2:]
	outputNum = [int(item) for item in binNum]
	if len(outputNum) < _length:
		outputNum = np.concatenate((np.zeros((_length-len(outputNum),)),np.array(outputNum)))
	else:
		outputNum = np.array(outputNum)
	return outputNum


# dtype = torch.DoubleTensor

## Define a FOUR qubit system
dev = qml.device('default.qubit', wires=num_qubits)
def statepreparation(a):

    """Quantum circuit to encode a the input vector into variational params

    Args:
        a: feature vector of rad and rad_square => np.array([rad_X_0, rad_X_1, rad_square_X_0, rad_square_X_1])
    """

    # Rot to computational basis encoding
    # a = [a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8]

    for ind in range(len(a)):
        qml.RX(np.pi * a[ind], wires=ind)
        qml.RZ(np.pi * a[ind], wires=ind)


# defining a basic block
def layer(W):
    for j in range(num_qubits - 1):
        qml.CNOT(wires=[j, j + 1])
    for i in range(num_qubits):
        qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)

# defining the quantum circuit, in which the number of layers depends on the number of weights taken
@qml.qnode(dev, interface='tf')
def circuit(weights, state=None):
    if((state==np.ones(num_qubits)).all()):
        for i in range(num_qubits):
            qml.PauliX(i)
    elif((state==np.zeros(num_qubits)).all()):
        pass
    else:
        qml.templates.embeddings.AmplitudeEmbedding(np.array(state, requires_grad=False), wires=range(num_qubits))
    for W in weights:
        layer(W)
    return qml.expval(qml.PauliZ(0)),qml.expval(qml.PauliZ(1)),qml.expval(qml.PauliZ(2)),qml.expval(qml.PauliZ(3))



def variational_classifier(var_Q_circuit, var_Q_bias , angles=None):
    """The variational classifier."""

    # Change to SoftMax???
    weights = var_Q_circuit

    raw_output = tf.Variable(circuit(weights, angles=angles)) + var_Q_bias
    # We are approximating Q Value
    # Maybe softmax is no need
    # softMaxOutPut = np.exp(raw_output) / np.exp(raw_output).sum()

    return raw_output



#@tf.function
def update(state, action, opt, var_Q_circuit, var_Q_bias):
    state = tf.convert_to_tensor(state)
#
#
#     '''# 变形，例如将 state 变形为形状 (batch_size, new_shape)
#     new_shape = (20, -1)  # -1 自动计算维度
#     reshaped_state = tf.reshape(state, new_shape)'''
#
#
    action = tf.convert_to_tensor(action)
    #breakpoint()
    # print(state)
    # print(action)

    with tf.GradientTape() as tape:
        action_prob = [variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles=decimalToBinaryFixLength(4, state))[action] for action in batch_sampled]
        p_actions = tf.gather_nd(action_prob, action)

        log_probs = tf.math.log(p_actions)
        loss = tf.math.reduce_sum(-log_probs)

        # loss = tf.abs(circuit4(phi, theta) - 0.5)**2
        # loss = cost(var_Q_circuit, var_Q_bias, batch_sampled, Q_target)

    # gradients = tape.gradient(loss, [phi, theta])
    gradients = tape.gradient(loss, [var_Q_circuit, var_Q_bias])
    # opt.apply_gradients(zip(gradients, [phi, theta]))
    opt.apply_gradients(zip(gradients, [var_Q_circuit, var_Q_bias]))

    return loss



# def square_loss(labels, predictions):
#     """ Square loss function
#
#     Args:
#         labels (array[float]): 1-d array of labels
#         predictions (array[float]): 1-d array of predictions
#     Returns:
#         float: square loss
#     """
#     loss = 0
#     for l, p in zip(labels, predictions):
#         loss = loss + (l - p) ** 2
#     loss = loss / len(labels)
#     return loss
#
# def cost(var_Q_circuit, var_Q_bias, features, labels):
#     """Cost (error) function to be minimized."""
#
#     # predictions = [variational_classifier(weights, angles=f) for f in features]
#     # Torch data type??
#
#     predictions = [variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles=decimalToBinaryFixLength(4,item.state))[item.action] for item in features]
#
#
#     return square_loss(labels, predictions)


#############################

def epsilon_greedy(var_Q_circuit, var_Q_bias, epsilon, n_actions, s, train=False):
    """
    @param Q Q values state x action -> value
    @param epsilon for exploration
    @param s number of states
    @param train if true then no random actions selected
    """

    # Modify to incorporate with Variational Quantum Classifier
    # epsilon should change along training
    # In the beginning => More Exploration
    # In the end => More Exploitation

    # More Random
    #np.random.seed(int(datetime.now().strftime("%S%f")))


    if train or np.random.rand() < ((epsilon/n_actions)+(1-epsilon)):
        action = tf.argmax(variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles = decimalToBinaryFixLength(4,s)))
    else:
        # need to be torch tensor
        action = tf.Variable(np.random.randint(0, n_actions))
    return action



def deep_Q_Learning(state, action, alpha, gamma, epsilon, episodes, max_steps, n_tests, render = False, test=False):
    """
    @param alpha learning rate
    @param gamma decay factor
    @param epsilon for exploration
    @param max_steps for max step in each episode
    @param n_tests number of test episodes
    """

    # env = gym.make('Deterministic-ShortestPath-4x4-FrozenLake-v0')
    # # env = gym.make('Deterministic-4x4-FrozenLake-v0')
    # n_states, n_actions = env.observation_space.n, env.action_space.n
    # print("NUMBER OF STATES:" + str(n_states))
    # print("NUMBER OF ACTIONS:" + str(n_actions))

    # Initialize Q function approximator variational quantum circuit
    # initialize weight layers

    # var_init = (0.01 * np.random.randn(num_layers, num_qubits, 3), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    var_init_circuit = tf.Variable(0.01 * np.random.randn(num_layers, num_qubits, 3).astype(np.float32), dtype=tf.dtypes.float32, trainable=True)
    var_init_bias = tf.Variable(np.zeros(4, dtype=np.float32), dtype=tf.dtypes.float32, trainable=True)

    # Define the two Q value function initial parameters
    # Use np copy() function to DEEP COPY the numpy array
    var_Q_circuit = var_init_circuit
    var_Q_bias = var_init_bias
    # print("INIT PARAMS")
    # print(var_Q_circuit)

    var_target_Q_circuit = var_Q_circuit.clone().detach()
    var_target_Q_bias = var_Q_bias.clone().detach()

    ##########################
    # Optimization method => random select train batch from replay memory
    # and opt

    # opt = NesterovMomentumOptimizer(0.01)

    # opt = torch.optim.Adam([var_Q_circuit, var_Q_bias], lr = 0.1)
    # opt = torch.optim.SGD([var_Q_circuit, var_Q_bias], lr=0.1, momentum=0.9)
    opt = tf.keras.optimizers.RMSprop([var_Q_circuit, var_Q_bias], lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

    ## NEed to move out of the function
    TARGET_UPDATE = 20
    batch_size = 5
    OPTIMIZE_STEPS = 5
    ##


    target_update_counter = 0

    iter_index = []
    iter_reward = []
    iter_total_steps = []

    cost_list = []


    timestep_reward = []


    # Demo of generating a ACTION
    # Output a numpy array of value for each action

    # Define the replay memory
    # Each transition:
    # (s_t_0, a_t_0, r_t, s_t_1, 'DONE')

    memory = ReplayMemory(80)

    # train the variational classifier


    for episode in range(episodes):
        print(f"Episode: {episode}")
        # Output a s in decimal format
        s = env.reset()
        # Doing epsilog greedy action selection
        # With var_Q
        a = epsilon_greedy(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, epsilon = epsilon, n_actions = n_actions, s = s).item()
        t = 0
        total_reward = 0
        done = False


        while t < max_steps:
            if render:
                print("###RENDER###")
                env.render()
                print("###RENDER###")
            t += 1

            target_update_counter += 1

            # Execute the action
            s_, reward, done, info = env.step(a)
            total_reward += reward
            # a_ = np.argmax(Q[s_, :])
            a_ = epsilon_greedy(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, epsilon = epsilon, n_actions = n_actions, s = s_).item()
            memory.push(s, a, reward, s_, done)

            if len(memory) > batch_size:

                # Sampling Mini_Batch from Replay Memory

                batch_sampled = memory.sample(batch_size = batch_size)

                # Transition = (s_t, a_t, r_t, s_t+1, done(True / False))

                # item.state => state
                # item.action => action taken at state s
                # item.reward => reward given based on (s,a)
                # item.next_state => state arrived based on (s,a)

                Q_target = [item.reward + (1 - int(item.done)) * gamma * tf.keras.max(variational_classifier(var_Q_circuit = var_target_Q_circuit, var_Q_bias = var_target_Q_bias, angles=decimalToBinaryFixLength(4,item.next_state))) for item in batch_sampled]
                # Q_prediction = [variational_classifier(var_Q, angles=decimalToBinaryFixLength(9,item.state))[item.action] for item in batch_sampled ]

                # Gradient Descent
                # cost(weights, features, labels)
                # square_loss_training = square_loss(labels = Q_target, Q_predictions)
                # print("UPDATING PARAMS...")



                # # CHANGE TO TORCH OPTIMIZER
                # def closure():
                #     opt.zero_grad()
                #     loss = cost(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, features = batch_sampled, labels = Q_target)
                #     # print(loss)
                #     loss.backward()
                #     print(type(loss))
                #     return loss
                # opt.step(closure)
                loss = update(state, action, opt, var_Q_circuit, var_Q_bias, batch_sampled, Q_target)
                # with tf.GradientTape() as tape:
                #     # loss = tf.abs(circuit4(phi, theta) - 0.5)**2
                #     loss = cost(var_Q_circuit, var_Q_bias, batch_sampled, Q_target)
                #
                # # gradients = tape.gradient(loss, [phi, theta])
                # gradients = tape.gradient(loss, [var_Q_circuit, var_Q_bias, batch_sampled, Q_target])
                # # opt.apply_gradients(zip(gradients, [phi, theta]))
                # opt.apply_gradients(zip(gradients, [var_Q_circuit, var_Q_bias, batch_sampled, Q_target]))


                # print("UPDATING PARAMS COMPLETED")
                current_replay_memory = memory.output_all()
            if target_update_counter > TARGET_UPDATE:
                print("UPDATEING TARGET CIRCUIT...")

                var_target_Q_circuit = var_Q_circuit.clone().detach()
                var_target_Q_bias = var_Q_bias.clone().detach()

                target_update_counter = 0

            s, a = s_, a_

            if done:
                if render:
                    print("###FINAL RENDER###")
                    env.render()
                    print("###FINAL RENDER###")
                    print(f"This episode took {t} timesteps and reward: {total_reward}")
                epsilon = epsilon / ((episode/100) + 1)
                # print("Q Circuit Params:")
                # print(var_Q_circuit)
                print(f"This episode took {t} timesteps and reward: {total_reward}")
                timestep_reward.append(total_reward)
                iter_index.append(episode)
                iter_reward.append(total_reward)
                iter_total_steps.append(t)
                break
    # if render:
    # 	print(f"Here are the Q values:\n{Q}\nTesting now:")
    # if test:
    # 	test_agent(Q, env, n_tests, n_actions)
    # return loss, timestep_reward, iter_index, iter_reward, iter_total_steps, var_Q_circuit, var_Q_bias
    return loss





# def one_qubit_rotation(qubit, symbols):
#     """
#     Returns Cirq gates that apply a rotation of the bloch sphere about the X,
#     Y and Z axis, specified by the values in `symbols`.
#     """
#     return [cirq.rx(symbols[0])(qubit),
#             cirq.ry(symbols[1])(qubit),
#             cirq.rz(symbols[2])(qubit)]
#
#
# def entangling_layer(qubits):
#     """
#     Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
#     """
#     cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
#     cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
#     return cz_ops
#
#
# def generate_circuit(qubits, n_layers):
#     """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
#     # Number of qubits
#     n_qubits = len(qubits)
#
#     # Sympy symbols for variational angles
#     params = sympy.symbols(f'theta(0:{3*(n_layers+1)*n_qubits})')
#     params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))
#
#     # Sympy symbols for encoding angles
#     inputs = sympy.symbols(f'x(0:{n_layers})'+f'_(0:{n_qubits})')
#     inputs = np.asarray(inputs).reshape((n_layers, n_qubits))
#
#     # Define circuit
#     circuit = cirq.Circuit()
#     for l in range(n_layers):
#         # Variational layer
#         circuit += cirq.Circuit(one_qubit_rotation(q, params[l, i]) for i, q in enumerate(qubits))
#         circuit += entangling_layer(qubits)
#         # Encoding layer
#         circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(qubits))
#
#     # Last varitional layer
#     circuit += cirq.Circuit(one_qubit_rotation(q, params[n_layers, i]) for i,q in enumerate(qubits))
#
#     return circuit, list(params.flat), list(inputs.flat)
#
#
#
# # 1.2 ReUploadingPQC layer using ControlledPQC
# class ReUploadingPQC(tf.keras.layers.Layer):
#     """
#     Performs the transformation (s_1, ..., s_d) -> (theta_1, ..., theta_N, lmbd[1][1]s_1, ..., lmbd[1][M]s_1,
#         ......., lmbd[d][1]s_d, ..., lmbd[d][M]s_d) for d=input_dim, N=theta_dim and M=n_layers.
#     An activation function from tf.keras.activations, specified by `activation` ('linear' by default) is
#         then applied to all lmbd[i][j]s_i.
#     All angles are finally permuted to follow the alphabetical order of their symbol names, as processed
#         by the ControlledPQC.
#     """
#
#     def __init__(self, qubits, n_layers, observables, activation="linear", name="re-uploading_PQC"):
#         super(ReUploadingPQC, self).__init__(name=name)
#         self.n_layers = n_layers
#         self.n_qubits = len(qubits)
#
#         circuit, theta_symbols, input_symbols = generate_circuit(qubits, n_layers)
#
#         theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
#         self.theta = tf.Variable(
#             initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
#             trainable=True, name="thetas"
#         )
#
#         lmbd_init = tf.ones(shape=(self.n_qubits * self.n_layers,))
#         self.lmbd = tf.Variable(
#             initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas"
#         )
#
#         # Define explicit symbol order.
#         symbols = [str(symb) for symb in theta_symbols + input_symbols]
#         self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])
#
#         self.activation = activation
#         self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
#         self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)
#
#     def call(self, inputs):
#         # inputs[0] = encoding data for the state.
#         batch_dim = tf.gather(tf.shape(inputs[0]), 0)
#         tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
#         tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
#         tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
#         scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
#         squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)
#
#         joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
#         joined_vars = tf.gather(joined_vars, self.indices, axis=1)
#
#         return self.computation_layer([tiled_up_circuits, joined_vars])
#
#
# # 2. Policy-gradient RL with PQC policies
# class Alternating(tf.keras.layers.Layer):
#     def __init__(self, output_dim):
#         super(Alternating, self).__init__()
#         self.w = tf.Variable(
#             initial_value=tf.constant([[(-1.)**i for i in range(output_dim)]]), dtype="float32",
#             trainable=True, name="obs-weights")
#
#     def call(self, inputs):
#         return tf.matmul(inputs, self.w)
#
#
#
#
#
#
# n_qubits = 20 # Dimension of the state vectors in CartPole
# n_layers = 5 # Number of layers in the PQC
# n_actions = 400 # Number of actions in CartPole
#
# qubits = cirq.GridQubit.rect(1, n_qubits)
#
# ops = [cirq.Z(q) for q in qubits]
# observables = [reduce((lambda x, y: x * y), ops)] # Z_0*Z_1*Z_2*Z_3
#
#
# def generate_model_policy(qubits, n_layers, n_actions, beta, observables):
#     """Generates a Keras model for a data re-uploading PQC policy."""
#     # 输出是action的概率
#     input_tensor = tf.keras.Input(shape=(len(qubits), ), dtype=tf.dtypes.float32, name='input')
#     re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables)([input_tensor])
#     process = tf.keras.Sequential([
#         Alternating(n_actions),
#         tf.keras.layers.Lambda(lambda x: x * beta),
#         tf.keras.layers.Softmax()
#     ], name="observables-policy")
#     policy = process(re_uploading_pqc)
#     model = tf.keras.Model(inputs=[input_tensor], outputs=policy)
#
#     return model
#
#
# model = generate_model_policy(qubits, n_layers, n_actions, 1.0, observables)



'''class SupervisedPolicy(tf.keras.Model):
    """docstring for SupervisedPolicy"""
    def __init__(self, learning_rate = 0.001):
        self.initializer = tf.keras.initializers.glorot_normal()
        with tf.variable_scope('supervised_policy'):
            self.state = tf.placeholder(tf.float32, [None, state_dim], name = 'state')
            self.action = tf.placeholder(tf.int32, [None], name = 'action')
            # self.action_prob = policy_nn(self.state, state_dim, action_space, self.initializer)
            self.action_prob = policy_nn(self.state)

            action_mask = tf.cast(tf.one_hot(self.action, depth = action_space), tf.bool)
            self.picked_action_prob = tf.boolean_mask(self.action_prob, action_mask)

            self.loss = tf.reduce_sum(-tf.log(self.picked_action_prob)) + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope = 'supervised_policy'))
            self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess = None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_prob, {self.state: state})

    def update(self, state, action, sess = None):
        sess = sess or tf.get_default_session()
        print('------------------------------------------------')
        _, loss = sess.run([self.train_op, self.loss], {self.state: state, self.action: action})
        return loss




def train():
    policy_nn = SupervisedPolicy()

    f = open(relationPath)
    train_data = f.readlines()
    f.close()

    num_samples = len(train_data)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if num_samples > 500:
            num_samples = 500
        else:
            num_episodes = num_samples

        for episode in range(num_samples):
            print("Episode %d" % episode)
            print('Training Sample:', train_data[episode%num_samples][:-1])

            env = Env(dataPath, train_data[episode%num_samples])
            sample = train_data[episode%num_samples].split()

            try:
                good_episodes = teacher(sample[0], sample[1], 5, env, graphpath)
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
                policy_nn.update(state_batch, action_batch)

        saver.save(sess, 'models/policy_supervised_' + relation)
        print('Model saved')'''


# optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)
# optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
# optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)
#
# # Assign the model parameters to each optimizer
# w_in, w_var, w_out = 1, 0, 2
#
# #@tf.function
# def update(state, action, model):
#     state = tf.convert_to_tensor(state)
#
#
#     '''# 变形，例如将 state 变形为形状 (batch_size, new_shape)
#     new_shape = (20, -1)  # -1 自动计算维度
#     reshaped_state = tf.reshape(state, new_shape)'''
#
#
#     action = tf.convert_to_tensor(action)
#     #breakpoint()
#     print(state)
#     print(action)
#     with tf.GradientTape() as tape:
#         tape.watch(model.trainable_variables)
#         breakpoint()
#         action_prob = model(state)
#
#
#         #breakpoint()
#         print(action_prob)
#         p_actions = tf.gather_nd(action_prob, action)
#
#         log_probs = tf.math.log(p_actions)
#         loss = tf.math.reduce_sum(-log_probs)
#
#         '''action_mask = tf.one_hot(action, depth=action_space)
#         print(action_mask)
#         picked_action_prob = tf.math.reduce_sum(action_prob * action_mask, axis=1)
#         print(picked_action_prob)
#         loss = -tf.math.reduce_sum(tf.math.log(picked_action_prob))
#         print(loss)'''
#
#     grads = tape.gradient(loss, model.trainable_variables)
#
#     for optimizer, w in zip([optimizer_in, optimizer_var, optimizer_out], [w_in, w_var, w_out]):
#         optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])
#         breakpoint()
#     return loss


var_init_circuit = tf.Variable(0.01 * np.random.randn(num_layers, num_qubits, 3).astype(np.float64), dtype=tf.dtypes.float64, trainable=True)
var_init_bias = tf.Variable(np.zeros(4, dtype=np.float64), dtype=tf.dtypes.float64, trainable=True)

var_Q_circuit = var_init_circuit
var_Q_bias = var_init_bias

# var_target_Q_circuit = var_Q_circuit.clone().detach()
# var_target_Q_bias = var_Q_bias.clone().detach()
opt = tf.keras.optimizers.RMSprop([var_Q_circuit, var_Q_bias], lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)



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
            # print(state_batch)
            # model(state_batch)

            loss = update(state_batch, action_result, opt, var_Q_circuit, var_Q_bias)
            print(f'Loss: {loss.numpy()}')

    # model.save_weights('models/policy_supervised_' + relation)
    print('Model saved')



def test(test_episodes):
    tf.reset_default_graph()
    policy_nn = SupervisedPolicy()

    f = open(relationPath)
    test_data = f.readlines()
    f.close()

    test_num = len(test_data)

    test_data = test_data[-test_episodes:]
    print(len(test_data))

    success = 0

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'models/policy_supervised_'+ relation)
        print('Model reloaded')
        for episode in range(len(test_data)):
            print('Test sample %d: %s' % (episode,test_data[episode][:-1]))
            env = Env(dataPath, test_data[episode])
            sample = test_data[episode].split()
            state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]
            for t in count():
                state_vec = env.idx_state(state_idx)
                action_probs = policy_nn.predict(state_vec)
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


'''if __name__ == "__main__":
    train()
    # test(50)'''

train()
