import pennylane as qml
from pennylane import numpy as np
import random
import autograd.numpy as anp
import math
import copy

class AgentQVC(object):
    def __init__(self, env, qnode, params, alpha=0.1, gamma=0.9, epsilon_max=1.0, epsilon_min=0.2, epsilon_halflife=10, memory_size=100000, memory_sampling=10, update_freq=100):
        self.env = env
        self.learner = qnode
        self.params = np.array(params, requires_grad=True)
        self.actions = [0,1,2,3]
        self.alpha = alpha
        self.gamma = gamma
        self.opt = qml.NesterovMomentumOptimizer(self.alpha)
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_halflife = epsilon_halflife
        self.visits = {}
        self.training = True
        self.memory_size = memory_size
        self.memory_sampling = memory_sampling
        self.memory_table = []
        self.update_freq = 100
        self.counter = 0

    def set_epsilon_max(self, epsilon_max):
        self.epsilon_max=epsilon_max

    def set_epsilon_min(self, epsilon_min):
        self.epsilon_min=epsilon_min

    def set_epsilon_halflife(self, epsilon_halflife):
        self.epsilon_halflife=epsilon_halflife

    def train_mode(self, flag):
        self.training=flag

    def statevector2int(self, state):
        state_int=0
        for i in range(len(state)):
            state_int+=state[i] * (2**(i))
        return state_int

    def check_if_state_exist(self, state):
        state_int=self.statevector2int(state)
        if state_int not in self.visits:
            self.visits[state_int]=0.

    def get_action(self, state):
        self.check_if_state_exist(state)
        if self.training==True and np.random.rand() < self.epsilon_min + (self.epsilon_max-self.epsilon_min)*pow(0.5, self.visits[self.statevector2int(state)]/self.epsilon_halflife):
            target_action = np.random.choice(self.actions)
        else:
            qvalues = self.learner(self.params, state)[:len(self.actions)]
            idx_list=list(range(len(qvalues)))
            random.shuffle(idx_list)
            reordered=qvalues[idx_list]
            target_action = idx_list[np.argmax(reordered)]
        return target_action

    def cost(self, params, experiences):
        count = 0
        total_cost = 0
        for exp in experiences:
            state = exp[0]
            # state = np.array(exp[0].astype(int), requires_grad=False)
            action = exp[1]
            reward = exp[2]
            state_next = exp[3]
            # state_next = np.array(exp[3].astype(int), requires_grad=False)
            terminal = exp[4]

            q_value_predict = self.learner(params, state)[action]
            if terminal == False:
                q_value_real = reward + self.gamma * np.amax(self.learner(params, state_next))
            else:
                q_value_real = reward
            total_cost+=(q_value_real - q_value_predict)**2
            count+=1

            # print(type(total_cost))
            # print(type(self.learner(params, state_next)))
            # print(type(params))
            average_cost = total_cost/count

        return average_cost

    def update_weights(self, experiences):
        for i in range(len(experiences)):
            self.check_if_state_exist(experiences[i][3])
            self.visits[self.statevector2int(experiences[i][0])] += 1
        self.params = self.opt.step(lambda w: self.cost(w, experiences), self.params)

    # def train(self, state):
    #     # Get first action.
    #     action = self.get_action(state)
    #     # Get next state.
    #     state_next, reward, terminal = self.env.step(action)
    #     # Store transition experience in memory
    #     if(self.counter > self.memory_size):
    #         np.delete(self.memory_table, 0)
    #         self.counter-=1
    #     self.memory_table.append(tuple((state, action, reward, state_next, terminal)))
    #     self.counter+=1
    #     if (self.counter%self.update_freq==0):
    #         # Sample random minibatch of transition experiences from memory
    #         experience_idx = np.random.choice(self.counter, size=min(self.counter,self.memory_sampling), replace=False)
    #         experiences = [self.memory_table[i] for i in experience_idx]
    #         experience_idx=np.sort(experience_idx)[::-1]
    #         print(f"experience_idx:{experience_idx}")
    #         print(type(experience_idx))
    #         for i in experience_idx:
    #             del self.memory_table[i]
    #         self.counter-=len(experience_idx)
    #
    #         # Update Q table with past experiences
    #         self.update_weights( experiences )
    #
    #         # Update Q table with the latest experience
    #         self.update_weights( [tuple((state, action, reward, state_next, terminal))] )
    #         self.counter=0
    #     return state_next, reward, terminal

    def train(self, state):
        action = self.get_action(state)
        state_next, reward, terminal = self.env.step(action)

        # Add the new experience
        if len(self.memory_table) >= self.memory_size:
            print(f"Removing oldest experience. Memory size before: {len(self.memory_table)}")
            self.memory_table.pop(0)  # Remove oldest experience

        self.memory_table.append((state, action, reward, state_next, terminal))
        self.counter = len(self.memory_table)

        if self.counter % self.update_freq == 0:
            print(f"Sampling experiences. Memory size: {self.counter}")
            experience_idx = np.random.choice(self.counter, size=min(self.counter, self.memory_sampling), replace=False)
            print(f"Experience indices: {experience_idx}")

            experiences = [self.memory_table[i] for i in experience_idx]
            print(f"Sampled experiences: {experiences}")

            # Remove sampled experiences from memory
            experience_set = set(experience_idx)
            self.memory_table = [self.memory_table[i] for i in range(len(self.memory_table)) if i not in experience_set]
            print(f"Memory size after removal: {len(self.memory_table)}")

            self.counter = len(self.memory_table)

            self.update_weights(experiences)
            self.update_weights([(state, action, reward, state_next, terminal)])

        return state_next, reward, terminal




















# import pennylane as qml
# from pennylane import numpy as np
# import random
#
# class AgentQVC(object):
#     def __init__(self, env, qnode, params, alpha=0.1, gamma=0.9, epsilon_max=1.0, epsilon_min=0.2, epsilon_halflife=10, memory_size=100000, memory_sampling=10, update_freq=100):
#         self.env = env
#         self.learner = qnode
#         self.params = np.array(params, requires_grad=True)
#         self.actions = [0,1,2,3]
#         self.alpha = alpha
#         self.gamma = gamma
#         self.opt = qml.NesterovMomentumOptimizer(self.alpha)
#         self.epsilon_max = epsilon_max
#         self.epsilon_min = epsilon_min
#         self.epsilon_halflife = epsilon_halflife
#         self.visits = {}
#         self.training = True
#         self.memory_size = memory_size
#         self.memory_sampling = memory_sampling
#         self.memory_table = []
#         self.update_freq = 100
#         self.counter = 0
#
#     def get_action(self, state):
#         self.check_if_state_exist(state)
#         if self.training==True and np.random.rand() < self.epsilon_min + (self.epsilon_max-self.epsilon_min)*pow(0.5, self.visits[self.statevector2int(state)]/self.epsilon_halflife):
#             target_action = np.random.choice(self.actions)
#         else:
#             qvalues = self.learner(self.params, state)[:len(self.actions)]
#             idx_list=list(range(len(qvalues)))
#             random.shuffle(idx_list)
#             reordered=qvalues[idx_list]
#             target_action = idx_list[np.argmax(reordered)]
#         return target_action
#
#     def cost(self, params, experiences):
#         count = 0
#         total_cost = 0
#         for exp in experiences:
#             state = exp[0]
#             action = exp[1]
#             reward = exp[2]
#             state_next = exp[3]
#             terminal = exp[4]
#             q_value_predict = self.learner(params, state)[action]
#             if terminal == False:
#                 q_value_real = reward + self.gamma * np.amax(self.learner(params, state_next))
#             else:
#                 q_value_real = reward
#             total_cost+=(q_value_real - q_value_predict)**2
#             count+=1
#             print(type(total_cost))
#             print(type(self.learner(params, state_next)))
#             print(type(params))
#             average_cost = total_cost/count
#
#         return average_cost
#
#     def update_weights(self, experiences):
#         for i in range(len(experiences)):
#             self.check_if_state_exist(experiences[i][3])
#             self.visits[self.statevector2int(experiences[i][0])] += 1
#         self.params = self.opt.step(lambda w: self.cost(w, experiences), self.params)
#
#     def train(self, state):
#         action = self.get_action(state)
#         state_next, reward, terminal = self.env.step(action)
#         if(self.counter > self.memory_size):
#             np.delete(self.memory_table, 0)
#             self.counter-=1
#         self.memory_table.append(tuple((state, action, reward, state_next, terminal)))
#         self.counter+=1
#         if (self.counter%self.update_freq==0):
#             experience_idx = np.random.choice(self.counter, size=min(self.counter,self.memory_sampling), replace=False)
#             experiences = [self.memory_table[i] for i in experience_idx]
#             experience_idx=np.sort(experience_idx)[::-1]
#             print(f"experience_idx:{experience_idx}")
#             print(type(experience_idx))
#             for i in experience_idx:
#                 del self.memory_table[i]
#             self.counter-=len(experience_idx)
#             self.update_weights( experiences )
#             self.update_weights( [tuple((state, action, reward, state_next, terminal))] )
#             self.counter=0
#         return state_next, reward, terminal
# 这段代码报错ValueError: setting an array element with a sequence.为什么，如何修改