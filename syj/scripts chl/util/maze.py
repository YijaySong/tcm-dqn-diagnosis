import pennylane as qml
from pennylane import numpy as np
import random
import math
import copy

class Maze(object):
    def __init__(self, layout, reward):
        self.reward_init = reward
        self.reward = reward
        self.layout = layout
        self.max_i = np.shape(layout)[0]-1
        self.max_j = np.shape(layout)[1]-1
        self.state = [self.max_i,self.max_j]

    def action_space(self):
        return range(4)

    def coordinates2statevector(self, state):
        i_coordinate=state[0]
        j_coordinate=state[1]
        numQubits_i=int(math.log(self.max_i+1,2))
        numQubits_j=int(math.log(self.max_j+1,2))
        state_vector=np.array(np.zeros(numQubits_i+numQubits_j), requires_grad=False)
        for i in range(numQubits_i):
            if(i_coordinate & 1 == 1):
                state_vector[numQubits_i-1-i]=1
            i_coordinate=i_coordinate//2
        for j in range(numQubits_j):
            if(j_coordinate & 1 == 1):
                state_vector[numQubits_i+numQubits_j-1-j]=1
            j_coordinate=j_coordinate//2
        return state_vector

    def check_wall(self, action):
        hit_wall=False
        next_state=copy.copy(self.state)
        if(action==0):
            next_state[0]-=1
            if(next_state[0] < 0):
                hit_wall=True
        elif(action==1):
            next_state[0]+=1
            if(next_state[0] > self.max_i):
                hit_wall=True
        elif(action==2):
            next_state[1]-=1
            if(next_state[1] < 0):
                hit_wall=True
        elif(action==3):
            next_state[1]+=1
            if(next_state[1] > self.max_j):
                hit_wall=True

        if(hit_wall==False and self.layout[next_state[0]][next_state[1]]=='W'):
            hit_wall=True
        return hit_wall

    def step(self, action):
        terminal=False
        reward=-1
        state_next=self.state
        if(self.check_wall(action)==False):
            if(action==0):
                state_next[0] -= 1
            elif(action==1):
                state_next[0] += 1
            elif(action==2):
                state_next[1] -= 1
            elif(action==3):
                state_next[1] += 1
        reward+=self.reward[state_next[0]][state_next[1]]
        if(self.layout[state_next[0]][state_next[1]]=='r'):
            self.reward[state_next[0]][state_next[1]]=0
        if(self.layout[state_next[0]][state_next[1]]=='R'):
            terminal=True
        next_state=self.coordinates2statevector(state_next)
        return next_state, reward, terminal

    def reset(self):
        self.reward = self.reward_init
        self.state = [self.max_i,self.max_j]
        return self.coordinates2statevector(self.state)
