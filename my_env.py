#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 23:06:17 2021

@author: apple
"""

import gym
from gym import spaces
from nbformat import write
import numpy as np
# from gym.utils.single import Reward
import random
import csv
import os
import math


# act = spaces.Box(low=1.0, high=50.0, shape=(1,), dtype=np.int64)
# ace = spaces.Box(low=0, high=320000000, shape=(1,51,4),dtype=np.int64)
# print(ace.sample())

class MyEnvrec(gym.Env):
    def __init__(self):
        # 单用户
        # self.States_file = np.load('/home/slience/Downloads/code/St.npy')
        # 多用户
        self.States_file = np.load(r'G:\doctor\DR_TWO\experiment\morvanzhou\contents\my_experiment\my_k_100set.npy')

        self.States_list = self.States_file
        # self.action_space = OneHotEncoding(size=50) # action space
        # self.action_space = spaces.Box(low=1.0, high=50.0, shape=(1,), dtype=np.int64) # action space

        # 单用户
        # self.action_space = spaces.Discrete(50) # action space
        # self.observation_space = spaces.Box(low=0, high=1, shape=(1,51,3),dtype=np.float64)

        # 多用户
        # self.action_space = spaces.Box(low=0, high=100, shape=(1, 100), dtype=int)  # action space
        # print(self.action_space.sample())
        self.action_space = spaces.Box(low=1, high=100, shape=(8, ), dtype=np.float64)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5000, 8, 2), dtype=np.float64)

        self.turns = 0
        # self.count = 0
        self.state = np.zeros((8, 2))

    def reset(self):
        self.state = np.zeros((8, 2), dtype=np.int64)
        self.turns = 0
        done = False
        print('Starts training')
        return self.state

    def mathfunc(self, lada, deta, cost):
        f = (1 - math.exp(-lada * deta)) / (lada * deta) - (cost / deta)
        return f

    def Reward(self, ob_state, action):
        print(action)
        # r1 = self.mathfunc(ob_state[0][0], action[0], ob_state[0][1]) + 0.8*0.5*self.mathfunc(ob_state[1][0], action[1], ob_state[1][1]) + 0.6*0.5*self.mathfunc(ob_state[5][0], action[5], ob_state[5][1])
        # r2 = self.mathfunc(ob_state[5][0], action[5], ob_state[5][1]) + 0.8*0.5*self.mathfunc(ob_state[2][0], action[2], ob_state[2][1]) + 0.8*0.5*mathfunc(ob_state[7][0], action[7], ob_state[7][1])
        # r3 = self.mathfunc(ob_state[2][0], action[2], ob_state[2][1]) + 0.9*0.5*self.mathfunc(ob_state[4][0], action[4], ob_state[4][1]) + 0.6*0.5*self.mathfunc(ob_state[3][0], action[3], ob_state[3][1]) + 0.6*0.5*self.mathfunc(ob_state[7][0], action[7], ob_state[7][1])
        # r4 = self.mathfunc(ob_state[3][0], action[3], ob_state[3][1]) + 0.8*0.5*self.mathfunc(ob_state[6][0], action[6], ob_state[6][1])
        # r5 = self.mathfunc(ob_state[4][0], action[4], ob_state[4][1]) + 0.9*0.5*self.mathfunc(ob_state[6][0], action[6], ob_state[6][1])
        # r6 = self.mathfunc(ob_state[5][0], action[5], ob_state[5][1]) + 0.8*0.5*self.mathfunc(ob_state[2][0], action[2], ob_state[2][1]) + 0.8*0.5*self.mathfunc(ob_state[6][0], action[6], ob_state[6][1])
        # r7 = self.mathfunc(ob_state[6][0], action[6], ob_state[6][1]) + 0.6*0.5*self.mathfunc(ob_state[3][0], action[3], ob_state[3][1])
        # r8 = self.mathfunc(ob_state[7][0], action[7], ob_state[7][1]) + 0.9*0.5*self.mathfunc(ob_state[0][0], action[0], ob_state[0][1]) + 0.8*0.5*self.mathfunc(ob_state[3][0], action[3], ob_state[3][1])
        r4 = (self.mathfunc(ob_state[3][0], action[3], ob_state[3][1]) + 0.5*0.8*self.mathfunc(ob_state[6][0], action[6], ob_state[6][1]))/(1 - (0.5*0.5*0.8*0.6))
        r7 = self.mathfunc(ob_state[6][0], action[6], ob_state[6][1]) + 0.5*0.6*r4
        r5 = self.mathfunc(ob_state[4][0], action[4], ob_state[4][1]) + 0.9*0.5*r7
        m1 = self.mathfunc(ob_state[0][0], action[0], ob_state[0][1]) + 0.5*0.8*self.mathfunc(ob_state[1][0], action[1], ob_state[1][1]) +0.5*0.6*self.mathfunc(ob_state[5][0], action[5], ob_state[5][1]+0.5*0.6*0.5*0.8*r7)
        r8 = (self.mathfunc(ob_state[7][0], action[7], ob_state[7][1]) + 0.5*0.8*r4 + 0.5*0.9*m1 + 0.5*0.9*(0.5*0.8*0.5*0.7+0.5*0.6*0.5*0.8)*(self.mathfunc(ob_state[2][0], action[2], ob_state[2][1])+0.5*0.9*r5+0.5*0.6*r4))/(1-0.5*0.5*0.9*0.6*(0.5*0.8*0.5*0.7+0.5*0.6*0.5*0.8)-0.5*0.5*0.8*0.4*0.5*0.9)
        r3 = self.mathfunc(ob_state[2][0], action[2], ob_state[2][1]) +0.5*0.9*self.mathfunc(ob_state[4][0], action[4], ob_state[4][1])+0.5*0.6*r4+0.5*0.6*r8
        r6 = self.mathfunc(ob_state[5][0], action[5], ob_state[5][1]) + 0.8*0.5*r3 + 0.8*0.5*r7
        r2 = self.mathfunc(ob_state[1][0], action[1], ob_state[1][1]) + 0.7*0.5*r3 + 0.4*0.5*r8
        r1 = self.mathfunc(ob_state[0][0], action[0], ob_state[0][1]) +  0.5*0.8*r2 + 0.6*0.5*r6

        # r1 = self.mathfunc(ob_state[0][0], action[0], ob_state[0][1]) + 0.8*0.5*r2 + 0.6*0.5*r6
        # r2 = self.mathfunc(ob_state[1][0], action[1], ob_state[1][1]) + 0.7*0.5*r3 + 0.4*0.5*r8
        # r3 = self.mathfunc(ob_state[2][0], action[2], ob_state[2][1]) + 0.9*0.5*r5 + 0.6*0.5*r4 + 0.6*0.5*r8
        # r4 = self.mathfunc(ob_state[3][0], action[3], ob_state[3][1]) + 0.8*0.5*r7
        # r5 = self.mathfunc(ob_state[4][0], action[4], ob_state[4][1]) + 0.9*0.5*r7
        # r6 = self.mathfunc(ob_state[5][0], action[5], ob_state[5][1]) + 0.8*0.5*r3 + 0.8*0.5*r7
        # r7 = self.mathfunc(ob_state[6][0], action[6], ob_state[6][1]) + 0.6*0.5*r4
        # r8 = self.mathfunc(ob_state[7][0], action[7], ob_state[7][1]) + 0.9*0.5*r1 + 0.8*0.5*r4
        R = r1 + r2 + r3 +r4 + r5 + r6 + r7 + r8
        print(r1, r2, r3, r4, r5, r6, r7, r8, R)
        with open(r'G:\doctor\DR_TWO\experiment\morvanzhou\contents\my_experiment\rw_reward_one.csv', 'a+', encoding='utf-8', newline="") as f:
            titles = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8']
            w = csv.DictWriter(f, fieldnames=titles)
            if not os.path.getsize(r'G:\doctor\DR_TWO\experiment\morvanzhou\contents\my_experiment\rw_reward_one.csv'):
                w.writeheader()
            w.writerows([{'a1': action[0], 'a2': action[1], 'a3': action[2], 'a4': action[3], 'a5': action[4], 'a6': action[5], 'a7': action[6], 'a8': action[7], 'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4, 'r5': r5, 'r6': r6, 'r7': r7, 'r8': r8}])
        return R

    def step(self, action):
        self.turns += 1
        # self.count += 1
        state_index = random.choice(range(5000))

        obs = self.States_list[state_index]  # this is the obs for nn from St1
        # print(obs)
        reward = self.Reward(obs, action)

        # reward = Reward(obs, action)[0]  # using St instead of St1 to calculate the reward
        # qos = Reward(obs, action)[1]
        # risk = Reward(obs, action)[2]

        with open(r'G:\doctor\DR_TWO\experiment\morvanzhou\contents\my_experiment\rw.csv', 'a+', encoding='utf-8', newline="") as f:
            titles = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']
            w = csv.DictWriter(f, fieldnames=titles)
            if not os.path.getsize(r'G:\doctor\DR_TWO\experiment\morvanzhou\contents\my_experiment\rw.csv'):
                w.writeheader()
            w.writerows([{'a1': action[0], 'a2': action[1], 'a3': action[2], 'a4': action[3], 'a5': action[4], 'a6': action[5], 'a7': action[6], 'a8': action[7]}])

        done = False
        info = {}
        # print(obs)
        # print(action)
        # print(type(reward))
        # print('reward',reward)

        return obs, reward, done, info

    def render(self, mode='human'):
        return

# MyEnvrec()










