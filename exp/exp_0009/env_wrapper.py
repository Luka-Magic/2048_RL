import numpy as np
import torch
import gym
from gym.spaces import Box

class AvoidStackWrapper(gym.Wrapper):
    def __init__(self, env, threshold=50):
        super().__init__(env)
        self.env = env
        self.no_change_counter = 0
        self.threshold = threshold
        
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if done:
            reward = -1
        else:
            reward += 1
        return next_state, reward, done, info
