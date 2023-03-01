import numpy as np
import torch
import gym
from gym.spaces import Box

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, rew):
        return rew

def env_wrappers(env, cfg, init_episode):
    env = RewardWrapper(env)
    return env
