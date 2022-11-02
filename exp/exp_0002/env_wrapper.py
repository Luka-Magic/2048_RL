import numpy as np
import torch
import gym
from gym.spaces import Box


class ChangeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reference = np.tile(2**np.arange(16), (4, 4, 1)).transpose(2, 1, 0)
        self.reference[0, :, :] = 0
        self.observation_spape = Box(
            low=0, high=255, shape=(16, 4, 4), dtype=np.uint8)
    
    def observation(self, obs):
        new_obs = (self.reference == np.tile(obs, (16, 1, 1))).astype(np.uint8)
        return new_obs

def env_wrappers(env, cfg, init_episode):
    env = ChangeObservation(env)
    return env
