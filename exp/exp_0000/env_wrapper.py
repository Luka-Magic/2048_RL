import numpy as np
import torch
import gym
from gym.spaces import Box


class ChangeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_spape = Box(
            low=0, high=255, shape=(4, 4), dtype=np.uint8)

    def observation(self, observation):
        observation = np.clip(np.log2(observation).astype(np.uint8), 0, 15)
        return observation


def env_wrappers(env, cfg, init_episode):
    env = ChangeObservation(env)
    return env
