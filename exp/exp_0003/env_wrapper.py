import numpy as np
import torch
import gym
from gym.spaces import Box


# class ChangeObservation(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.reference = np.tile(2**np.arange(16), (4, 4, 1)).transpose(2, 1, 0)
#         self.reference[0, :, :] = 0
#         self.observation_spape = Box(
#             low=0, high=255, shape=(16, 4, 4), dtype=np.uint8)
    
#     def observation(self, obs):
#         new_obs = (self.reference == np.tile(obs, (16, 1, 1))).astype(np.uint8)
#         return new_obs

# class AvoidStackWrapper(gym.Wrapper):
#     def __init__(self, env, threshold=10):
#         super().__init__(env)
#         self.env = env
#         self.no_reward_counter = 0
#         self.threshold = threshold
        
#     def step(self, action):
#         next_state, next_after_state, reward, done, info = self.env.step(action)
#         if next_state == next_after_state:
#             self.no_reward_counter += 1
#             if self.no_reward_counter == self.threshold:
#                 done = True
#         return next_state, next_after_state, reward, done, info

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, rew):
        # rew = max(1, rew)
        # rew = np.log2(rew).astype(np.uint8)
        return rew

def env_wrappers(env, cfg, init_episode):
    env = AvoidStackWrapper(env)
    env = RewardWrapper(env)
    return env
