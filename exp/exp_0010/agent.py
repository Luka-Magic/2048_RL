from math import gamma
import random
import numpy as np
from utils.sum_tree import SumTree
from utils.state_converter import StateConverter
from collections import deque, namedtuple
import pickle
import time
import datetime
import wandb


Transition = namedtuple(
    'Transition', ('state', 'next_state', 'action', 'reward', 'done'))

'''
    N Tuple Networkの作成
    1. agent.action(state) -> action
        stateから4方向に動かした時の4つのVをtupleを用いて計算
        最も良いVになるようなactionを返す
    2. env.step(action) -> s', r, s", done
    3. agent.learn(s', r, s", done)
'''

class LUT:
    def __init__(self):
        self.lut = []
        '''
            0  1  2  3
            4  5  6  7
            8  9  10 11
            12 13 14 15
        '''

        self.tuples = [
            [0, 1, 4, 5], [1, 2, 5, 6], [2, 3, 6, 7],
            [4, 5, 8, 9], [5, 6, 9, 10], [6, 7, 10, 11],
            [8, 9, 12, 13], [9, 10, 13, 14], [10, 11, 14, 15],
            [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
            [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]
        ]
        # self.tuples_arr = np.array(self.tuples)
        self.reset_model()

    def _state_convert(self, state):
        '''
            0, 2, 4, 8... -> 0, 1, 2, 3... に変換
        '''
        return np.log2(np.clip(state, a_min=1, a_max=None)).astype(np.int8)
    
    def get_value(self, state):
        '''
            state -> 
            for ntuple in ntuples:
                ntupleでindexに変換 -> 検索
        '''

        value = 0
        state = self._state_convert(state)
        for tuple_idx, tuple in enumerate(self.tuples):
            index = self._state2index(tuple, tuple_idx, state)
            value += self.lut[index]
        return value

    def _state2index(self, tuple, tuple_idx, state):
        idx_start = self.lut_split[tuple_idx]
        idx = sum(state.reshape(-1)[tuple] * (15 ** np.array([0, 1, 2, 3])))    
        return idx_start + idx

    def learn(self, state, grad):
        state = self._state_convert(state)
        for tuple_idx, tuple in enumerate(self.tuples):
            index = self._state2index(tuple, tuple_idx, state)
            self.lut[index] += grad
    
    def reset_model(self):
        lut_split = [0]
        length = 0
        for ntuple in self.tuples:
            length += 15**len(ntuple)
            lut_split.append(length)
        self.lut = [0.0] * length
        self.lut_split = lut_split


class Agent:
    def __init__(self, cfg, save_dir):
        self.LUT = LUT()
        self.cfg = cfg

        self.lr = cfg.lr
        self.save_dir = save_dir
        self.converter = StateConverter()
        self.step = 0
        self.episode = 0
        self.wandb = cfg.wandb

        self.restart_step = 0
        self.restart_episode = 0

        self.save_checkpoint_interval = cfg.save_checkpoint_interval

        self.logger = Logger()
        self.eval_logger = EvalLogger()

    def action(self, state):
        self.step += 1
        after_states, can_actions, scores = self.converter.make_after_states(state)
        value_after_states = np.array([self.LUT.get_value(after_state) for after_state in after_states]) \
                            + np.array(scores)
        action = can_actions[np.argmax(value_after_states)]
        return action
    
    def learn(self, after_state, next_state, action, reward, done):
        '''
            1. Vを計算
                V = get_v(after_state)
            2. newVを計算する
                next_V = max(get_v(next_state) + next_r)
        '''
        v = self.LUT.get_value(after_state)
        next_after_states, _, scores = self.converter.make_after_states(next_state)
        value_after_states = np.array([self.LUT.get_value(next_after_state) for next_after_state in next_after_states]) \
                           + np.array(scores)
        if done:
            return
        next_v = np.max(value_after_states)
        grad = (next_v - v) * self.lr
        self.LUT.learn(after_state, grad)

        if self.wandb:
            self.logger.step(reward, v, grad)

    def log_episode(self, episode, info):
        '''
            学習でepisode終了時におくるlog
        '''
        self.episode = episode
        if self.wandb:
            self.logger.log_episode(
                self.step, episode, info)

        if episode != 0 and episode != self.restart_episode:
            if episode % self.save_checkpoint_interval == 0:
                self._save_checkpoint()
        
    def eval_observe(self, reward):
        '''
            評価時に観測した値をアップデート
        '''
        if self.wandb:
            self.eval_logger.step(reward)
    
    def eval_log_episode(self, info):
        '''
            評価時1episodeでの値をアップデート
        '''
        self.eval_logger.eval_episode(info)


    def eval_log(self, episode):
        '''
            1回分の評価を送る
        '''
        update_flag = self.eval_logger.log_eval(episode)
        if update_flag:
            self._save()
    
    def _save_checkpoint(self):
        checkpoint_path = (self.save_dir / f'agent_net.pkl')
        save_dict = dict(
            model=self.LUT.lut,
            step=self.step,
            episode=self.episode
        )
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(save_dict, f)

        datetime_now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print(
            f"Episode {self.episode} - "
            f"Step {self.step} - "
            f"Time {datetime_now}"
        )
    
    def _save(self):
        checkpoint_path = (self.save_dir / f'best_model.pkl')
        save_dict = dict(
            model=self.LUT.lut,
            step=self.step,
            episode=self.episode
        )
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(save_dict, f)

class Logger:
    def __init__(self):
        self.episode_last_time = time.time()
        self._reset_episode_log()

    def _reset_episode_log(self):
        # 変数名どうしよう、logとかつけたらわかりやすそう
        self.episode_steps = 0
        self.episode_sum_rewards = 0.0
        self.episode_max_reward = 0.0
        self.episode_grad = 0.0
        self.episode_v = 0.0
        self.episode_learn_steps = 0
        self.episode_start_time = self.episode_last_time

    def step(self, reward, v, grad):
        self.episode_steps += 1
        self.episode_sum_rewards += reward
        self.episode_max_reward = max(reward, self.episode_max_reward)
        self.episode_learn_steps += 1
        self.episode_grad += abs(grad)
        self.episode_v += v

    def log_episode(self, step, episode, info):
        self.episode_last_time = time.time()
        episode_time = self.episode_last_time - self.episode_start_time
        episode_clear = np.max(info['state']) >= 2048
        if self.episode_learn_steps == 0:
            episode_average_grad = 0
            episode_average_v = 0
            episode_step_per_second = 0
        else:
            episode_average_grad = self.episode_grad / self.episode_learn_steps
            episode_average_v = self.episode_v / self.episode_learn_steps
            episode_step_per_second = self.episode_learn_steps / episode_time  # 一回の学習に何秒かけたか

        wandb_dict = dict(
            episode=episode,
            step=step,
            step_per_second=episode_step_per_second,
            sum_rewards=self.episode_sum_rewards,
            max_reward=self.episode_max_reward,
            length=self.episode_steps,
            average_grad=episode_average_grad,
            average_v=episode_average_v,
            clear=episode_clear
        )
        wandb.log(wandb_dict)

        self._reset_episode_log()


class EvalLogger:
    def __init__(self):
        self.n_episodes = 0
        self._reset_eval()
        self._reset_episode_log()
        self.best_reward = 0

    def _reset_eval(self):
        self.n_episodes = 0
        self.eval_sum_rewards = 0.0
        self.eval_max_reward = 0.0
        self.eval_clear = 0

    def _reset_episode_log(self):
        # 変数名どうしよう、logとかつけたらわかりやすそう
        self.episode_sum_rewards = 0.0
        self.episode_max_reward = 0.0
    
    def step(self, reward):
        self.episode_sum_rewards += reward
        self.episode_max_reward = max(reward, self.episode_max_reward)

    def eval_episode(self, info):
        self.n_episodes += 1
        self.eval_sum_rewards += self.episode_sum_rewards
        self.eval_max_reward += self.episode_max_reward
        episode_clear = int(np.max(info['state']) >= 2048)
        self.eval_clear += episode_clear
        self._reset_episode_log()

    def log_eval(self, episode):
        update_flag = False
        mean_reward = self.eval_sum_rewards / self.n_episodes
        mean_max_reward = self.eval_max_reward / self.n_episodes
        clear_rate = self.eval_clear / self.n_episodes
        wandb_dict = dict(
            episode = episode,
            mean_reward = mean_reward,
            mean_max_reward = mean_max_reward,
            clear_rate = clear_rate
        )
        wandb.log(wandb_dict)
        print(f'\n    EVAL [{episode}] - mean_reward: {mean_reward}, mean_max_reward: {mean_max_reward}')
        self._reset_episode_log()
        self._reset_eval()
        
        if self.best_reward < mean_reward:
            self.best_reward = mean_reward
            update_flag = True
        return update_flag




