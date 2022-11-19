from math import gamma
import random
import numpy as np
import torch
from torch import nn
from torchsummary import summary
from torch.cuda.amp import autocast, GradScaler
from model import Model
from utils.sum_tree import SumTree
from utils.state_converter import StateConverter
from collections import deque, namedtuple
import pickle
import time
import datetime
import wandb
import zlib
import pickle


Transition = namedtuple(
    'Transition', ('state', 'next_state', 'action', 'reward', 'done'))


class Memory:
    def __init__(self, cfg):
        self.memory_size = cfg.memory_size
        self.memory = deque(maxlen=self.memory_size)
        self.memory_compress = cfg.memory_compress

        self.batch_size = cfg.batch_size

    def _compress(self, exp):
        if self.memory_compress:
            exp = zlib.compress(pickle.dumps(exp))
        return exp

    def _decompress(self, exp):
        if self.memory_compress:
            exp = pickle.loads(zlib.decompress(exp))
        return exp

    def push(self, exp):
        exp = self._compress(exp)
        self.memory.append(exp) 

    def sample(self, episode):
        sample_indices = np.random.choice(
            np.arange(len(self.memory)), replace=False, size=self.batch_size)
        batch = [self._decompress(self.memory[idx]) for idx in sample_indices]
        # batch = Transition(*map(np.stack, zip(*batch)))
        return (None, batch, None)

    def update(self, indices, td_error):
        pass

    def __len__(self):
        return len(self.memory)


class PERMemory(Memory):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.memory = SumTree(self.memory_size)

        self.n_episodes = cfg.n_episodes

        self.priority_alpha = cfg.priority_alpha
        self.priority_epsilon = cfg.priority_epsilon
        self.priority_use_IS = cfg.priority_use_IS
        self.priority_beta = cfg.priority_beta

    def push(self, exp):
        exp = self._compress(exp)
        priority = self.memory.max()
        if priority <= 0:
            priority = 1
        self.memory.add(priority, exp)

    def sample(self, episode):
        batch = []
        indices = []
        weights = np.empty(self.batch_size, dtype='float32')
        total = self.memory.total()
        beta = self.priority_beta + \
            (1 - self.priority_beta) * episode / self.n_episodes

        for i, rand in enumerate(np.random.uniform(0, total, self.batch_size)):
            idx, priority, exp = self.memory.get(rand)

            weights[i] = (self.memory_size * priority / total) ** (-beta)

            exp = self._decompress(exp)
            batch.append(exp)
            indices.append(idx)
        
        weights /= weights.max()
        # batch = Transition(*map(np.stack, zip(*batch)))
        return (indices, batch, weights)

    def update(self, indices, td_error):
        if indices != None:
            return

        for i in range(len(indices)):
            priority = (
                td_error[i] + self.priority_epsilon) ** self.priority_alpha
            self.memory.update(indices[i], priority)


class Brain:
    def __init__(self, cfg):
        # init
        self.cfg = cfg
        self.wandb = cfg.wandb
        self.episode = 0
        self.batch_size = cfg.batch_size

        self.n_episodes = cfg.n_episodes
        self.gamma = cfg.gamma
        self.n_actions = cfg.n_actions

        self.converter = StateConverter()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # model
        self.input_size = (cfg.state_channel, cfg.state_height, cfg.state_width)
        self.output_size = cfg.n_actions
        
        self.policy_net, self.target_net = self._create_model()
        self.synchronize_model()
        self.target_net.eval()

        if cfg.loss_fn == 'SmoothL1Loss':
            self.loss_fn = nn.SmoothL1Loss()

        self.scaler = GradScaler()
        if cfg.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.policy_net.parameters(), lr=cfg.lr)

        # exploration
        self.exploration_rate = cfg.exploration_rate
        self.exploration_rate_decay = cfg.exploration_rate_decay
        self.exploration_rate_min = cfg.exploration_rate_min
        if self.exploration_rate_decay is None:
            self.exploration_rate_decay = np.exp((np.log(self.exploration_rate_min) - np.log(self.exploration_rate)) / self.n_episodes)

        # momory
        if cfg.use_PER:
            self.memory = PERMemory(cfg)
        else:
            self.memory = Memory(cfg)
        
        # multi step learning
        self.multi_step_learning = cfg.multi_step_learning
        if self.multi_step_learning:
            self.n_multi_steps = cfg.n_multi_steps
            self.multi_step_trainsitions = deque(maxlen=self.n_multi_steps)
            self.gamma = cfg.gamma

    def synchronize_model(self):
        # モデルの同期
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _create_model(self):
        policy_net = Model(self.input_size, self.output_size).float().to(self.device)
        target_net = Model(self.input_size, self.output_size).float().to(self.device)
        print(summary(policy_net, (16, 4, 4)))
        return policy_net, target_net

    def select_action(self, state, eval=False):
        after_states = []
        action_candidates = []
        for action in range(self.n_actions):
            after_state, _, no_change = self.converter.make_after_state(state, action)
            after_states.append(after_state)
            if not no_change:
                action_candidates.append(action)

        if np.random.rand() < self.exploration_rate and not eval:
            action = random.choice(action_candidates)
        else:
            after_states = torch.from_numpy(np.stack(after_states, axis=0)).float().to(self.device)
            with torch.no_grad():
                v = self.policy_net(after_states)
            action = torch.argmax(v, axis=0).item()
        return action

    def update_exploration_rate(self):
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(
            self.exploration_rate_min, self.exploration_rate)

    def send_memory(self, after_state, next_after_state, action, reward, done):
        exp = Transition(after_state, next_after_state, [action],
                         [reward], [done])
        self.memory.push(exp)

    def update(self, episode):
        # メモリからサンプル
        indices, batch, weights = self.memory.sample(episode)
        # サンプルした経験から損失を計算
        loss, td_error, v = self._loss(batch, weights)
        # PERがONの場合はメモリを更新
        self.memory.update(indices, td_error)
        # policy_netを学習
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        return loss.detach().cpu(), v.mean()

    def _loss(self, batch, weights):
        new_batch = []
        for exp in batch:
            state = self.converter.convert(exp.state)
            next_state = self.converter.convert(exp.next_state)
            new_batch.append(Transition(state, next_state, exp.action, exp.reward, exp.done))
        
        batch = Transition(*map(np.stack, zip(*new_batch)))

        state = torch.tensor(batch.state).to(self.device).float()
        next_state = torch.tensor(batch.next_state).to(self.device).float()
        action = torch.tensor(batch.action).to(self.device)
        reward = torch.tensor(batch.reward).to(self.device)
        done = torch.tensor(batch.done).to(self.device)

        with torch.no_grad():
            next_v = self.target_net(next_state)
            td_target = (reward + (1. - done.float())
                         * self.gamma * next_v).float()

        with autocast():
            td_estimate = self.policy_net(state)
            td_error = torch.abs(td_target - td_estimate)
            loss = self.loss_fn(td_estimate, td_target)
        return loss, td_error.detach().cpu(), td_estimate.detach().cpu()


class Agent:
    def __init__(self, cfg, save_dir):
        self.cfg = cfg
        self.step = 0
        self.episode = 0
        self.restart_step = 0
        self.restart_episode = 0

        self.synchronize_interval = cfg.synchronize_interval
        self.burnin = cfg.burnin
        self.learn_interval = cfg.learn_interval

        self.save_dir = save_dir
        self.save_checkpoint_interval = cfg.save_checkpoint_interval
        self.save_model_interval = cfg.save_model_interval

        self.brain = Brain(cfg)

        self.wandb = cfg.wandb
        if self.wandb:
            self.logger = Logger()
            self.eval_logger = EvalLogger()

    def set_mode(self, mode):
        if mode == 'train':
            self.brain.policy_net.train()
        elif mode == 'eval':
            self.brain.policy_net.eval()

    def action(self, state):
        self.step += 1
        action = self.brain.select_action(state)
        return action
    
    def eval_action(self, state):
        action = self.brain.select_action(state, eval=True)
        return action
    
    def observe(self, after_state, next_after_state, action, reward, done):
        self.brain.send_memory(after_state, next_after_state, action, reward, done)
        if self.wandb:
            self.logger.step(reward)

    def eval_observe(self, reward):
        if self.wandb:
            self.eval_logger.step(reward)

    def learn(self):
        if self.step % self.synchronize_interval == 0:
            self.brain.synchronize_model()
        if self.step < self.burnin + self.restart_episode:
            return
        if self.step % self.learn_interval != 0:
            return

        # メモリからサンプリングして学習を行い、損失とqの値を出力
        loss, v = self.brain.update(self.episode)

        if self.wandb:
            self.logger.step_learn(loss, v)

    def restart_learning(self, checkpoint_path):
        self._reset_episode_log()

        checkpoint = torch.load(checkpoint_path)
        self.brain.policy_net.load_state_dict(checkpoint['model'])
        self.brain.synchronize_model()

        self.brain.exploration_rate = checkpoint['exploration_rate']
        self.restart_step = checkpoint['step']
        self.restart_episode = checkpoint['episode']
        print(f'Restart learning from episode {self.restart_episode}')
        self.step = self.restart_step
        self.episode = self.restart_episode
        return self.restart_episode

    def log_episode(self, episode, info):
        self.brain.update_exploration_rate()
        
        if self.wandb == False:
            return
        self.episode = episode

        if self.wandb:
            self.logger.log_episode(
                self.step, episode, self.brain.exploration_rate, info)

        if episode != 0 and episode != self.restart_episode:
            if episode % self.save_checkpoint_interval == 0:
                self._save_checkpoint()

    def eval_episode(self):
        self.eval_logger.eval_episode()
    
    def log_eval(self, episode):
        update_flag = self.eval_logger.log_eval(episode)
        if update_flag:
            self._save()
        
    def _save_checkpoint(self):
        checkpoint_path = (self.save_dir / f'agent_net.ckpt')
        torch.save(dict(
            model=self.brain.policy_net.state_dict(),
            exploration_rate=self.brain.exploration_rate,
            step=self.step,
            episode=self.episode
        ), checkpoint_path)
        datetime_now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print(
            f"Episode {self.episode} - "
            f"Step {self.step} - "
            f"Epsilon {self.brain.exploration_rate:.3f} - "
            f"Time {datetime_now}"
        )

    def _save(self):
        checkpoint_path = (self.save_dir / f'best_model.ckpt')
        torch.save(dict(
            model=self.brain.policy_net.state_dict(),
            exploration_rate=self.brain.exploration_rate,
            step=self.step,
            episode=self.episode
        ), checkpoint_path)


class Logger:
    def __init__(self):
        self.episode_last_time = time.time()
        self._reset_episode_log()

    def _reset_episode_log(self):
        # 変数名どうしよう、logとかつけたらわかりやすそう
        self.episode_steps = 0
        self.episode_sum_rewards = 0.0
        self.episode_max_reward = 0.0
        self.episode_loss = 0.0
        self.episode_v = 0.0
        self.episode_learn_steps = 0
        self.episode_start_time = self.episode_last_time

    def step(self, reward):
        self.episode_steps += 1
        self.episode_sum_rewards += reward
        self.episode_max_reward = max(reward, self.episode_max_reward)

    def step_learn(self, loss, v):
        # 一回の学習につき (learn_interval分飛ばしている)
        self.episode_learn_steps += 1
        self.episode_loss += loss
        self.episode_v += v

    def log_episode(self, step, episode, exploration_rate, info):
        self.episode_last_time = time.time()
        episode_time = self.episode_last_time - self.episode_start_time
        if self.episode_learn_steps == 0:
            episode_average_loss = 0
            episode_average_v = 0
            episode_step_per_second = 0
        else:
            episode_average_loss = self.episode_loss / self.episode_learn_steps
            episode_average_v = self.episode_v / self.episode_learn_steps
            episode_step_per_second = self.episode_learn_steps / episode_time  # 一回の学習に何秒かけたか

        wandb_dict = dict(
            episode=episode,
            step=step,
            epsilon=exploration_rate,
            step_per_second=episode_step_per_second,
            sum_rewards=self.episode_sum_rewards,
            max_reward=self.episode_max_reward,
            length=self.episode_steps,
            average_loss=episode_average_loss,
            average_v=episode_average_v,
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

    def _reset_episode_log(self):
        # 変数名どうしよう、logとかつけたらわかりやすそう
        self.episode_steps = 0
        self.episode_sum_rewards = 0.0
        self.episode_max_reward = 0.0
    
    def step(self, reward):
        self.episode_sum_rewards += reward
        self.episode_max_reward = max(reward, self.episode_max_reward)

    def eval_episode(self):
        self.n_episodes += 1
        self.eval_sum_rewards += self.episode_sum_rewards
        self.eval_max_reward += self.episode_max_reward

    def log_eval(self, episode):
        update_flag = False
        mean_reward = self.eval_sum_rewards / self.n_episodes
        mean_max_reward = self.eval_max_reward / self.n_episodes
        wandb_dict = dict(
            episode = episode,
            mean_reward = mean_reward,
            mean_max_reward = mean_max_reward,
        )
        wandb.log(wandb_dict)
        print(f'\n    EVAL [{episode}] - mean_reward: {mean_reward}, mean_max_reward: {mean_max_reward}')
        self._reset_episode_log()
        self._reset_eval()
        
        if self.best_reward < mean_reward:
            self.best_reward = mean_reward
            update_flag = True
        return update_flag




