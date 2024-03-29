from math import gamma
import random
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from model import Model
from utils.SumTree import SumTree
from collections import deque, namedtuple
import pickle
import time
import datetime
import wandb
import zlib

Transition = namedtuple(
    'Transition', ('state', 'next_state', 'action', 'reward', 'done'))


class Memory:
    def __init__(self, cfg):
        self.memory_size = cfg.memory_size
        self.memory = deque(maxlen=self.memory_size)
        self.batch_size = cfg.batch_size

    def push(self, exp):
        self.memory.append(exp)

    def sample(self, episode):
        sample_indices = np.random.choice(
            np.arange(len(self.memory)), replace=False, size=self.batch_size)
        batch = [self.memory[idx] for idx in sample_indices]
        batch = Transition(*map(np.stack, zip(*batch)))
        return (None, batch, None)

    def update(self, indices, td_error):
        pass

    def __len__(self):
        return len(self.memory)


class Brain:
    def __init__(self, cfg, n_actions):
        self.n_actions = n_actions
        self.input_dim = (cfg.state_channel, cfg.state_height, cfg.state_width)

        # init
        self.cfg = cfg
        self.wandb = cfg.wandb
        self.episode = 0
        self.batch_size = cfg.batch_size

        self.n_episodes = cfg.n_episodes
        self.gamma = cfg.gamma

        # model
        self.policy_net, self.target_net = self._create_model(cfg)
        self.synchronize_model()
        self.target_net.eval()

        self.scaler = GradScaler()
        if cfg.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.policy_net.parameters(), lr=cfg.lr)

        if cfg.loss_fn == 'SmoothL1Loss':
            self.loss_fn = nn.SmoothL1Loss()

        # exploration
        self.exploration_rate = cfg.exploration_rate
        self.exploration_rate_decay = cfg.exploration_rate_decay
        self.exploration_rate_min = cfg.exploration_rate_min

        self.memory = Memory(cfg)

        self.noisy = cfg.noisy
        self.double = cfg.double

    def synchronize_model(self):
        # モデルの同期
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _create_model(self, cfg):
        # modelを選べるように改変
        policy_net = Model(
            self.input_dim, self.n_actions).float().to('cuda')
        target_net = Model(
            self.input_dim, self.n_actions).float().to('cuda')
        return policy_net, target_net

    def select_action(self, state):
        epsilon = 0. if self.noisy else self.exploration_rate

        if np.random.rand() < epsilon:
            action = np.random.randint(self.n_actions)
        else:
            # state = state.__array__()
            state = torch.tensor(state).float().cuda().unsqueeze(0)
            with torch.no_grad():
                Q = self._get_Q(self.policy_net, state)
            action = torch.argmax(Q, axis=1).item()

        if not self.noisy:
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(
                self.exploration_rate_min, self.exploration_rate)

        return action

    def send_memory(self, state, next_state, action, reward, done):
        exp = Transition([state], [next_state], [action],
                         [reward], [done])
        self.memory.push(exp)

    def update(self, episode):
        # メモリからサンプル
        indices, batch, weights = self.memory.sample(episode)
        # サンプルした経験から損失を計算
        loss, td_error, q = self._loss(batch, weights)
        # PERがONの場合はメモリを更新
        self.memory.update(indices, td_error)
        # policy_netを学習
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        return loss.detach().cpu(), q

    def _get_Q(self, model, x):
        return model(x)

    def _loss(self, batch, weights):
        state = torch.tensor(batch.state).to('cuda').float()
        next_state = torch.tensor(batch.next_state).to('cuda').float()
        action = torch.tensor(batch.action).to('cuda')
        reward = torch.tensor(batch.reward).to('cuda')
        done = torch.tensor(batch.done).to('cuda')

        # non_final_mask = torch.tensor(~.done).to('cuda')
        # non_final_next_state = torch.stack([next_state for not])
        with torch.no_grad():
            if self.double:
                next_state_Q = self.policy_net(next_state)
                best_action = torch.argmax(next_state_Q, axis=1)
                next_Q = self.target_net(next_state)[
                    np.arange(0, self.batch_size), best_action
                ]
            else:
                next_Q = torch.max(self.target_net(next_state), axis=1)
            td_target = (reward + (1. - done.float())
                         * self.gamma * next_Q).float()

        with autocast():
            Q = self.policy_net(state)
            td_estimate = Q[np.arange(0, self.batch_size), action.squeeze()]

            td_error = torch.abs(td_target - td_estimate)

            loss = self.loss_fn(td_estimate, td_target)
        return loss, td_error.detach().cpu(), td_estimate.detach().cpu()


class Agent:
    def __init__(self, cfg, n_actions, save_dir):
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

        self.brain = Brain(cfg, n_actions)

        self.wandb = cfg.wandb
        if self.wandb:
            self.logger = Logger()

    def action(self, state):
        self.step += 1
        action = self.brain.select_action(state)
        return action

    def observe(self, state, next_state, action, reward, done):
        self.brain.send_memory(state, next_state, action, reward, done)
        if self.wandb:
            self.logger.step(reward)

    def learn(self):
        if self.step % self.synchronize_interval == 0:
            self.brain.synchronize_model()
        if self.step % self.burnin + self.restart_episode:
            return
        if self.step % self.learn_interval != 0:
            return

        # メモリからサンプリングして学習を行い、損失とqの値を出力
        loss, q = self.brain.update(self.episode)

        if self.wandb:
            self.logger.step_learn(loss, q)

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
        if self.wandb == False:
            return
        self.episode = episode

        if self.wandb:
            self.logger.log_episode(
                self.step, episode, self.brain.exploration_rate, info)

        if episode != 0 and episode != self.restart_episode:
            if episode % self.save_checkpoint_interval == 0:
                self._save_checkpoint()
            if episode % self.save_model_interval == 0:
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
        checkpoint_path = (self.save_dir / f'agent_net_{self.episode}.ckpt')
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
        self.episode_reward = 0.0
        self.episode_max_reward = 0.0
        self.episode_loss = 0.0
        self.episode_q = 0.0
        self.episode_learn_steps = 0
        self.episode_start_time = self.episode_last_time

    def step(self, reward):
        self.episode_steps += 1
        self.episode_reward += reward
        self.episode_max_reward = max(reward, self.episode_max_reward)

    def step_learn(self, loss, q):
        # 一回の学習につき (learn_interval分飛ばしている)
        self.episode_learn_steps += 1
        self.episode_loss += loss
        self.episode_q += q

    def log_episode(self, step, episode, exploration_rate, info):
        self.episode_last_time = time.time()
        episode_time = self.episode_last_time - self.episode_start_time
        if self.episode_learn_steps == 0:
            episode_average_loss = 0
            episode_average_q = 0
            episode_step_per_second = 0
        else:
            episode_average_loss = self.episode_loss / self.episode_learn_steps
            episode_average_q = self.episode_q / self.episode_learn_steps
            episode_step_per_second = self.episode_learn_steps / episode_time  # 一回の学習に何秒かけたか

        episode_reward = self.episode_reward / \
            self.episode_steps if self.episode_steps != 0 else 0
        episode_max_reward = self.episode_max_reward

        wandb_dict = dict(
            episode=episode,
            step=step,
            epsilon=exploration_rate,
            step_per_second=episode_step_per_second,
            reward=episode_reward,
            max_reward=episode_max_reward,
            length=self.episode_steps,
            average_loss=episode_average_loss,
            average_q=episode_average_q,
        )
        wandb.log(wandb_dict)

        self._reset_episode_log()
