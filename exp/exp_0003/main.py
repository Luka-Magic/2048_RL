import os
import hydra
import wandb
from pathlib import Path
from omegaconf import DictConfig
from tqdm import tqdm
# 環境
import gym
from env_wrapper import env_wrappers
# エージェント
from agent import Agent
import warnings
import sys

@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):
    # 設定
    save_dir = Path('/'.join(os.getcwd().split('/')
                    [:-6])) / f"outputs/{os.getcwd().split('/')[-4]}"
    save_dir.mkdir(exist_ok=True)
    warnings.simplefilter('ignore')

    # wandb
    if cfg.wandb:
        wandb.login()
        wandb.init(project=cfg.wandb_project, entity='luka-magic',
                   name=os.getcwd().split('/')[-4], config=cfg)

    # 環境
    env = gym.make(cfg.environment)

    # エージェント
    agent = Agent(cfg, save_dir)

    checkpoint_path = save_dir / 'agent_net.ckpt'

    if cfg.reset_learning or not checkpoint_path.exists():
        init_episode = 0
    else:
        init_episode = agent.restart_episode

    env = env_wrappers(env, cfg, init_episode=init_episode)

    # 学習
    for episode in tqdm(range(init_episode, cfg.n_episodes)):
        state = env.reset()
        while True:
            action = agent.action(state)
            next_state, reward, done, info = env.step(action)
            agent.observe(state, next_state, action, reward, done)
            agent.learn()
            state = next_state
            if done:
                break
        agent.log_episode(episode, info)


if __name__ == '__main__':
    main()
