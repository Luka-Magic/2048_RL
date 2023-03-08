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
import numpy as np
from pathlib import Path
warnings.simplefilter('ignore')

@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):
    EXP_NAME = Path.cwd().parents[2].name
    ROOT_DIR = Path.cwd().parents[4]

    # 環境をインポート
    sys.path.append(str(ROOT_DIR / f'env/{cfg.env_name}'))
    import gym_2048

    # 設定
    save_dir = ROOT_DIR / 'outputs' / EXP_NAME
    
    save_dir.mkdir(exist_ok=True, parents=True)

    # wandb
    if cfg.wandb:
        wandb.login()
        wandb.init(project=cfg.wandb_project, entity='luka-magic',
                   name=EXP_NAME, config=cfg)

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
    for episode in tqdm(range(1+init_episode, 1+cfg.n_episodes)):
        state = env.reset()
        after_state = np.zeros_like(state)
        while True:
            action = agent.action(state)
            next_state, reward, done, info = env.step(action)
            after_state = info['after_state']
            agent.learn(after_state, next_state, action, reward, done)
            state = next_state
            if done:
                break

        agent.log_episode(episode, {'state': state})

        if episode % cfg.eval_interval == 0:
            for _ in range(cfg.n_eval_episodes):
                state = env.reset()
                while True:
                    action = agent.action(state)
                    next_state, reward, done, info = env.step(action)
                    agent.eval_observe(reward)
                    state = next_state
                    if done:
                        break
                agent.eval_log_episode({'state': state})
            agent.eval_log(episode)

if __name__ == '__main__':
    main()
