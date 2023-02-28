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

@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):
    # 環境をインポート
    sys.path.append(f'../../../../../env/{cfg.env_name}')
    import gym_2048

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
    agent.set_mode('train')

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
            next_after_state = info['after_state']
            agent.observe(after_state, next_after_state, action, reward, done)
            agent.learn()
            state = next_state
            after_state = next_after_state
            if done:
                break

        agent.log_episode(episode, {})

        if episode % cfg.eval_interval == 0:
            agent.set_mode('eval')
            for _ in range(cfg.n_eval_episodes):
                state = env.reset()
                no_change_counter = 0
                while True:
                    action = agent.eval_action(state)
                    next_state, reward, done, info = env.step(action)
                    agent.eval_observe(reward)
                    if info['no_change']:
                        no_change_counter += 1
                        if no_change_counter == 50:
                            break
                    else:
                        no_change_counter = 0
                    if done:
                        break
                    state = next_state
                agent.eval_episode()
            agent.log_eval(episode)
            agent.set_mode('train')


if __name__ == '__main__':
    main()
