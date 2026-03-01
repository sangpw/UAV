# train_td3.py
import numpy as np
import torch
import os
from rl_core import TD3, ReplayBuffer
from env_wrapper import UAVEnv


def train():
    env = UAVEnv(T_sim=600, dt=0.1)

    state_dim = env.observation_space_dim
    action_dim = env.action_space_dim
    max_action = 1.0  # 动作已归一化到 [-1, 1]

    policy = TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # 训练参数
    max_episodes = 200  # 训练200个回合
    max_timesteps = 6000  # 每个回合最大步数
    start_timesteps = 5000  # 纯随机探索步数
    batch_size = 256

    total_timesteps = 0

    if not os.path.exists("./models"):
        os.makedirs("./models")

    print("开始 TD3 训练...")

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for t in range(max_timesteps):
            total_timesteps += 1

            # 选择动作
            if total_timesteps < start_timesteps:
                action = np.random.uniform(-1, 1, size=(action_dim,))
            else:
                action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, 0.1, size=action_dim)
                ).clip(-max_action, max_action)

            # 与环境交互
            next_state, reward, done, _ = env.step(action)

            # 存入 buffer
            replay_buffer.add(state, action, next_state, reward, done)

            state = next_state
            episode_reward += reward

            # 训练网络
            if total_timesteps >= start_timesteps:
                policy.train(replay_buffer, batch_size)

            if done:
                break

        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}, Timesteps = {t}")

        # 每20个回合保存一次模型
        if (episode + 1) % 20 == 0:
            policy.save(f"./models/td3_ems_{episode + 1}")

    print("训练完成！模型已保存到 ./models 文件夹")


if __name__ == "__main__":
    train()