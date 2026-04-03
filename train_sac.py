import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from envs.env_wrapper_hierarchical import HierarchicalUAVEnv
from rl_core import SAC, ReplayBuffer
from controllers.ems import TD3_EMS
from utils import compute_hierarchical_reward  # 导入统一奖励


def train():
    MAX_EPISODES = 500
    MAX_STEPS = 500
    BATCH_SIZE = 256
    START_STEPS = 5000

    save_dir = "./models/SAC"
    os.makedirs(save_dir, exist_ok=True)

    # 直接使用测试环境：双层架构 (1s/0.1s)
    # fixed_map=False 增加训练泛化性
    env = HierarchicalUAVEnv(planner_dt=1.0, ems_dt=0.1, fixed_map=False)
    ems_agent = TD3_EMS(model_path="models/TD3/td3_latest_success_actor.pth")
    env.set_ems(ems_agent)
    state_dim = 12
    action_dim = 3  # [vx, vy, vz]

    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=1.0,  # 网络输出 [-1, 1]
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_tune_alpha=True
    )

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=1000000)
    reward_history = []
    success_record = []

    total_steps = 0
    for episode in range(MAX_EPISODES):
        state = env.reset(seed=episode)
        prev_info = {'distance': np.linalg.norm(env.uav.get_position() - env.target_pos), 'h2_total': 0.0}
        episode_reward = 0

        for t in range(MAX_STEPS):
            total_steps += 1

            # 动作选择与映射
            if total_steps < START_STEPS:
                action = np.random.uniform(-1, 1, size=(action_dim,))
            else:
                action = agent.select_action(state)

            # 执行步进：HierarchicalUAVEnv.step(action) 内部会将 [-1, 1] 映射到物理速度
            # 这里的 action 对应 [vx, vy, vz]
            rescaled_action = action * 15.0
            next_state, _, done, info = env.step(rescaled_action)

            # 使用统一奖励函数
            reward = compute_hierarchical_reward(info, prev_info, action, done, False)

            replay_buffer.add(state, action, next_state, reward * 0.1, float(done))

            state = next_state
            prev_info = info.copy()
            episode_reward += reward

            if total_steps >= START_STEPS:
                agent.train(replay_buffer, BATCH_SIZE)

            if done:
                success = 1 if (info['distance'] < 15.0) else 0
                success_record.append(success)
                break

        reward_history.append(episode_reward)
        if (episode + 1) % 10 == 0:
            avg_r = np.mean(reward_history[-10:])
            sr = np.mean(success_record[-50:]) if success_record else 0
            print(f"Ep: {episode + 1} | Steps: {total_steps} | AvgR: {avg_r:.1f} | Success: {sr:.2%}")

        if (episode + 1) % 100 == 0:
            agent.save(f"{save_dir}/sac_ep{episode + 1}")

    plt.figure(figsize=(10, 5))
    plt.plot(reward_history)
    plt.title("SAC Training on Hierarchical Env")
    plt.savefig("sac_training_unified.png")


if __name__ == "__main__":
    train()