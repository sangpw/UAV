# train_sac.py
import numpy as np
import os
import matplotlib.pyplot as plt
from envs.env_wrapper_hierarchical import HierarchicalUAVEnv
from rl_core import SAC, ReplayBuffer
from controllers.ems import TD3_EMS


def train():
    MAX_EPISODES = 500
    MAX_STEPS = 400
    BATCH_SIZE = 256
    START_STEPS = 5000

    save_dir = "./models/SAC"
    os.makedirs(save_dir, exist_ok=True)

    env = HierarchicalUAVEnv(planner_dt=1.0, ems_dt=0.1, fixed_map=False)
    # 确保此处 TD3 路径正确
    ems_agent = TD3_EMS(model_path="models/TD3/td3_latest_success_actor.pth")
    env.set_ems(ems_agent)

    state_dim = 12
    action_dim = 3

    agent = SAC(
        state_dim=state_dim, action_dim=action_dim, max_action=1.0,
        lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, auto_tune_alpha=True
    )

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=1000000)
    reward_history = []
    success_record = []
    total_steps = 0

    for episode in range(MAX_EPISODES):
        state = env.reset(seed=episode)
        episode_reward = 0

        for t in range(MAX_STEPS):
            total_steps += 1

            if total_steps < START_STEPS:
                action = np.random.uniform(-1, 1, size=(action_dim,))
            else:
                action = agent.select_action(state)

            # 直接执行，环境内部会处理所有奖励逻辑
            next_state, reward, done, info = env.step(action)

            # 存储到 buffer，通常 SAC 对奖励量级敏感，可以在此 * 0.1，但建议先尝试原值
            replay_buffer.add(state, action, next_state, reward, float(done))

            state = next_state
            episode_reward += reward

            if total_steps >= START_STEPS:
                agent.train(replay_buffer, BATCH_SIZE)

            if done:
                success = 1 if (info['distance'] < 15.0 and not info['collision']) else 0
                success_record.append(success)
                break

        reward_history.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_r = np.mean(reward_history[-10:])
            sr = np.mean(success_record[-50:]) if success_record else 0
            print(f"Ep: {episode + 1} | Steps: {total_steps} | AvgR: {avg_r:.2f} | Success: {sr:.2%}")

        if (episode + 1) % 100 == 0:
            agent.save(f"{save_dir}/sac_ep{episode + 1}")

    plt.figure(figsize=(10, 5))
    plt.plot(reward_history)
    plt.title("SAC Training (Unified Reward System)")
    plt.savefig("sac_training_unified.png")


if __name__ == "__main__":
    train()