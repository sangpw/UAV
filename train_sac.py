# train_sac.py
import numpy as np
import torch
import os
from rl_core import SAC, ReplayBuffer
from env_wrapper_hierarchical import HierarchicalUAVEnv
from ems import RuleBasedEMS


def train():
    T_sim = 600
    planner_dt = 0.5
    ems_dt = 0.1

    env = HierarchicalUAVEnv(
        planner_dt=planner_dt,
        ems_dt=ems_dt,
        T_sim=T_sim,
        start_pos=np.array([0., 0., -10.]),
        target_pos=np.array([800., 600., -100.])
    )

    env.set_ems(RuleBasedEMS(soc_min=0.3, soc_max=0.85))

    state_dim = 11
    action_dim = 3
    max_action = 15.0

    policy = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        auto_tune_alpha=True,
        lr=3e-4
    )

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))

    max_episodes = 500
    max_timesteps = int(T_sim / planner_dt)  # 1200步
    start_timesteps = 1000
    batch_size = 256

    total_timesteps = 0

    if not os.path.exists("./models"):
        os.makedirs("./models")

    print("开始 SAC 路径规划训练...")
    print(f"Max timesteps per episode: {max_timesteps}")

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        step_count = 0

        for t in range(max_timesteps):
            total_timesteps += 1
            step_count += 1

            # 选择动作
            if total_timesteps < start_timesteps:
                # 纯随机探索，但要确保合理范围
                action = np.zeros(3)
                action[0] = np.random.uniform(0, 5)  # x 方向主要向前（假设目标在 +x）
                action[1] = np.random.uniform(0, 5)  # y 方向主要向前
                action[2] = np.random.uniform(-3, 0)  # z 方向主要向上（ENU方式，会被 env 反转）
                # 再添加小幅随机扰动
                action += np.random.normal(0, 1.0, size=3)
            else:
                action = policy.select_action(np.array(state), deterministic=False)
                # 添加探索噪声
                noise = np.random.normal(0, 0.1, size=action_dim)
                action = np.clip(action + noise, -max_action, max_action)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 存储transition
            replay_buffer.add(state, action, next_state, reward, float(done))

            state = next_state
            episode_reward += reward

            # 训练
            if total_timesteps >= start_timesteps:
                losses = policy.train(replay_buffer, batch_size=batch_size)

            # 如果第一步就done，打印调试信息
            if t < 100 and done:
                print(f"  [Debug] Episode {episode + 1} ended at step 1!")
                print(f"    Action: {action}")
                print(f"    Position: {info['position']}")
                print(f"    Distance: {info['distance']:.1f}m")
                print(f"    SOC: {info['soc']:.2f}")
                break

            if done:
                break

        # 打印回合信息
        if (episode + 1) % 10 == 0 or step_count < 5:  # 如果步数异常也打印
            print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                  f"Steps={step_count}, Alpha={policy.alpha:.3f}")

        # 定期保存
        if (episode + 1) % 50 == 0:
            policy.save(f"./models/sac_planner_{episode + 1}")
            print(f"  -> Saved model at episode {episode + 1}")

    print("训练完成！")


if __name__ == "__main__":
    train()