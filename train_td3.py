# train_td3.py
import numpy as np
import os
from rl_core import TD3, ReplayBuffer
from env_wrapper import UAVEnv
import matplotlib.pyplot as plt
import glob
import sqlite3
import time
import json


def find_latest_model(models_dir):
    actor_files = glob.glob(os.path.join(models_dir, 'td3_ems_*_actor'))
    critic_files = glob.glob(os.path.join(models_dir, 'td3_ems_*_critic'))
    if not actor_files or not critic_files:
        return None
    # 提取回合数
    get_ep = lambda f: int(f.split('_')[-2])
    latest_ep = max(get_ep(f) for f in actor_files)
    base_path = os.path.join(models_dir, f'td3_ems_{latest_ep}')
    if os.path.exists(base_path + '_actor') and os.path.exists(base_path + '_critic'):
        return base_path
    return None


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS episode_log (
        episode INTEGER PRIMARY KEY,
        reward REAL,
        h2 REAL,
        timesteps INTEGER,
        duration REAL,
        params_json TEXT
    )''')
    conn.commit()
    return conn


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

    models_dir = "./models"
    db_path = "training_log.db"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    conn = init_db(db_path)

    # 检查是否有已有模型，加载最新模型继续训练
    latest_model = find_latest_model(models_dir)
    if latest_model:
        print(f"检测到已有模型 {latest_model}，将继续训练。")
        policy.load(latest_model)
    else:
        print("未检测到已有模型，将从头训练。")

    print("开始 TD3 训练...")

    episode_rewards = []
    episode_h2s = []
    valid_episodes = []
    valid_rewards = []
    valid_h2s = []
    valid_durations = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        episode_h2 = 0
        t = 0  # 初始化 t，防止未赋值引用
        step_params = []
        start_time = time.time()

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
            # 获取氢耗
            h2_step = env.fc.step((action[0] + 1) / 2 * env.max_power, env.dt)[1]

            # 存入 buffer
            replay_buffer.add(state, action, next_state, reward, done)

            state = next_state
            episode_reward += reward
            episode_h2 += h2_step

            # 记录每步参数变化
            step_params.append({
                'step': t,
                'reward': reward,
                'h2': h2_step,
                'soc': env.bat.SOC,
                'soh': env.bat.SOH
            })

            # 训练网络
            if total_timesteps >= start_timesteps:
                policy.train(replay_buffer, batch_size)

            if done:
                break

        duration = time.time() - start_time
        episode_rewards.append(episode_reward)
        episode_h2s.append(episode_h2)
        # 仅记录有效episode
        if t >= int(0.9 * max_timesteps):
            valid_episodes.append(episode)
            valid_rewards.append(episode_reward)
            valid_h2s.append(episode_h2)
            valid_durations.append(duration)
            # 写入数据库
            conn.execute('''INSERT INTO episode_log (episode, reward, h2, timesteps, duration, params_json) VALUES (?, ?, ?, ?, ?, ?)''',
                         (episode, episode_reward, episode_h2, t, duration, json.dumps(step_params)))
            conn.commit()
            print(f"[有效] Episode {episode + 1}: Reward={episode_reward:.2f}, H2={episode_h2:.4f}g, Steps={t}, Time={duration:.2f}s")
        else:
            print(f"[失败] Episode {episode + 1}: 未能完成90%步数，未记录。")

        # 每20个回合保存一次模型
        if (episode + 1) % 20 == 0:
            policy.save(f"./models/td3_ems_{episode + 1}")

    print("训练完成！模型已保存到 ./models 文件夹")

    # 绘制有效episode的reward和氢耗变化曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(valid_episodes, valid_rewards, label='Valid Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward per Valid Episode')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(valid_episodes, valid_h2s, label='Valid H2 Consumption (g)')
    plt.xlabel('Episode')
    plt.ylabel('Total H2 Consumption (g)')
    plt.title('Hydrogen Consumption per Valid Episode')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(valid_episodes, valid_durations, label='Training Time (s)')
    plt.xlabel('Episode')
    plt.ylabel('Duration (s)')
    plt.title('Training Time per Valid Episode')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curve_td3_valid.png')
    plt.show()

    conn.close()


if __name__ == "__main__":
    train()