import numpy as np
import os
import glob
import sqlite3
import time
import json
import matplotlib.pyplot as plt

from rl_core import TD3, ReplayBuffer
from envs.env_wrapper import UAVEnv


# ==========================================
# 0. 辅助函数：安全的模型保存逻辑
# ==========================================
def safe_save_model(policy, path_prefix):
    """
    安全保存模型，增加异常处理，防止保存失败导致训练中断。
    注意：假设你的 policy.save(path) 会自动保存 actor 和 critic
    """
    try:
        policy.save(path_prefix)
    except Exception as e:
        print(f"  >>> [Warning] 模型保存失败 ({path_prefix})! 错误信息: {e}")


def find_latest_model(models_dir):
    """
    寻找最新的模型进行断点续训。
    优先寻找最新成功的模型，其次寻找回合数最大的存档。
    """
    # 1. 优先检查是否存在“最近一次成功”的模型
    latest_success_base = os.path.join(models_dir, 'td3_latest_success')
    if os.path.exists(latest_success_base + '_actor') or os.path.exists(latest_success_base + '_actor.pth'):
        return latest_success_base

    # 2. 否则按周期存档寻找最大的 Episode
    actor_files = glob.glob(os.path.join(models_dir, 'td3_ep*_actor*'))
    if not actor_files:
        return None

    try:
        # 提取形如 td3_ep50_actor.pth 中的 50
        import re
        get_ep = lambda f: int(re.search(r'ep(\d+)', f).group(1))
        latest_ep = max(get_ep(f) for f in actor_files)
        base_path = os.path.join(models_dir, f'td3_ep{latest_ep}')
        return base_path
    except Exception:
        return None


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS episode_log
                 (
                     episode
                     INTEGER
                     PRIMARY
                     KEY,
                     reward
                     REAL,
                     h2
                     REAL,
                     timesteps
                     INTEGER,
                     duration
                     REAL,
                     params_json
                     TEXT
                 )''')
    conn.commit()
    return conn


# ==========================================
# 1. 训练主循环
# ==========================================
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
    max_episode_steps = 0  # 记录单次实验存活的最远步数
    success_count = 0  # 记录有效/成功回合数

    # 统一路径管理
    save_dir = "./models/TD3"
    db_path = "training_log.db"
    os.makedirs(save_dir, exist_ok=True)

    conn = init_db(db_path)

    # 断点续训逻辑
    latest_model = find_latest_model(save_dir)
    if latest_model:
        try:
            policy.load(latest_model)
            print(f"[Info] 检测到已有模型 {latest_model}，将继续训练。")
        except Exception as e:
            print(f"[Warning] 模型加载失败 ({e})，将从头训练。")
    else:
        print("[Info] 未检测到已有模型，将从头训练。")

    print(f"Start Training TD3... State Dim: {state_dim}, Action Dim: {action_dim}")

    episode_rewards = []
    avg_reward_history = []
    episode_h2s = []

    # 有效数据记录
    valid_episodes = []
    valid_rewards = []
    valid_h2s = []
    valid_durations = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        episode_h2 = 0
        step_params = []
        start_time = time.time()
        episode_success = False

        # 初始化 t 防止由于提早退出引发未绑定错误
        t = 0

        for t in range(max_timesteps):
            total_timesteps += 1

            # 选择动作 (TD3的探索方式: 动作加上正态分布噪声)
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

            # 核心修改: 存入 buffer 前对 reward 进行缩放，防止 Q 值爆炸
            scaled_reward = reward * 0.1
            replay_buffer.add(state, action, next_state, scaled_reward, float(done))

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

        # 结算步数
        current_episode_steps = t + 1
        if current_episode_steps > max_episode_steps:
            max_episode_steps = current_episode_steps

        # 判断是否为“有效/成功”回合 (存活超过 90% 步数)
        if current_episode_steps >= int(0.9 * max_timesteps):
            episode_success = True
            success_count += 1
            duration = time.time() - start_time

            valid_episodes.append(episode + 1)
            valid_rewards.append(episode_reward)
            valid_h2s.append(episode_h2)
            valid_durations.append(duration)

            # 写入数据库 (加上 try-except 保护，防止 JSON 序列化失败导致程序崩溃)
            try:
                conn.execute('''INSERT INTO episode_log (episode, reward, h2, timesteps, duration, params_json)
                                VALUES (?, ?, ?, ?, ?, ?)''',
                             (episode + 1, episode_reward, episode_h2, current_episode_steps, duration,
                              json.dumps(step_params)))
                conn.commit()
            except Exception as e:
                print(f"  >>> [DB Error] 数据库写入失败: {e}")

        # 数据统计与打印
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-50:])
        avg_reward_history.append(avg_reward)

        print(f"Ep: {episode + 1} | Reward: {episode_reward:.1f} | Avg: {avg_reward:.1f} | "
              f"Steps: {current_episode_steps} | Max Steps: {max_episode_steps} | Valid: {'Yes' if episode_success else 'No'}")

        # ==========================================
        # 模型保存逻辑优化
        # ==========================================

        # 1. 主存：保存最近一次成功的模型 (不断覆盖)
        if episode_success:
            safe_save_model(policy, f"{save_dir}/td3_latest_success")
            print(f"  -> [Saved] 发现有效轨迹，最优模型已更新 (H2: {episode_h2:.4f}g)")

        # 2. 辅存：每隔 20 回合保存一次归档记录
        if (episode + 1) % 20 == 0:
            safe_save_model(policy, f"{save_dir}/td3_ep{episode + 1}")
            print(f"  -> [Saved] 周期模型已存档 (Ep: {episode + 1})")

    print("训练完成！模型已保存到 ./models/TD3 文件夹")

    # 绘制有效episode的reward和氢耗变化曲线
    if len(valid_episodes) > 0:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.plot(valid_episodes, valid_rewards, label='Valid Episode Reward', marker='.')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reward per Valid Episode')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(valid_episodes, valid_h2s, label='Valid H2 Consumption (g)', marker='.', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Total H2 Consumption (g)')
        plt.title('H2 Consumption per Valid Episode')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(valid_episodes, valid_durations, label='Training Time (s)', marker='.', color='green')
        plt.xlabel('Episode')
        plt.ylabel('Duration (s)')
        plt.title('Training Time per Valid Episode')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_curve_td3_valid.png')
        plt.show()
    else:
        print("警告: 没有产生任何有效(Valid)的回合，跳过绘图。")

    conn.close()


if __name__ == "__main__":
    train()