import numpy as np
import torch
import os
import gymnasium as gym
import matplotlib.pyplot as plt

# 导入环境、算法和工具函数
from envs.urban_env import UrbanPlanningEnv
from rl_core import SAC, ReplayBuffer, ObservationBuilder
from utils import check_collision  # 确保导入了统一的碰撞检测


# ==========================================
# 1. 增强型环境包装器 (加入能量感知与高度保护)
# ==========================================
class EnergyAwareWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.scale_pos = 1000.0
        self.scale_vel = 15.0
        self.scale_pwr = 1000.0  # 功率归一化基准
        self.last_dist = None
        self.obs_builder = ObservationBuilder()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_dist = np.linalg.norm(info['target'] - info['position'])
        return self._normalize_obs(obs, info), info

    def step(self, action):
        # 1. 动作映射 [-1, 1] -> [-15, 15]
        rescaled_action = action * 15.0

        # 2. 执行环境步进
        next_obs, reward, terminated, truncated, info = self.env.step(rescaled_action)

        # 3. 跨层奖励重塑 (Reward Shaping)
        curr_pos = info['position']
        curr_dist = np.linalg.norm(info['target'] - curr_pos)

        # (a) 进度奖励 (引导向目标飞行)
        reward_progress = (self.last_dist - curr_dist) * 15.0

        # (b) 能量消耗惩罚 (感知下层EMS)
        # 假设 info['power_state'] 包含了当前功耗 p_load
        p_load = info.get('power_load', 500.0)  # 兜底值
        reward_energy = -(p_load / self.scale_pwr) * 0.5

        # (c) 高度与边界保护 (防止撞地)
        reward_safety = 0
        if curr_pos[2] > -5.0:  # 距离地面不足5米时开始警告惩罚
            reward_safety = -2.0
        if curr_pos[2] > 0:  # 彻底撞地
            reward_safety = -50.0

        # (d) 终端奖励
        reward_terminal = 0
        if terminated:
            if curr_dist < 15.0:
                reward_terminal = 1000.0
                print(f"  >>> [Success] Arrived! H2: {info['power_state']['h2_cum']:.2f}g")
            else:
                reward_terminal = -500.0  # 碰撞惩罚

        # 综合奖励 (权重：进度 > 终端 > 能量 > 安全)
        total_reward = reward_progress + reward_energy + reward_safety + reward_terminal

        self.last_dist = curr_dist
        norm_obs = self._normalize_obs(next_obs, info)

        return norm_obs, total_reward, terminated, truncated, info

    def _normalize_obs(self, obs, info):
        """
        构建增强版状态向量: [pos(3), vel(3), error(3), soc, p_load, h2]
        维度: 12
        """
        pos = info['position']
        vel = info['velocity']
        error = info['target'] - pos
        soc = info['power_state']['soc']
        h2 = info['power_state']['h2_cum']
        p_load = info.get('power_load', 500.0)

        norm_obs = np.concatenate([
            pos / self.scale_pos,
            vel / self.scale_vel,
            error / self.scale_pos,
            [soc],
            [p_load / self.scale_pwr],
            [h2 / 100.0]
        ]).astype(np.float32)
        return norm_obs


# ==========================================
# 2. 训练主循环
# ==========================================
def train():
    # 参数配置
    MAX_EPISODES = 3000  # 增加回合数以适应更复杂的状态空间
    MAX_STEPS = 500
    BATCH_SIZE = 256
    START_STEPS = 5000  # 初始随机探索步数

    save_dir = "./models/SAC"
    os.makedirs(save_dir, exist_ok=True)

    # 初始化环境 (fixed_map=False 增加泛化性)
    raw_env = UrbanPlanningEnv(num_obstacles=15, fixed_map=False)
    env = EnergyAwareWrapper(raw_env)

    # 自动获取状态维度 (现在应该是 12)
    state_dim = 12
    action_dim = env.action_space.shape[0]

    print(f"Training Start. State Dim: {state_dim}, Action Dim: {action_dim}")

    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=1.0,
        lr=3e-4,
        gamma=0.98,  # 略微调低衰减率，让它更看重近期收益
        tau=0.005,
        alpha=0.1,  # 适中的探索温度
        auto_tune_alpha=True
    )

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=1000000)

    reward_history = []
    success_record = []  # 记录最近100次成功率

    total_steps = 0

    for episode in range(MAX_EPISODES):
        # 每次重置使用不同的种子，生成不同的随机地图
        state, info = env.reset(seed=episode)
        episode_reward = 0

        for t in range(MAX_STEPS):
            total_steps += 1

            # 动作选择
            if total_steps < START_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            # 执行步进
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 存储经验 (Reward 缩放有助于稳定训练)
            replay_buffer.add(state, action, next_state, reward * 0.1, float(terminated))

            state = next_state
            episode_reward += reward

            # 训练步
            if total_steps >= START_STEPS:
                agent.train(replay_buffer, BATCH_SIZE)

            if done:
                success = 1 if (info['distance'] < 20.0) else 0
                success_record.append(success)
                break

        reward_history.append(episode_reward)

        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_r = np.mean(reward_history[-10:])
            sr = np.mean(success_record[-50:]) if success_record else 0
            print(f"Ep: {episode + 1} | Steps: {total_steps} | AvgReward: {avg_r:.1f} | SuccessRate: {sr:.2%}")

        # 定期保存与最优保存
        if len(success_record) > 0 and success_record[-1] == 1:
            if episode_reward >= max(reward_history if reward_history else [-inf]):
                agent.save(f"{save_dir}/sac_best")

        if (episode + 1) % 100 == 0:
            agent.save(f"{save_dir}/sac_ep{episode + 1}")

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history)
    plt.title("SAC Energy-Aware Training")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("sac_training.png")


if __name__ == "__main__":
    train()