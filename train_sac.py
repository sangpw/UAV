import numpy as np
import torch
import os
import gymnasium as gym
import matplotlib.pyplot as plt

# 导入你的环境和算法
from envs.urban_env import UrbanPlanningEnv
from rl_core import SAC, ReplayBuffer


# ==========================================
# 0. 辅助函数：安全的模型保存逻辑
# ==========================================
def safe_save_model(agent, prefix):
    """
    安全保存模型，增加异常处理，防止保存失败导致训练中断。
    """
    try:
        # 1. 保存 Actor (推理必须)
        torch.save(agent.actor.state_dict(), f"{prefix}_actor.pth")

        # 2. 保存 Critics (如果想恢复训练需要)
        try:
            torch.save(agent.critic_1.state_dict(), f"{prefix}_critic_1.pth")
            torch.save(agent.critic_2.state_dict(), f"{prefix}_critic_2.pth")
        except AttributeError:
            # 如果 rl_core 写法不同，可能叫 critic，这里做个兼容
            if hasattr(agent, 'critic'):
                torch.save(agent.critic.state_dict(), f"{prefix}_critic.pth")

    except Exception as e:
        # 捕获异常，打印错误日志但不中断训练
        print(f"  >>> [Warning] 模型保存失败 ({prefix})! 错误信息: {e}")


# ==========================================
# 1. 环境包装器 (关键：归一化 + 奖励重塑)
# ==========================================
class NormalizedEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.scale_pos = 1000.0  # 地图大概 1000m
        self.scale_vel = 15.0  # 最大速度

        # 记录上一步距离，用于计算引导奖励
        self.last_dist = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_dist = np.linalg.norm(info['target'] - info['position'])
        return self._normalize_obs(obs), info

    def step(self, action):
        # 1. 动作缩放: 网络输出[-1, 1] -> 环境[-15, 15]
        rescaled_action = action * 15.0

        # 2. 环境步进
        next_obs, original_reward, terminated, truncated, info = self.env.step(rescaled_action)

        # 3. 奖励重塑 (Reward Shaping) - 修复收敛慢的核心
        curr_dist = np.linalg.norm(info['target'] - info['position'])

        # (a) 进度引导奖励: 靠近给正分，远离给负分 (系数下调至 10.0，使单步奖励平滑)
        reward_progress = (self.last_dist - curr_dist) * 10.0

        # (b) 移除原有的绝对距离惩罚 (解决模型故意去撞墙的“自杀悖论”)
        # reward_dist = -curr_dist * 0.005  <-- 已删除

        # (c) 生存/耗时惩罚: 鼓励尽快到达，不要绕圈子 (加大惩罚力度)
        reward_step = -0.5

        # (d) 稀疏大奖励: 明确终点目标与失败后果
        reward_terminal = 0.0
        if terminated:
            if curr_dist < 20.0:  # 成功到达
                reward_terminal = 500.0  # 必须远大于存活期间可能扣除的总分
                print(f"  >>> [Success] Reached Target! Dist: {curr_dist:.1f}")
            else:  # 撞墙/撞地
                reward_terminal = -200.0  # 严厉惩罚撞墙行为

        # 总奖励
        reward = reward_progress + reward_step + reward_terminal

        # 更新距离
        self.last_dist = curr_dist

        # 4. 观测归一化
        norm_obs = self._normalize_obs(next_obs)

        return norm_obs, reward, terminated, truncated, info

    def _normalize_obs(self, obs):
        """
        obs结构:[pos(3), vel(3), error(3), soc, h2]
        """
        new_obs = np.array(obs, dtype=np.float32).copy()
        new_obs[0:3] /= self.scale_pos
        new_obs[3:6] /= self.scale_vel
        new_obs[6:9] /= self.scale_pos
        return new_obs


# ==========================================
# 2. 训练主循环
# ==========================================
def train():
    # 配置参数
    MAX_EPISODES = 2000
    MAX_STEPS = 400
    BATCH_SIZE = 256
    START_STEPS = 2000  # 预热步数 (保持 2000 即可)

    # 初始化环境
    raw_env = UrbanPlanningEnv(
        num_obstacles=15,
        fixed_map=True,
        map_seed=42
    )
    env = NormalizedEnvWrapper(raw_env)

    # 初始化 SAC
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = 1.0

    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.05  # <--- 核心修改: 降低探索参数(原0.2)，减少模型动作抖动，加速收敛
    )

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=1000000)

    # 记录
    reward_history = []
    avg_reward_history = []
    success_count = 0
    total_steps = 0
    max_episode_steps = 0

    save_dir = "./models/SAC"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print(f"Start Training SAC... State Dim: {state_dim}, Action Dim: {action_dim}")

    for episode in range(MAX_EPISODES):
        state, info = env.reset(seed=np.random.randint(0, 1000))
        episode_reward = 0
        episode_success = False

        for t in range(MAX_STEPS):
            total_steps += 1

            # 选择动作
            if total_steps < START_STEPS:
                action = env.action_space.sample() / 15.0
            else:
                action = agent.select_action(state, deterministic=False)

            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # <--- 核心修改: 加入 Buffer 时对 reward 进行缩放，防止神经网络 Q 值爆炸
            # 注意：只是存进池子里的缩小了，打印和画图用的 episode_reward 还是真实的直观分数
            scaled_reward = reward * 0.1
            replay_buffer.add(state, action, next_state, scaled_reward, float(terminated))

            state = next_state
            episode_reward += reward

            # 模型更新
            if total_steps >= START_STEPS:
                agent.train(replay_buffer, batch_size=BATCH_SIZE)

            if done:
                dist = np.linalg.norm(info['target'] - info['position'])
                if dist < 20.0:
                    success_count += 1
                    episode_success = True
                break

        # 结算本回合的生存步数，并更新全局最大步数
        current_episode_steps = t + 1
        if current_episode_steps > max_episode_steps:
            max_episode_steps = current_episode_steps

        # 记录与打印
        reward_history.append(episode_reward)
        avg_reward = np.mean(reward_history[-50:])
        avg_reward_history.append(avg_reward)

        if (episode + 1) % 10 == 0:
            print(
                f"Ep: {episode + 1} | Reward: {episode_reward:.1f} | Avg: {avg_reward:.1f} | Total Steps: {total_steps} | Max Steps: {max_episode_steps} | Success: {success_count}")

        # 模型保存逻辑
        if episode_success:
            safe_save_model(agent, f"{save_dir}/sac_latest_success")
            print(f"  -> [Saved] 最优模型已更新 (Ep: {episode + 1})")

        if (episode + 1) % 50 == 0:
            safe_save_model(agent, f"{save_dir}/sac_ep{episode + 1}")
            print(f"  -> [Saved] 周期模型已存档 (Ep: {episode + 1})")

    # 绘图
    plt.plot(avg_reward_history)
    plt.title("SAC Training Curve")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.savefig("training_curve.png")
    plt.show()


if __name__ == "__main__":
    train()