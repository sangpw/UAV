import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import List, Tuple

# 尝试导入物理模型，如果没有则使用占位符
from models import MultirotorUAV, LithiumBattery, FuelCellStack



class UrbanPlanningEnv(gym.Env):
    """
    城区物流配送路径规划验证环境 (长方体建筑版)
    """

    def __init__(self,
                 start_pos=[0, 0, -10],
                 target_pos=[800, 600, -50],
                 dt=0.5,
                 num_obstacles=15):
        super().__init__()

        self.dt = dt
        self.start_pos = np.array(start_pos, dtype=np.float32)
        self.target_pos = np.array(target_pos, dtype=np.float32)

        # 物理模型
        self.uav = MultirotorUAV()
        self.bat = LithiumBattery()
        self.fc = FuelCellStack()

        # 障碍物生成: List of [cx, cy, width, length, height]
        self.obstacles = self._generate_city_blocks(num_obstacles)

        # 动作空间 & 观测空间
        self.action_space = spaces.Box(low=-15, high=15, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

        self.time_step = 0
        self.max_steps = 1000

    def _generate_city_blocks(self, n):
        """生成长方体建筑"""
        obs = []
        center_line = self.target_pos - self.start_pos

        for _ in range(n):
            # 1. 随机位置 (沿途分布)
            t = np.random.uniform(0.1, 0.9)
            noise = np.random.normal(0, 150, 2)
            cx = self.start_pos[0] + center_line[0] * t + noise[0]
            cy = self.start_pos[1] + center_line[1] * t + noise[1]

            # 2. 随机尺寸
            width = np.random.uniform(40, 100)  # x方向长度
            length = np.random.uniform(40, 100)  # y方向长度
            height = np.random.uniform(50, 180)  # 高度

            # 格式: [cx, cy, w, l, h]
            obs.append(np.array([cx, cy, width, length, height]))
        return obs

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.uav.reset(self.start_pos)
        self.time_step = 0
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        # 简单的向量观测
        pos = self.uav.get_position()
        vel = self.uav.get_velocity()
        error = self.target_pos - pos
        obs = np.concatenate([pos / 1000, vel / 15, error / 1000, [self.bat.SOC, 0.0]], dtype=np.float32)
        return obs

    def _get_info(self):
        return {
            'position': self.uav.get_position(),
            'velocity': self.uav.get_velocity(),
            'target': self.target_pos,
            'obstacles': self.obstacles,  # 长方体列表
            'dt': self.dt,
            'power_state': {
                'soc': self.bat.SOC,
                'h2_cum': 0.0  # 占位，如果 FC 模型未完全启用
            }
        }

    def _check_collision(self, pos):
        # 1. 地面
        if pos[2] > 0: return True, "Ground"

        # 2. 长方体碰撞检测 (AABB)
        x, y, z = pos
        for o in self.obstacles:
            cx, cy, w, l, h = o
            # x范围: [cx - w/2, cx + w/2]
            # y范围: [cy - l/2, cy + l/2]
            # z范围: [-h, 0] (NED系)

            if (cx - w / 2 <= x <= cx + w / 2) and \
                    (cy - l / 2 <= y <= cy + l / 2) and \
                    (z > -h):  # NED: z > -h 意味着在楼顶下方
                return True, "Building"
        return False, None

    def step(self, action):
        self.time_step += 1
        cmd_vel = np.clip(action, -15, 15)
        self.uav.step(cmd_vel, self.dt)

        pos = self.uav.get_position()
        dist = np.linalg.norm(pos - self.target_pos)

        terminated = False
        truncated = False
        reward = -dist * 0.01

        collision, c_type = self._check_collision(pos)
        if collision:
            reward = -1000
            terminated = True
            print(f"Collision with {c_type} at {pos}")

        if dist < 15.0:
            reward = 1000
            terminated = True
            print("Target Reached!")

        if self.time_step >= self.max_steps: truncated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self, path_history=None):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(*self.start_pos, c='g', s=100, label='Start')
        ax.scatter(*self.target_pos, c='r', s=100, label='Target', marker='*')

        # 绘制长方体
        for o in self.obstacles:
            cx, cy, w, l, h = o
            # bar3d 需要左下角坐标 (anchor)
            x_anchor = cx - w / 2
            y_anchor = cy - l / 2
            z_anchor = 0  # 地面

            # bar3d: x, y, z, dx, dy, dz
            # NED注意：我们在画图时，为了直观，通常让Z轴向上或手动反转。
            # 这里我们画在 Z=0 到 Z=-h
            ax.bar3d(x_anchor, y_anchor, -h, w, l, h, color='gray', alpha=0.4, edgecolor='k')

        if path_history is not None:
            path = np.array(path_history)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], c='b', linewidth=2, label='Path')

        ax.set_xlabel('X');
        ax.set_ylabel('Y');
        ax.set_zlabel('Z')
        ax.set_zlim(-200, 10)
        ax.invert_zaxis()  # 保持 NED 视觉习惯
        plt.show()