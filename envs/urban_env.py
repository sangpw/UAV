import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# 导入utils中的地图生成工具
from utils import generate_city_blocks, set_fixed_map_flag

# 尝试导入物理模型 (保持你的原始逻辑)
try:
    from models import MultirotorUAV, LithiumBattery, FuelCellStack
except ImportError:
    # Mock 类用于独立运行测试
    class MultirotorUAV:
        def __init__(self): self.state = np.zeros(6)

        def reset(self, pos): self.state = np.zeros(6); self.state[:3] = pos

        def step(self, vel, dt): self.state[3:] = vel; self.state[:3] += vel * dt

        def get_position(self): return self.state[:3]

        def get_velocity(self): return self.state[3:]


    class LithiumBattery:
        def __init__(self): self.SOC = 1.0


    class FuelCellStack:
        pass


class UrbanPlanningEnv(gym.Env):
    """
    城区物流配送路径规划验证环境 (优化版)
    特点：
    1. 楼房生成增加防重叠检测，确保街道间距。
    2. 起点和终点周围设有安全保护区。
    """

    def __init__(self,
                 start_pos=[0, 0, -10],
                 target_pos=[800, 600, -50],
                 dt=1,
                 num_obstacles=20,  # 稍微增加数量，测试密集城区
                 fixed_map: bool = False,
                 map_seed: int = 42,
                 min_building_dist: float = 30.0):  # <--- 新参数：楼宇间最小间距
        super().__init__()

        self.dt = dt
        self.start_pos = np.array(start_pos, dtype=np.float32)
        self.target_pos = np.array(target_pos, dtype=np.float32)
        self.num_obstacles = num_obstacles
        self.min_building_dist = min_building_dist  # 街道宽度

        self.fixed_map = fixed_map
        self.map_seed = map_seed
        self.obstacles = []

        # 设置全局固定地图开关
        set_fixed_map_flag(fixed_map)

        # 物理模型
        self.uav = MultirotorUAV()
        self.bat = LithiumBattery()
        self.fc = FuelCellStack()

        # 动作 & 观测空间
        self.action_space = spaces.Box(low=-15, high=15, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

        self.time_step = 0
        self.max_steps = 1000

        # 如果是固定地图，初始化时生成
        if self.fixed_map:
            self.obstacles = generate_city_blocks(
                n=self.num_obstacles,
                start_pos=self.start_pos,
                target_pos=self.target_pos,
                min_building_dist=self.min_building_dist,
                seed=self.map_seed
            )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # 如果不是固定地图，每次重新生成
        if not self.fixed_map:
            # 使用 seed 初始化的随机生成器
            self.obstacles = generate_city_blocks(
                n=self.num_obstacles,
                start_pos=self.start_pos,
                target_pos=self.target_pos,
                min_building_dist=self.min_building_dist,
                seed=seed
            )

        self.uav.reset(self.start_pos)
        self.bat = LithiumBattery()
        self.time_step = 0
        return self._get_obs(), self._get_info()

    def _get_obs(self):
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
            'obstacles': self.obstacles,           # 障碍物列表
            'dt': self.dt,                         # 仿真步长
            'power_state': {                       # 能源状态
                'soc': self.bat.SOC,
                'h2_cum': getattr(self.fc, 'h2_consumed', 0.0)
            }
        }

    def step(self, action):
        self.time_step += 1
        cmd_vel = np.clip(action, -15, 15)
        self.uav.step(cmd_vel, self.dt)
        pos = self.uav.get_position()

        # 简单的碰撞检测逻辑
        terminated = False
        truncated = False
        reward = -np.linalg.norm(pos - self.target_pos) * 0.01

        # 地面检测
        if pos[2] > 0:
            terminated = True
            reward = -1000

        # 建筑物检测
        for o in self.obstacles:
            cx, cy, w, l, h = o
            # NED: z > -h (高度低于楼顶) 且 在xy范围内
            if (cx - w / 2 <= pos[0] <= cx + w / 2) and \
                    (cy - l / 2 <= pos[1] <= cy + l / 2) and \
                    (pos[2] > -h):
                terminated = True
                reward = -1000
                break

        dist = np.linalg.norm(pos - self.target_pos)
        if dist < 15.0:
            terminated = True
            reward = 1000
            print("Target Reached!")

        if self.time_step >= self.max_steps: truncated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self, path_history=None):
        """
        可视化环境状态
        :param path_history: (可选) 历史路径点列表，用于绘制轨迹
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 1. 绘制起点和终点
        ax.scatter(*self.start_pos, c='g', s=100, label='Start', depthshade=False)
        ax.scatter(*self.target_pos, c='r', s=100, label='Target', marker='*', depthshade=False)

        # 2. 绘制建筑物
        for o in self.obstacles:
            cx, cy, w, l, h = o
            x_anchor = cx - w / 2
            y_anchor = cy - l / 2

            # 颜色区分：高楼(>150m)用深蓝，矮楼用灰
            color = 'navy' if h > 150 else 'gray'
            alpha = 0.6 if h > 150 else 0.4

            # 绘制长方体
            ax.bar3d(x_anchor, y_anchor, -h, w, l, h,
                     color=color, alpha=alpha, edgecolor='k', linewidth=0.5)

        # 3. 绘制飞行轨迹 (关键修复)
        if path_history is not None and len(path_history) > 0:
            path = np.array(path_history)
            # 确保路径数据格式正确
            if path.ndim == 2 and path.shape[1] == 3:
                ax.plot(path[:, 0], path[:, 1], path[:, 2],
                        c='magenta', linewidth=2.5, label='Trajectory')
                # 标记当前末端位置
                ax.scatter(*path[-1], c='orange', s=50, marker='o')

        # 4. 视图设置 (自动居中)
        mid_x = (self.start_pos[0] + self.target_pos[0]) / 2
        mid_y = (self.start_pos[1] + self.target_pos[1]) / 2

        # 计算显示范围
        range_x = abs(self.target_pos[0] - self.start_pos[0])
        range_y = abs(self.target_pos[1] - self.start_pos[1])
        max_range = max(range_x, range_y) / 2 + 200

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(-300, 50)  # 固定Z轴范围，适配建筑高度

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Urban UAV Planning Env')
        ax.legend()
        plt.show()