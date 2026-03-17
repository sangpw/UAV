import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

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
            rng = np.random.default_rng(self.map_seed)
            self.obstacles = self._generate_city_blocks(self.num_obstacles, rng)

    def _generate_city_blocks(self, n, rng):
        """
        生成互不重叠的城市建筑 (Rejection Sampling)
        """
        obs = []

        # 定义地图生成范围 (在起点终点周围扩展)
        margin = 100
        min_x = min(self.start_pos[0], self.target_pos[0]) - margin
        max_x = max(self.start_pos[0], self.target_pos[0]) + margin
        min_y = min(self.start_pos[1], self.target_pos[1]) - margin
        max_y = max(self.start_pos[1], self.target_pos[1]) + margin

        # 最大尝试次数，防止死循环
        max_attempts = n * 50
        attempts = 0

        while len(obs) < n and attempts < max_attempts:
            attempts += 1

            # 1. 随机生成尺寸 (更加多样化)
            # 区分高瘦楼(写字楼)和矮胖楼(商场)
            if rng.random() < 0.3:
                # 高楼
                width = rng.uniform(30, 60)
                length = rng.uniform(30, 60)
                height = rng.uniform(100, 250)
            else:
                # 普通建筑
                width = rng.uniform(50, 120)
                length = rng.uniform(50, 120)
                height = rng.uniform(40, 120)

            # 2. 随机位置
            cx = rng.uniform(min_x, max_x)
            cy = rng.uniform(min_y, max_y)

            new_block = np.array([cx, cy, width, length, height])

            # 3. 冲突检测
            if not self._is_valid_placement(new_block, obs):
                continue

            obs.append(new_block)

        if len(obs) < n:
            print(f"Warning: Only placed {len(obs)}/{n} obstacles due to space constraints.")

        return obs

    def _is_valid_placement(self, new_block, existing_blocks):
        """检测新生成的方块是否有效"""
        cx, cy, w, l, _ = new_block

        # --- A. 保护区检测 (Start/Target Safety Zone) ---
        # 保护半径
        safe_radius = 60.0

        # 计算新楼房的矩形边界 (AABB)
        min_bx, max_bx = cx - w / 2, cx + w / 2
        min_by, max_by = cy - l / 2, cy + l / 2

        # 检查是否覆盖 Start 点
        if (min_bx - safe_radius < self.start_pos[0] < max_bx + safe_radius) and \
                (min_by - safe_radius < self.start_pos[1] < max_by + safe_radius):
            return False

        # 检查是否覆盖 Target 点
        if (min_bx - safe_radius < self.target_pos[0] < max_bx + safe_radius) and \
                (min_by - safe_radius < self.target_pos[1] < max_by + safe_radius):
            return False

        # --- B. 建筑物间距检测 (Overlap Check) ---
        for ex_block in existing_blocks:
            ex_cx, ex_cy, ex_w, ex_l, _ = ex_block

            # 两个矩形是否重叠逻辑:
            # 如果 (RectA.left < RectB.right) && (RectA.right > RectB.left) ...
            # 我们加上 min_building_dist 作为缓冲区
            buffer = self.min_building_dist

            # 检查 X 轴投影是否重叠 (带缓冲)
            overlap_x = (abs(cx - ex_cx) * 2) < (w + ex_w + buffer * 2)
            # 检查 Y 轴投影是否重叠 (带缓冲)
            overlap_y = (abs(cy - ex_cy) * 2) < (l + ex_l + buffer * 2)

            if overlap_x and overlap_y:
                return False  # 发生重叠或间距过小

        return True

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # 如果不是固定地图，每次重新生成
        if not self.fixed_map:
            # 使用 seed 初始化的随机生成器
            self.obstacles = self._generate_city_blocks(self.num_obstacles, self.np_random)

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
            terminated = True;
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
        ax.set_zlim(-300, 10)  # 假设最高楼不超过300米

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m) NED')

        # 反转Z轴，符合NED坐标系直觉（天空在上方）
        ax.invert_zaxis()

        ax.set_title(f"Urban Mission (Obstacles: {len(self.obstacles)})")
        plt.legend(loc='upper right')
        plt.draw()
        plt.pause(0.001)  # 给绘图窗口刷新时间


# ==========================================
# 验证与测试
# ==========================================
if __name__ == "__main__":
    # 测试随机地图的生成质量
    print("生成随机城区地图...")
    env = UrbanPlanningEnv(num_obstacles=30, fixed_map=False, min_building_dist=30)

    # 重置并渲染
    env.reset(seed=2024)
    env.render()

    # 再次重置看是否变化
    print("生成另一个随机地图...")
    env.reset(seed=999)
    env.render()