# envs/urban_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Optional

# 导入核心组件
from rl_core import ObservationBuilder
from utils import generate_city_blocks, set_fixed_map_flag, plot_uav_path, check_collision
from models import MultirotorUAV, LithiumBattery, FuelCellStack


class UrbanPlanningEnv(gym.Env):
    """
    城区物流配送路径规划环境 (标准版)
    坐标系：NED (North-East-Down), Z轴向下为正，地面为 Z=0
    观测维度：12维 (由 ObservationBuilder 统一定义)
    """

    def __init__(self,
                 start_pos=[0.0, 0.0, -10.0],  # 初始高度10m
                 target_pos=[800.0, 600.0, -100.0],  # 目标高度100m
                 dt=1.0,  # 决策步长 (s)
                 num_obstacles=15,
                 fixed_map: bool = False,
                 map_seed: int = 42,
                 min_building_dist: float = 30.0):
        super().__init__()

        # 1. 基础参数设置
        self.dt = dt
        self.start_pos = np.array(start_pos, dtype=np.float32)
        self.target_pos = np.array(target_pos, dtype=np.float32)
        self.num_obstacles = num_obstacles
        self.min_building_dist = min_building_dist
        self.fixed_map = fixed_map
        self.map_seed = map_seed

        # 2. 核心组件初始化
        self.obs_builder = ObservationBuilder()
        self.uav = MultirotorUAV(mass=5.0, rotor_radius=0.15, num_rotors=4)
        self.bat = LithiumBattery(capacity_ah=10.0, initial_soc=0.6)
        self.fc = FuelCellStack(num_cells=50)

        # 3. 定义空间
        # 动作：[Vx, Vy, Vz] 归一化在 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # 观测：12维标准向量
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_builder.state_dim,),
            dtype=np.float32
        )

        # 4. 运行状态变量
        self.time_step = 0
        self.max_steps = 500
        self.obstacles = []
        self.path_history = []
        self.current_power_load = 500.0  # 默认悬停功率

        # 初始化固定地图
        set_fixed_map_flag(fixed_map)
        if self.fixed_map:
            self.obstacles = generate_city_blocks(
                n=self.num_obstacles,
                start_pos=self.start_pos,
                target_pos=self.target_pos,
                min_building_dist=self.min_building_dist,
                seed=self.map_seed
            )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """环境重置"""
        super().reset(seed=seed)

        # 地图生成逻辑
        if not self.fixed_map:
            # 如果没有固定地图，每次使用传入的 seed 生成随机地图
            self.obstacles = generate_city_blocks(
                n=self.num_obstacles,
                start_pos=self.start_pos,
                target_pos=self.target_pos,
                min_building_dist=self.min_building_dist,
                seed=seed
            )

        # 重置物理模型
        self.uav.reset(self.start_pos)
        self.bat = LithiumBattery(capacity_ah=10.0, initial_soc=0.6)
        self.fc = FuelCellStack(num_cells=50)

        self.time_step = 0
        self.current_power_load = 500.0
        self.path_history = [self.start_pos.copy()]

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        """调用标准的 ObservationBuilder 构建观测"""
        pos = self.uav.get_position()
        vel = self.uav.get_velocity()

        return self.obs_builder.build(
            pos=pos,
            vel=vel,
            target=self.target_pos,
            soc=self.bat.SOC,
            p_load=self.current_power_load,
            h2_cum=self.fc.operating_hours * 10.0  # 氢耗近似转换
        )

    def _get_info(self):
        """返回详细的状态字典"""
        pos = self.uav.get_position()
        vel = self.uav.get_velocity()
        dist = np.linalg.norm(pos - self.target_pos)

        return {
            'position': pos.copy(),
            'velocity': vel.copy(),
            'target': self.target_pos.copy(),
            'distance': dist,
            'power_load': self.current_power_load,
            'obstacles': self.obstacles,
            'power_state': {
                'soc': self.bat.SOC,
                'soh': self.bat.SOH,
                'h2_cum': self.fc.operating_hours * 10.0
            }
        }

    def step(self, action):
        """执行环境步进 (1s)"""
        self.time_step += 1

        # 1. 动作映射与物理执行
        # 网络输出 [-1, 1] -> 实际速度 [-15, 15] m/s
        target_vel = np.clip(action * 15.0, -15.0, 15.0)

        # 无人机模型步进
        # step 返回: (new_state, power_demand)
        state, p_load = self.uav.step(target_vel, self.dt)
        self.current_power_load = p_load

        # 2. 能量系统步进 (简单同步更新，确保 SOC 变化)
        # 假设 EMS 这里按简单的功率平衡处理
        p_fc_act, h2_step = self.fc.step(p_load * 0.8, self.dt)  # FC承担80%负荷
        p_bat_req = p_load - p_fc_act
        self.bat.step(p_bat_req, self.dt)

        # 3. 记录轨迹
        pos = self.uav.get_position()
        self.path_history.append(pos.copy())

        # 4. 碰撞与终止判定
        terminated = False
        truncated = False

        # 距离判定
        dist = np.linalg.norm(pos - self.target_pos)

        # 使用统一工具函数进行碰撞检测 (地面+建筑)
        is_collision = check_collision(pos, self.obstacles)

        # 核心奖励 (基础部分，重塑部分在 Wrapper 中完成)
        # 这里给一个微小的距离惩罚，确保 reward 不是 0
        reward = -dist * 0.001

        if is_collision:
            terminated = True
            reward = -200.0  # 碰撞重罚 (Wrapper 会进一步增强)

        elif dist < 15.0:
            terminated = True
            reward = 500.0  # 成功奖励

        elif self.bat.SOC < 0.15:
            terminated = True
            reward = -200.0  # 电量耗尽惩罚

        if self.time_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        """调用标准可视化工具"""
        plot_uav_path(
            start_pos=self.start_pos,
            target_pos=self.target_pos,
            obstacles=self.obstacles,
            path_history=self.path_history,
            title=f"Urban Planning Env - Step {self.time_step}"
        )

    def close(self):
        pass