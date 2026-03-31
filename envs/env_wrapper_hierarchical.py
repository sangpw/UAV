# env_wrapper_hierarchical.py
from math import inf
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List

# 导入物理模型
from models import MultirotorUAV, FuelCellStack, LithiumBattery
# 导入工具函数
from utils import generate_city_blocks, set_fixed_map_flag, check_collision
# 核心修改：导入统一的观测构造器
from rl_core import ObservationBuilder


class HierarchicalUAVEnv:
    """
    双层优化环境封装 (支持跨层观测与标准重构)
    """

    def __init__(self,
                 planner_dt: float = 1.0,
                 ems_dt: float = 0.1,
                 T_sim: float = 600.0,
                 start_pos: np.ndarray = np.array([0., 0., -10.]),
                 target_pos: np.ndarray = np.array([800., 600., -100.]),
                 num_obstacles: int = 15,
                 fixed_map: bool = True,
                 map_seed: int = 42,
                 min_building_dist: float = 30.0):

        self.planner_dt = planner_dt
        self.ems_dt = ems_dt
        self.T_sim = T_sim
        self.substeps = int(planner_dt / ems_dt)

        # 1. 初始化统一观测构造器
        self.obs_builder = ObservationBuilder()

        # 2. 物理模型
        self.uav = MultirotorUAV(mass=5.0, rotor_radius=0.15, num_rotors=4)
        self.fc = FuelCellStack(num_cells=50, cell_area=100, max_slew_rate=20.0)
        self.bat = LithiumBattery(capacity_ah=10, initial_soc=0.6)

        self.start_pos = start_pos
        self.target_pos = target_pos

        # 3. 地图配置
        self.num_obstacles = num_obstacles
        self.fixed_map = fixed_map
        self.map_seed = map_seed
        self.min_building_dist = min_building_dist
        set_fixed_map_flag(fixed_map)

        self.obstacles = []
        if self.fixed_map:
            self.obstacles = generate_city_blocks(
                n=self.num_obstacles,
                start_pos=self.start_pos,
                target_pos=self.target_pos,
                min_building_dist=self.min_building_dist,
                seed=self.map_seed
            )

        self.planner = None
        self.ems = None

        # 4. 状态变量
        self.current_planned_vel = np.zeros(3)
        self.last_p_load = 500.0  # 初始功耗（假设为悬停功耗）
        self.time_step = 0
        self.max_steps = int(T_sim / planner_dt)

    def set_planner(self, planner):
        self.planner = planner

    def set_ems(self, ems):
        self.ems = ems

    def reset(self, seed: Optional[int] = None):
        """重置环境并返回初始 12 维观测"""
        self.uav.reset(self.start_pos)
        self.fc = FuelCellStack(num_cells=50, cell_area=100, max_slew_rate=20.0)
        self.bat = LithiumBattery(capacity_ah=10, initial_soc=0.6)

        self.time_step = 0
        self.current_planned_vel = np.zeros(3)
        self.last_p_load = 500.0  # 重置功耗记录

        if not self.fixed_map:
            self.obstacles = generate_city_blocks(
                n=self.num_obstacles,
                start_pos=self.start_pos,
                target_pos=self.target_pos,
                min_building_dist=self.min_building_dist,
                seed=seed
            )

        init_pos = self.uav.get_position()
        dist_to_target = np.linalg.norm(init_pos - self.target_pos)
        print(f"  [Reset] Start: {init_pos}, Target: {self.target_pos}, Dist: {dist_to_target:.1f}m")

        if check_collision(init_pos, self.obstacles):
            print(f"  [Warning] Initial position collision detected!")

        return self._get_planner_obs()

    def _get_planner_obs(self):
        """
        使用 rl_core.ObservationBuilder 构建标准 12 维观测
        """
        return self.obs_builder.build(
            pos=self.uav.get_position(),
            vel=self.uav.get_velocity(),
            target=self.target_pos,
            soc=self.bat.SOC,
            p_load=self.last_p_load,  # 传入功耗状态
            h2_cum=self.fc.operating_hours * 10  # 氢耗近似
        )

    def _get_ems_obs(self, power_load):
        """构建下层EMS观测 (保持不变)"""
        return np.array([
            power_load / 1000.0,
            self.bat.SOC,
            self.bat.SOH,
            self.fc.current_power_act / 1000.0
        ], dtype=np.float32)

    def step(self, action=None):
        """
        执行上层步进 (1s)
        """
        # 1. 确定规划指令
        if action is not None:
            # 训练模式：接受 12 维状态对应的 3 维动作
            self.current_planned_vel = np.clip(np.array(action), -15.0, 15.0)
        else:
            # 推理模式：调用 planner (SAC/AStar/APF)
            if self.planner is None:
                raise ValueError("Planner not set")

            # 核心修改：通过 kwargs 将 power_load 传入推理侧
            self.current_planned_vel = self.planner.compute_velocity_command(
                current_pos=self.uav.get_position(),
                current_vel=self.uav.get_velocity(),
                target_pos=self.target_pos,
                obstacles=self.obstacles,
                power_state={'soc': self.bat.SOC, 'h2_cum': self.fc.operating_hours * 10},
                dt=self.planner_dt,
                power_load=self.last_p_load  # 跨层参数
            )

        # 2. 指令保护
        if np.linalg.norm(self.current_planned_vel) < 0.1:
            self.current_planned_vel[2] = -2.0  # 防止无指令坠落

        self.time_step += 1
        ems_reward_accum = 0
        step_p_load_accum = 0  # 用于计算该 planner_dt 内的平均功耗

        # 3. 高频下层循环 (10 x 0.1s)
        for _ in range(self.substeps):
            # 物理步进
            state, p_load = self.uav.step(self.current_planned_vel, self.ems_dt)
            pos = self.uav.get_position()
            step_p_load_accum += p_load

            if check_collision(pos, self.obstacles):
                break

            # EMS 决策
            if self.ems is not None:
                fc_power = self.ems.compute_fc_command(
                    p_load, self.bat.SOC, self.ems_dt
                )
            else:
                fc_power = np.clip(p_load, 0, 500) if self.bat.SOC > 0.3 else 0

            # 能量分配
            p_fc_act, h2_step = self.fc.step(fc_power, self.ems_dt)
            p_bat_req = p_load - p_fc_act
            p_bat_act, soc_new, soh_new = self.bat.step(p_bat_req, self.ems_dt)

            # 累计 EMS 奖励 (供上层感知能效)
            ems_reward_accum += (-h2_step * 10 - abs(self.bat.SOC - 0.6) * 50)

        # 更新上一时刻平均功耗，供下一次 obs 使用
        self.last_p_load = step_p_load_accum / self.substeps

        # 4. 结算奖励与终止条件
        pos = self.uav.get_position()
        dist_to_target = np.linalg.norm(pos - self.target_pos)
        reward = -dist_to_target * 0.01
        done = False

        if dist_to_target < 15.0:
            reward += 500.0
            done = True
            print(f"  [Done] Arrived! Dist: {dist_to_target:.1f}m")

        if check_collision(pos, self.obstacles):
            reward -= 2000.0
            done = True
            print(f"  [Done] Collision! Pos: {pos}")

        if self.bat.SOC < 0.15:
            reward -= 2000.0
            done = True
            print(f"  [Done] Battery Depleted!")

        if self.time_step >= self.max_steps:
            done = True

        # 加入能效反馈：上层奖励包含部分下层能耗表现
        reward += ems_reward_accum * 0.1

        obs = self._get_planner_obs()
        info = {
            'distance': dist_to_target,
            'position': pos.copy(),
            'velocity': self.uav.get_velocity().copy(),
            'soc': self.bat.SOC,
            'power_load': self.last_p_load,
            'h2_total': self.fc.operating_hours * 10
        }

        return obs, reward, done, info