# env_wrapper_hierarchical.py
from math import inf

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List
from models import MultirotorUAV, FuelCellStack, LithiumBattery


class HierarchicalUAVEnv:
    """
    双层优化环境封装
    """

    def __init__(self,
                 planner_dt: float = 0.5,
                 ems_dt: float = 0.1,
                 T_sim: float = 600.0,
                 start_pos: np.ndarray = np.array([0., 0., -10.]),
                 target_pos: np.ndarray = np.array([800., 600., -100.])):

        self.planner_dt = planner_dt
        self.ems_dt = ems_dt
        self.T_sim = T_sim
        self.substeps = int(planner_dt / ems_dt)

        # 物理模型
        self.uav = MultirotorUAV(mass=5.0, rotor_radius=0.15, num_rotors=4)
        self.fc = FuelCellStack(num_cells=50, cell_area=100, max_slew_rate=20.0)
        self.bat = LithiumBattery(capacity_ah=10, initial_soc=0.6)

        self.start_pos = start_pos
        self.target_pos = target_pos

        # 确保障碍物不会与起点重叠
        self.obstacles = [
            np.array([300, 300, -50, 50]),
            np.array([600, 700, -80, 80]),
        ]

        self.planner = None
        self.ems = None

        self.current_planned_vel = np.zeros(3)
        self.time_step = 0
        self.max_steps = int(T_sim / planner_dt)

    def set_planner(self, planner):
        self.planner = planner

    def set_ems(self, ems):
        self.ems = ems

    def reset(self):
        """重置环境并返回初始观测"""
        self.uav.reset(self.start_pos)
        self.fc = FuelCellStack(num_cells=50, cell_area=100, max_slew_rate=20.0)
        self.bat = LithiumBattery(capacity_ah=10, initial_soc=0.6)
        self.time_step = 0
        self.current_planned_vel = np.zeros(3)

        # 检查初始状态是否合法（调试用）
        init_pos = self.uav.get_position()
        dist_to_target = np.linalg.norm(init_pos - self.target_pos)
        print(f"  [Reset] Start pos: {init_pos}, Target: {self.target_pos}, Dist: {dist_to_target:.1f}m")

        # 检查是否在障碍物内
        if self._check_collision(init_pos):
            print(f"  [Warning] Initial position collision detected!")

        return self._get_planner_obs()

    def _get_planner_obs(self):
        """构建上层观测"""
        pos = self.uav.get_position()
        vel = self.uav.get_velocity()
        to_target = self.target_pos - pos

        obs = np.array([
            *pos, *vel, *to_target,
            self.bat.SOC,
            self.fc.operating_hours * 10  # 累计氢耗近似
        ], dtype=np.float32)
        return obs

    def _get_ems_obs(self, power_load):
        """构建下层EMS观测"""
        return np.array([
            power_load / 1000.0,
            self.bat.SOC,
            self.bat.SOH,
            self.fc.current_power_act / 1000.0
        ], dtype=np.float32)

    def _check_collision(self, pos):
        """碰撞检测"""
        # 地面碰撞 (z > 0 表示在地面上方，但起飞高度z<0，所以z>0是坠地)
        if pos[2] > 0:
            return True

        # 障碍物碰撞
        for obs in self.obstacles:
            if np.linalg.norm(pos - obs[:3]) < obs[3]:
                return True
        return False

    def step(self, action=None):
        """
        执行环境步进

        关键修改：接受 action 参数用于训练，如果不传则使用 planner
        """
        # ===== 修复 1: 正确处理动作输入 =====
        if action is not None:
            # 训练模式：直接使用传入的动作 (numpy array)
            action = np.array(action, dtype=np.float32)
            if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                print(f"  [Warning] Invalid action: {action}")
                action = np.zeros(3)
            self.current_planned_vel = np.clip(action, -15.0, 15.0)
        else:
            # 推理模式：使用 planner
            if self.planner is None:
                raise ValueError("Planner not set for inference mode")
            obs = self._get_planner_obs()
            self.current_planned_vel = self.planner.compute_velocity_command(
                current_pos=self.uav.get_position(),
                current_vel=self.uav.get_velocity(),
                target_pos=self.target_pos,
                obstacles=self.obstacles,
                power_state={'soc': self.bat.SOC, 'h2_cum': self.fc.operating_hours * 10},
                dt=self.planner_dt
            )

        self.time_step += 1

        # ===== 修复 2: 确保动作不为零导致坠落 =====
        if np.linalg.norm(self.current_planned_vel) < 0.1:
            # 如果速度指令太小，给一个向上的力防止坠地
            self.current_planned_vel[2] = -2.0  # 向上(z为负)

        # ===== 下层循环: 高频EMS步进 =====
        ems_reward_accum = 0

        for _ in range(self.substeps):
            # UAV动力学
            state, power_load = self.uav.step(self.current_planned_vel, self.ems_dt)
            pos = self.uav.get_position()

            # 检查是否已坠地或碰撞（在子步骤中检查）
            if self._check_collision(pos):
                break

            # EMS决策
            if self.ems is not None:
                ems_obs = self._get_ems_obs(power_load)
                fc_power = self.ems.compute_fc_command(
                    power_load, self.bat.SOC, self.ems_dt, future_load=None
                )
            else:
                # 默认EMS
                fc_power = np.clip(power_load, 0, 500) if self.bat.SOC > 0.3 else 0

            # 执行能量管理
            p_fc_act, h2_step = self.fc.step(fc_power, self.ems_dt)
            p_bat_req = power_load - p_fc_act
            p_bat_act, soc_new, soh_new = self.bat.step(p_bat_req, self.ems_dt)

            # 累计EMS奖励（氢耗惩罚 + SOC维持）
            r_h2 = -h2_step * 10
            r_soc = -abs(self.bat.SOC - 0.6) * 50
            ems_reward_accum += (r_h2 + r_soc)

        # ===== 计算上层奖励和终止条件 =====
        pos = self.uav.get_position()
        dist_to_target = np.linalg.norm(pos - self.target_pos)

        reward = 0
        done = False

        # 距离奖励（每步惩罚距离）
        reward += -dist_to_target * 0.01

        # 到达奖励
        if dist_to_target < 15.0:
            reward += 200.0
            done = True
            print(f"  [Episode Done] Arrived at target! Dist: {dist_to_target:.1f}m")

        # 碰撞惩罚
        if self._check_collision(pos):
            reward -= inf
            done = True
            print(f"  [Episode Done] Collision/Out of bounds! Pos: {pos}")

        # SOC耗尽
        if self.bat.SOC < 0.15:
            reward -= 100.0
            done = True
            print(f"  [Episode Done] Battery depleted! SOC: {self.bat.SOC:.2f}")

        # 超时
        if self.time_step >= self.max_steps:
            done = True
            print(f"  [Episode Done] Max steps reached!")

        # 能量效率奖励
        reward += ems_reward_accum * 0.1

        obs = self._get_planner_obs()
        info = {
            'distance': dist_to_target,
            'position': pos.copy(),
            'velocity': self.uav.get_velocity().copy(),
            'soc': self.bat.SOC,
            'soh': self.bat.SOH,
            'h2_total': self.fc.operating_hours * 10
        }

        return obs, reward, done, info