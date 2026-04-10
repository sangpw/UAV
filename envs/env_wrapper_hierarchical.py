# env_wrapper_hierarchical.py
import numpy as np
from typing import Optional

from models import MultirotorUAV, FuelCellStack, LithiumBattery
from utils import generate_city_blocks, set_fixed_map_flag, check_collision, compute_hierarchical_reward
from rl_core import ObservationBuilder


class HierarchicalUAVEnv:
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

        self.obs_builder = ObservationBuilder()
        self.uav = MultirotorUAV(mass=5.0, rotor_radius=0.15, num_rotors=4)
        self.fc = FuelCellStack(num_cells=50, cell_area=100, max_slew_rate=20.0)
        self.bat = LithiumBattery(capacity_ah=10, initial_soc=0.6)

        self.start_pos = start_pos
        self.target_pos = target_pos
        self.num_obstacles = num_obstacles
        self.fixed_map = fixed_map
        self.map_seed = map_seed
        self.min_building_dist = min_building_dist
        set_fixed_map_flag(fixed_map)

        self.obstacles = []
        if self.fixed_map:
            self.obstacles = generate_city_blocks(
                n=self.num_obstacles, start_pos=self.start_pos,
                target_pos=self.target_pos, min_building_dist=self.min_building_dist, seed=self.map_seed
            )

        self.planner = None
        self.ems = None

        # 统计变量
        self.last_p_load = 500.0
        self.last_fc_power = 0.0
        self.total_h2_consumed = 0.0  # 精确统计氢气消耗量(g)
        self.time_step = 0
        self.max_steps = int(T_sim / planner_dt)
        self.prev_info = None

    def set_planner(self, planner):
        self.planner = planner

    def set_ems(self, ems):
        self.ems = ems

    def reset(self, seed: Optional[int] = None):
        self.uav.reset(self.start_pos)
        self.fc = FuelCellStack(num_cells=50, cell_area=100, max_slew_rate=20.0)
        self.bat = LithiumBattery(capacity_ah=10, initial_soc=0.6)

        self.time_step = 0
        self.last_p_load = 500.0
        self.last_fc_power = 0.0
        self.total_h2_consumed = 0.0  # 重置氢耗

        if not self.fixed_map:
            self.obstacles = generate_city_blocks(
                n=self.num_obstacles, start_pos=self.start_pos,
                target_pos=self.target_pos, min_building_dist=self.min_building_dist, seed=seed
            )

        init_dist = np.linalg.norm(self.uav.get_position() - self.target_pos)
        # 初始化用于奖励计算的 prev_info
        self.prev_info = {
            'distance': init_dist,
            'h2_total': 0.0,
            'soc': self.bat.SOC,
            'fc_power': 0.0,
            'position': self.uav.get_position().copy()
        }
        return self._get_planner_obs()

    def _get_planner_obs(self):
        # 观测空间：h2_cum 传入实际消耗量
        return self.obs_builder.build(
            pos=self.uav.get_position(), vel=self.uav.get_velocity(),
            target=self.target_pos, soc=self.bat.SOC,
            p_load=self.last_p_load, h2_cum=self.total_h2_consumed
        )

    def step(self, action=None):
        if action is not None:
            current_v_cmd = np.clip(np.array(action) * 15.0, -15.0, 15.0)
        else:
            current_v_cmd = self.planner.compute_velocity_command(
                current_pos=self.uav.get_position(), current_vel=self.uav.get_velocity(),
                target_pos=self.target_pos, obstacles=self.obstacles,
                power_state={'soc': self.bat.SOC, 'h2_cum': self.total_h2_consumed},
                dt=self.planner_dt, power_load=self.last_p_load
            )

        if np.linalg.norm(current_v_cmd) < 0.1: current_v_cmd[2] = -2.0

        self.time_step += 1
        step_p_load_accum = 0
        step_fc_p_accum = 0
        collision_flag = False

        # 高频子循环
        for _ in range(self.substeps):
            _, p_load = self.uav.step(current_v_cmd, self.ems_dt)
            step_p_load_accum += p_load
            if check_collision(self.uav.get_position(), self.obstacles):
                collision_flag = True
                break

            fc_cmd = self.ems.compute_fc_command(p_load, self.bat.SOC, self.ems_dt) if self.ems \
                else (np.clip(p_load, 0, 500) if self.bat.SOC > 0.3 else 0)

            # 物理步进：获取该步实际功率和氢耗(g)
            p_fc_act, h2_step = self.fc.step(fc_cmd, self.ems_dt)

            self.total_h2_consumed += h2_step  # 累加总氢耗
            step_fc_p_accum += p_fc_act
            self.bat.step(p_load - p_fc_act, self.ems_dt)

        self.last_p_load = step_p_load_accum / self.substeps
        self.last_fc_power = step_fc_p_accum / self.substeps

        dist_now = np.linalg.norm(self.uav.get_position() - self.target_pos)
        terminated = (dist_now < 15.0) or collision_flag or (self.bat.SOC < 0.15)
        truncated = (self.time_step >= self.max_steps)

        # 构造 info 字典
        info = {
            'distance': dist_now,
            'position': self.uav.get_position().copy(),
            'velocity': self.uav.get_velocity().copy(),
            'soc': self.bat.SOC,
            'fc_power': self.last_fc_power,
            'h2_total': self.total_h2_consumed,  # 实际氢耗(g)
            'SOH': self.fc.operating_hours,  # 运行小时数
            'collision': collision_flag,
            'time_out': truncated
        }

        # 计算奖励
        reward_action = action if action is not None else (current_v_cmd / 15.0)
        reward = compute_hierarchical_reward(
            info, self.prev_info, reward_action, terminated, truncated
        )

        self.prev_info = info.copy()
        done = terminated or truncated
        return self._get_planner_obs(), reward, done, info