# env_wrapper.py
import numpy as np
from utils import generate_flight_profile
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List
from models import MultirotorUAV, FuelCellStack, LithiumBattery

class UAVEnv:
    """
    将物理模型封装为 Gym-Like 环境
    """

    def __init__(self, T_sim=600, dt=0.1):
        self.T_sim = T_sim
        self.dt = dt

        # 动作空间：FC功率 [0, 500] (稍后归一化到 -1~1)
        self.max_power = 500.0

        # 状态空间: [SOC, Load_normalized, SOH]
        self.observation_space_dim = 3
        self.action_space_dim = 1

        # 初始化内部变量
        self.time_step = 0
        self.t_arr, self.load_arr = None, None
        self.fc = None
        self.bat = None

    def reset(self):
        # 重置物理模型
        self.fc = FuelCellStack(num_cells=50, max_slew_rate=50.0)
        self.bat = LithiumBattery(capacity_ah=5.0, initial_soc=np.random.uniform(0.5, 0.7))

        # 重置工况 (每次随机一点扰动，防止过拟合)
        self.t_arr, self.load_arr = generate_flight_profile(self.T_sim, self.dt)
        self.time_step = 0

        return self._get_state()

    def step(self, action):
        """
        :param action: 归一化动作 [-1, 1]
        """
        # 1. 动作还原 (-1~1 -> 0~500W)
        p_fc_cmd = (action[0] + 1) / 2 * self.max_power

        # 2. 获取当前负载
        p_load = self.load_arr[self.time_step]

        # 3. 物理模型步进
        p_fc_act, h2_step = self.fc.step(p_fc_cmd, self.dt)
        p_bat_req = p_load - p_fc_act
        p_bat_act, soc, soh = self.bat.step(p_bat_req, self.dt)

        # 4. 计算奖励 (Reward Function) - 核心部分！
        # 目标：最小化氢耗，维持SOC在0.6，减少电池老化

        # H2 惩罚 (归一化大约在 0~1 之间)
        r_h2 = - (h2_step / 0.02)

        # SOC 维持惩罚 (二次型)
        r_soc = - 20.0 * ((soc - 0.6) ** 2)

        # SOH 衰减惩罚 (放大系数)
        delta_soh = (1.0 - soh)  # 这里简化为总衰减量，实际应为 delta
        # 由于SOH单步变化极微小，通常忽略或给极大权重，这里暂时忽略单步SOH reward

        # 边界惩罚 (Soft Constraint)
        r_constraint = 0
        if soc < 0.2 or soc > 0.9:
            r_constraint = -10.0

        reward = r_h2 + r_soc + r_constraint

        # 5. 状态更新
        self.time_step += 1
        done = False
        if self.time_step >= len(self.t_arr) - 1:
            done = True

        # 提前终止 (如果电池耗尽)
        if soc <= 0.05:
            reward -= 100  # 失败惩罚
            done = True

        next_state = self._get_state()

        return next_state, reward, done, {}

    def _get_state(self):
        # 归一化状态 [SOC(0-1), Load(0-1), SOH(0-1)]
        current_load = self.load_arr[self.time_step]
        norm_load = current_load / 1000.0
        return np.array([self.bat.SOC, norm_load, self.bat.SOH])

