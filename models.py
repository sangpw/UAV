import numpy as np
from typing import Tuple, Dict

class FuelCellStack:
    """
    燃料电池物理模型
    包含：极化曲线、动态响应滞后(Slew Rate)、寿命衰退
    """

    def __init__(self, num_cells=50, cell_area=100, max_slew_rate=20.0):
        self.N = num_cells
        self.A = cell_area
        self.max_slew_rate = max_slew_rate  # W/s

        # 物理状态
        self.current_power_act = 0.0
        self.operating_hours = 0.0

        # 电化学参数 (Amphlett模型简化)
        self.T = 333  # K
        self.R_int = 0.0003
        self.B = 0.016
        self.deg_rate = 1e-5  # V/h

    def get_voltage(self, current_density):
        """内部计算：根据电流密度算电压"""
        i = max(current_density, 1e-4)
        E_nernst = 1.229 - 0.85e-3 * (self.T - 298.15)
        v_act = 0.9514 - 0.00312 * self.T - 0.000187 * self.T * np.log(i)
        v_act = max(0, 0.2 + 0.05 * np.log(i / 0.001))
        v_ohm = i * self.R_int * 300
        v_conc = self.B * np.log(1 - i / 1.5)
        if np.isnan(v_conc): v_conc = 1.0

        v_cell = E_nernst - v_act - v_ohm + v_conc
        v_deg = self.deg_rate * self.operating_hours
        return max(0, v_cell - v_deg)

    def step(self, power_cmd, dt):
        """
        物理步进
        :param power_cmd: 控制器给出的功率指令 (W)
        :param dt: 时间步长 (s)
        :return: (实际功率, 氢耗量g)
        """
        # 1. 动态滞后模拟 (Slew Rate Limit)
        power_diff = power_cmd - self.current_power_act
        max_change = self.max_slew_rate * dt

        if abs(power_diff) <= max_change:
            self.current_power_act = power_cmd
        else:
            self.current_power_act += np.sign(power_diff) * max_change

        # 2. 计算电流和氢耗
        v_guess = self.N * 0.7
        i_act = 0
        # 简单迭代反解电流
        for _ in range(3):
            if v_guess < 0.1: v_guess = 1.0
            i_total = self.current_power_act / v_guess
            i_dens = min(i_total / self.A, 1.5)
            v_single = self.get_voltage(i_dens)
            v_guess = v_single * self.N
            i_act = i_total

        # 3. 寿命统计
        if self.current_power_act > 10:
            self.operating_hours += dt / 3600.0

        h2_grams = (self.N * i_act * 2.016) / (2 * 96485) * dt
        return self.current_power_act, h2_grams


class LithiumBattery:
    """
    锂电池物理模型 (R-int)
    包含：SOC计算、SOH衰退
    """

    def __init__(self, capacity_ah=10, initial_soc=0.8):
        self.Q_design = capacity_ah
        self.Q_actual = capacity_ah
        self.SOC = initial_soc
        self.SOH = 1.0
        self.R_int = 0.05
        self.cycle_fade_factor = 2e-5

    def get_ocv(self, soc):
        # 6S LiPo OCV 曲线近似
        return (3.0 + 1.0 * soc + 0.2 * (soc ** 3)) * 6

    def step(self, power_req, dt):
        """
        :param power_req: 负载需求功率 (W, >0放电, <0充电)
        """
        ocv = self.get_ocv(self.SOC)
        if ocv <= 0: ocv = 0.1

        # 近似电流计算 I = P/V
        current = power_req / ocv

        # 物理限制保护 (BMS底层切断)
        if self.SOC <= 0 and current > 0: current = 0
        if self.SOC >= 1 and current < 0: current = 0

        # 状态更新
        delta_ah = (current * dt) / 3600.0
        self.SOC = np.clip(self.SOC - delta_ah / self.Q_actual, 0, 1)

        # 寿命衰减
        stress = 1.0 + 1.5 * ((abs(current) / self.Q_design) ** 1.5)
        self.SOH -= (abs(delta_ah) * self.cycle_fade_factor * stress) / self.Q_design
        self.Q_actual = self.Q_design * self.SOH

        p_act = ocv * current - (current ** 2) * self.R_int
        return p_act, self.SOC, self.SOH


class MultirotorUAV:
    """
    多旋翼无人机飞行动力学模型
    """

    def __init__(self, mass=5.0, rotor_radius=0.15, num_rotors=4, max_speed=20.0):
        self.m = mass
        self.R = rotor_radius
        self.n_rotors = num_rotors
        self.max_speed = max_speed
        self.rho = 1.225

        # 状态: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        self.A_rotor = np.pi * rotor_radius ** 2

        # 功率计算基准 (悬停功率)
        # 修正变量名：统一使用 self.hover_power_theoretical 以匹配新逻辑，或者都在此处改为 hover_power
        # 这里我们将 self.hover_power 赋值给 self.hover_power_theoretical 以兼容两种写法
        self.hover_power = (self.m * 9.81) ** 1.5 / np.sqrt(2 * self.n_rotors * self.rho * self.A_rotor)
        self.hover_power_theoretical = self.hover_power  # [FIX] 添加别名，防止报错

    def reset(self, position: np.ndarray):
        self.state = np.zeros(6)
        self.state[:3] = position
        return self.state.copy()

    def compute_power(self, velocity: np.ndarray, acceleration: np.ndarray) -> float:
        """计算当前飞行状态所需功率 (W) - [Final Fixed Version]"""
        v = np.linalg.norm(velocity)

        # 1. 诱导功率 (平滑过渡修正)
        v_tip = 150.0
        mu = v / v_tip

        # 平滑过渡系数：从悬停(1.0)平滑过渡到高速前飞(1.15)
        k_ind = 1.0 + 0.15 * np.clip(mu / 0.1, 0.0, 1.0)

        # 核心公式：随速度增加，诱导功率下降 (Translational Lift)
        # 这里使用 self.hover_power_theoretical (已在init中定义)
        P_ind = self.hover_power_theoretical * k_ind * max(0.4, 1.0 - 0.5 * mu ** 2)

        # 2. 型阻功率
        P_prof = 0.1 * self.rho * self.n_rotors * self.A_rotor * (v ** 3)

        # 3. 爬升功率 (NED坐标系修正)
        # Z轴向下为正，velocity[2] < 0 表示向上爬升
        v_climb = -velocity[2]
        P_climb = self.m * 9.81 * v_climb if v_climb > 0 else 0

        # 4. 机动功率
        P_accel = self.m * np.linalg.norm(acceleration) * v * 0.5

        total = (P_ind + P_prof + P_climb + P_accel) / 0.85
        return max(total, 50.0)

    def step(self, target_velocity: np.ndarray, dt: float) -> Tuple[np.ndarray, float]:
        tau = 0.3
        current_v = self.state[3:]
        new_v = current_v + (target_velocity - current_v) * (dt / tau)
        new_v = np.clip(new_v, -self.max_speed, self.max_speed)

        acc = (new_v - current_v) / dt
        self.state[:3] += new_v * dt
        self.state[3:] = new_v

        power = self.compute_power(new_v, acc)
        return self.state.copy(), power

    def get_position(self) -> np.ndarray:
        return self.state[:3].copy()

    def get_velocity(self) -> np.ndarray:
        return self.state[3:].copy()
