import numpy as np
from typing import Tuple, Dict

import numpy as np


class FuelCellStack:
    """
    燃料电池电堆模型
    包含：极化曲线计算、动态响应滞后、寿命衰退
    适用于能量管理算法仿真验证
    """

    def __init__(self, num_cells=50, cell_area=100, max_slew_rate=20.0):
        self.N = num_cells  # 单体电池数量
        self.A = cell_area  # 单体电池有效面积 cm²
        self.max_slew_rate = max_slew_rate  # 最大功率变化率 W/s

        # 工作状态变量
        self.current_power_act = 0.0  # 当前实际输出功率
        self.operating_hours = 0.0  # 累计运行时长 h

        # 电化学模型固定参数
        self.T = 343.15  # 电堆工作温度 K
        self.R = 8.3143  # 通用气体常数
        self.F = 96485  # 法拉第常数
        self.alpha = 0.5  # 电荷转移系数
        self.i0 = 1e-3  # 交换电流密度 A/cm²
        self.R_int = 0.0003  # 欧姆内阻 Ω·cm²
        self.B = 0.016  # 浓差极化系数
        self.i_lim = 1.2  # 极限电流密度 A/cm²
        self.E0 = 1.229  # 标准状态下可逆电压

        # 寿命衰退参数
        self.deg_rate = 1e-5  # 电压衰退速率 V/h

    def get_voltage(self, i):
        """
        计算单体电池输出电压
        i：电流密度 A/cm²
        """
        # 限制电流密度在有效区间内
        i = np.clip(i, 1e-4, self.i_lim - 1e-3)

        # 计算能斯特开路电压
        E_nernst = self.E0 - 0.85e-3 * (self.T - 298.15)

        # 计算活化极化过电势
        v_act = (self.R * self.T / (self.alpha * self.F)) * np.log(i / self.i0)

        # 计算欧姆极化过电势
        v_ohm = i * self.R_int

        # 计算浓差极化过电势
        v_conc = self.B * np.log(self.i_lim / (self.i_lim - i))

        # 计算理论单体电压
        v_cell = E_nernst - v_act - v_ohm - v_conc

        # 计算寿命衰退导致的电压损失
        v_deg = self.deg_rate * self.operating_hours

        # 返回最终单体电压，设置最小输出限制
        return max(0.1, v_cell - v_deg)

    def step(self, power_cmd, dt):
        """
        单步动态仿真
        :param power_cmd: 功率指令 W
        :param dt: 仿真步长 s
        :return: 实际输出功率 W, 氢气消耗量 g
        """
        # 功率动态变化率约束
        power_diff = power_cmd - self.current_power_act
        max_change = self.max_slew_rate * dt
        if abs(power_diff) <= max_change:
            self.current_power_act = power_cmd
        else:
            self.current_power_act += np.sign(power_diff) * max_change

        # 限制输出功率非负
        self.current_power_act = max(0.0, self.current_power_act)

        # 低功率工况直接返回零输出
        P = self.current_power_act
        if P < 1.0:
            return 0.0, 0.0

        # 迭代求解电堆工作电流与电压
        v_stack = self.N * 0.65
        i_total = 0.0
        for _ in range(5):
            i_total = P / v_stack if v_stack > 0 else 0.0
            i_dens = i_total / self.A
            v_single = self.get_voltage(i_dens)
            v_stack = v_single * self.N

        # 累计有效运行时长
        if P > 5:
            self.operating_hours += dt / 3600.0

        # 计算氢气消耗量
        h2_grams = (self.N * i_total * 2.016) / (2 * self.F) * dt

        return self.current_power_act, max(0, h2_grams)


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
