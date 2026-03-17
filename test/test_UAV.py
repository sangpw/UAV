import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


# ==========================================
# 1. 多旋翼无人机物理模型类
# ==========================================
class MultirotorUAV:
    """
    多旋翼无人机飞行动力学模型 (优化版)
    """

    def __init__(self, mass=5.0, rotor_radius=0.15, num_rotors=4, max_speed=20.0):
        self.m = mass
        self.R = rotor_radius
        self.n_rotors = num_rotors
        self.max_speed = max_speed
        self.rho = 1.225

        # 状态: [x, y, z, vx, vy, vz] (NED坐标系: z向下为正)
        self.state = np.zeros(6)
        self.A_rotor = np.pi * rotor_radius ** 2

        # 功率计算基准 (悬停功率)
        # 理论悬停推力 = mg
        thrust_hover = self.m * 9.81
        # 理论悬停诱导功率
        self.hover_power_theoretical = (thrust_hover ** 1.5) / \
                                       np.sqrt(2 * self.n_rotors * self.rho * self.A_rotor)

        # 考虑到电机效率及其他损耗的实际悬停功率估算 (仅用于参考/打印)
        self.hover_power_elec = self.hover_power_theoretical / 0.85

    def reset(self, position: np.ndarray):
        """重置状态"""
        self.state = np.zeros(6)
        self.state[:3] = position
        return self.state.copy()

    def compute_power(self, velocity: np.ndarray, acceleration: np.ndarray) -> float:
        """计算当前飞行状态所需功率 (W)"""
        v = np.linalg.norm(velocity)

        # 1. 诱导功率 (Induced Power) - 平滑过渡修正
        v_tip = 150.0  # 桨尖速度 (典型值)
        mu = v / v_tip

        # 平滑过渡系数：从悬停(1.0)平滑过渡到高速前飞(1.15)
        # 模拟前飞时的湍流增加
        k_ind = 1.0 + 0.15 * np.clip(mu / 0.1, 0.0, 1.0)

        # 核心公式：随速度增加，诱导功率下降 (Translational Lift)
        # max(0.4, ...) 限制诱导功率最低降至悬停的40%，防止数值过低
        P_ind = self.hover_power_theoretical * k_ind * max(0.4, 1.0 - 0.5 * mu ** 2)

        # 2. 型阻功率 (Profile Drag) - 随速度立方增加
        # 模拟空气阻力对旋翼旋转的阻碍
        P_prof = 0.1 * self.rho * self.n_rotors * self.A_rotor * (v ** 3)

        # 3. 爬升/重力功率 (Gravity / Climb Power)
        # NED坐标系: z轴向下。velocity[2] < 0 代表向上爬升。
        # 只有向上爬升时，重力才做负功(消耗能量)。下降时电机通常不回收能量，视为0或维持怠速。
        v_climb = -velocity[2]  # 向上速度
        P_climb = 0
        if v_climb > 0:
            P_climb = self.m * 9.81 * v_climb

        # 4. 机动/加速度功率 (Inertial Power)
        # F = ma, P = F*v
        P_accel = self.m * np.linalg.norm(acceleration) * v * 0.5

        # 总功率 / 综合效率 (0.85) + 基础待机功耗 (50W)
        total = (P_ind + P_prof + P_climb + P_accel) / 0.85
        return max(total, 50.0)

    def step(self, target_velocity: np.ndarray, dt: float) -> Tuple[np.ndarray, float]:
        """
        执行一步仿真
        :param target_velocity: 目标速度 [vx, vy, vz]
        :param dt: 时间步长
        :return: (新状态, 瞬时功率)
        """
        # 动力学滞后模拟 (一阶低通滤波, tau = 0.3s)
        tau = 0.3
        current_v = self.state[3:]

        # 计算新速度
        new_v = current_v + (target_velocity - current_v) * (dt / tau)

        # 速度截断 (安全限制)
        new_v = np.clip(new_v, -self.max_speed, self.max_speed)

        # 计算加速度 (用于功率计算)
        acc = (new_v - current_v) / dt

        # 更新位置 (欧拉积分)
        self.state[:3] += new_v * dt
        # 更新速度
        self.state[3:] = new_v

        # 计算功率
        power = self.compute_power(new_v, acc)

        return self.state.copy(), power


# ==========================================
# 2. 验证脚本逻辑
# ==========================================

def validate_physics():
    uav = MultirotorUAV()
    dt = 0.1
    print(f"=== 无人机参数 ===")
    print(f"质量: {uav.m} kg")
    print(f"理论悬停机械功率: {uav.hover_power_theoretical:.2f} W")
    print(f"预估悬停电功率 (eff=0.85): {uav.hover_power_elec:.2f} W")
    print("=" * 30)

    # --- 测试 1: 速度-功率曲线 (水平飞行) ---
    print("\n[Test 1] 生成水平飞行 P-V 曲线...")
    speeds = np.linspace(0, 20, 50)
    powers = []

    for v_x in speeds:
        # 重置位置，高度设为-10m
        uav.reset(np.array([0, 0, -10.0]))

        target = np.array([v_x, 0, 0])
        # 预热 3秒 (30 steps) 让速度稳定，消除加速度功率的影响
        for _ in range(30):
            uav.step(target, dt)

        # 记录稳定飞行时的功率
        _, p = uav.step(target, dt)
        powers.append(p)

    # --- 测试 2: 垂直运动测试 (爬升 vs 下降) ---
    print("[Test 2] 垂直运动测试 (爬升/悬停/下降)...")
    t_vertical = []
    v_z_act = []
    p_vertical = []

    uav.reset(np.array([0, 0, -10.0]))

    steps = 300  # 30秒
    for i in range(steps):
        t = i * dt

        # 0-10s: 悬停
        if t < 10:
            cmd = np.array([0, 0, 0])
        # 10-20s: 爬升 3m/s (NED: -3)
        elif t < 20:
            cmd = np.array([0, 0, -3])
        # 20-30s: 下降 3m/s (NED: +3)
        else:
            cmd = np.array([0, 0, 3])

        state, p = uav.step(cmd, dt)

        t_vertical.append(t)
        v_z_act.append(state[5])  # vz
        p_vertical.append(p)

    # --- 测试 3: 阶跃响应 (惯性测试) ---
    print("[Test 3] 速度阶跃响应测试...")
    t_step = []
    v_x_act = []

    uav.reset(np.array([0, 0, -10.0]))

    for i in range(100):  # 10秒
        t = i * dt
        # 1秒时给出 10m/s 指令
        cmd = np.array([10, 0, 0]) if t > 1.0 else np.array([0, 0, 0])
        state, _ = uav.step(cmd, dt)
        t_step.append(t)
        v_x_act.append(state[3])  # vx

    # ==========================================
    # 3. 绘图验证
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图1: P-V 曲线
    ax1 = axes[0, 0]
    ax1.plot(speeds, powers, 'b-', linewidth=2)
    ax1.set_title("Test 1: Power vs Horizontal Speed (Bucket Curve)")
    ax1.set_xlabel("Speed (m/s)")
    ax1.set_ylabel("Power (W)")
    ax1.grid(True)

    # 寻找最佳巡航速度（功率最低点）
    min_idx = np.argmin(powers)
    min_power = powers[min_idx]
    best_speed = speeds[min_idx]
    ax1.plot(best_speed, min_power, 'ro', label=f'Best Endurance: {best_speed:.1f}m/s')

    ax1.text(0.5, powers[0], "Hover", color='blue', fontweight='bold')
    ax1.legend()

    # 图2: 垂直速度跟踪
    ax2 = axes[0, 1]
    ax2.plot(t_vertical, v_z_act, 'k-', label='Actual Vz')
    # 绘制参考线
    ax2.plot([0, 10, 10, 20, 20, 30], [0, 0, -3, -3, 3, 3], 'r--', alpha=0.5, label='Command')
    ax2.set_title("Test 2: Vertical Velocity (NED Frame)")
    ax2.set_ylabel("Vz (m/s) [Neg=Up, Pos=Down]")
    ax2.grid(True)
    ax2.legend()

    # 图3: 垂直功率变化
    ax3 = axes[1, 0]
    ax3.plot(t_vertical, p_vertical, 'g-', linewidth=2)
    ax3.set_title("Test 2: Vertical Power Consumption")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Power (W)")
    ax3.grid(True)
    # 标注区域
    ax3.axvspan(10, 20, color='red', alpha=0.1, label='Climb Phase')
    ax3.axvspan(20, 30, color='green', alpha=0.1, label='Descent Phase')
    ax3.legend()

    # 验证逻辑：爬升功率应显著大于下降功率
    # 索引转换: dt=0.1, 10s=100 steps
    avg_hover = np.mean(p_vertical[50:90])  # 5-9s
    avg_climb = np.mean(p_vertical[150:190])  # 15-19s
    avg_descent = np.mean(p_vertical[250:290])  # 25-29s

    print(f"\n[验证结果]")
    print(f"悬停平均功率: {avg_hover:.1f} W")
    print(f"爬升平均功率: {avg_climb:.1f} W (预期: > 悬停)")
    print(f"下降平均功率: {avg_descent:.1f} W (预期: ≈ 悬停或略低)")

    if avg_climb > avg_hover + 100:
        print(">> ✅ 爬升物理模型正常 (克服重力做功)")
    else:
        print(">> ❌ 爬升模型异常 (功率未显著增加)")

    if avg_descent < avg_climb:
        print(">> ✅ 下降物理模型正常 (重力辅助)")

    # 图4: 阶跃响应
    ax4 = axes[1, 1]
    ax4.plot(t_step, v_x_act, 'b-', label='Response')
    ax4.axvline(1.0, color='k', linestyle=':', label='Step Input (10m/s)')
    # 计算63.2%响应时间点 (Tau)
    target_val = 10.0
    tau_val = target_val * 0.632
    ax4.axhline(tau_val, color='r', linestyle=':', alpha=0.5, label='63.2% Target')
    ax4.set_title("Test 3: Step Response (Inertia Check)")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Vx (m/s)")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    validate_physics()