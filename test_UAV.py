import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. 修正后的无人机物理模型 (包含在此脚本中以便独立运行)
# ==========================================
class MultirotorUAV_Fixed:
    def __init__(self, mass=5.0, rotor_radius=0.15, num_rotors=4):
        self.m = mass
        self.R = rotor_radius
        self.n_rotors = num_rotors
        self.rho = 1.225

        # 状态: [x, y, z, vx, vy, vz] (NED坐标系: z向下为正)
        self.state = np.zeros(6)
        self.A_rotor = np.pi * rotor_radius ** 2

        # 理论悬停功率计算 (动量理论)
        self.thrust_hover = self.m * 9.81
        self.hover_power_theoretical = (self.thrust_hover ** 1.5) / \
                                       np.sqrt(2 * self.n_rotors * self.rho * self.A_rotor)

        # 考虑电机效率
        self.hover_power_elec = self.hover_power_theoretical / 0.85

    def reset(self):
        self.state = np.zeros(6)
        # 初始高度设为 -10m (空中)
        self.state[2] = -10.0

    def compute_power(self, velocity: np.ndarray, acceleration: np.ndarray) -> float:
        v = np.linalg.norm(velocity)

        # 1. 诱导功率 (Induced Power) - 随前飞速度增加而降低
        v_tip = 150.0
        mu = v / v_tip
        # 简单的诱导功率修正系数
        if mu < 0.1:
            k_ind = 1.0
        else:
            k_ind = 1.15  # 高速下湍流增加

        # 核心公式：随速度增加，诱导功率下降 (Translational Lift)
        # max(0.2, ...) 防止功率降得太低
        P_ind = self.hover_power_theoretical * k_ind * max(0.2, 1.0 - (mu * 15))
        # 注意：原代码的公式比较简单，这里为了验证物理特性，使用原代码逻辑，
        # 但原代码: max(0.5, 1.0 - 0.5 * mu ** 2) 下降得不够快，我们保持原逻辑进行验证。
        P_ind = self.hover_power_theoretical * k_ind * max(0.5, 1.0 - 0.5 * mu ** 2)

        # 2. 型阻功率 (Profile Drag) - 随速度立方增加
        # 假设 Cd_mean = 0.05 (桨叶阻力系数)
        # P = 1/8 * rho * N * A * Cd * v_tip^3 ... 简化为:
        P_prof = 0.1 * self.rho * self.n_rotors * self.A_rotor * (v ** 3)
        # 原代码系数修正：
        P_prof = 0.1 * self.rho * self.n_rotors * self.A_rotor * (v ** 3)

        # 3. 爬升/重力功率 [已修正逻辑]
        # NED坐标系: z轴向下。velocity[2] < 0 代表向上爬升。
        v_vertical = velocity[2]
        P_climb = 0
        if v_vertical < 0:  # 向上飞
            # P = F * v
            P_climb = self.m * 9.81 * abs(v_vertical)

        # 4. 机动/加速度功率
        P_accel = self.m * np.linalg.norm(acceleration) * v * 0.5

        # 总功率 / 效率 + 待机功耗
        total = (P_ind + P_prof + P_climb + P_accel) / 0.85
        return max(total, 50.0)

    def step(self, target_vel, dt):
        # 动力学滞后模拟 (tau = 0.3s)
        curr_v = self.state[3:]
        tau = 0.3

        # 一阶低通滤波
        new_v = curr_v + (target_vel - curr_v) * (dt / tau)

        acc = (new_v - curr_v) / dt
        self.state[3:] = new_v
        self.state[:3] += new_v * dt

        power = self.compute_power(new_v, acc)
        return self.state, power


# ==========================================
# 2. 验证脚本逻辑
# ==========================================

def validate_physics():
    uav = MultirotorUAV_Fixed()
    dt = 0.1
    print(f"=== 无人机参数 ===")
    print(f"质量: {uav.m} kg")
    print(f"理论悬停机械功率: {uav.hover_power_theoretical:.2f} W")
    print(f"预估悬停电功率 (eff=0.85): {uav.hover_power_elec + 50:.2f} W (含待机)")
    print("=" * 30)

    # --- 测试 1: 速度-功率曲线 (水平飞行) ---
    print("\n[Test 1] 生成水平飞行 P-V 曲线...")
    speeds = np.linspace(0, 20, 50)
    powers = []

    for v_x in speeds:
        uav.reset()
        # 让无人机稳定在这个速度
        target = np.array([v_x, 0, 0])
        # 预热 2秒让速度稳定
        for _ in range(20): uav.step(target, dt)

        _, p = uav.step(target, dt)
        powers.append(p)

    # --- 测试 2: 垂直运动测试 (爬升 vs 下降) ---
    print("[Test 2] 垂直运动测试 (爬升/悬停/下降)...")
    t_vertical = []
    v_z_act = []
    p_vertical = []
    uav.reset()

    steps = 300  # 30秒
    for i in range(steps):
        t = i * dt

        # 0-10s: 悬停
        if t < 10:
            cmd = np.array([0, 0, 0])
            status = "Hover"
        # 10-20s: 爬升 3m/s (NED: -3)
        elif t < 20:
            cmd = np.array([0, 0, -3])
            status = "Climb"
        # 20-30s: 下降 3m/s (NED: +3)
        else:
            cmd = np.array([0, 0, 3])
            status = "Descent"

        state, p = uav.step(cmd, dt)

        t_vertical.append(t)
        v_z_act.append(state[5])  # vz
        p_vertical.append(p)

    # --- 测试 3: 阶跃响应 (惯性测试) ---
    print("[Test 3] 速度阶跃响应测试...")
    t_step = []
    v_x_act = []
    uav.reset()
    for i in range(100):  # 10秒
        t = i * dt
        # 1秒时给出 10m/s 指令
        cmd = np.array([10, 0, 0]) if t > 1.0 else np.array([0, 0, 0])
        state, _ = uav.step(cmd, dt)
        t_step.append(t)
        v_x_act.append(state[3])

    # ==========================================
    # 3. 绘图验证
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图1: P-V 曲线
    ax1 = axes[0, 0]
    ax1.plot(speeds, powers, 'b-', linewidth=2)
    ax1.set_title("Test 1: Power vs Horizontal Speed")
    ax1.set_xlabel("Speed (m/s)")
    ax1.set_ylabel("Power (W)")
    ax1.grid(True)
    # 理论分析标记
    min_idx = np.argmin(powers)
    ax1.plot(speeds[min_idx], powers[min_idx], 'ro', label=f'Max Range Speed (~{speeds[min_idx]:.1f}m/s)')
    ax1.legend()
    ax1.text(0, powers[0] + 20, "Hover", color='blue')
    ax1.text(15, powers[-1] - 100, "Drag Dominates", color='blue')

    # 图2: 垂直速度跟踪
    ax2 = axes[0, 1]
    ax2.plot(t_vertical, v_z_act, 'k-', label='Actual Vz')
    ax2.plot(t_vertical, [0] * 100 + [-3] * 100 + [3] * 100, 'r--', alpha=0.5, label='Command')
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
    ax3.axvspan(10, 20, color='red', alpha=0.1, label='Climb')
    ax3.axvspan(20, 30, color='green', alpha=0.1, label='Descent')
    ax3.legend()

    # 验证逻辑：爬升功率应显著大于下降功率
    avg_hover = np.mean(p_vertical[50:90])
    avg_climb = np.mean(p_vertical[150:190])
    avg_descent = np.mean(p_vertical[250:290])

    print(f"\n[验证结果]")
    print(f"悬停平均功率: {avg_hover:.1f} W")
    print(f"爬升平均功率: {avg_climb:.1f} W (预期: > 悬停)")
    print(f"下降平均功率: {avg_descent:.1f} W (预期: ≈ 悬停或略低)")

    if avg_climb > avg_hover + 100:
        print(">> ✅ 爬升物理模型正常 (重力做功被计入)")
    else:
        print(">> ❌ 爬升模型异常 (功率未显著增加)")

    # 图4: 阶跃响应
    ax4 = axes[1, 1]
    ax4.plot(t_step, v_x_act, 'b-', label='Response')
    ax4.axvline(1.0, color='k', linestyle=':', label='Step Input')
    ax4.axvline(1.0 + 0.3, color='r', linestyle='--', label='Tau (0.3s)')
    ax4.axhline(10 * 0.632, color='r', linestyle=':', alpha=0.5, label='63.2% Target')
    ax4.set_title("Test 3: Step Response (Inertia Check)")
    ax4.set_xlabel("Time (s)")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    validate_physics()