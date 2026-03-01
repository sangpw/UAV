# utils.py
import numpy as np

# 尝试导入 airsim，如果未安装则跳过，防止报错
try:
    import airsim

    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False


def apply_turbulence_and_gusts(load_profile, dt, intensity=1.0):
    """
    给光滑的功率曲线添加风阻和湍流
    :param intensity: 强度系数
    """
    steps = len(load_profile)
    t = np.linspace(0, steps * dt, steps)

    # 1. 高频湍流 (White Noise) - 模拟空气的不稳定性
    turbulence = np.random.normal(0, 15 * intensity, steps)

    # 2. 低频阵风 (Low Frequency Drift) - 模拟侧风或逆风变化
    # 使用几个不同频率的正弦波叠加
    gusts = 30 * np.sin(2 * np.pi * 0.01 * t) + \
            15 * np.sin(2 * np.pi * 0.05 * t + 1)

    # 3. 叠加
    noisy_load = load_profile + turbulence + gusts * intensity

    # 4. 物理约束：功率不能小于待机功率 (防止出现负值)
    noisy_load = np.clip(noisy_load, 50, None)

    return noisy_load


# ==========================================
# 复杂工况生成器：测绘/巡检任务
# ==========================================
def generate_complex_profile(total_time, dt):
    """
    生成高复杂度的无人机工况：包含起飞、多段巡航、转弯机动、悬停、降落
    """
    steps = int(total_time / dt)
    t = np.linspace(0, total_time, steps)
    load = np.zeros_like(t)

    # 定义基础功率水平 (W)
    P_IDLE = 50
    P_TAKEOFF = 950
    P_CLIMB = 750
    P_CRUISE = 450
    P_TURN = 600  # 转弯时需要增加推力维持高度
    P_HOVER = 500
    P_DESCENT = 350

    # --- 阶段规划 (按时间切片) ---
    # 假设总时长 600s

    # 1. 0% - 5%: 待机与自检
    idx_1 = int(steps * 0.05)
    load[:idx_1] = P_IDLE

    # 2. 5% - 10%: 起飞 (最大功率)
    idx_2 = int(steps * 0.10)
    load[idx_1:idx_2] = P_TAKEOFF

    # 3. 10% - 20%: 爬升到任务高度
    idx_3 = int(steps * 0.20)
    load[idx_2:idx_3] = P_CLIMB

    # 4. 20% - 80%: 巡航任务 (加入“弓”字形走线，包含多次转弯)
    idx_4 = int(steps * 0.80)

    # 基础巡航
    load[idx_3:idx_4] = P_CRUISE

    # 在巡航段模拟“周期性转弯”
    # 假设每隔 60秒 转弯一次，转弯持续 8秒
    turn_interval = 60  # 秒
    turn_duration = 8  # 秒

    cruise_start_time = t[idx_3]
    cruise_end_time = t[idx_4]

    current_t = cruise_start_time
    while current_t < cruise_end_time:
        # 找到转弯的时间索引范围
        t_start_turn = current_t + turn_interval
        if t_start_turn + turn_duration > cruise_end_time:
            break

        idx_turn_start = int(t_start_turn / dt)
        idx_turn_end = int((t_start_turn + turn_duration) / dt)

        # 叠加转弯功率 (模拟掉头时的侧倾和加速)
        load[idx_turn_start:idx_turn_end] = P_TURN

        current_t = t_start_turn

    # 5. 80% - 90%: 悬停拍照/投掷 (定点作业)
    idx_5 = int(steps * 0.90)
    load[idx_4:idx_5] = P_HOVER

    # 6. 90% - 98%: 返航降落
    idx_6 = int(steps * 0.98)
    load[idx_5:idx_6] = P_DESCENT

    # 7. 98% - 100%: 落地待机
    load[idx_6:] = P_IDLE

    # --- 后处理：添加环境噪声 ---
    load = apply_turbulence_and_gusts(load, dt, intensity=0.8)

    # --- 后处理：平滑滤波 (模拟电机惯性) ---
    # 真实的电机功率不会呈阶跃变化
    window_size = int(1.0 / dt)  # 1秒的移动平均
    if window_size > 0:
        kernel = np.ones(window_size) / window_size
        load = np.convolve(load, kernel, mode='same')

    return t, load


# 为了兼容旧代码，保留原函数名，但内部指向新函数 (可选)
def generate_flight_profile(total_time, dt):
    return generate_complex_profile(total_time, dt)


# ==========================================
# 第二部分：AirSim 接口封装 (未来使用)
# ==========================================

class AirSimBridge:
    """
    负责与 AirSim 建立连接，并将飞行状态转换为功率数据
    """

    def __init__(self, ip="", port=41451, prop_coeff=1e-8, avionics_power=50.0):
        if not AIRSIM_AVAILABLE:
            raise ImportError("AirSim library not found. Please install using 'pip install airsim'")

        self.client = airsim.MultirotorClient(ip=ip, port=port)
        self.prop_coeff = prop_coeff  # 螺旋桨功率系数 (需标定)
        self.avionics_power = avionics_power  # 航电功耗 (W)
        self.is_connected = False

    def connect(self):
        try:
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.is_connected = True
            print("[AirSimBridge] Connected successfully.")
        except Exception as e:
            print(f"[AirSimBridge] Connection failed: {e}")
            self.is_connected = False

    def get_current_power_demand(self):
        """
        获取当前时刻的功率需求 (W)
        原理: P_total = sum(k * w^3) + P_avionics
        """
        if not self.is_connected:
            return 0.0

        # 获取电机状态
        try:
            rotor_states = self.client.getRotorStates()
            total_prop_power = 0.0

            # 假设是4旋翼
            for i in range(4):
                # 获取转速 (AirSim 返回的通常是 rad/s 或 RPM，需根据模型确认)
                # 这里假设返回的是 rad/s
                w = rotor_states.rotors[i]['speed']

                # 过滤极小值防止噪音
                if w < 0: w = 0

                # 机械功率 P = K * w^3
                total_prop_power += self.prop_coeff * (w ** 3)

            # 考虑电机效率 (如 85%) 转换为电功率
            elec_power = total_prop_power / 0.85

            return elec_power + self.avionics_power

        except Exception as e:
            print(f"Error reading AirSim data: {e}")
            return 0.0

    def land_vehicle(self):
        """紧急降落指令"""
        if self.is_connected:
            self.client.landAsync()