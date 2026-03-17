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


# ========================
# 地图生成工具 (从 urban_env.py 迁移)
# ========================
# 全局固定地图开关
_FIXED_MAP_ENABLE = False

def set_fixed_map_flag(enable: bool):
    """设置是否启用固定地图的全局开关"""
    global _FIXED_MAP_ENABLE
    _FIXED_MAP_ENABLE = enable

def _is_valid_placement(new_block, existing_blocks, start_pos, target_pos, min_building_dist):
    """检测新生成的方块是否有效（内部辅助函数）"""
    cx, cy, w, l, _ = new_block

    # --- A. 保护区检测 (Start/Target Safety Zone) ---
    # 保护半径
    safe_radius = 60.0

    # 计算新楼房的矩形边界 (AABB)
    min_bx, max_bx = cx - w / 2, cx + w / 2
    min_by, max_by = cy - l / 2, cy + l / 2

    # 检查是否覆盖 Start 点
    if (min_bx - safe_radius < start_pos[0] < max_bx + safe_radius) and \
            (min_by - safe_radius < start_pos[1] < max_by + safe_radius):
        return False

    # 检查是否覆盖 Target 点
    if (min_bx - safe_radius < target_pos[0] < max_bx + safe_radius) and \
            (min_by - safe_radius < target_pos[1] < max_by + safe_radius):
        return False

    # --- B. 建筑物间距检测 (Overlap Check) ---
    for ex_block in existing_blocks:
        ex_cx, ex_cy, ex_w, ex_l, _ = ex_block

        # 两个矩形是否重叠逻辑:
        # 如果 (RectA.left < RectB.right) && (RectA.right > RectB.left) ...
        # 我们加上 min_building_dist 作为缓冲区
        buffer = min_building_dist

        # 检查 X 轴投影是否重叠 (带缓冲)
        overlap_x = (abs(cx - ex_cx) * 2) < (w + ex_w + buffer * 2)
        # 检查 Y 轴投影是否重叠 (带缓冲)
        overlap_y = (abs(cy - ex_cy) * 2) < (l + ex_l + buffer * 2)

        if overlap_x and overlap_y:
            return False  # 发生重叠或间距过小

    return True

def generate_city_blocks(n, start_pos, target_pos, min_building_dist, seed=None):
    """
    生成互不重叠的城市建筑 (Rejection Sampling)
    全局复用版本，从 urban_env.py 迁移而来
    :param n: 障碍物数量
    :param start_pos: 起点坐标
    :param target_pos: 终点坐标
    :param min_building_dist: 楼宇最小间距
    :param seed: 随机种子（固定地图时生效）
    :return: 障碍物列表
    """
    # 如果启用固定地图，强制使用指定seed
    if _FIXED_MAP_ENABLE and seed is None:
        seed = 42  # 默认固定种子

    # 初始化随机数生成器
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    obs = []

    # 定义地图生成范围 (在起点终点周围扩展)
    margin = 100
    min_x = min(start_pos[0], target_pos[0]) - margin
    max_x = max(start_pos[0], target_pos[0]) + margin
    min_y = min(start_pos[1], target_pos[1]) - margin
    max_y = max(start_pos[1], target_pos[1]) + margin

    # 最大尝试次数，防止死循环
    max_attempts = n * 50
    attempts = 0

    while len(obs) < n and attempts < max_attempts:
        attempts += 1

        # 1. 随机生成尺寸 (更加多样化)
        # 区分高瘦楼(写字楼)和矮胖楼(商场)
        if rng.random() < 0.3:
            # 高楼
            width = rng.uniform(30, 60)
            length = rng.uniform(30, 60)
            height = rng.uniform(100, 250)
        else:
            # 普通建筑
            width = rng.uniform(50, 120)
            length = rng.uniform(50, 120)
            height = rng.uniform(40, 120)

        # 2. 随机位置
        cx = rng.uniform(min_x, max_x)
        cy = rng.uniform(min_y, max_y)

        new_block = np.array([cx, cy, width, length, height])

        # 3. 冲突检测
        if not _is_valid_placement(new_block, obs, start_pos, target_pos, min_building_dist):
            continue

        obs.append(new_block)

    if len(obs) < n:
        print(f"Warning: Only placed {len(obs)}/{n} obstacles due to space constraints.")

    return obs





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