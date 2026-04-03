# utils.py
import numpy as np
from math import inf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


def plot_uav_path(start_pos, target_pos, obstacles, path_history, title="UAV Flight Path (Alt = -Z)"):
    """
    修正后的 3D 路径规划可视化函数
    将 NED 坐标系转换成视觉直观的 Alt (高度) 坐标
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 定义转换函数：NED 中的 Z 转换成绘图中的高度 H (H = -Z)
    def to_alt(z):
        return -z

    # 1. 绘制起点和终点 (Z 取反)
    ax.scatter(start_pos[0], start_pos[1], to_alt(start_pos[2]),
               c='g', s=100, label='Start (Takeoff)', depthshade=False)
    ax.scatter(target_pos[0], target_pos[1], to_alt(target_pos[2]),
               c='r', s=150, label='Target (Landing)', marker='*', depthshade=False)

    # 2. 绘制建筑物
    for o in obstacles:
        cx, cy, w, l, h = o
        x_anchor = cx - w / 2
        y_anchor = cy - l / 2

        # 在视觉直观图中：
        # 底座在地面 (Z_visual = 0)
        # 高度增加到 h (dZ_visual = h)
        color = 'navy' if h > 150 else 'gray'
        ax.bar3d(x_anchor, y_anchor, 0, w, l, h,
                 color=color, alpha=0.4, edgecolor='k', linewidth=0.5)

    # 3. 绘制飞行轨迹
    if path_history is not None and len(path_history) > 0:
        path = np.array(path_history)
        # 将路径中的 Z 坐标全部取反
        visual_path_z = to_alt(path[:, 2])

        ax.plot(path[:, 0], path[:, 1], visual_path_z,
                c='magenta', linewidth=2.5, label='Trajectory', alpha=0.9)

        # 标记当前终点位置
        ax.scatter(path[-1, 0], path[-1, 1], visual_path_z[-1], c='orange', s=50)

    # 4. 视图设置
    mid_x = (start_pos[0] + target_pos[0]) / 2
    mid_y = (start_pos[1] + target_pos[1]) / 2

    # 动态调整显示范围
    range_val = max(abs(target_pos[0] - start_pos[0]), abs(target_pos[1] - start_pos[1])) / 1.5 + 100
    ax.set_xlim(mid_x - range_val, mid_x + range_val)
    ax.set_ylim(mid_y - range_val, mid_y + range_val)

    # 设置 Z 轴（高度）从 0 开始往上
    ax.set_zlim(0, 300)

    ax.set_xlabel('North (X) [m]')
    ax.set_ylabel('East (Y) [m]')
    ax.set_zlabel('Altitude (-Z) [m]')  # 明确标注 Z 轴已取反
    ax.set_title(title)
    ax.legend()

    ax.view_init(elev=25, azim=-135)  # 调整到一个更好的观察角度
    plt.show()



def check_collision(pos: np.ndarray, obstacles: list) -> bool:
    """
    通用碰撞检测函数 (NED 坐标系)
    :param pos: 无人机当前位置 [x, y, z]
    :param obstacles: 障碍物列表，每个元素为 [cx, cy, w, l, h]
    :return: True 表示碰撞，False 表示安全
    """
    # 1. 地面碰撞检测
    # NED坐标系中，Z轴向下为正，地面为 Z=0。因此 Z > 0 表示进入地下或触地
    if pos[2] > 0:
        return True

    # 2. 建筑物碰撞检测
    for obs in obstacles:
        cx, cy, w, l, h = obs

        # 计算 AABB (轴对齐包围盒) 边界
        xmin, xmax = cx - w / 2, cx + w / 2
        ymin, ymax = cy - l / 2, cy + l / 2

        # 建筑物顶部在 NED 中的坐标是 -h (因为高度是向上，即负方向)
        # 如果 Z > -h，说明无人机的高度低于楼顶
        if (xmin <= pos[0] <= xmax) and \
                (ymin <= pos[1] <= ymax) and \
                (pos[2] > -h):
            return True

    return False


def compute_hierarchical_reward(info, prev_info, action, terminated, truncated):
    # 1. 时间与进度奖励 (目标：最短时间)
    curr_dist = info['distance']
    prev_dist = prev_info['distance']
    reward_progress = (prev_dist - curr_dist) * 15.0  # 进度
    reward_time = -2.0  # 步时长惩罚 (鼓励尽快到达)

    # 2. 能量消耗奖励 (目标：最少能耗)
    # 计算当前步的氢气消耗 (g)
    h2_step = info['h2_total'] - prev_info['h2_total']
    reward_energy = -h2_step * 500.0  # 权重需根据H2价格与时间的价值折算

    # 3. 寿命损耗惩罚 (目标：燃料电池耐久性)
    # 从 info 中获取燃料电池当前功率 (需确保 env.step 返回此信息)
    curr_p_fc = info.get('fc_power', 0.0)
    prev_p_fc = prev_info.get('fc_power', 0.0)

    # (a) 功率变动率惩罚 (防止剧烈震荡)
    p_slew_rate = abs(curr_p_fc - prev_p_fc)
    reward_health = -0.05 * (p_slew_rate ** 2)

    # (b) 电池健康平衡 (防止深充深放)
    reward_soc = -50.0 * (abs(info['soc'] - 0.6) ** 2)

    # 4. 安全与终端奖励
    reward_terminal = 0
    if terminated:
        if curr_dist < 15.0:
            reward_terminal = 2000.0  # 大额成功奖
        else:
            reward_terminal = -2000.0  # 碰撞重罚

    # 5. 统筹求和
    total_reward = (
            reward_progress +  # 引导
            reward_time +  # 促使快
            reward_energy +  # 促使省
            reward_health +  # 促使稳(寿命)
            reward_soc +  # 促使续航稳定
            reward_terminal  # 促使安全
    )

    return total_reward