import numpy as np
import heapq
import math
import torch
from typing import List, Optional

# 尝试导入 SAC 模型，如果不存在则通过，避免影响其他规划器运行
try:
    from rl_core import ActorSAC
except ImportError:
    # 仅作为静默处理，只有在实例化 SACPlanner 时才会报错
    ActorSAC = None


class BasePlanner:
    """路径规划器基类"""

    def compute_velocity_command(self,
                                 current_pos: np.ndarray,
                                 current_vel: np.ndarray,
                                 target_pos: np.ndarray,
                                 obstacles: List[np.ndarray],
                                 power_state: dict,
                                 dt: float,
                                 future_trajectory: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError


class RuleBasedPlanner(BasePlanner):
    """
    基于规则的规划器 (人工势场法 + NED坐标系适配)
    """

    def __init__(self, max_speed: float = 15.0, arrive_radius: float = 10.0):
        self.max_speed = max_speed
        self.arrive_radius = arrive_radius

    def compute_velocity_command(self, current_pos, current_vel, target_pos, obstacles,
                                 power_state, dt, future_trajectory=None):
        # 1. 吸引力 (飞向目标)
        error = target_pos - current_pos
        dist = np.linalg.norm(error)

        if dist < self.arrive_radius:
            return np.zeros(3)

        # 比例控制
        desired_vel = (error / dist) * self.max_speed

        # 2. 斥力 (避障)
        repulsive_force = np.zeros(3)

        for obs in obstacles:
            # 解析新环境的障碍物格式: [cx, cy, w, l, h]
            cx, cy, w, l, h = obs

            # 将长方体简化为圆柱体进行快速避障
            # 半径取宽长的平均值的一半，再加一点余量
            radius = (w + l) / 4.0
            # 障碍物中心 (NED坐标系: z 从 0 到 -h)
            # 我们关注水平避障
            obs_center_2d = np.array([cx, cy])
            curr_pos_2d = current_pos[:2]

            dist_vec_2d = curr_pos_2d - obs_center_2d
            dist_val_2d = np.linalg.norm(dist_vec_2d)

            # 高度判断：如果无人机已经飞得比楼顶还高 (z < -h)，则不需要避障
            # 注意 NED: z越小越高。例如 楼顶-100, 无人机-120.
            # 加上 10m 的垂直安全余量
            if current_pos[2] < (-h - 10.0):
                continue

            safe_dist = radius + 25.0  # 安全半径 (稍微加大一点)

            if dist_val_2d < safe_dist:
                # 产生水平斥力
                rep_dir = dist_vec_2d / (dist_val_2d + 1e-6)
                # 距离越近，斥力越大
                strength = 4.0 * (safe_dist - dist_val_2d)
                repulsive_force[:2] += rep_dir * strength

        # 3. 合成速度
        final_cmd = desired_vel + repulsive_force

        # 4. 高度保持/限制
        # 如果斥力太大导致合力偏离目标太远，稍微增加向上爬升的倾向以越过障碍
        if np.linalg.norm(repulsive_force) > 5.0:
            # NED: 负数是向上。尝试向上爬升避障
            final_cmd[2] -= 3.0

            # 5. 限速
        v_norm = np.linalg.norm(final_cmd)
        if v_norm > self.max_speed:
            final_cmd = final_cmd / v_norm * self.max_speed

        return final_cmd


class AStarPlanner(BasePlanner):
    """
    适配 NED 坐标系与 3D 障碍物的稳健 A* 规划器
    """

    def __init__(self, grid_res: float = 20.0, replan_interval: int = 20, safety_margin: float = 10.0):
        self.res = grid_res
        self.replan_interval = replan_interval
        self.margin = safety_margin

        self.replan_step_count = 0
        self.path_queue = []
        self.current_waypoint_idx = 0

        # 地图缓存
        self.grid = None
        self.min_bounds = None
        self.grid_shape = None
        self.initialized = False

    def _init_grid(self, start_pos, target_pos, obstacles):
        """动态构建 3D 占据栅格地图"""
        # 1. 确定边界
        all_x = [start_pos[0], target_pos[0]]
        all_y = [start_pos[1], target_pos[1]]
        # NED: 地面是0, 楼顶是 -h. 这是一个从负数到0的区间
        all_z = [start_pos[2], target_pos[2], 0]

        for o in obstacles:
            cx, cy, w, l, h = o
            all_x.extend([cx - w / 2, cx + w / 2])
            all_y.extend([cy - l / 2, cy + l / 2])
            all_z.append(-h)

        buffer = 60.0  # 边界余量
        self.min_bounds = np.array([min(all_x) - buffer, min(all_y) - buffer, min(all_z) - buffer])
        self.max_bounds = np.array([max(all_x) + buffer, max(all_y) + buffer, 10.0])  # 上限略高于地面

        self.grid_shape = np.ceil((self.max_bounds - self.min_bounds) / self.res).astype(int)
        self.grid = np.zeros(self.grid_shape, dtype=np.uint8)

        print(f"[A*] Initializing Grid: Shape={self.grid_shape}, Res={self.res}m")

        # 2. 障碍物栅格化
        for o in obstacles:
            cx, cy, w, l, h = o
            # 障碍物 AABB (NED: z in [-h, 0])
            ox_min = cx - w / 2 - self.margin
            ox_max = cx + w / 2 + self.margin
            oy_min = cy - l / 2 - self.margin
            oy_max = cy + l / 2 + self.margin
            oz_min = -h - self.margin  # 楼顶上方一点
            oz_max = 0  # 地面

            idx_x_min, idx_x_max = self._pos_to_idx_range(ox_min, ox_max, 0)
            idx_y_min, idx_y_max = self._pos_to_idx_range(oy_min, oy_max, 1)
            idx_z_min, idx_z_max = self._pos_to_idx_range(oz_min, oz_max, 2)

            self.grid[idx_x_min:idx_x_max, idx_y_min:idx_y_max, idx_z_min:idx_z_max] = 1

        # 3. 地面栅格化 (z > 0)
        # [FIX] 修复了这里的解包错误
        g_z_min, g_z_max = self._pos_to_idx_range(0, 100, 2)

        if g_z_min < self.grid_shape[2]:
            self.grid[:, :, g_z_min:] = 1  # 地面以下全堵死

        self.initialized = True

    def _pos_to_idx(self, pos):
        idx = np.floor((pos - self.min_bounds) / self.res).astype(int)
        for i in range(3):
            idx[i] = np.clip(idx[i], 0, self.grid_shape[i] - 1)
        return tuple(idx)

    def _idx_to_pos(self, idx):
        return self.min_bounds + (np.array(idx) + 0.5) * self.res

    def _pos_to_idx_range(self, v_min, v_max, dim):
        i_min = int(np.floor((v_min - self.min_bounds[dim]) / self.res))
        i_max = int(np.ceil((v_max - self.min_bounds[dim]) / self.res))
        return max(0, i_min), min(self.grid_shape[dim], i_max)

    def _astar_search(self, start_idx, target_idx):
        if self.grid[start_idx] == 1:
            # 起点在障碍物内，A* 无法开始
            # 简单策略：向 z 轴负方向（向上）搜索最近的自由点
            print("[A*] Start in obstacle. Searching for free space upwards...")
            curr = list(start_idx)
            found = False
            for z in range(curr[2], -1, -1):
                if self.grid[curr[0], curr[1], z] == 0:
                    start_idx = (curr[0], curr[1], z)
                    found = True
                    break
            if not found:
                return None

        open_set = []
        heapq.heappush(open_set, (0, start_idx))
        came_from = {}
        g_score = {start_idx: 0}

        # 3D 邻域 (26连通)
        neighbors = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if
                     not (dx == 0 and dy == 0 and dz == 0)]

        max_steps = 50000
        steps = 0

        while open_set:
            _, current = heapq.heappop(open_set)
            steps += 1

            if current == target_idx:
                path = []
                while current in came_from:
                    path.append(self._idx_to_pos(current))
                    current = came_from[current]
                path.reverse()
                return path

            if steps > max_steps:
                print("[A*] Search timeout.")
                return None

            for dx, dy, dz in neighbors:
                nxt = (current[0] + dx, current[1] + dy, current[2] + dz)

                # 越界检查
                if not (0 <= nxt[0] < self.grid_shape[0] and
                        0 <= nxt[1] < self.grid_shape[1] and
                        0 <= nxt[2] < self.grid_shape[2]):
                    continue

                # 碰撞检查
                if self.grid[nxt] == 1:
                    continue

                dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                new_g = g_score[current] + dist

                if nxt not in g_score or new_g < g_score[nxt]:
                    g_score[nxt] = new_g
                    # 启发式函数乘 1.001 打破对称性
                    h_val = math.sqrt(sum((np.array(nxt) - np.array(target_idx)) ** 2))
                    priority = new_g + h_val * 1.001
                    came_from[nxt] = current
                    heapq.heappush(open_set, (priority, nxt))
        return None

    def compute_velocity_command(self, current_pos, current_vel, target_pos, obstacles, power_state, dt, **kwargs):
        # 初始化
        if not self.initialized:
            self._init_grid(current_pos, target_pos, obstacles)

        # 靠近目标时直接飞
        if np.linalg.norm(current_pos - target_pos) < 20.0:
            return (target_pos - current_pos)

        # 重规划逻辑
        need_replan = (not self.path_queue) or (self.replan_step_count >= self.replan_interval)

        if need_replan:
            self.replan_step_count = 0
            start_idx = self._pos_to_idx(current_pos)
            target_idx = self._pos_to_idx(target_pos)

            new_path = self._astar_search(start_idx, target_idx)
            if new_path:
                self.path_queue = new_path
                self.current_waypoint_idx = 0
                # print(f"[A*] Replan success. Path Len: {len(new_path)}")
            else:
                print("[A*] Replan failed. Climbing up.")
                # 失败策略：如果高度不够高，先向上爬升 (NED 负方向)
                if current_pos[2] > -150:
                    return np.array([0, 0, -3.0])
                # 否则直接朝目标冲（死马当活马医）
                else:
                    dir_vec = target_pos - current_pos
                    return dir_vec / np.linalg.norm(dir_vec) * 5.0

        self.replan_step_count += 1

        # 路径跟踪 (Pure Pursuit)
        if self.path_queue:
            # 寻找预瞄点 (Looking ahead)
            lookahead_idx = min(self.current_waypoint_idx + 2, len(self.path_queue) - 1)
            target_pt = self.path_queue[lookahead_idx]

            # 检查是否通过了当前点
            dist_to_curr = np.linalg.norm(current_pos - self.path_queue[self.current_waypoint_idx])
            if dist_to_curr < 15.0:
                self.current_waypoint_idx = min(self.current_waypoint_idx + 1, len(self.path_queue) - 1)

            direction = target_pt - current_pos
            if np.linalg.norm(direction) > 0.1:
                return (direction / np.linalg.norm(direction)) * 15.0

        return np.zeros(3)


class SACPlanner(BasePlanner):
    """
    SAC 深度强化学习规划器
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        if ActorSAC is None:
            raise ImportError("Cannot use SACPlanner because rl_core.py is missing.")

        self.device = torch.device(device)
        self.max_speed = 15.0
        self.state_dim = 11
        self.action_dim = 3

        # 初始化网络
        self.actor = ActorSAC(self.state_dim, self.action_dim, max_action=1.0).to(self.device)

        try:
            self.actor.load_state_dict(torch.load(model_path, map_location=self.device))
            self.actor.eval()
            print(f"[SACPlanner] Model loaded from {model_path}")
        except Exception as e:
            print(f"[SACPlanner] Error loading model: {e}")
            print("[SACPlanner] Running with UNTRAINED random weights!")

    def compute_velocity_command(self, current_pos, current_vel, target_pos, obstacles,
                                 power_state, dt, future_trajectory=None):

        # 构建观测向量 (需要与训练时的 Env._get_obs 保持一致)
        # 假设训练环境也做了类似的归一化
        to_target = target_pos - current_pos

        obs = np.concatenate([
            current_pos / 1000.0,  # 位置归一化
            current_vel / 15.0,  # 速度归一化
            to_target / 1000.0,  # 相对距离归一化
            [power_state.get('soc', 0.6)],
            [power_state.get('h2_cum', 0) / 100.0]
        ]).astype(np.float32)

        # 维度对齐
        if len(obs) != self.state_dim:
            obs = np.resize(obs, self.state_dim)

        state_t = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)

        with torch.no_grad():
            # 推理模式 deterministic=True
            action = self.actor.get_action(state_t, deterministic=True).cpu().data.numpy().flatten()

        return action * self.max_speed