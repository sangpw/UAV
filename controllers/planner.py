# controllers/planner.py
import numpy as np
import torch
import heapq
import math
from typing import List, Optional

# 核心修改：导入统一的观测构造器，确保 SAC 输入维度一致
from rl_core import ObservationBuilder, ActorSAC


class BasePlanner:
    """路径规划器基类"""

    def compute_velocity_command(self,
                                 current_pos: np.ndarray,
                                 current_vel: np.ndarray,
                                 target_pos: np.ndarray,
                                 obstacles: List[np.ndarray],
                                 power_state: dict,
                                 dt: float,
                                 **kwargs) -> np.ndarray:
        """
        统一接口定义
        :param kwargs: 接收额外的参数（如 power_load），防止接口不匹配报错
        """
        raise NotImplementedError


class RuleBasedPlanner(BasePlanner):
    """
    基于规则的规划器 (人工势场法 APF)
    特点：计算速度极快，适合避障，但容易陷入局部最优
    """

    def __init__(self, max_speed: float = 15.0, arrive_radius: float = 10.0):
        self.max_speed = max_speed
        self.arrive_radius = arrive_radius

    def compute_velocity_command(self, current_pos, current_vel, target_pos, obstacles,
                                 power_state, dt, **kwargs):
        # 1. 吸引力 (飞向目标)
        error = target_pos - current_pos
        dist = np.linalg.norm(error)

        if dist < self.arrive_radius:
            return np.zeros(3)

        # 基础期望速度方向
        desired_vel = (error / dist) * self.max_speed

        # 2. 斥力 (避障逻辑)
        repulsive_force = np.zeros(3)
        for obs in obstacles:
            cx, cy, w, l, h = obs
            # 简化为圆柱体避障：中心点 (cx, cy)
            obs_center_2d = np.array([cx, cy])
            curr_pos_2d = current_pos[:2]

            dist_vec_2d = curr_pos_2d - obs_center_2d
            dist_val_2d = np.linalg.norm(dist_vec_2d)

            # 高度判断：如果无人机已经在楼顶上方安全距离，不需要避障
            # NED: z 越小越高 (例如 -150 比 -100 高)
            if current_pos[2] < (-h - 10.0):
                continue

            # 斥力触发半径 (建筑物半径 + 安全余量)
            safe_dist = (w + l) / 4.0 + 20.0

            if dist_val_2d < safe_dist:
                rep_dir = dist_vec_2d / (dist_val_2d + 1e-6)
                # 距离越近，斥力越强
                strength = 5.0 * (safe_dist - dist_val_2d)
                repulsive_force[:2] += rep_dir * strength
                # 配合向上爬升躲避
                repulsive_force[2] -= 2.0

                # 3. 合成速度
        final_cmd = desired_vel + repulsive_force

        # 4. 限速
        v_norm = np.linalg.norm(final_cmd)
        if v_norm > self.max_speed:
            final_cmd = final_cmd / v_norm * self.max_speed

        return final_cmd


class AStarPlanner(BasePlanner):
    """
    完整的 3D A* 全局路径规划器
    """

    def __init__(self, grid_res: float = 15.0, replan_interval: int = 30, margin: float = 10.0):
        self.res = grid_res  # 栅格分辨率 (米)
        self.replan_interval = replan_interval
        self.margin = margin  # 建筑物安全膨胀距离
        self.step_count = 0
        self.path_history = []  # 记录计算出的世界坐标路径点
        self.initialized = False

        # 地图定义
        self.grid = None
        self.min_bounds = None
        self.grid_shape = None

    def _init_grid(self, start_pos, target_pos, obstacles):
        """将世界地图离散化为 3D 占据栅格"""
        # 1. 确定地图边界 (包含缓冲区)
        all_pts = np.array([start_pos, target_pos])
        buffer = 100.0
        self.min_bounds = np.floor(np.min(all_pts, axis=0) - buffer)
        self.max_bounds = np.ceil(np.max(all_pts, axis=0) + buffer)
        # 强制 Z 轴范围：地面(0) 到 高空(-300)
        self.min_bounds[2] = -300.0
        self.max_bounds[2] = 20.0

        self.grid_shape = np.ceil((self.max_bounds - self.min_bounds) / self.res).astype(int)
        self.grid = np.zeros(self.grid_shape, dtype=np.int8)

        print(f"[A*] Map Initialized. Shape: {self.grid_shape}, Res: {self.res}m")

        # 2. 障碍物膨胀与栅格化
        for o in obstacles:
            cx, cy, w, l, h = o
            # 计算带安全余量的 AABB 边界
            # NED: z 从 -h 到 0
            b_min = np.array([cx - w / 2 - self.margin, cy - l / 2 - self.margin, -h - self.margin])
            b_max = np.array([cx + w / 2 + self.margin, cy + l / 2 + self.margin, 5.0])  # 地面稍微往下一点

            # 转换为索引
            idx_min = np.floor((b_min - self.min_bounds) / self.res).astype(int)
            idx_max = np.ceil((b_max - self.min_bounds) / self.res).astype(int)

            # 边界剪裁与填充
            idx_min = np.clip(idx_min, [0, 0, 0], self.grid_shape - 1)
            idx_max = np.clip(idx_max, [0, 0, 0], self.grid_shape - 1)

            self.grid[idx_min[0]:idx_max[0] + 1,
            idx_min[1]:idx_max[1] + 1,
            idx_min[2]:idx_max[2] + 1] = 1

        self.initialized = True

    def _pos_to_idx(self, pos):
        idx = np.floor((np.array(pos) - self.min_bounds) / self.res).astype(int)
        return tuple(np.clip(idx, [0, 0, 0], self.grid_shape - 1))

    def _idx_to_pos(self, idx):
        return self.min_bounds + (np.array(idx) + 0.5) * self.res

    def _astar_search(self, start_pos, target_pos):
        """核心 Heapq 搜索算法"""
        start_idx = self._pos_to_idx(start_pos)
        target_idx = self._pos_to_idx(target_pos)

        if self.grid[start_idx] == 1 or self.grid[target_idx] == 1:
            print("[A*] Warning: Start or Target inside obstacle. Finding nearest free cell.")

        # 优先级队列: (f_score, current_idx)
        open_set = []
        heapq.heappush(open_set, (0, start_idx))

        came_from = {}
        g_score = {start_idx: 0}

        # 定义 26 连通邻域
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0: continue
                    neighbors.append((dx, dy, dz, math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)))

        max_iter = 10000
        count = 0

        while open_set:
            count += 1
            current = heapq.heappop(open_set)[1]

            if current == target_idx:
                # 路径回溯
                path = []
                while current in came_from:
                    path.append(self._idx_to_pos(current))
                    current = came_from[current]
                path.reverse()
                return path

            if count > max_iter: break

            for dx, dy, dz, dist in neighbors:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)

                # 边界检查
                if not (0 <= neighbor[0] < self.grid_shape[0] and
                        0 <= neighbor[1] < self.grid_shape[1] and
                        0 <= neighbor[2] < self.grid_shape[2]):
                    continue

                # 碰撞检查
                if self.grid[neighbor] == 1:
                    continue

                tentative_g_score = g_score[current] + dist

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    # h_score: 欧几里得距离
                    h_score = math.sqrt(sum((np.array(neighbor) - np.array(target_idx)) ** 2))
                    f_score = tentative_g_score + h_score
                    heapq.heappush(open_set, (f_score, neighbor))

        return None  # 没找到路径

    def compute_velocity_command(self, current_pos, current_vel, target_pos, obstacles,
                                 power_state, dt, **kwargs):
        # 1. 延迟初始化地图
        if not self.initialized:
            self._init_grid(current_pos, target_pos, obstacles)

        # 2. 定期重规划或当路径跑完时计算
        if self.step_count % self.replan_interval == 0 or not self.path_history:
            new_path = self._astar_search(current_pos, target_pos)
            if new_path:
                self.path_history = new_path
                # print(f"[A*] New path found with {len(new_path)} nodes.")
            else:
                # 降级方案：如果 A* 失败，返回直线（可能碰撞）
                self.path_history = [target_pos]

        self.step_count += 1

        # 3. 路径跟踪 (寻找最近的预瞄点)
        if not self.path_history:
            return np.zeros(3)

        # 找到路径中距离当前位置一定范围外的第一个点作为预瞄点
        look_ahead_dist = 25.0
        target_pt = self.path_history[-1]
        for pt in self.path_history:
            if np.linalg.norm(pt - current_pos) > look_ahead_dist:
                target_pt = pt
                break

        # 移除已经经过的路径点 (简单处理)
        if np.linalg.norm(self.path_history[0] - current_pos) < look_ahead_dist and len(self.path_history) > 1:
            self.path_history.pop(0)

        # 4. 计算速度指令
        direction = target_pt - current_pos
        dist = np.linalg.norm(direction)
        if dist < 1.0: return np.zeros(3)

        return (direction / dist) * 15.0


class SACPlanner(BasePlanner):
    """
    深度强化学习规划器 (SAC)
    特点：具备能量感知能力，能通过学习得到平滑且节能的路径
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.obs_builder = ObservationBuilder()
        self.state_dim = self.obs_builder.state_dim  # 自动设为 12
        self.action_dim = 3

        # 初始化 Actor 网络
        self.actor = ActorSAC(self.state_dim, self.action_dim, max_action=1.0).to(self.device)

        # 加载权重 (适配不同的保存后缀)
        try:
            if os.path.exists(model_path):
                self.actor.load_state_dict(torch.load(model_path, map_location=self.device))
            elif os.path.exists(model_path + ".pth"):
                self.actor.load_state_dict(torch.load(model_path + ".pth", map_location=self.device))
            elif os.path.exists(model_path + "_actor.pth"):
                self.actor.load_state_dict(torch.load(model_path + "_actor.pth", map_location=self.device))

            self.actor.eval()
            print(f"[SACPlanner] Model successfully loaded from {model_path}")
        except Exception as e:
            print(f"[SACPlanner] Error loading model: {e}. Using random weights!")

    def compute_velocity_command(self, current_pos, current_vel, target_pos, obstacles,
                                 power_state, dt, **kwargs):
        """
        跨层推理核心：利用 ObservationBuilder 将 EMS 信息喂给网络
        """
        # 1. 提取能量管理层的实时参数 (通过 kwargs 获取 power_load)
        soc = power_state.get('soc', 0.6)
        h2 = power_state.get('h2_cum', 0.0)
        p_load = kwargs.get('power_load', 500.0)  # 跨层感知的关键

        # 2. 构建标准的 12 维观测向量
        obs = self.obs_builder.build(
            pos=current_pos,
            vel=current_vel,
            target=target_pos,
            soc=soc,
            p_load=p_load,
            h2_cum=h2
        )

        # 3. 神经网络推理
        state_t = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            # 推理时使用 deterministic=True 获得稳定路径
            action = self.actor.get_action(state_t, deterministic=True).cpu().data.numpy().flatten()

        # 4. 动作反归一化 [-1, 1] -> [-15, 15] m/s
        velocity_cmd = action * 15.0

        return velocity_cmd


import os  # 用于路径检查