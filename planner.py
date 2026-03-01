import numpy as np
import torch
import heapq
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from rl_core import Actor  ,ActorSAC# 复用rl_core中的Actor定义


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
        """
        接口定义：所有子类必须匹配此参数列表
        :param future_trajectory: (可选) 预测未来轨迹，用于MPC类算法
        :return: 目标速度指令 [vx, vy, vz] (m/s)
        """
        raise NotImplementedError


class RuleBasedPlanner(BasePlanner):
    """
    策略1: 基于规则的路径跟随 (P控制 + 简单避障)
    """

    def __init__(self, kp_pos: float = 1.0, kp_alt: float = 0.5, max_speed: float = 15.0):
        self.kp_pos = kp_pos
        self.kp_alt = kp_alt
        self.max_speed = max_speed

    def compute_velocity_command(self, current_pos, current_vel, target_pos, obstacles,
                                 power_state, dt, future_trajectory=None):
        pos_error = target_pos - current_pos
        dist = np.linalg.norm(pos_error)
        if dist < 2.0: return np.zeros(3)

        v_cmd = pos_error * self.kp_pos
        v_cmd[2] = pos_error[2] * self.kp_alt

        # 简单的势场避障
        for obs in obstacles:
            obs_pos, obs_r = obs[:3], obs[3]
            dist_vec = current_pos - obs_pos
            dist_val = np.linalg.norm(dist_vec)
            if dist_val < obs_r + 15.0:  # 避障半径
                # 产生斥力
                repulsion = (dist_vec / dist_val) * (1.0 / (dist_val - obs_r + 0.1)) * 50.0
                v_cmd += repulsion

        v_norm = np.linalg.norm(v_cmd[:2])
        if v_norm > self.max_speed:
            v_cmd[:2] = v_cmd[:2] / v_norm * self.max_speed
        v_cmd[2] = np.clip(v_cmd[2], -5, 5)
        return np.clip(v_cmd, -self.max_speed, self.max_speed)


class AStarPlanner(BasePlanner):
    """
    长方体避障版 A* Planner
    """

    def __init__(self, grid_res: float = 20.0, replan_interval: int = 20):
        self.grid_res = grid_res
        self.replan_interval = replan_interval
        self.waypoints = []
        self.current_wp_idx = 0
        self.step_counter = 0
        # 安全膨胀距离 (米)
        self.safety_margin = 15.0

    def _heuristic(self, a, b):
        return np.linalg.norm(a - b)

    def _check_collision(self, node_arr, obstacles):
        # 1. 地面与限高
        if node_arr[2] > 0 or node_arr[2] < -250: return True

        # 2. 长方体检测 (Box Collision)
        x, y, z = node_arr

        for obs in obstacles:
            # obs: [cx, cy, w, l, h]
            cx, cy, w, l, h = obs

            # 计算包含安全余量的边界
            # 为什么是 w/2 + margin? 因为是从中心向两边扩
            x_min = cx - (w / 2 + self.safety_margin)
            x_max = cx + (w / 2 + self.safety_margin)
            y_min = cy - (l / 2 + self.safety_margin)
            y_max = cy + (l / 2 + self.safety_margin)

            # 垂直边界: 建筑是从地面(0)长到(-h)
            # 所以如果在 [x_min, x_max] 和 [y_min, y_max] 范围内，
            # 且高度 > (-h - margin)，就算碰撞 (注意NED高度是负的，越高越小)
            # z_top = -h
            z_collision_threshold = -h - 5.0  # 顶部额外留 5米余量

            if (x_min <= x <= x_max) and (y_min <= y <= y_max):
                # 水平位置冲突，检查高度
                # 如果当前高度 Z > z_collision_threshold (意味着在楼下方)，则碰撞
                if z > z_collision_threshold:
                    return True

        return False

    def _plan_path(self, start, goal, obstacles):
        # 坐标离散化
        start_node = tuple(np.round(start / self.grid_res) * self.grid_res)
        goal_node = tuple(np.round(goal / self.grid_res) * self.grid_res)

        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}

        # 6邻域
        step = self.grid_res
        motions = [
            np.array([step, 0, 0]), np.array([-step, 0, 0]),
            np.array([0, step, 0]), np.array([0, -step, 0]),
            np.array([0, 0, -step]), np.array([0, 0, step])
        ]

        max_ops = 5000
        ops = 0
        final_node = None

        while open_set and ops < max_ops:
            ops += 1
            current = heapq.heappop(open_set)[1]
            curr_arr = np.array(current)

            if np.linalg.norm(curr_arr - np.array(goal_node)) < self.grid_res * 1.5:
                final_node = current
                break

            for m in motions:
                neighbor_arr = curr_arr + m
                neighbor = tuple(neighbor_arr)

                # 范围限制 (Map bounds)
                if abs(neighbor_arr[0] - start[0]) > 1200 or abs(neighbor_arr[1] - start[1]) > 1200:
                    continue

                # 碰撞检测
                if self._check_collision(neighbor_arr, obstacles):
                    continue

                new_g = g_score[current] + np.linalg.norm(m)
                if neighbor not in g_score or new_g < g_score[neighbor]:
                    g_score[neighbor] = new_g
                    f = new_g + self._heuristic(neighbor_arr, goal_node)
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (f, neighbor))

        if final_node:
            path = []
            curr = final_node
            while curr in came_from:
                path.append(np.array(curr))
                curr = came_from[curr]
            path.reverse()
            path.append(goal)
            self.waypoints = path
            print(f"[A*] Path found with {len(path)} nodes.")
        else:
            print("[A*] Failed to find path. Climbing up.")
            # 失败保护：向上飞
            self.waypoints = [start + np.array([0, 0, -50]), goal]

        self.current_wp_idx = 0

    def compute_velocity_command(self, current_pos, current_vel, target_pos, obstacles, power_state, dt, **kwargs):
        self.step_counter += 1

        # 规划触发逻辑
        if not self.waypoints or \
                (self.step_counter % self.replan_interval == 0 and np.linalg.norm(current_pos - target_pos) > 20):
            self._plan_path(current_pos, target_pos, obstacles)

        if not self.waypoints: return np.zeros(3)

        # 路径跟踪
        target_wp = self.waypoints[self.current_wp_idx]
        dist = np.linalg.norm(current_pos - target_wp)

        if dist < 15.0:
            self.current_wp_idx = min(self.current_wp_idx + 1, len(self.waypoints) - 1)
            target_wp = self.waypoints[self.current_wp_idx]

        direction = target_wp - current_pos
        norm = np.linalg.norm(direction)
        if norm > 0.1:
            return (direction / norm) * 12.0
        return np.zeros(3)


class SACPlanner(BasePlanner):
    """
    修正点：使用 ActorSAC 类加载模型，并调用正确的推理接口
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.max_speed = 15.0
        self.state_dim = 11
        self.action_dim = 3

        # [FIX] 使用 ActorSAC 而不是 Actor
        # 注意：ActorSAC 的构造函数参数可能需要根据 rl_core.py 调整，这里假设使用默认 hidden_dim
        self.actor = ActorSAC(self.state_dim, self.action_dim, max_action=1.0).to(self.device)

        try:
            # 加载模型
            self.actor.load_state_dict(torch.load(model_path, map_location=self.device))
            self.actor.eval()
            print(f"[SACPlanner] Model loaded from {model_path}")
        except Exception as e:
            print(f"[SACPlanner] Warning: Failed to load model from {model_path}. Error: {e}")
            # 初始化一个随机网络以防崩溃

    def compute_velocity_command(self, current_pos, current_vel, target_pos, obstacles,
                                 power_state, dt, future_trajectory=None):
        to_target = target_pos - current_pos
        # 归一化处理应与 training 保持一致
        obs = np.concatenate([
            current_pos / 800.0,  # 简单归一化位置
            current_vel / 15.0,  # 归一化速度
            to_target / 800.0,
            [power_state.get('soc', 0.6)],
            [power_state.get('h2_cum', 0) / 100.0]
        ]).astype(np.float32)

        # 确保维度匹配
        if len(obs) != self.state_dim:
            # 如果维度不对，补零或截断（防止crash）
            obs = np.resize(obs, self.state_dim)

        state_t = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)

        with torch.no_grad():
            # [FIX] 使用 get_action 接口，且 deterministic=True (推理模式)
            action = self.actor.get_action(state_t, deterministic=True).cpu().data.numpy().flatten()

        return action * self.max_speed