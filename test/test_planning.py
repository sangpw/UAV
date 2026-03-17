import numpy as np
import time
import matplotlib.pyplot as plt

# 假设你将刚才定义的环境保存为了 urban_env.py
from urban_env import UrbanPlanningEnv

# 尝试导入规划器，如果没有则定义简单的占位符以便代码能运行
try:
    from planner import RuleBasedPlanner, AStarPlanner
except ImportError:
    print("Warning: planner.py not found. Using Mock Planners.")


    class PlannerBase:
        def compute_velocity_command(self, current_pos, target_pos, **kwargs):
            # 简单的 P 控制器向目标飞
            err = target_pos - current_pos
            v = err / np.linalg.norm(err) * 10.0
            return v


    class RuleBasedPlanner(PlannerBase):
        def __init__(self, max_speed): pass


    class AStarPlanner(PlannerBase):
        def __init__(self, grid_res, replan_interval): pass

# ==========================================
# 全局配置 (确保两个规划器面对完全相同的场景)
# ==========================================
CONFIG = {
    "start_pos": [0, 0, -10],
    "target_pos": [800, 600, -50],
    "num_obstacles": 20,  # 增加障碍物数量以测试避障
    "map_seed": 12345,  # <--- 关键：固定种子
    "fixed_map": True,  # <--- 关键：锁定地图
    "dt": 0.5
}


def run_planning_demo(planner_name, planner):
    print(f"\n" + "=" * 40)
    print(f"=== Testing Planner: {planner_name} ===")
    print(f"=" * 40)

    # 1. 初始化环境 (使用全局 CONFIG 确保地图一致)
    env = UrbanPlanningEnv(
        start_pos=CONFIG["start_pos"],
        target_pos=CONFIG["target_pos"],
        dt=CONFIG["dt"],
        num_obstacles=CONFIG["num_obstacles"],
        fixed_map=CONFIG["fixed_map"],  # 开启固定地图模式
        map_seed=CONFIG["map_seed"]  # 传入相同的种子
    )

    # 重置环境
    obs, info = env.reset()

    path = []
    total_reward = 0
    done = False

    # 记录起始时间
    start_time = time.time()

    print(f"Map initialized with Seed {CONFIG['map_seed']}. Start simulation...")

    while not done:
        current_pos = info['position']
        path.append(current_pos)

        # ------------------------------------------------------
        # 2. Planner 决策
        # ------------------------------------------------------
        # 我们直接将 info 中的数据传给 planner
        # 注意：planner 需要能处理 info['obstacles'] (列表格式: [cx, cy, w, l, h])
        velocity_cmd = planner.compute_velocity_command(
            current_pos=current_pos,
            current_vel=info['velocity'],
            target_pos=info['target'],
            obstacles=info['obstacles'],
            power_state=info.get('power_state', {}),
            dt=info['dt']
        )

        # 简单限速保护 (防止 planner 输出过大)
        velocity_cmd = np.clip(velocity_cmd, -20, 20)

        # ------------------------------------------------------
        # 3. 环境步进
        # ------------------------------------------------------
        obs, reward, terminated, truncated, info = env.step(velocity_cmd)

        total_reward += reward
        done = terminated or truncated

        # 打印进度
        if env.time_step % 20 == 0:
            dist = np.linalg.norm(current_pos - CONFIG["target_pos"])
            print(f"Step {env.time_step:3d}: Dist to Target = {dist:.1f}m")

    # ------------------------------------------------------
    # 4. 结果统计与可视化
    # ------------------------------------------------------
    duration = time.time() - start_time
    status = "Success" if total_reward > 0 else "Collision/Timeout"

    print(f"[{planner_name}] Finished.")
    print(f"  - Status: {status}")
    print(f"  - Time: {duration:.2f}s")
    print(f"  - Total Steps: {env.time_step}")
    print(f"  - Final Reward: {total_reward:.1f}")

    # 渲染结果 (传入路径历史)
    # 标题显示规划器名称
    print("Rendering...")
    plt.ion()  # 如果你在非阻塞模式下运行
    env.render(path_history=path)
    plt.show()  # 阻塞显示，关闭窗口后继续下一个


if __name__ == "__main__":
    # --- 实验 1: 规则/人工势场 规划器 ---
    # 假设它是反应式的，不需要预处理地图
    planner_rule = RuleBasedPlanner(max_speed=15.0)
    run_planning_demo("RuleBased Planner", planner_rule)

    # --- 实验 2: A* 规划器 ---
    # 确保 A* 能够处理 3D 障碍物信息
    # grid_res=20 表示栅格分辨率 20m，replan_interval=10 表示每10步重规划一次
    planner_astar = AStarPlanner(grid_res=20.0, replan_interval=10)
    run_planning_demo("A* Planner", planner_astar)