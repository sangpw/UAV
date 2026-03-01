import numpy as np
import time
# 导入我们刚才定义的单层环境
from urban_env import UrbanPlanningEnv
# 导入你的 Planner (假设文件名是 planner.py)
from planner import RuleBasedPlanner, AStarPlanner


def run_planning_demo(planner_name, planner):
    print(f"=== Testing {planner_name} ===")

    # 初始化环境
    env = UrbanPlanningEnv(
        start_pos=[0, 0, -10],
        target_pos=[800, 600, -100],
        num_obstacles=10  # 增加障碍物密度
    )

    obs, info = env.reset()
    path = []

    total_reward = 0
    done = False

    start_time = time.time()

    while not done:
        # 记录路径
        path.append(info['position'])

        # 1. Planner 决策
        # 注意：这里我们利用 env 提供的 info 中的原始数据传给 Planner
        velocity_cmd = planner.compute_velocity_command(
            current_pos=info['position'],
            current_vel=info['velocity'],
            target_pos=info['target'],
            obstacles=info['obstacles'],  # 适配后的障碍物列表
            power_state=info['power_state'],
            dt=info['dt']
        )

        # 2. 环境步进
        obs, reward, terminated, truncated, info = env.step(velocity_cmd)

        total_reward += reward
        done = terminated or truncated

        # 简单打印进度
        if env.time_step % 50 == 0:
            dist = np.linalg.norm(info['position'] - info['target'])
            print(f"Step {env.time_step}: Dist={dist:.1f}m, Reward={total_reward:.1f}")

    duration = time.time() - start_time
    print(f"[{planner_name}] Finished. Time: {duration:.2f}s, Total Steps: {env.time_step}")

    # 3. 可视化
    env.render(path_history=path)


if __name__ == "__main__":
    # 1. 测试规则规划器
    run_planning_demo("RuleBased", RuleBasedPlanner(max_speed=15.0))

    # 2. 测试 A* 规划器 (确保你已经修复了 A* 的代码)
    # A* 需要较大的栅格和重规划间隔来平衡速度
    run_planning_demo("A*", AStarPlanner(grid_res=30.0, replan_interval=30))