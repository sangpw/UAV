#evaluate_system.py
import numpy as np
from envs.env_wrapper_hierarchical import HierarchicalUAVEnv
from controllers.planner import SACPlanner, RuleBasedPlanner, AStarPlanner
from controllers.ems import TD3_EMS, RuleBasedEMS, ECMS_EMS, MPC_EMS
from utils import plot_uav_path

def evaluate_combination(planner_name, ems_name, planner_model_path=None, ems_model_path=None):
    # 1. 初始化 1s / 0.1s 双层环境
    env = HierarchicalUAVEnv(planner_dt=1.0, ems_dt=0.1, T_sim=1000)

    # 2. 实例化规划器
    if planner_name == "SAC":
        planner = SACPlanner(model_path=planner_model_path)
    elif planner_name == "APF":
        planner = RuleBasedPlanner()
    elif planner_name == "AStar":
        planner = AStarPlanner()

    # 3. 实例化能量管理
    if ems_name == "TD3":
        ems = TD3_EMS(model_path=ems_model_path)
    elif ems_name == "Rule":
        ems = RuleBasedEMS()
    elif ems_name == "ECMS":
        ems = ECMS_EMS()
    elif ems_name == "MPC":
        ems = MPC_EMS()

    # 4. 注入环境
    env.set_planner(planner)
    env.set_ems(ems)

    # 5. 开始仿真运行
    obs = env.reset()
    done = False
    total_h2 = 0
    path_history = []
    print(f"\nEvaluating: {planner_name} + {ems_name}...")

    while not done:
        path_history.append(env.uav.get_position().copy())
        # HierarchicalUAVEnv.step() 如果不传 action，内部会自动调用 env.planner 和 env.ems
        obs, reward, done, info = env.step()

    print(f"[{planner_name} + {ems_name}] Test Finished.")
    print(f"Distance to target: {info['distance']:.2f} m")
    print(f"Final SOC: {info['soc']:.4f}, H2 Consumption: {info['h2_total']:.4f} g, SOH: {info['SOH']:.4f} h")

    plot_uav_path(
        start_pos=env.start_pos,
        target_pos=env.target_pos,
        obstacles=env.obstacles,
        path_history=path_history,
        title=f"Path Plan: {planner_name} + {ems_name}"
    )


    return info['h2_total'], info['distance']


if __name__ == "__main__":
    # 您需要提前通过 train_sac.py 和 train_td3.py 获得这两个 pth 文件
    sac_path = "models/SAC/sac_ep500_actor"
    td3_path = "models/TD3/td3_latest_success_actor.pth"

    #--- 对比实验组设计 ---
    # 1. 您提出的目标系统
    evaluate_combination("SAC", "TD3", sac_path, td3_path)

    # 2. 控制变量：只替换下层 EMS (验证 TD3 优于传统方法)
    evaluate_combination("SAC", "Rule", sac_path, None)
    evaluate_combination("SAC", "ECMS", sac_path, None)
    evaluate_combination("SAC", "MPC", sac_path, None)

    # 3. 控制变量：只替换上层 Planner (验证 SAC 优于传统方法)
    evaluate_combination("APF", "TD3", None, td3_path)
    evaluate_combination("AStar", "TD3", None, td3_path)

    # 4. 传统双层基线
    evaluate_combination("AStar", "ECMS")