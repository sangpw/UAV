import numpy as np
import matplotlib.pyplot as plt
import time
from models import MultirotorUAV, FuelCellStack, LithiumBattery
from planner import RuleBasedPlanner, AStarPlanner, SACPlanner
from ems import RuleBasedEMS, ECMS_EMS, MPC_EMS, TD3_EMS
from env_wrapper_hierarchical import HierarchicalUAVEnv


def run_simulation_core(strategy_name, planner_obj, ems_obj, T_sim=600):
    """
    单次仿真内核 (仿照 main.py 的 run_simulation_core)
    """
    print(f"[{strategy_name}] Simulation started...")
    start_time = time.time()

    # 环境实例化
    env = HierarchicalUAVEnv(
        planner_dt=0.5,  # 2Hz
        ems_dt=0.1,  # 10Hz
        T_sim=T_sim,
        start_pos=np.array([0., 0., -10.]),
        target_pos=np.array([800., 600., -100.])
    )
    env.set_planner(planner_obj)
    env.set_ems(ems_obj)

    # 重置
    obs = env.reset()

    # 数据记录容器 (仿照 main.py 的 results 结构)
    results = {
        'name': strategy_name,
        't': [],
        'pos': [],
        'vel': [],
        'p_load': [],
        'p_fc': [],
        'p_bat': [],
        'soc': [],
        'soh': [],
        'h2_cum': [],
        'dist': []
    }

    done = False
    while not done:
        obs, reward, done, info = env.step()

        # 记录数据
        t = env.time_step * env.planner_dt
        results['t'].append(t)
        results['pos'].append(info['position'])
        results['vel'].append(info['velocity'])
        results['dist'].append(info['distance'])
        results['soc'].append(info['soc'])
        results['soh'].append(info['soh'])
        results['h2_cum'].append(info['h2_total'])

        # 功率记录 (最后一次substep的状态)
        results['p_load'].append(env.uav.compute_power(info['velocity'], np.zeros(3)))
        results['p_fc'].append(env.fc.current_power_act)
        results['p_bat'].append(results['p_load'][-1] - results['p_fc'][-1])

        if env.time_step % 50 == 0:
            print(f"  > Step {env.time_step}, Dist: {info['distance']:.1f}m, "
                  f"SOC: {info['soc']:.2f}")

    elapsed = time.time() - start_time
    print(f"[{strategy_name}] Done. Time: {elapsed:.2f}s\n")
    return results


def plot_comparison(results_list):
    """
    绘制多策略对比图 (仿照 main.py 的 plot_comparison)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 颜色映射 (仿照原风格)
    colors = {
        'Rule+Rule': 'green',
        'Rule+ECMS': 'blue',
        'A*+ECMS': 'red',
        'SAC+TD3': 'purple'
    }

    # 1. 3D轨迹
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    target = np.array([800, 600, -100])
    for res in results_list:
        name = res['name']
        pos = np.array(res['pos'])
        ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                 label=name, color=colors.get(name, 'k'), linewidth=2)
    ax1.scatter([0], [0], [-10], color='g', s=100, label='Start')
    ax1.scatter([target[0]], [target[1]], [target[2]], color='r', s=100, marker='*')
    ax1.set_title('3D Flight Path')
    ax1.legend()

    # 2. 距离收敛
    ax2 = axes[0, 1]
    for res in results_list:
        ax2.plot(res['t'], res['dist'], label=res['name'],
                 color=colors.get(res['name'], 'k'))
    ax2.set_ylabel('Distance to Target (m)')
    ax2.set_title('Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 功率分配示例 (第一个结果)
    ax3 = axes[0, 2]
    res = results_list[0]
    ax3.plot(res['t'], res['p_load'], 'k--', alpha=0.3, label='Load')
    ax3.plot(res['t'], res['p_fc'], label='FC', color='red')
    ax3.plot(res['t'], res['p_bat'], label='Bat', color='blue')
    ax3.set_title(f'{res["name"]} Power Split')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. SOC对比
    ax4 = axes[1, 0]
    for res in results_list:
        ax4.plot(res['t'], res['soc'], label=res['name'],
                 color=colors.get(res['name'], 'k'))
    ax4.axhline(0.6, color='r', linestyle=':', alpha=0.5)
    ax4.set_ylabel('SOC')
    ax4.set_title('Battery SOC')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 氢耗累积
    ax5 = axes[1, 1]
    for res in results_list:
        ax5.plot(res['t'], res['h2_cum'], label=res['name'],
                 color=colors.get(res['name'], 'k'))
    ax5.set_ylabel('H2 Consumption (g)')
    ax5.set_title('Cumulative H2')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. SOH衰减
    ax6 = axes[1, 2]
    for res in results_list:
        soh_loss = (1.0 - np.array(res['soh'])) * 100
        ax6.plot(res['t'], soh_loss, label=res['name'],
                 color=colors.get(res['name'], 'k'))
    ax6.set_ylabel('SOH Loss (%)')
    ax6.set_title('Battery Degradation')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hierarchical_comparison.png', dpi=150)
    plt.show()


def print_statistics(results_list):
    """打印统计报告 (仿照 main.py)"""
    print("=" * 70)
    print(f"{'Strategy':<15} | {'Final Dist':<10} | {'H2 (g)':<10} | {'SOH Loss %':<10} | {'Time (s)':<8}")
    print("-" * 70)

    for res in results_list:
        name = res['name']
        final_dist = res['dist'][-1]
        final_h2 = res['h2_cum'][-1]
        soh_loss = (1.0 - res['soh'][-1]) * 100
        duration = res['t'][-1]

        print(f"{name:<15} | {final_dist:.1f}      | {final_h2:.4f}   | {soh_loss:.4f}    | {duration:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    # 定义实验组合 (Planner + EMS)
    experiments = [
        ("Rule+Rule", RuleBasedPlanner(), RuleBasedEMS()),
        ("Rule+ECMS", RuleBasedPlanner(), ECMS_EMS(soc_target=0.6, k_p=2.5)),
        # ("SAC+TD3", SACPlanner("./models/sac_planner_best_actor"),
        #  TD3_EMS("./models/td3_ems_best_actor")),
    ]

    # 批量运行
    all_results = []
    for name, planner, ems in experiments:
        res = run_simulation_core(name, planner, ems, T_sim=600)
        all_results.append(res)

    # 输出结果
    print_statistics(all_results)
    plot_comparison(all_results)