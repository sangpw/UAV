import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd  # 用于美化表格输出 (如果没有安装pandas, 可删除相关打印代码)
import torch
# 导入你的模块
from models import FuelCellStack, LithiumBattery
from ems import RuleBasedEMS, ECMS_EMS, MPC_EMS,TD3_EMS
from utils import generate_flight_profile


def run_simulation_core(strategy_name, strategy_obj, t_arr, p_load_arr, dt):
    """
    单次仿真内核
    """
    print(f"[{strategy_name}] Simulation started...")
    start_time = time.time()

    # 1. 每次仿真必须重新实例化物理模型，确保起点一致
    # 设定：50片电堆，爬坡率限制 50W/s
    fc = FuelCellStack(num_cells=50, max_slew_rate=10.0)
    # 设定：22.2V 5Ah 电池 (约110Wh), 初始SOC 0.6
    # 注意：为了让SOH变化在短时间内可见，我们将 cycle_fade_factor 设得比较大
    bat = LithiumBattery(capacity_ah=5.0, initial_soc=0.6)
    bat.cycle_fade_factor = 1e-4  # 人为放大衰减以便观测 (实际约为 1e-5)

    ems = strategy_obj

    # 数据记录容器
    results = {
        'name': strategy_name,
        't': t_arr,
        'p_load': p_load_arr,
        'p_fc': [],
        'p_bat': [],
        'soc': [],
        'soh': [],
        'h2_cum': []
    }

    h2_total = 0
    steps = len(t_arr)

    # 2. 时间步进循环
    for i in range(steps):
        p_req = p_load_arr[i]

        # 准备 MPC 需要的未来信息 (切片)
        future_load = p_load_arr[i:]

        # A. EMS 计算指令
        fc_cmd = ems.compute_fc_command(p_req, bat.SOC, dt, future_load=future_load)

        # B. 物理模型响应
        p_fc_act, h2_step = fc.step(fc_cmd, dt)

        # 电池填补剩余 (Load - FC)
        p_bat_req = p_req - p_fc_act
        p_bat_act, soc, soh = bat.step(p_bat_req, dt)

        # C. 记录数据
        h2_total += h2_step

        results['p_fc'].append(p_fc_act)
        results['p_bat'].append(p_bat_act)
        results['soc'].append(soc)
        results['soh'].append(soh)
        results['h2_cum'].append(h2_total)

        # 进度打印 (防止MPC太慢用户以为卡死)
        if i % 1000 == 0 and i > 0:
            print(f"  > Progress: {i}/{steps} steps...")

    elapsed = time.time() - start_time
    print(f"[{strategy_name}] Done. Calc Time: {elapsed:.2f}s\n")
    return results


def plot_comparison(results_list):
    """
    绘制多策略对比图 (2x2)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 颜色映射
    colors = {'Rule Based': 'green', 'ECMS': 'blue', 'MPC': 'red'}
    linestyles = {'Rule Based': '--', 'ECMS': '-.', 'MPC': '-'}

    # 1. 功率分配示例 (取其中一个策略展示负载，其他策略展示FC功率)
    ax1 = axes[0, 0]
    # 画负载背景
    t = results_list[0]['t']
    p_load = results_list[0]['p_load']
    ax1.plot(t, p_load, color='gray', alpha=0.3, label='Load Demand')

    for res in results_list:
        name = res['name']
        ax1.plot(res['t'], res['p_fc'], label=f'{name} FC',
                 color=colors.get(name, 'k'), linestyle=linestyles.get(name, '-'), alpha=0.8)

    ax1.set_ylabel('Power (W)')
    ax1.set_title('Fuel Cell Power Response')
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(True, alpha=0.3)

    # 2. SOC 曲线对比
    ax2 = axes[0, 1]
    for res in results_list:
        name = res['name']
        ax2.plot(res['t'], res['soc'], label=name,
                 color=colors.get(name, 'k'), linestyle=linestyles.get(name, '-'))

    ax2.set_ylabel('SOC')
    ax2.set_title('Battery SOC Trajectory')
    ax2.axhline(0.3, color='r', linestyle=':', alpha=0.5)  # 下限参考
    ax2.axhline(0.85, color='r', linestyle=':', alpha=0.5)  # 上限参考
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 累计氢耗对比
    ax3 = axes[1, 0]
    for res in results_list:
        name = res['name']
        ax3.plot(res['t'], res['h2_cum'], label=name,
                 color=colors.get(name, 'k'), linestyle=linestyles.get(name, '-'))

    ax3.set_ylabel('H2 Consumption (g)')
    ax3.set_title('Cumulative Hydrogen Consumption')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. SOH 衰减对比
    ax4 = axes[1, 1]
    for res in results_list:
        name = res['name']
        # 为了让微小的SOH变化看清楚，我们画 (1 - SOH) * 100，即“健康度损失百分比”
        soh_loss_pct = (1.0 - np.array(res['soh'])) * 100
        ax4.plot(res['t'], soh_loss_pct, label=name,
                 color=colors.get(name, 'k'), linestyle=linestyles.get(name, '-'))

    ax4.set_ylabel('SOH Loss (%)')
    ax4.set_title('Battery Degradation (SOH Loss)')
    ax4.set_xlabel('Time (s)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_statistics(results_list):
    """
    打印数值统计报告
    """
    print("=" * 60)
    print(f"{'Strategy':<15} | {'Final SOC':<10} | {'H2 Used (g)':<12} | {'SOH Loss (%)':<12}")
    print("-" * 60)

    for res in results_list:
        name = res['name']
        final_soc = res['soc'][-1]
        final_h2 = res['h2_cum'][-1]
        start_soh = res['soh'][0]
        final_soh = res['soh'][-1]
        soh_loss_pct = (start_soh - final_soh) * 100

        print(f"{name:<15} | {final_soc:.2%}     | {final_h2:.4f}       | {soh_loss_pct:.4f}%")
    print("=" * 60)
    print("注: SOH Loss 取决于电流倍率波动，曲线越平滑，寿命损耗通常越小。")


if __name__ == "__main__":
    # 1. 设置仿真总参数
    T_SIM = 600  # 秒 (10分钟)
    DT = 0.1  # 100ms (为了照顾MPC的计算速度，不用10ms)

    # 2. 生成统一的复杂工况 (所有策略公用此数据)
    print("Generating complex flight profile...")
    t_arr, p_load_arr = generate_flight_profile(T_SIM, DT)

    # 3. 定义要对比的策略列表
    strategies = [
        # 策略 A: 规则控制
        ("Rule Based", RuleBasedEMS(soc_min=0.3, soc_max=0.85)),

        # 策略 B: ECMS (参数需根据你的模型调优，这里使用上一轮优化过的参数)
        # k_p 越大，对 SOC 偏差越敏感
        ("ECMS", ECMS_EMS(soc_target=0.6, k_p=2.5)),

        # 策略 C: MPC (预测未来 20个点 = 2秒)
        # 注意: 纯Python的 minimize 比较慢，这里 horizon 设小一点
        ("MPC", MPC_EMS(horizon=15, dt_mpc=DT)),
        ("TD3 RL", TD3_EMS(model_path="./models/td3_ems_200_actor"))
    ]

    # 4. 批量运行
    all_results = []
    for name, strat_obj in strategies:
        res = run_simulation_core(name, strat_obj, t_arr, p_load_arr, DT)
        all_results.append(res)

    # 5. 输出结果
    print_statistics(all_results)
    plot_comparison(all_results)