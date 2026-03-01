import numpy as np
from scipy.optimize import minimize
from rl_core import Actor # 需要从 rl_core 导入 Actor 定义
import torch
class BaseEMS:
    """EMS 策略基类"""

    def compute_fc_command(self, load_power, battery_soc, dt, future_load=None):
        """
        接口定义：所有子类必须匹配此参数列表
        :param future_load: (可选) 未来负载预测数组，用于MPC
        """
        raise NotImplementedError


class RuleBasedEMS(BaseEMS):
    """
    策略1: 基于规则 (State Machine)
    """

    def __init__(self, soc_min=0.3, soc_max=0.85):
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.P_fc_max = 500
        self.P_fc_min = 100

    # --- 修复点：添加 future_load=None ---
    def compute_fc_command(self, load, soc, dt, future_load=None):
        cmd = 0

        # 1. 强制充电区
        if soc < self.soc_min:
            cmd = min(load + 200, self.P_fc_max)

        # 2. 正常调节区
        elif self.soc_min <= soc <= self.soc_max:
            if load > self.P_fc_max:
                cmd = self.P_fc_max
            elif load < self.P_fc_min:
                # 迟滞逻辑
                if soc < (self.soc_min + 0.15):
                    cmd = self.P_fc_min
                else:
                    cmd = 0
            else:
                # 功率跟随
                cmd = load

                # 3. 强制放电区
        else:
            cmd = min(load * 0.5, self.P_fc_max)
            if load < 100: cmd = 0

        return cmd


class ECMS_EMS(BaseEMS):
    """
    策略2: ECMS (等效消耗最小化)
    """

    def __init__(self, soc_target=0.6, k_p=2.0):
        self.soc_target = soc_target
        self.k_p = k_p
        self.p_fc_candidates = np.arange(0, 501, 5)
        # 拟合系数: c2*P^2 + c1*P + c0
        self.fc_coeff = [2.5e-6, 4e-5, 1e-4]

    def _get_h2_consumption(self, p_fc_array):
        cost = self.fc_coeff[0] * (p_fc_array ** 2) + \
               self.fc_coeff[1] * p_fc_array + \
               self.fc_coeff[2]
        cost[p_fc_array < 10] = 1e-5
        return cost

    # 同样确保这里有 future_load=None，即使不用它
    def compute_fc_command(self, load, soc, dt, future_load=None):
        # 1. 燃料电池成本
        cost_fc = self._get_h2_consumption(self.p_fc_candidates)

        # 2. 电池成本 (等效因子 s)
        p_bat_candidates = load - self.p_fc_candidates
        s0 = 3.0
        penalty = 1 - self.k_p * (soc - self.soc_target)
        s = s0 * penalty

        cost_bat = p_bat_candidates * 4.5e-4 * s

        # 3. 总成本最小化
        cost_total = cost_fc + cost_bat
        idx_min = np.argmin(cost_total)

        return self.p_fc_candidates[idx_min]


class MPC_EMS(BaseEMS):
    """
    策略3: MPC (模型预测控制)
    """

    def __init__(self, horizon=10, dt_mpc=1.0):
        self.N = horizon
        self.dt = dt_mpc
        self.Q_cap = 2.0 * 22.2 * 3600
        self.soc_ref = 0.6
        self.last_p_fc = 0

        # 权重
        self.w_h2 = 0.1
        self.w_soc = 200000

    def cost_function(self, u_control, *args):
        soc_k = args[0]
        load_profile = args[1]

        J = 0
        soc_curr = soc_k

        for k in range(self.N):
            p_fc = u_control[k]
            p_load = load_profile[k]
            p_bat = p_load - p_fc

            # Cost 计算
            h2_cost = 1e-4 * p_fc

            delta_soc = (p_bat * self.dt) / self.Q_cap
            soc_next = soc_curr - delta_soc

            soc_cost = (soc_next - self.soc_ref) ** 2

            J += self.w_h2 * h2_cost + self.w_soc * soc_cost

            if soc_next < 0.2 or soc_next > 0.9:
                J += 1e8  # 软约束惩罚

            soc_curr = soc_next

        return J

    def compute_fc_command(self, load, soc, dt, future_load=None):
        # 1. 处理预测时域
        if future_load is None:
            load_horizon = np.ones(self.N) * load
        else:
            steps_needed = self.N
            if len(future_load) >= steps_needed:
                load_horizon = future_load[:steps_needed]
            else:
                pad = np.ones(steps_needed - len(future_load)) * future_load[-1]
                load_horizon = np.concatenate((future_load, pad))

        # 2. 优化
        bounds = [(0, 500) for _ in range(self.N)]
        u0 = np.ones(self.N) * 0

        res = minimize(
            self.cost_function,
            u0,
            args=(soc, load_horizon),
            method='SLSQP',
            bounds=bounds,
            options={'ftol': 1e-4, 'maxiter': 5, 'disp': False}
        )

        p_opt = res.x[0]
        return p_opt


class TD3_EMS(BaseEMS):
    """
    基于深度强化学习 (TD3) 的策略
    推理模式
    """

    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.max_action = 1.0

        # 定义网络结构 (必须与训练时一致)
        self.actor = Actor(state_dim=3, action_dim=1, max_action=1.0).to(self.device)

        # 加载权重
        try:
            self.actor.load_state_dict(torch.load(model_path, map_location=self.device))
            self.actor.eval()  # 切换到评估模式 (关闭Dropout/BatchNorm等)
            print(f"[TD3_EMS] Model loaded from {model_path}")
        except FileNotFoundError:
            print(f"[TD3_EMS] Error: Model file not found at {model_path}")
            # 如果没找到模型，可以用随机网络跑跑看(虽然结果很烂)，或者抛出异常

    def compute_fc_command(self, load, soc, dt, future_load=None):
        """
        :param load: 当前负载 (W)
        :param soc: 电池 SOC (0-1)
        """
        # 1. 构建状态向量 (必须与 env_wrapper 中的 _get_state 一致)
        # 假设我们无法获得 SOH，或者 SOH 变化很慢近似为 1.0 (如果是完全观测，这里要填写真实SOH)
        # State: [SOC, Normalized_Load, SOH]
        norm_load = load / 1000.0
        state = np.array([soc, norm_load, 1.0])  # 这里SOH暂时硬编码为1，或者你需要从电池对象获取

        # 2. 转为 Tensor
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        # 3. 网络推理
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()

        # 4. 动作还原 (-1~1 -> 0~500W)
        # 对应 env_wrapper 中的: (action + 1) / 2 * 500
        p_fc_cmd = (action[0] + 1) / 2 * 500.0

        return p_fc_cmd