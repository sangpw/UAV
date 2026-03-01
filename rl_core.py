# rl_core.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 经验回放池
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1e6):
        self.ptr = 0
        self.size = 0
        self.max_size = int(max_size)

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, action_dim))
        self.next_state = np.zeros((self.max_size, state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device)
        )

# 2. 策略网络 (Actor)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # 输出范围 [-1, 1] * max_action
        return self.max_action * torch.tanh(self.l3(a))

# 3. 价值网络 (Critic) - Twin Architecture
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        # Q1
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        # Q2
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

# 4. TD3 算法主体
class TD3:
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256, discount=0.99, tau=0.005,
              policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (not_done * discount * target_Q)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % policy_freq == 0:
            actor_loss = -self.critic.forward(state, self.actor(state))[0].mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic.state_dict(), filename + "_critic")

    def load(self, filename):
        # 推理时只加载 Actor 即可
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=device))




class ActorSAC(nn.Module):
    """
    SAC策略网络：输出高斯分布的均值和对数标准差
    使用重参数化技巧进行采样
    """

    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(ActorSAC, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)

        # 均值和对数标准差分别输出
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))

        mean = self.mean(a)
        log_std = self.log_std(a)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state):
        """
        重参数化采样，返回动作和log_prob
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # 重参数化技巧
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 带梯度的采样

        # 通过tanh压缩到[-1, 1]，再映射到动作空间
        action = torch.tanh(x_t) * self.max_action

        # 计算log_prob，考虑tanh变换的修正
        # log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) / self.max_action**2 + 1e-6)
        # 修正：action已经乘以max_action，需要归一化回去计算
        log_prob = normal.log_prob(x_t) - torch.log(1 - torch.tanh(x_t).pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

    def get_action(self, state, deterministic=False):
        """
        用于推理的确定性或随机动作获取
        """
        mean, log_std = self.forward(state)

        if deterministic:
            return torch.tanh(mean) * self.max_action
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.sample()
            return torch.tanh(x_t) * self.max_action


class CriticSAC(nn.Module):
    """
    SAC使用的Critic，与TD3的Critic结构相同但使用方式不同
    只使用一个Q网络（SAC通常使用双胞胎Q网络防止过估计）
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CriticSAC, self).__init__()

        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


class SAC:
    """
    Soft Actor-Critic算法主体
    特点：最大熵框架、自动温度调节、重参数化技巧
    """

    def __init__(self, state_dim, action_dim, max_action, lr=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2, auto_tune_alpha=True):
        self.device = device

        # 策略网络（随机策略）
        self.actor = ActorSAC(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Critic网络（双胞胎Q网络，与TD3类似）
        self.critic_1 = CriticSAC(state_dim, action_dim).to(device)
        self.critic_2 = CriticSAC(state_dim, action_dim).to(device)

        self.critic_1_target = CriticSAC(state_dim, action_dim).to(device)
        self.critic_2_target = CriticSAC(state_dim, action_dim).to(device)

        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr)

        # 最大熵温度系数alpha
        self.auto_tune_alpha = auto_tune_alpha
        if auto_tune_alpha:
            # 自动调节：目标熵通常为 -dim(A)
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.total_it = 0

    def select_action(self, state, deterministic=False):
        """
        选择动作，与TD3接口保持一致
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor.get_action(state, deterministic=deterministic)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        """
        SAC训练步骤
        """
        self.total_it += 1

        # 采样batch
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # ------------------- Critic更新 -------------------
        with torch.no_grad():
            # 采样下一个动作和log_prob
            next_action, next_log_prob = self.actor.sample(next_state)

            # 目标Q值计算（考虑熵正则）
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob

            target_q = reward + (not_done * self.gamma * target_q)

        # 当前Q值
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        # Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # 更新Critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # ------------------- Actor更新 -------------------
        # 采样动作和log_prob（重参数化）
        new_action, log_prob = self.actor.sample(state)

        # 计算当前策略的Q值
        q1_new = self.critic_1(state, new_action)
        q2_new = self.critic_2(state, new_action)
        q_new = torch.min(q1_new, q2_new)

        # Actor损失：最大化Q - alpha * log_prob（即最小化alpha*log_prob - Q）
        actor_loss = (self.alpha * log_prob - q_new).mean()

        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------- Alpha更新 -------------------
        if self.auto_tune_alpha:
            # 计算alpha损失
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

        # ------------------- 软更新目标网络 -------------------
        for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha
        }

    def save(self, filename):
        """保存模型"""
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic_1.state_dict(), filename + "_critic1")
        torch.save(self.critic_2.state_dict(), filename + "_critic2")
        if self.auto_tune_alpha:
            torch.save(self.log_alpha, filename + "_log_alpha")

    def load(self, filename):
        """加载模型（推理时使用）"""
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=self.device))
        self.critic_1.load_state_dict(torch.load(filename + "_critic1", map_location=self.device))
        self.critic_2.load_state_dict(torch.load(filename + "_critic2", map_location=self.device))
        if self.auto_tune_alpha:
            self.log_alpha = torch.load(filename + "_log_alpha", map_location=self.device)
            self.alpha = self.log_alpha.exp().item()