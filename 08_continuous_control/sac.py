"""08 · SAC: 最大熵 RL (squashed Gaussian + 双 Q + 自动温度).

对应 README:
  公式 (4)  最大熵目标 J = E[sum r + alpha * H(pi)]
  公式 (5)  软贝尔曼: V(s) = E_a[Q(s,a) - alpha*log pi(a|s)]
  4.3 节    重参数化 + tanh 雅可比修正 + 自动温度 alpha

运行: python 08_continuous_control/sac.py
"""

import collections
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

LOG_STD_MIN, LOG_STD_MAX = -20, 2


class GaussianActor(nn.Module):
    """squashed Gaussian 策略: a = act_limit * tanh(m + sigma*xi)."""

    def __init__(self, obs_dim, act_dim, act_limit, hidden=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)
        self.act_limit = act_limit

    def forward(self, s, deterministic=False):
        h = self.body(s)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        u = mu if deterministic else dist.rsample()   # 重参数化: 梯度穿过 u
        a = torch.tanh(u)
        # tanh 换元的雅可比修正: log pi(a) = log rho(u) - sum log(1 - tanh^2(u))
        logp = dist.log_prob(u).sum(-1) - \
            (2 * (np.log(2) - u - nn.functional.softplus(-2 * u))).sum(-1)
        return self.act_limit * a, logp


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, cap):
        self.buf = collections.deque(maxlen=cap)

    def push(self, *tr):
        self.buf.append(tr)

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = map(np.array, zip(*batch))
        to = lambda x: torch.as_tensor(x, dtype=torch.float32)
        return to(s), to(a), to(r), to(s2), to(d)

    def __len__(self):
        return len(self.buf)


def soft_update(net, tgt, tau):
    with torch.no_grad():
        for p, p_t in zip(net.parameters(), tgt.parameters()):
            p_t.mul_(1 - tau).add_(tau * p)


def train(total_steps=30_000, start_steps=1_000, batch=128, gamma=0.99,
          tau=0.005, lr=3e-4, seed=0, solve_at=-200.0):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    actor = GaussianActor(obs_dim, act_dim, act_limit)
    q1, q2 = Critic(obs_dim, act_dim), Critic(obs_dim, act_dim)
    q1_t, q2_t = Critic(obs_dim, act_dim), Critic(obs_dim, act_dim)
    q1_t.load_state_dict(q1.state_dict()); q2_t.load_state_dict(q2.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_c = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)

    # 自动温度: 目标熵 = -dim(A) (惯例), 学 log_alpha 保证 alpha > 0
    target_entropy = -float(act_dim)
    log_alpha = torch.zeros(1, requires_grad=True)
    opt_alpha = torch.optim.Adam([log_alpha], lr=lr)

    buf = ReplayBuffer(100_000)
    obs, _ = env.reset(seed=seed)
    ep_ret, ep_returns = 0.0, []
    for step in range(total_steps):
        if step < start_steps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a, _ = actor(torch.as_tensor(obs, dtype=torch.float32))
            a = a.numpy()
        obs2, r, terminated, truncated, _ = env.step(a)
        buf.push(obs, a, r, obs2, float(terminated))
        obs = obs2; ep_ret += r
        if terminated or truncated:
            ep_returns.append(ep_ret); ep_ret = 0.0
            obs, _ = env.reset()

        if len(buf) >= batch and step >= start_steps:
            s, a_b, r_b, s2, d = buf.sample(batch)
            alpha = log_alpha.exp().detach()

            # ---- Critic: 软贝尔曼目标, 公式 (5) ----
            with torch.no_grad():
                a2, logp2 = actor(s2)
                q_min = torch.min(q1_t(s2, a2), q2_t(s2, a2))
                y = r_b + gamma * (1 - d) * (q_min - alpha * logp2)
            c_loss = nn.functional.mse_loss(q1(s, a_b), y) + \
                nn.functional.mse_loss(q2(s, a_b), y)
            opt_c.zero_grad(); c_loss.backward(); opt_c.step()

            # ---- Actor: 最小化 E[alpha*log pi - Q] (公式 6 的 KL 投影) ----
            a_new, logp = actor(s)
            q_new = torch.min(q1(s, a_new), q2(s, a_new))
            a_loss = (alpha * logp - q_new).mean()
            opt_a.zero_grad(); a_loss.backward(); opt_a.step()

            # ---- 温度: 对偶上升, 维持熵约束 ----
            alpha_loss = -(log_alpha.exp() * (logp.detach() + target_entropy)).mean()
            opt_alpha.zero_grad(); alpha_loss.backward(); opt_alpha.step()

            soft_update(q1, q1_t, tau)
            soft_update(q2, q2_t, tau)

        if ep_returns and (step + 1) % 2000 == 0:
            avg20 = np.mean(ep_returns[-20:])
            print(f"  step {step+1:6d}  episodes {len(ep_returns):4d}  "
                  f"avg20 {avg20:8.1f}  alpha {float(log_alpha.exp()):.3f}")
            if avg20 >= solve_at and len(ep_returns) >= 20:
                print(f"  ✓ 达标 (avg20 = {avg20:.1f} >= {solve_at})")
                break
    env.close()
    return ep_returns


if __name__ == "__main__":
    print("=== SAC on Pendulum-v1 ===")
    train()
