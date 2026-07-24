"""08 · DDPG: 确定性策略梯度 + DQN 全家桶.

对应 README:
  公式 (1) DPG 定理: grad J = E[grad_theta mu(s) * grad_a Q(s,a)|a=mu(s)]
  公式 (2) Critic 目标 (连续动作版 DQN)

运行: python 08_continuous_control/ddpg.py
"""

import collections
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, act_dim), nn.Tanh())
        self.act_limit = act_limit

    def forward(self, s):
        return self.act_limit * self.net(s)


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
        to = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt)
        return to(s), to(a), to(r), to(s2), to(d)

    def __len__(self):
        return len(self.buf)


def soft_update(net, tgt, tau):
    with torch.no_grad():
        for p, p_t in zip(net.parameters(), tgt.parameters()):
            p_t.mul_(1 - tau).add_(tau * p)


def train(total_steps=30_000, start_steps=1_000, batch=128, gamma=0.99,
          tau=0.005, act_noise=0.1, lr=1e-3, seed=0, env_name="Pendulum-v1",
          solve_at=-200.0):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    actor, critic = Actor(obs_dim, act_dim, act_limit), Critic(obs_dim, act_dim)
    actor_t, critic_t = Actor(obs_dim, act_dim, act_limit), Critic(obs_dim, act_dim)
    actor_t.load_state_dict(actor.state_dict())
    critic_t.load_state_dict(critic.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr)
    buf = ReplayBuffer(100_000)

    obs, _ = env.reset(seed=seed)
    ep_ret, ep_returns = 0.0, []
    for step in range(total_steps):
        if step < start_steps:
            a = env.action_space.sample()  # 初期均匀随机探索
        else:
            with torch.no_grad():
                a = actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()
            a = np.clip(a + act_noise * act_limit * np.random.randn(act_dim),
                        -act_limit, act_limit)  # 高斯探索噪声
        obs2, r, terminated, truncated, _ = env.step(a)
        buf.push(obs, a, r, obs2, float(terminated))
        obs = obs2; ep_ret += r
        if terminated or truncated:
            ep_returns.append(ep_ret); ep_ret = 0.0
            obs, _ = env.reset()

        if len(buf) >= batch and step >= start_steps:
            s, a_b, r_b, s2, d = buf.sample(batch)
            with torch.no_grad():  # 公式 (2): 目标网络 + 目标动作
                y = r_b + gamma * (1 - d) * critic_t(s2, actor_t(s2))
            c_loss = nn.functional.mse_loss(critic(s, a_b), y)
            opt_c.zero_grad(); c_loss.backward(); opt_c.step()

            # 公式 (1): 最大化 Q(s, mu(s)) <=> 最小化其负值
            a_loss = -critic(s, actor(s)).mean()
            opt_a.zero_grad(); a_loss.backward(); opt_a.step()

            soft_update(actor, actor_t, tau)
            soft_update(critic, critic_t, tau)

        if ep_returns and (step + 1) % 2000 == 0:
            avg20 = np.mean(ep_returns[-20:])
            print(f"  step {step+1:6d}  episodes {len(ep_returns):4d}  avg20 {avg20:8.1f}")
            if avg20 >= solve_at and len(ep_returns) >= 20:
                print(f"  ✓ 达标 (avg20 = {avg20:.1f} >= {solve_at})")
                break
    env.close()
    return ep_returns


if __name__ == "__main__":
    print("=== DDPG on Pendulum-v1 ===")
    train()
