"""08 · TD3: DDPG + 三板斧 (Clipped Double Q / 目标策略平滑 / 延迟更新).

对应 README:
  公式 (3)  y = r + gamma * min_j Q_j^-(s', a~'),  a~' = clip(mu^-(s') + clip(eps,-c,c))
  第 3 节   延迟策略更新 (policy_delay=2)

运行: python 08_continuous_control/td3.py
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
        to = lambda x: torch.as_tensor(x, dtype=torch.float32)
        return to(s), to(a), to(r), to(s2), to(d)

    def __len__(self):
        return len(self.buf)


def soft_update(net, tgt, tau):
    with torch.no_grad():
        for p, p_t in zip(net.parameters(), tgt.parameters()):
            p_t.mul_(1 - tau).add_(tau * p)


def train(total_steps=30_000, start_steps=1_000, batch=128, gamma=0.99,
          tau=0.005, act_noise=0.1, target_noise=0.2, noise_clip=0.5,
          policy_delay=2, lr=1e-3, seed=0, solve_at=-200.0):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    actor = Actor(obs_dim, act_dim, act_limit)
    actor_t = Actor(obs_dim, act_dim, act_limit)
    actor_t.load_state_dict(actor.state_dict())
    q1, q2 = Critic(obs_dim, act_dim), Critic(obs_dim, act_dim)
    q1_t, q2_t = Critic(obs_dim, act_dim), Critic(obs_dim, act_dim)
    q1_t.load_state_dict(q1.state_dict()); q2_t.load_state_dict(q2.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_c = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    buf = ReplayBuffer(100_000)

    obs, _ = env.reset(seed=seed)
    ep_ret, ep_returns, n_updates = 0.0, [], 0
    for step in range(total_steps):
        if step < start_steps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()
            a = np.clip(a + act_noise * act_limit * np.random.randn(act_dim),
                        -act_limit, act_limit)
        obs2, r, terminated, truncated, _ = env.step(a)
        buf.push(obs, a, r, obs2, float(terminated))
        obs = obs2; ep_ret += r
        if terminated or truncated:
            ep_returns.append(ep_ret); ep_ret = 0.0
            obs, _ = env.reset()

        if len(buf) >= batch and step >= start_steps:
            s, a_b, r_b, s2, d = buf.sample(batch)
            with torch.no_grad():
                # ② 目标策略平滑: 目标动作加截断噪声
                eps = (torch.randn_like(a_b) * target_noise
                       ).clamp(-noise_clip, noise_clip)
                a2 = (actor_t(s2) + eps).clamp(-act_limit, act_limit)
                # ① Clipped Double Q: 两个目标 Critic 取 min, 公式 (3)
                y = r_b + gamma * (1 - d) * torch.min(q1_t(s2, a2), q2_t(s2, a2))
            c_loss = nn.functional.mse_loss(q1(s, a_b), y) + \
                nn.functional.mse_loss(q2(s, a_b), y)
            opt_c.zero_grad(); c_loss.backward(); opt_c.step()
            n_updates += 1

            # ③ 延迟策略更新
            if n_updates % policy_delay == 0:
                a_loss = -q1(s, actor(s)).mean()
                opt_a.zero_grad(); a_loss.backward(); opt_a.step()
                soft_update(actor, actor_t, tau)
                soft_update(q1, q1_t, tau)
                soft_update(q2, q2_t, tau)

        if ep_returns and (step + 1) % 2000 == 0:
            avg20 = np.mean(ep_returns[-20:])
            print(f"  step {step+1:6d}  episodes {len(ep_returns):4d}  avg20 {avg20:8.1f}")
            if avg20 >= solve_at and len(ep_returns) >= 20:
                print(f"  ✓ 达标 (avg20 = {avg20:.1f} >= {solve_at})")
                break
    env.close()
    return ep_returns


if __name__ == "__main__":
    print("=== TD3 on Pendulum-v1 ===")
    train()
