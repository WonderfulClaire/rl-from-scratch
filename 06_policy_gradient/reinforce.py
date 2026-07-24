"""06 · REINFORCE (蒙特卡洛策略梯度) + 滑动平均基线.

对应 README 公式 (4):
  grad J = E[ sum_t grad log pi(a_t|s_t) * (G_t - b) ]

运行: python 06_policy_gradient/reinforce.py
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return torch.distributions.Categorical(logits=self.net(x))


def train(episodes=1500, gamma=0.99, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make("CartPole-v1")
    pi = PolicyNet(env.observation_space.shape[0], env.action_space.n)
    opt = torch.optim.Adam(pi.parameters(), lr=lr)

    baseline = 0.0  # 回报的指数滑动平均, 作为常数基线 b
    ep_returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        log_probs, rewards, done = [], [], False
        while not done:
            dist = pi(torch.as_tensor(obs, dtype=torch.float32))
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            obs, r, terminated, truncated, _ = env.step(int(action))
            rewards.append(r)
            done = terminated or truncated

        # reward-to-go: G_t = r_{t+1} + gamma G_{t+1} (公式 3 的因果版目标)
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()
        returns_t = torch.as_tensor(returns, dtype=torch.float32)

        ep_ret = sum(rewards)
        baseline = 0.95 * baseline + 0.05 * ep_ret if ep else ep_ret

        # 标准化优势 (工程技巧: 进一步压方差, 不改变期望方向)
        adv = returns_t - baseline / max(len(rewards), 1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        loss = -(torch.stack(log_probs) * adv).sum()  # 负号: 梯度上升
        opt.zero_grad()
        loss.backward()
        opt.step()

        ep_returns.append(ep_ret)
        avg20 = np.mean(ep_returns[-20:])
        if (ep + 1) % 50 == 0:
            print(f"  ep {ep+1:4d}  return {ep_ret:6.1f}  avg20 {avg20:6.1f}")
        if avg20 >= 475 and len(ep_returns) >= 20:
            print(f"  ✓ 在第 {ep+1} 回合解决 CartPole (avg20 = {avg20:.1f})")
            break
    env.close()
    return ep_returns


if __name__ == "__main__":
    print("=== REINFORCE on CartPole-v1 ===")
    train()
