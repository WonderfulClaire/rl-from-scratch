"""06 · A2C: Advantage Actor-Critic (含 GAE 与熵正则).

对应 README:
  第 4 节   Actor 损失 -log pi * A_hat, Critic 损失 TD 误差平方, 熵正则
  公式 (5)  GAE: A_t = delta_t + gamma * lambda * A_{t+1} (反向递推)

运行: python 06_policy_gradient/a2c.py
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


def ortho_init(layer, gain=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    """策略/价值分离网络: 避免价值梯度污染策略特征 (共享体的常见坑)."""

    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.pi_body = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            ortho_init(nn.Linear(hidden, hidden)), nn.Tanh())
        self.pi_head = ortho_init(nn.Linear(hidden, n_actions), gain=0.01)
        self.v_body = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            ortho_init(nn.Linear(hidden, hidden)), nn.Tanh())
        self.v_head = ortho_init(nn.Linear(hidden, 1), gain=1.0)

    def forward(self, x):
        return torch.distributions.Categorical(logits=self.pi_head(self.pi_body(x))), \
            self.v_head(self.v_body(x)).squeeze(-1)


def compute_gae(rewards, values, last_value, terminated_flags, gamma, lam):
    """公式 (5) 的反向递推: A_t = delta_t + gamma*lambda*(1-term)*A_{t+1}."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    next_v, gae = last_value, 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - terminated_flags[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
        next_v = values[t]
    return adv


def train(total_steps=400_000, rollout_len=256, gamma=0.99, lam=0.95,
          lr=3e-4, ent_coef=0.005, vf_coef=0.5, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make("CartPole-v1")
    ac = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    opt = torch.optim.Adam(ac.parameters(), lr=lr)

    obs, _ = env.reset(seed=seed)
    ep_ret, ep_returns, step = 0.0, [], 0
    while step < total_steps:
        # ---- 收集一段 rollout ----
        obs_buf, act_buf, rew_buf, val_buf, logp_buf, term_buf = [], [], [], [], [], []
        for _ in range(rollout_len):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                dist, v = ac(obs_t)
            a = dist.sample()
            next_obs, r, terminated, truncated, _ = env.step(int(a))
            obs_buf.append(obs); act_buf.append(int(a)); rew_buf.append(r)
            val_buf.append(float(v)); logp_buf.append(float(dist.log_prob(a)))
            term_buf.append(float(terminated))  # truncated 仍 bootstrap
            ep_ret += r
            step += 1
            obs = next_obs
            if terminated or truncated:
                ep_returns.append(ep_ret)
                ep_ret = 0.0
                obs, _ = env.reset()
        with torch.no_grad():
            _, last_v = ac(torch.as_tensor(obs, dtype=torch.float32))

        adv = compute_gae(np.array(rew_buf), np.array(val_buf), float(last_v),
                          np.array(term_buf), gamma, lam)
        ret = adv + np.array(val_buf)                      # TD(lambda) 回报 = A + V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        s = torch.as_tensor(np.array(obs_buf), dtype=torch.float32)
        a = torch.as_tensor(act_buf, dtype=torch.int64)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)

        dist, v = ac(s)
        logp = dist.log_prob(a)
        pi_loss = -(logp * adv_t).mean()                   # Actor: 策略梯度
        v_loss = nn.functional.mse_loss(v, ret_t)          # Critic: 回归 TD(lambda) 目标
        ent = dist.entropy().mean()                        # 熵正则: 防止过早确定化
        loss = pi_loss + vf_coef * v_loss - ent_coef * ent

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
        opt.step()

        if ep_returns:
            avg20 = np.mean(ep_returns[-20:])
            if step % (rollout_len * 20) == 0:
                print(f"  step {step:7d}  episodes {len(ep_returns):4d}  avg20 {avg20:6.1f}")
            if avg20 >= 475 and len(ep_returns) >= 20:
                print(f"  ✓ 在 {step} 步解决 CartPole (avg20 = {avg20:.1f})")
                break
    env.close()
    return ep_returns


if __name__ == "__main__":
    print("=== A2C (GAE + entropy) on CartPole-v1 ===")
    train()
