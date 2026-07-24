"""07 · PPO-clip 单文件实现 (GAE + minibatch 多 epoch + 熵正则).

对应 README:
  公式 (4)  L_CLIP = E[min(rho*A, clip(rho,1-eps,1+eps)*A)]
  3.1 节    完整算法流程
  3.2 节    实现细节 (优势标准化 / 正交初始化 / 梯度裁剪 / KL 监控)

运行: python 07_trpo_ppo/ppo.py
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
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.pi_body = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            ortho_init(nn.Linear(hidden, hidden)), nn.Tanh())
        self.pi_head = ortho_init(nn.Linear(hidden, n_actions), gain=0.01)
        self.v_body = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            ortho_init(nn.Linear(hidden, hidden)), nn.Tanh())
        self.v_head = ortho_init(nn.Linear(hidden, 1), gain=1.0)

    def dist(self, x):
        return torch.distributions.Categorical(logits=self.pi_head(self.pi_body(x)))

    def value(self, x):
        return self.v_head(self.v_body(x)).squeeze(-1)


def compute_gae(rew, val, last_v, term, gamma, lam):
    T = len(rew)
    adv, gae, next_v = np.zeros(T, dtype=np.float32), 0.0, last_v
    for t in reversed(range(T)):
        nonterm = 1.0 - term[t]
        delta = rew[t] + gamma * next_v * nonterm - val[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
        next_v = val[t]
    return adv


def train(total_steps=200_000, rollout_len=2048, epochs=10, minibatch=64,
          gamma=0.99, lam=0.95, clip_eps=0.2, lr=3e-4,
          ent_coef=0.01, vf_coef=0.5, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make("CartPole-v1")
    ac = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    opt = torch.optim.Adam(ac.parameters(), lr=lr, eps=1e-5)

    obs, _ = env.reset(seed=seed)
    ep_ret, ep_returns, step = 0.0, [], 0
    while step < total_steps:
        # ---------- 1) 用旧策略收集 rollout ----------
        S, A, R, V, LP, TERM = [], [], [], [], [], []
        for _ in range(rollout_len):
            s_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                dist, v = ac.dist(s_t), ac.value(s_t)
            a = dist.sample()
            next_obs, r, terminated, truncated, _ = env.step(int(a))
            S.append(obs); A.append(int(a)); R.append(r)
            V.append(float(v)); LP.append(float(dist.log_prob(a)))
            TERM.append(float(terminated))
            ep_ret += r; step += 1; obs = next_obs
            if terminated or truncated:
                ep_returns.append(ep_ret); ep_ret = 0.0
                obs, _ = env.reset()
        with torch.no_grad():
            last_v = float(ac.value(torch.as_tensor(obs, dtype=torch.float32)))

        # ---------- 2) GAE ----------
        adv = compute_gae(np.array(R), np.array(V), last_v,
                          np.array(TERM), gamma, lam)
        ret = adv + np.array(V)

        S_t = torch.as_tensor(np.array(S), dtype=torch.float32)
        A_t = torch.as_tensor(A, dtype=torch.int64)
        LP_old = torch.as_tensor(LP, dtype=torch.float32)
        ADV = torch.as_tensor(adv, dtype=torch.float32)
        RET = torch.as_tensor(ret, dtype=torch.float32)

        # ---------- 3) 多 epoch minibatch 更新 ----------
        idx = np.arange(rollout_len)
        kls, clip_fracs = [], []
        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, rollout_len, minibatch):
                mb = idx[start:start + minibatch]
                mb_adv = ADV[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)  # 优势标准化

                dist = ac.dist(S_t[mb])
                logp = dist.log_prob(A_t[mb])
                rho = torch.exp(logp - LP_old[mb])                # 重要性采样比
                surr1 = rho * mb_adv
                surr2 = torch.clamp(rho, 1 - clip_eps, 1 + clip_eps) * mb_adv
                pi_loss = -torch.min(surr1, surr2).mean()          # 公式 (4)

                v_loss = nn.functional.mse_loss(ac.value(S_t[mb]), RET[mb])
                ent = dist.entropy().mean()
                loss = pi_loss + vf_coef * v_loss - ent_coef * ent

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                opt.step()

                with torch.no_grad():
                    kls.append(float((LP_old[mb] - logp).mean()))  # approx KL
                    clip_fracs.append(float(((rho - 1).abs() > clip_eps).float().mean()))

        if ep_returns:
            avg20 = np.mean(ep_returns[-20:])
            print(f"  step {step:7d}  episodes {len(ep_returns):4d}  avg20 {avg20:6.1f}"
                  f"  KL {np.mean(kls):.4f}  clipfrac {np.mean(clip_fracs):.2f}")
            if avg20 >= 475 and len(ep_returns) >= 20:
                print(f"  ✓ 在 {step} 步解决 CartPole (avg20 = {avg20:.1f})")
                break
    env.close()
    return ep_returns


if __name__ == "__main__":
    print("=== PPO-clip on CartPole-v1 ===")
    train()
