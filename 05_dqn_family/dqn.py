"""05 · DQN 家族单文件实现: DQN / Double DQN / Dueling DQN / 优先经验回放.

对应 README:
  公式 (1) TD 目标 + 目标网络 (semi-gradient)
  公式 (2) Double DQN: 在线网络选动作, 目标网络评动作
  公式 (3) Dueling: Q = V + (A - mean A)
  公式 (4)(5) PER: 比例优先级 + 重要性采样权重

运行:
  python 05_dqn_family/dqn.py                     # 原始 DQN
  python 05_dqn_family/dqn.py --double --dueling  # Double + Dueling
  python 05_dqn_family/dqn.py --double --per      # Double + PER
"""

import argparse
import collections
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------- 网络结构
class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, dueling=False, hidden=128):
        super().__init__()
        self.dueling = dueling
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        if dueling:
            self.v_head = nn.Linear(hidden, 1)          # V(s)
            self.a_head = nn.Linear(hidden, n_actions)  # A(s,a)
        else:
            self.q_head = nn.Linear(hidden, n_actions)

    def forward(self, x):
        h = self.body(x)
        if self.dueling:
            v, a = self.v_head(h), self.a_head(h)
            return v + a - a.mean(dim=1, keepdim=True)  # 公式 (3)
        return self.q_head(h)


# ---------------------------------------------------------------- 回放池
class ReplayBuffer:
    """均匀采样回放池."""

    def __init__(self, cap):
        self.buf = collections.deque(maxlen=cap)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return (*map(np.array, zip(*batch)), None, None)  # 对齐 PER 的返回签名

    def __len__(self):
        return len(self.buf)


class PrioritizedReplayBuffer:
    """比例优先级回放 (公式 4) + 重要性采样权重 (公式 5).

    教学实现用 O(n) 的 numpy 采样; 生产实现应换 SumTree 到 O(log n).
    """

    def __init__(self, cap, alpha=0.6):
        self.cap, self.alpha = cap, alpha
        self.buf, self.prios = [], np.zeros(cap, dtype=np.float64)
        self.pos = 0

    def push(self, *transition):
        max_p = self.prios[: len(self.buf)].max() if self.buf else 1.0
        if len(self.buf) < self.cap:
            self.buf.append(transition)
        else:
            self.buf[self.pos] = transition
        self.prios[self.pos] = max_p  # 新样本给最大优先级, 保证至少被抽一次
        self.pos = (self.pos + 1) % self.cap

    def sample(self, batch_size, beta=0.4):
        p = self.prios[: len(self.buf)] ** self.alpha
        p /= p.sum()                                     # 公式 (4)
        idx = np.random.choice(len(self.buf), batch_size, p=p)
        w = (len(self.buf) * p[idx]) ** (-beta)          # 公式 (5)
        w /= w.max()
        batch = [self.buf[i] for i in idx]
        return (*map(np.array, zip(*batch)), idx, w.astype(np.float32))

    def update_priorities(self, idx, td_errors, eps=1e-5):
        self.prios[idx] = np.abs(td_errors) + eps

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------- 训练
def train(use_double=False, use_dueling=False, use_per=False,
          episodes=400, seed=0, quiet=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net = QNet(obs_dim, n_actions, dueling=use_dueling)
    tgt_net = QNet(obs_dim, n_actions, dueling=use_dueling)
    tgt_net.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=1e-3)

    buffer = PrioritizedReplayBuffer(50_000) if use_per else ReplayBuffer(50_000)
    gamma, batch_size = 0.99, 64
    eps_start, eps_end, eps_decay_steps = 1.0, 0.02, 8_000
    target_sync, warmup = 500, 1_000
    beta_start, beta_frames = 0.4, 40_000

    step_count, ep_returns = 0, []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_ret, done = 0.0, False
        while not done:
            eps = max(eps_end, eps_start - step_count / eps_decay_steps)
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = int(q_net(torch.as_tensor(obs, dtype=torch.float32)
                                       .unsqueeze(0)).argmax())
            next_obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # 关键坑: truncated (超时) 不是真终止, 仍需 bootstrap -> 只存 terminated
            buffer.push(obs, action, r, next_obs, float(terminated))
            obs = next_obs
            ep_ret += r
            step_count += 1

            if len(buffer) >= warmup:
                beta = min(1.0, beta_start + step_count * (1 - beta_start) / beta_frames)
                if use_per:
                    s, a, rew, s2, term, idx, w = buffer.sample(batch_size, beta)
                else:
                    s, a, rew, s2, term, idx, w = buffer.sample(batch_size)
                s = torch.as_tensor(s, dtype=torch.float32)
                a = torch.as_tensor(a, dtype=torch.int64)
                rew = torch.as_tensor(rew, dtype=torch.float32)
                s2 = torch.as_tensor(s2, dtype=torch.float32)
                term = torch.as_tensor(term, dtype=torch.float32)

                q_sa = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():  # 目标不回传梯度 (semi-gradient)
                    if use_double:
                        a_star = q_net(s2).argmax(dim=1, keepdim=True)      # 在线网选
                        q_next = tgt_net(s2).gather(1, a_star).squeeze(1)   # 目标网评, 公式 (2)
                    else:
                        q_next = tgt_net(s2).max(dim=1).values
                    y = rew + gamma * (1.0 - term) * q_next                  # 公式 (1)

                td_err = y - q_sa
                if use_per:
                    w_t = torch.as_tensor(w, dtype=torch.float32)
                    loss = (w_t * nn.functional.smooth_l1_loss(
                        q_sa, y, reduction="none")).mean()
                    buffer.update_priorities(idx, td_err.detach().numpy())
                else:
                    loss = nn.functional.smooth_l1_loss(q_sa, y)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                opt.step()

                if step_count % target_sync == 0:
                    tgt_net.load_state_dict(q_net.state_dict())

        ep_returns.append(ep_ret)
        avg20 = np.mean(ep_returns[-20:])
        if not quiet and (ep + 1) % 20 == 0:
            print(f"  ep {ep+1:4d}  return {ep_ret:6.1f}  avg20 {avg20:6.1f}  eps {eps:.3f}")
        if avg20 >= 475 and len(ep_returns) >= 20:
            print(f"  ✓ 在第 {ep+1} 回合解决 CartPole (avg20 = {avg20:.1f})")
            break

    env.close()
    return ep_returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--double", action="store_true", help="Double DQN")
    parser.add_argument("--dueling", action="store_true", help="Dueling 结构")
    parser.add_argument("--per", action="store_true", help="优先经验回放")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    name = "DQN" + ("+Double" if args.double else "") + \
           ("+Dueling" if args.dueling else "") + ("+PER" if args.per else "")
    print(f"=== {name} on CartPole-v1 ===")
    train(args.double, args.dueling, args.per, args.episodes, args.seed)
