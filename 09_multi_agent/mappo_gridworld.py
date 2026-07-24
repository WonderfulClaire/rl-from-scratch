"""09 · IPPO vs MAPPO: 2 智能体合作会合任务.

环境: 5x5 网格, 两个智能体从 (0,0) 和 (4,4) 出发.
     两人**同时**位于中心 3x3 区域内的同一格才算会合成功: 团队奖励 +10, 回合结束.
     每步团队奖励 -0.1 (鼓励尽快), 最多 50 步.
     观测: 自己的位置 (one-hot 2 维坐标归一化) + 对方位置 + agent id.

对比:
  IPPO : Critic 只看自己的观测
  MAPPO: Critic 看全局状态 (两人位置拼接) —— CTDE

对应 README 第 4-5 节.

运行: python 09_multi_agent/mappo_gridworld.py
"""

import numpy as np
import torch
import torch.nn as nn

N = 5
MAX_STEPS = 50
CENTER = {(r, c) for r in range(1, 4) for c in range(1, 4)}
DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}  # 4=不动


class RendezvousEnv:
    """完全合作: 共享团队奖励."""

    def reset(self, rng):
        # 随机出生在边缘 (提高协调难度: "去哪个格会合" 每回合都要重新协商)
        edges = [(r, c) for r in range(N) for c in range(N)
                 if r in (0, N - 1) or c in (0, N - 1)]
        i, j = rng.choice(len(edges), size=2, replace=False)
        self.pos = [np.array(edges[i]), np.array(edges[j])]
        self.t = 0
        return self._obs()

    def _obs(self):
        """每个智能体: [自己坐标/N, 对方坐标/N, id] -> 5 维."""
        obs = []
        for i in range(2):
            me, other = self.pos[i] / (N - 1), self.pos[1 - i] / (N - 1)
            obs.append(np.concatenate([me, other, [float(i)]]).astype(np.float32))
        return obs

    def global_state(self):
        return np.concatenate([self.pos[0], self.pos[1]]).astype(np.float32) / (N - 1)

    def step(self, actions):
        for i, a in enumerate(actions):
            dr, dc = DELTAS[int(a)]
            self.pos[i] = np.clip(self.pos[i] + [dr, dc], 0, N - 1)
        self.t += 1
        p0, p1 = tuple(self.pos[0]), tuple(self.pos[1])
        if p0 == p1 and p0 in CENTER:
            return self._obs(), 10.0, True
        return self._obs(), -0.1, self.t >= MAX_STEPS


class ActorCritic(nn.Module):
    """共享参数的 Actor (输入含 agent id) + 可切换输入的 Critic."""

    def __init__(self, obs_dim, state_dim, n_actions, centralized, hidden=64):
        super().__init__()
        self.centralized = centralized
        self.actor = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh(),
                                   nn.Linear(hidden, n_actions))
        critic_in = state_dim if centralized else obs_dim
        self.critic = nn.Sequential(nn.Linear(critic_in, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh(),
                                    nn.Linear(hidden, 1))

    def dist(self, obs):
        return torch.distributions.Categorical(logits=self.actor(obs))

    def value(self, critic_input):
        return self.critic(critic_input).squeeze(-1)


def train(centralized, iters=300, episodes_per_iter=16, gamma=0.99, lam=0.95,
          clip_eps=0.2, epochs=4, lr=3e-4, seed=0, quiet=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = RendezvousEnv()
    ac = ActorCritic(obs_dim=5, state_dim=4, n_actions=5, centralized=centralized)
    opt = torch.optim.Adam(ac.parameters(), lr=lr)

    success_hist = []
    for it in range(iters):
        # ---------- 采样 ----------
        OBS, CIN, ACT, LP, REW, VAL, DONE = [], [], [], [], [], [], []
        n_success = 0
        for _ in range(episodes_per_iter):
            obs = env.reset(rng)
            done = False
            while not done:
                state = env.global_state()
                acts, lps, vals = [], [], []
                for i in range(2):
                    o = torch.as_tensor(obs[i])
                    dist = ac.dist(o)
                    a = dist.sample()
                    cin = torch.as_tensor(state) if centralized else o
                    acts.append(int(a))
                    lps.append(float(dist.log_prob(a)))
                    vals.append(float(ac.value(cin)))
                next_obs, r, done = env.step(acts)
                for i in range(2):  # 两个智能体各存一条 (共享团队奖励)
                    OBS.append(obs[i])
                    CIN.append(state if centralized else obs[i])
                    ACT.append(acts[i]); LP.append(lps[i])
                    REW.append(r); VAL.append(vals[i]); DONE.append(float(done))
                obs = next_obs
            if r > 0:
                n_success += 1
        success_hist.append(n_success / episodes_per_iter)

        # ---------- GAE (按 done 切断) ----------
        T = len(REW)
        adv, gae, next_v = np.zeros(T, dtype=np.float32), 0.0, 0.0
        for t in reversed(range(T)):
            nonterm = 1.0 - DONE[t]
            delta = REW[t] + gamma * next_v * nonterm - VAL[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
            next_v = VAL[t]
        ret = adv + np.array(VAL, dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.as_tensor(np.array(OBS))
        cin_t = torch.as_tensor(np.array(CIN))
        act_t = torch.as_tensor(ACT)
        lp_old = torch.as_tensor(LP)
        adv_t = torch.as_tensor(adv)
        ret_t = torch.as_tensor(ret)

        # ---------- PPO 更新 ----------
        for _ in range(epochs):
            dist = ac.dist(obs_t)
            logp = dist.log_prob(act_t)
            rho = torch.exp(logp - lp_old)
            surr = torch.min(rho * adv_t,
                             torch.clamp(rho, 1 - clip_eps, 1 + clip_eps) * adv_t)
            v = ac.value(cin_t)
            loss = -surr.mean() + 0.5 * nn.functional.mse_loss(v, ret_t) \
                - 0.01 * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
            opt.step()

        if not quiet and (it + 1) % 50 == 0:
            rate = np.mean(success_hist[-20:])
            print(f"  iter {it+1:4d}  近20轮会合成功率 {rate:.2f}")
    return success_hist


def iters_to_solve(curve, threshold=0.9, window=10):
    """达到滑动平均成功率 >= threshold 所需迭代数 (未达到返回 len)."""
    for t in range(window, len(curve)):
        if np.mean(curve[t - window:t]) >= threshold:
            return t
    return len(curve)


def main():
    n_seeds = 3
    print("=== 合作会合任务 (随机出生点): IPPO vs MAPPO, 各 3 种子平均 ===\n")
    results = {}
    for name, central in [("IPPO (局部 Critic)", False),
                          ("MAPPO (中心化 Critic)", True)]:
        print(f"--- {name} ---")
        curves = [train(central, seed=sd, quiet=(sd > 0)) for sd in range(n_seeds)]
        final = np.mean([np.mean(c[-20:]) for c in curves])
        speed = np.mean([iters_to_solve(c) for c in curves])
        results[name] = (speed, final)
        print(f"  达到90%成功率平均需 {speed:.0f} 轮   最终成功率 {final:.2f}\n")

    print("汇总 (协调信息只有中心化 Critic 能编码 -> MAPPO 预期收敛更快):")
    for k, (s, f) in results.items():
        print(f"  {k:24s} 收敛轮数 {s:5.0f}  最终成功率 {f:.2f}")


if __name__ == "__main__":
    main()
