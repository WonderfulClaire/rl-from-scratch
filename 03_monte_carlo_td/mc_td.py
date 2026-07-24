"""03 · Random Walk 上的 MC vs TD(0) 对比 (复现 Sutton & Barto 图 6.2 结论).

环境: 5 个非终止状态 A-B-C-D-E, 从 C 出发, 每步等概率左右移动.
      左端终止奖励 0, 右端终止奖励 +1, gamma = 1.
真实值: V(A..E) = 1/6, 2/6, 3/6, 4/6, 5/6 (可由线性方程组解析求出).

对应 README:
  公式 (1) first-visit MC 更新
  公式 (2) TD(0) 更新

运行: python 03_monte_carlo_td/mc_td.py
"""

import numpy as np

N_STATES = 5          # A=0 ... E=4
START = 2             # C
TRUE_V = np.arange(1, 6) / 6.0


def gen_episode(rng):
    """按均匀随机策略生成一条轨迹: [(state, reward), ...], 末尾奖励含终止."""
    s = START
    traj = []  # (s_t, r_{t+1})
    while True:
        step = rng.choice([-1, 1])
        s_next = s + step
        if s_next < 0:
            traj.append((s, 0.0)); return traj
        if s_next >= N_STATES:
            traj.append((s, 1.0)); return traj
        traj.append((s, 0.0))
        s = s_next


def run_mc(episodes, alpha, seed):
    """first-visit MC, 公式 (1). gamma=1 时 G_t = 最终奖励."""
    rng = np.random.default_rng(seed)
    V = np.full(N_STATES, 0.5)
    errs = []
    for _ in range(episodes):
        traj = gen_episode(rng)
        G = traj[-1][1]  # gamma=1 且中间奖励为 0 -> 所有 t 的回报都等于末奖励
        seen = set()
        for s, _ in traj:
            if s not in seen:      # first-visit
                V[s] += alpha * (G - V[s])
                seen.add(s)
        errs.append(np.sqrt(np.mean((V - TRUE_V) ** 2)))
    return np.array(errs)


def run_td0(episodes, alpha, seed):
    """TD(0), 公式 (2)."""
    rng = np.random.default_rng(seed)
    V = np.full(N_STATES, 0.5)
    errs = []
    for _ in range(episodes):
        s = START
        while True:
            step = rng.choice([-1, 1])
            s_next = s + step
            if s_next < 0:                       # 左端终止, V(terminal)=0
                V[s] += alpha * (0.0 + 0.0 - V[s])
                break
            if s_next >= N_STATES:               # 右端终止, 奖励 +1
                V[s] += alpha * (1.0 + 0.0 - V[s])
                break
            V[s] += alpha * (0.0 + V[s_next] - V[s])  # delta_t = r + V(s') - V(s)
            s = s_next
        errs.append(np.sqrt(np.mean((V - TRUE_V) ** 2)))
    return np.array(errs)


def main():
    episodes, n_seeds = 100, 100
    print(f"Random Walk: {n_seeds} 个种子平均, 每个 {episodes} 回合\n")
    print("真实值 V(A..E) =", np.round(TRUE_V, 4), "\n")

    settings = [("MC   alpha=0.03", run_mc, 0.03),
                ("MC   alpha=0.01", run_mc, 0.01),
                ("TD(0) alpha=0.10", run_td0, 0.10),
                ("TD(0) alpha=0.05", run_td0, 0.05)]

    checkpoints = [0, 4, 9, 24, 49, 99]
    header = "  ".join(f"ep{c+1:>3d}" for c in checkpoints)
    print(f"{'RMSE':22s}  {header}")
    results = {}
    for name, fn, alpha in settings:
        curves = np.stack([fn(episodes, alpha, seed) for seed in range(n_seeds)])
        mean_curve = curves.mean(axis=0)
        results[name] = mean_curve
        vals = "  ".join(f"{mean_curve[c]:.3f}" for c in checkpoints)
        print(f"{name:22s}  {vals}")

    best_td = min(results["TD(0) alpha=0.10"][-1], results["TD(0) alpha=0.05"][-1])
    best_mc = min(results["MC   alpha=0.03"][-1], results["MC   alpha=0.01"][-1])
    print(f"\n100 回合后最优 RMSE:  TD(0) = {best_td:.4f}   MC = {best_mc:.4f}")
    print("结论: 同等数据量下 TD(0) 误差更低 ->", "复现成功 ✓" if best_td < best_mc else "未复现 ✗")


if __name__ == "__main__":
    main()
