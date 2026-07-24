"""
benchmark.py — 统一学习曲线基准

把各章的 train() 拉过来，在同一任务上跑多个随机种子，
记录每回合回报，插值对齐到公共横轴（环境步数），画出
"均值 ± 标准差"对比曲线。所有数字都来自真实运行。

用法:
    python tools/benchmark.py --suite cartpole --seeds 3
    python tools/benchmark.py --suite pendulum --seeds 3

产物:
    results/<suite>_curves.png   对比曲线图
    results/<suite>_curves.npz   原始曲线数据（可复现绘图）

设计要点:
- CartPole 每步奖励 +1，所以"回合回报 == 回合步数"，
  cumsum(returns) 即环境步数，天然可对齐不同算法。
- Pendulum 每回合定长 200 步，步数 = 200 * 回合序号。
- 多种子用 np.interp 插值到公共步网格后求均值/标准差。
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")


def load_train(rel_path, func="train"):
    """从数字开头的章节目录里加载 train()（无法用普通 import）。"""
    path = os.path.join(REPO, rel_path)
    name = rel_path.replace("/", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, func)


# suite -> [(label, callable(seed)->ep_returns, steps_per_ep or None)]
def cartpole_suite():
    dqn = load_train("05_dqn_family/dqn.py")
    reinforce = load_train("06_policy_gradient/reinforce.py")
    a2c = load_train("06_policy_gradient/a2c.py")
    ppo = load_train("07_trpo_ppo/ppo.py")
    return [
        ("REINFORCE", lambda s: reinforce(episodes=1200, seed=s), None),
        ("A2C",       lambda s: a2c(total_steps=400_000, seed=s), None),
        ("PPO",       lambda s: ppo(total_steps=200_000, seed=s), None),
        ("DQN (Double+Dueling)",
         lambda s: dqn(use_double=True, use_dueling=True, episodes=600, seed=s, quiet=True), None),
    ]


def pendulum_suite():
    ddpg = load_train("08_continuous_control/ddpg.py")
    td3 = load_train("08_continuous_control/td3.py")
    sac = load_train("08_continuous_control/sac.py")
    return [
        ("DDPG", lambda s: ddpg(total_steps=30_000, seed=s), 200),
        ("TD3",  lambda s: td3(total_steps=30_000, seed=s), 200),
        ("SAC",  lambda s: sac(total_steps=30_000, seed=s), 200),
    ]


def steps_axis(returns, steps_per_ep):
    """由回合回报重建环境步数横轴。"""
    returns = np.asarray(returns, dtype=float)
    if steps_per_ep is None:            # CartPole: 回报=步数
        return np.cumsum(returns)
    return steps_per_ep * np.arange(1, len(returns) + 1)


def moving_average(x, w):
    if len(x) < w:
        return x.copy()
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


def run_algo(label, fn, steps_per_ep, seeds, smooth):
    """跑多种子，返回 (公共步网格, 均值曲线, 标准差, 每种子解决步数)。"""
    curves = []          # list of (steps, smoothed_returns)
    solve_steps = []
    for sd in seeds:
        t0 = time.time()
        rets = fn(sd)
        rets = np.asarray(rets, dtype=float)
        xs = steps_axis(rets, steps_per_ep)
        sm = moving_average(rets, smooth)
        xs_sm = xs[len(rets) - len(sm):]     # 对齐 valid 卷积后的横轴
        curves.append((xs_sm, sm))
        solve_steps.append(float(xs[-1]))
        print(f"    [{label}] seed={sd}  episodes={len(rets)}  "
              f"final_steps={xs[-1]:.0f}  t={time.time()-t0:.0f}s", flush=True)

    # 公共步网格：到所有种子里"最短的最大步数"，保证都有数据
    x_max = min(c[0][-1] for c in curves)
    grid = np.linspace(0, x_max, 300)
    stacked = np.vstack([np.interp(grid, xs, ys) for xs, ys in curves])
    return grid, stacked.mean(0), stacked.std(0), solve_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["cartpole", "pendulum"], required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--smooth", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    seeds = list(range(args.seeds))
    suite = cartpole_suite() if args.suite == "cartpole" else pendulum_suite()
    threshold = 475 if args.suite == "cartpole" else -200
    env_name = "CartPole-v1" if args.suite == "cartpole" else "Pendulum-v1"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]  # 红蓝绿紫
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=130)
    save = {}
    summary = []
    for i, (label, fn, spe) in enumerate(suite):
        print(f"  == running {label} ({args.seeds} seeds) ==", flush=True)
        grid, mean, std, solve = run_algo(label, fn, spe, seeds, args.smooth)
        c = colors[i % len(colors)]
        ax.plot(grid, mean, color=c, label=label, linewidth=2)
        ax.fill_between(grid, mean - std, mean + std, color=c, alpha=0.18)
        save[f"{label}_grid"] = grid
        save[f"{label}_mean"] = mean
        save[f"{label}_std"] = std
        summary.append((label, np.mean(solve), np.std(solve), mean[-1]))

    ax.axhline(threshold, color="gray", ls="--", lw=1,
               label=f"solved threshold ({threshold})")
    ax.set_xlabel("Environment steps", fontsize=12)
    ax.set_ylabel("Episode return (moving avg)", fontsize=12)
    ax.set_title(f"Learning curves on {env_name}  "
                 f"(mean ± std over {args.seeds} seeds)", fontsize=13)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = os.path.join(RESULTS, f"{args.suite}_curves.png")
    fig.savefig(png, bbox_inches="tight")
    np.savez(os.path.join(RESULTS, f"{args.suite}_curves.npz"), **save)

    print("\n=== SUMMARY (real runs) ===")
    print(f"{'algorithm':24s} {'final_steps(mean)':>18s} {'final_return':>14s}")
    for label, sm, ss, fret in summary:
        print(f"{label:24s} {sm:>15.0f}±{ss:<4.0f} {fret:>14.1f}")
    print(f"\nsaved: {png}")


if __name__ == "__main__":
    main()
