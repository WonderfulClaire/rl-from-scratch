"""
mappo_curve.py — MAPPO vs IPPO 学习曲线（Rendezvous 网格世界）

把第 09 章的 train() 拉过来，在"多智能体 rendezvous"任务上跑多个
随机种子，记录每轮的成功率，画出 "均值 ± 标准差" 对比曲线。
所有数字来自真实运行。

用法:
    python tools/mappo_curve.py --seeds 3 --iters 200

产物:
    results/mappo_gridworld_curves.png   对比曲线图
    results/mappo_gridworld_curves.npz   原始曲线数据（可复现绘图）
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
    path = os.path.join(REPO, rel_path)
    name = rel_path.replace("/", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, func)


def run(centralized, iters, seeds):
    train = load_train("09_multi_agent/mappo_gridworld.py")
    curves = []
    for sd in seeds:
        t0 = time.time()
        hist = train(centralized=centralized, iters=iters,
                     episodes_per_iter=16, seed=sd, quiet=True)
        hist = np.asarray(hist, dtype=float)
        curves.append(hist)
        print(f"  [{'MAPPO' if centralized else 'IPPO'}] seed={sd} "
              f"final={hist[-1]:.3f} t={time.time() - t0:.0f}s", flush=True)
    grid = np.arange(1, iters + 1)
    stacked = np.vstack([np.interp(grid, np.arange(1, len(c) + 1), c)
                         for c in curves])
    return grid, stacked.mean(0), stacked.std(0)


def solve_iter(mean, grid, thr=0.95):
    """均值曲线首次稳定超过 thr 的迭代（向前看 5 轮取平滑）。"""
    smoothed = np.convolve(mean, np.ones(5) / 5, mode="same")
    idx = np.where(smoothed >= thr)[0]
    return int(grid[idx[0]]) if len(idx) else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS, exist_ok=True)
    seeds = list(range(args.seeds))

    g1, m1, s1 = run(True, args.iters, seeds)    # MAPPO (centralized critic)
    g2, m2, s2 = run(False, args.iters, seeds)   # IPPO (decentralized)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=130)
    ax.plot(g1, m1, color="#1f77b4", label="MAPPO (centralized critic)", lw=2)
    ax.fill_between(g1, m1 - s1, m1 + s1, color="#1f77b4", alpha=0.18)
    ax.plot(g2, m2, color="#d62728", label="IPPO (decentralized)", lw=2)
    ax.fill_between(g2, m2 - s2, m2 + s2, color="#d62728", alpha=0.18)
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="solved (success = 1.0)")
    ax.set_xlabel("Training iteration", fontsize=12)
    ax.set_ylabel("Success rate", fontsize=12)
    ax.set_title(f"MAPPO vs IPPO on Rendezvous gridworld "
                 f"(mean ± std over {args.seeds} seeds)", fontsize=13)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = os.path.join(RESULTS, "mappo_gridworld_curves.png")
    fig.savefig(png, bbox_inches="tight")
    np.savez(os.path.join(RESULTS, "mappo_gridworld_curves.npz"),
             MAPPO_grid=g1, MAPPO_mean=m1, MAPPO_std=s1,
             IPPO_grid=g2, IPPO_mean=m2, IPPO_std=s2)

    sm_mappo = solve_iter(m1, g1)
    sm_ippo = solve_iter(m2, g2)
    print("\n=== SUMMARY (real runs) ===")
    print(f"MAPPO final success = {m1[-1]:.3f},  solve_iter(>=0.95) = {sm_mappo}")
    print(f"IPPO  final success = {m2[-1]:.3f},  solve_iter(>=0.95) = {sm_ippo}")
    print(f"saved: {png}")


if __name__ == "__main__":
    main()
