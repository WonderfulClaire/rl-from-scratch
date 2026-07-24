"""02 · 动态规划: 策略迭代与值迭代.

对应 README:
  第 3 节  策略迭代 (评估到收敛 + 贪心改进, 有限步收敛)
  第 4 节  值迭代   (V <- T* V, 几何收敛 + 停机误差界)

运行: python 02_dynamic_programming/dp_solvers.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from importlib import import_module

GridWorld = import_module("01_mdp_bellman.gridworld").GridWorld if False else None

# 为避免包名以数字开头的 import 问题, 直接复用一个轻量 GridWorld 构造函数


def build_gridworld():
    """与 01 章相同的 4x4 GridWorld: 返回 (P, R, n_states, n_actions)."""
    N, nS, nA = 4, 16, 4
    terminals = (0, 15)
    P = np.zeros((nS, nA, nS))
    R = np.zeros((nS, nA))
    deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    for s in range(nS):
        if s in terminals:
            P[s, :, s] = 1.0
            continue
        r, c = divmod(s, N)
        for a, (dr, dc) in deltas.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < N and 0 <= nc < N):
                nr, nc = r, c
            P[s, a, nr * N + nc] = 1.0
            R[s, a] = -1.0
    return P, R, nS, nA


def policy_evaluation(P, R, pi, gamma, tol=1e-10):
    """迭代策略评估 (第 1 节)."""
    V = np.zeros(P.shape[0])
    while True:
        Q = R + gamma * np.einsum("sap,p->sa", P, V)
        V_new = np.einsum("sa,sa->s", pi, Q)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new


def greedy_policy(P, R, V, gamma):
    """对 V 贪心 (策略改进定理, 第 2 节)."""
    Q = R + gamma * np.einsum("sap,p->sa", P, V)
    pi = np.zeros_like(Q)
    pi[np.arange(len(Q)), np.argmax(Q, axis=1)] = 1.0
    return pi


def policy_iteration(P, R, gamma):
    """策略迭代 (第 3 节): 有限步收敛."""
    nS, nA = R.shape
    pi = np.full((nS, nA), 1.0 / nA)  # 从均匀随机策略出发
    for it in range(1, 100):
        V = policy_evaluation(P, R, pi, gamma)
        pi_new = greedy_policy(P, R, V, gamma)
        n_changed = int(np.sum(np.argmax(pi_new, 1) != np.argmax(pi, 1)))
        print(f"  第 {it} 轮: 贪心动作变化的状态数 = {n_changed}")
        if n_changed == 0:
            return V, pi_new, it
        pi = pi_new
    raise RuntimeError("策略迭代未收敛")


def value_iteration(P, R, gamma, eps=1e-8):
    """值迭代 (第 4 节): V <- T* V, 停机时 ||V - V*|| < gamma*eps/(1-gamma)."""
    V = np.zeros(P.shape[0])
    for it in range(1, 100_000):
        Q = R + gamma * np.einsum("sap,p->sa", P, V)
        V_new = np.max(Q, axis=1)
        diff = np.max(np.abs(V_new - V))
        V = V_new
        if it <= 5 or it % 20 == 0:
            print(f"  iter {it:4d}   ||V_k+1 - V_k||_inf = {diff:.3e}")
        if diff < eps:
            bound = gamma * eps / (1 - gamma)
            print(f"  停机于第 {it} 轮, 理论误差界 ||V - V*|| < {bound:.2e}")
            return V, greedy_policy(P, R, V, gamma), it
    raise RuntimeError("值迭代未收敛")


ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}


def render_policy(pi, terminals=(0, 15)):
    acts = np.argmax(pi, axis=1)
    lines = []
    for r in range(4):
        row = []
        for c in range(4):
            s = r * 4 + c
            row.append(" T " if s in terminals else f" {ARROWS[acts[s]]} ")
        lines.append("".join(row))
    return "\n".join(lines)


def main():
    gamma = 0.9
    P, R, nS, nA = build_gridworld()

    print("=== 策略迭代 ===")
    V_pi, pi_pi, iters_pi = policy_iteration(P, R, gamma)
    print(f"收敛轮数: {iters_pi}\n")

    print("=== 值迭代 ===")
    V_vi, pi_vi, iters_vi = value_iteration(P, R, gamma)
    print()

    err = np.max(np.abs(V_pi - V_vi))
    same_pi = np.array_equal(np.argmax(pi_pi, 1), np.argmax(pi_vi, 1))
    print(f"两算法 V* 最大偏差: {err:.2e}   贪心策略一致: {same_pi}")
    print("\nV* (4x4):")
    print(np.round(V_vi.reshape(4, 4), 3))
    print("\n最优策略:")
    print(render_policy(pi_vi))


if __name__ == "__main__":
    main()
