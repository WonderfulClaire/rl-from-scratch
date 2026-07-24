"""01 · GridWorld 环境 + 贝尔曼期望方程的解析解与迭代解.

对应 README 中的公式:
  (4) 贝尔曼期望方程   V^pi(s) = sum_a pi(a|s) [R(s,a) + gamma * sum_s' P(s'|s,a) V^pi(s')]
  (5) 解析解           V^pi = (I - gamma * P^pi)^{-1} R^pi
  (8) 压缩性           迭代误差以 gamma 的几何速率收敛

运行: python 01_mdp_bellman/gridworld.py
"""

import numpy as np


class GridWorld:
    """4x4 GridWorld (Sutton & Barto Example 4.1 风格).

    - 状态: 0..15, 其中 0 和 15 是终止状态(吸收态).
    - 动作: 0=上, 1=下, 2=左, 3=右. 撞墙则停在原地.
    - 奖励: 每步 -1, 进入终止状态后永远为 0.
    """

    N = 4
    N_STATES = 16
    N_ACTIONS = 4
    TERMINALS = (0, 15)

    def __init__(self):
        # P[s, a, s'] 转移概率张量, R[s, a] 期望即时奖励
        self.P = np.zeros((self.N_STATES, self.N_ACTIONS, self.N_STATES))
        self.R = np.zeros((self.N_STATES, self.N_ACTIONS))
        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        for s in range(self.N_STATES):
            if s in self.TERMINALS:
                # 吸收态: 停留原地, 奖励 0
                self.P[s, :, s] = 1.0
                continue
            r, c = divmod(s, self.N)
            for a, (dr, dc) in deltas.items():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.N and 0 <= nc < self.N):
                    nr, nc = r, c  # 撞墙
                self.P[s, a, nr * self.N + nc] = 1.0
                self.R[s, a] = -1.0


def analytic_v(env: GridWorld, pi: np.ndarray, gamma: float) -> np.ndarray:
    """公式 (5): V^pi = (I - gamma P^pi)^{-1} R^pi, 直接线性代数求解."""
    # P^pi[s, s'] = sum_a pi(a|s) P(s'|s,a);  R^pi[s] = sum_a pi(a|s) R(s,a)
    P_pi = np.einsum("sa,sap->sp", pi, env.P)
    R_pi = np.einsum("sa,sa->s", pi, env.R)
    return np.linalg.solve(np.eye(env.N_STATES) - gamma * P_pi, R_pi)


def iterative_v(env: GridWorld, pi: np.ndarray, gamma: float,
                tol: float = 1e-10, v_star: np.ndarray | None = None) -> np.ndarray:
    """迭代策略评估: V <- T^pi V, 由压缩映射定理保证几何收敛."""
    V = np.zeros(env.N_STATES)
    for k in range(10_000):
        # 公式 (4) 的向量化实现
        Q = env.R + gamma * np.einsum("sap,p->sa", env.P, V)  # Q^pi 的一步展开
        V_new = np.einsum("sa,sa->s", pi, Q)
        diff = np.max(np.abs(V_new - V))
        if v_star is not None and k % 20 == 0:
            err = np.max(np.abs(V_new - v_star))
            print(f"  iter {k:4d}   ||V_k - V*||_inf = {err:.3e}")
        V = V_new
        if diff < tol:
            print(f"  收敛于第 {k} 轮 (||V_k+1 - V_k||_inf < {tol})")
            break
    return V


def main():
    gamma = 0.9
    env = GridWorld()
    # 均匀随机策略 pi(a|s) = 1/4
    pi = np.full((env.N_STATES, env.N_ACTIONS), 1.0 / env.N_ACTIONS)

    print("=== 1) 解析解: V^pi = (I - gamma P^pi)^{-1} R^pi ===")
    v_exact = analytic_v(env, pi, gamma)
    print(np.round(v_exact.reshape(4, 4), 3), "\n")

    print("=== 2) 迭代解: V <- T^pi V (观察几何收敛) ===")
    v_iter = iterative_v(env, pi, gamma, v_star=v_exact)
    print(np.round(v_iter.reshape(4, 4), 3), "\n")

    err = np.max(np.abs(v_exact - v_iter))
    print(f"两种解法最大偏差: {err:.2e}  ->  {'一致 ✓' if err < 1e-8 else '不一致 ✗'}")


if __name__ == "__main__":
    main()
