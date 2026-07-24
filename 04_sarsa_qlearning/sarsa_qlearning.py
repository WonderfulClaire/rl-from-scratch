"""04 · CliffWalking 上的 SARSA / Q-learning / Double Q-learning 对比.

环境: 4x12 网格, 左下角起点 S, 右下角终点 G, 下边缘中间是悬崖.
      每步 -1; 掉崖 -100 并传送回起点; 到 G 回合结束.

对应 README:
  公式 (1) SARSA        Q <- Q + a [r + g Q(s',a') - Q]
  公式 (2) Q-learning   Q <- Q + a [r + g max_a' Q(s',a') - Q]
  公式 (3) Double Q     用 Q1 选动作、Q2 评动作

运行: python 04_sarsa_qlearning/sarsa_qlearning.py
"""

import numpy as np

ROWS, COLS = 4, 12
N_S, N_A = ROWS * COLS, 4  # 0=上 1=下 2=左 3=右
START, GOAL = 3 * COLS + 0, 3 * COLS + 11
CLIFF = {3 * COLS + c for c in range(1, 11)}
DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def step(s, a):
    """返回 (s_next, reward, done)."""
    r, c = divmod(s, COLS)
    dr, dc = DELTAS[a]
    nr, nc = min(max(r + dr, 0), ROWS - 1), min(max(c + dc, 0), COLS - 1)
    s_next = nr * COLS + nc
    if s_next in CLIFF:
        return START, -100.0, False
    if s_next == GOAL:
        return s_next, -1.0, True
    return s_next, -1.0, False


def eps_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return rng.integers(N_A)
    q = Q[s]
    return rng.choice(np.flatnonzero(q == q.max()))  # 随机破平


def train(algo, episodes=500, alpha=0.5, gamma=1.0, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_S, N_A))
    Q2 = np.zeros((N_S, N_A))  # 仅 double 使用
    returns = []
    for _ in range(episodes):
        s, ep_ret, done = START, 0.0, False
        Qsum = Q + Q2 if algo == "double" else Q
        a = eps_greedy(Qsum, s, eps, rng)
        while not done:
            s_next, r, done = step(s, a)
            ep_ret += r
            if algo == "sarsa":
                a_next = eps_greedy(Q, s_next, eps, rng)
                target = r + (0.0 if done else gamma * Q[s_next, a_next])
                Q[s, a] += alpha * (target - Q[s, a])
                s, a = s_next, a_next
            elif algo == "qlearning":
                target = r + (0.0 if done else gamma * Q[s_next].max())
                Q[s, a] += alpha * (target - Q[s, a])
                s = s_next
                a = eps_greedy(Q, s, eps, rng)
            else:  # double Q-learning, 公式 (3)
                if rng.random() < 0.5:
                    a_star = int(np.argmax(Q[s_next]))
                    target = r + (0.0 if done else gamma * Q2[s_next, a_star])
                    Q[s, a] += alpha * (target - Q[s, a])
                else:
                    a_star = int(np.argmax(Q2[s_next]))
                    target = r + (0.0 if done else gamma * Q[s_next, a_star])
                    Q2[s, a] += alpha * (target - Q2[s, a])
                s = s_next
                a = eps_greedy(Q + Q2, s, eps, rng)
        returns.append(ep_ret)
    return (Q + Q2 if algo == "double" else Q), np.array(returns)


ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}


def render_greedy_path(Q):
    """从 S 出发按贪心策略走, 标出路径."""
    grid = [["." for _ in range(COLS)] for _ in range(ROWS)]
    for c in range(1, 11):
        grid[3][c] = "C"
    grid[3][11] = "G"
    s, steps = START, 0
    while s != GOAL and steps < 100:
        a = int(np.argmax(Q[s]))
        r, c = divmod(s, COLS)
        grid[r][c] = ARROWS[a]
        s, _, done = step(s, a)
        steps += 1
        if done:
            break
    grid[3][0] = "S" if grid[3][0] == "." else grid[3][0]
    return "\n".join(" ".join(row) for row in grid)


def main():
    n_seeds, episodes = 20, 500
    print(f"CliffWalking: {n_seeds} 个种子平均, 每个 {episodes} 回合\n")
    for algo, label in [("sarsa", "SARSA"), ("qlearning", "Q-learning"),
                        ("double", "Double Q-learning")]:
        all_ret = np.stack([train(algo, episodes=episodes, seed=sd)[1]
                            for sd in range(n_seeds)])
        mean_last100 = all_ret[:, -100:].mean()
        Q, _ = train(algo, episodes=episodes, seed=0)
        print(f"=== {label} ===")
        print(f"最后 100 回合平均回报: {mean_last100:.1f}")
        print("贪心路径 (C=悬崖):")
        print(render_greedy_path(Q))
        print()

    print("预期结论: SARSA 训练期回报更高(安全路线), "
          "Q-learning 路径更短(贴悬崖), Double Q 介于两者之间且更稳.")


if __name__ == "__main__":
    main()
