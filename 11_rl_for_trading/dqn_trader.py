"""11 · DQN 交易智能体: 训练段学习 -> 测试段回测 -> 基准对比 + 成本敏感性.

对应 README 第 4 节:
  - Double DQN (第 05 章的算法, 换个环境直接用)
  - 时间切分: 前 80% 训练, 后 20% 测试 (测试段只碰一次)
  - 基准: 买入持有 / 完美后见之明(上界) / 随机(下界)
  - 成本敏感性: c 增大 10 倍 -> 换手率应显著下降

运行: python 11_rl_for_trading/dqn_trader.py
"""

import collections
import random

import numpy as np
import torch
import torch.nn as nn

from trading_env import TradingEnv, make_price_series


class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


def train_dqn(env, epochs=30, gamma=0.95, lr=1e-3, batch=64, seed=0):
    """在训练环境上反复过 epoch (每个 epoch 从头到尾走一遍序列)."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    q = QNet(env.obs_dim, env.n_actions)
    q_t = QNet(env.obs_dim, env.n_actions)
    q_t.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = collections.deque(maxlen=100_000)
    step_count = 0
    for ep in range(epochs):
        obs = env.reset()
        eps = max(0.05, 1.0 - ep / (epochs * 0.6))
        done = False
        while not done:
            if random.random() < eps:
                a = random.randrange(env.n_actions)
            else:
                with torch.no_grad():
                    a = int(q(torch.as_tensor(obs)).argmax())
            obs2, r, done = env.step(a)
            buf.append((obs, a, r, obs2 if obs2 is not None else obs * 0, float(done)))
            obs = obs2
            step_count += 1
            if len(buf) >= 1000 and step_count % 4 == 0:
                s, ab, rb, s2, d = map(np.array, zip(*random.sample(buf, batch)))
                s = torch.as_tensor(s, dtype=torch.float32)
                ab = torch.as_tensor(ab, dtype=torch.int64)
                rb = torch.as_tensor(rb, dtype=torch.float32)
                s2 = torch.as_tensor(s2, dtype=torch.float32)
                d = torch.as_tensor(d, dtype=torch.float32)
                q_sa = q(s).gather(1, ab.unsqueeze(1)).squeeze(1)
                with torch.no_grad():  # Double DQN
                    a_star = q(s2).argmax(1, keepdim=True)
                    y = rb + gamma * (1 - d) * q_t(s2).gather(1, a_star).squeeze(1)
                loss = nn.functional.smooth_l1_loss(q_sa, y)
                opt.zero_grad(); loss.backward(); opt.step()
                if step_count % 1000 == 0:
                    q_t.load_state_dict(q.state_dict())
    return q


def backtest(q, env):
    """贪心策略回测, 返回 (指标, 换手率)."""
    obs = env.reset()
    done, n_trades, n_steps = False, 0, 0
    prev_pos = 0
    while not done:
        with torch.no_grad():
            a = int(q(torch.as_tensor(obs)).argmax())
        obs, _, done = env.step(a)
        n_trades += int(env.pos != prev_pos)
        prev_pos = env.pos
        n_steps += 1
    return env.stats(), n_trades / n_steps


def benchmark(env, mode, seed=0):
    rng = np.random.default_rng(seed)
    obs = env.reset()
    done = False
    while not done:
        if mode == "buyhold":
            a = 2                                   # 恒定 +1
        elif mode == "random":
            a = int(rng.integers(3))
        elif mode == "oracle":                      # 后见之明: 偷看下一期收益
            a = 2 if env.ret[env.t] > env.cost else (0 if env.ret[env.t] < -env.cost else 1)
        obs, _, done = env.step(a)
    return env.stats()


def fmt(s):
    return (f"终值 {s['final_equity']:.3f}  年化 {s['ann_return']*100:6.1f}%  "
            f"夏普 {s['sharpe']:5.2f}  最大回撤 {s['max_drawdown']*100:5.1f}%")


def main():
    price, ret = make_price_series(T=6000, seed=42)
    split = int(len(ret) * 0.8)
    ret_train, ret_test = ret[:split], ret[split:]
    # 反前视: 标准化统计量只来自训练段
    stats_norm = (ret_train.mean(), ret_train.std() + 1e-12)

    print("=== 训练 (前 80% 序列) ===")
    env_train = TradingEnv(ret_train, cost=0.0005, norm_stats=stats_norm)
    q = train_dqn(env_train, epochs=30)
    tr_stats, tr_turn = backtest(q, env_train)
    print(f"  训练段: {fmt(tr_stats)}  换手率 {tr_turn:.2f}")

    print("\n=== 测试段回测 (后 20%, 未见过的数据) ===")
    env_test = TradingEnv(ret_test, cost=0.0005, norm_stats=stats_norm)
    te_stats, te_turn = backtest(q, env_test)
    print(f"  DQN      : {fmt(te_stats)}  换手率 {te_turn:.2f}")
    for mode, label in [("buyhold", "买入持有"), ("random", "随机策略"),
                        ("oracle", "后见之明(上界)")]:
        env_b = TradingEnv(ret_test, cost=0.0005, norm_stats=stats_norm)
        print(f"  {label:9s}: {fmt(benchmark(env_b, mode))}")

    print("\n=== 成本敏感性: c x10 后重新训练, 换手率应显著下降 ===")
    env_hc = TradingEnv(ret_train, cost=0.005, norm_stats=stats_norm)
    q_hc = train_dqn(env_hc, epochs=30)
    _, turn_hc = backtest(q_hc, env_hc)
    print(f"  低成本(c=0.0005) 换手率 {tr_turn:.3f}   高成本(c=0.005) 换手率 {turn_hc:.3f}")
    print(f"  成本内生 -> 学出惰性: {'✓' if turn_hc < tr_turn else '✗'}")

    print("\n提醒: 合成数据含 AR(1) 动量结构, 真实市场的可学结构远弱于此;"
          "\n本 demo 的意义是验证建模纪律 (反前视/成本内生/时间切分), 不是策略本身.")


if __name__ == "__main__":
    main()
