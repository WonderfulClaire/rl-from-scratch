"""11 · 交易环境: 带交易成本与风险惩罚的仓位管理 MDP.

对应 README 第 2 节:
  状态  = [过去 lookback 期收益, 已实现波动, 当前仓位]
  动作  = 目标仓位 {-1(空), 0(平), +1(多)}
  奖励  = 公式 (2): pnl - cost*|仓位变化| - (lambda/2)*pnl^2

数据: 合成价格 = AR(1) 动量趋势 + 高斯噪声 (低信噪比但存在可学结构).
反前视纪律: 状态只含 t 及以前的信息; 标准化统计量只来自训练段.
"""

import numpy as np


def make_price_series(T=6000, seed=0, trend_phi=0.97, trend_sigma=0.0004,
                      noise_sigma=0.008):
    """合成对数收益: r_t = mu_t + eps_t, mu_t 是缓慢均值回复的动量项."""
    rng = np.random.default_rng(seed)
    mu = np.zeros(T)
    for t in range(1, T):
        mu[t] = trend_phi * mu[t - 1] + trend_sigma * rng.standard_normal()
    ret = mu + noise_sigma * rng.standard_normal(T)
    price = 100.0 * np.exp(np.cumsum(ret))
    return price, ret


class TradingEnv:
    """离散仓位交易环境 (gym 风格接口, 无 gym 依赖)."""

    ACTIONS = (-1, 0, 1)  # 目标仓位

    def __init__(self, returns, lookback=8, cost=0.0005, risk_lambda=1.0,
                 norm_stats=None):
        self.ret = returns.astype(np.float64)
        self.lookback = lookback
        self.cost = cost
        self.risk_lambda = risk_lambda
        # 反前视: 标准化统计量必须由训练段传入 (测试环境复用训练段的 stats)
        if norm_stats is None:
            norm_stats = (self.ret.mean(), self.ret.std() + 1e-12)
        self.mu_n, self.sd_n = norm_stats

    @property
    def obs_dim(self):
        return self.lookback + 2  # 收益窗口 + 波动 + 当前仓位

    @property
    def n_actions(self):
        return len(self.ACTIONS)

    def _obs(self):
        w = self.ret[self.t - self.lookback: self.t]      # 只用 t 以前
        w_norm = (w - self.mu_n) / self.sd_n
        vol = w.std() / self.sd_n
        return np.concatenate([w_norm, [vol, float(self.pos)]]).astype(np.float32)

    def reset(self):
        self.t = self.lookback
        self.pos = 0
        self.equity = [1.0]
        return self._obs()

    def step(self, action_idx):
        new_pos = self.ACTIONS[action_idx]
        y_next = self.ret[self.t]                          # t 时刻决策 -> 吃 t 的收益
        pnl = new_pos * y_next
        trade_cost = self.cost * abs(new_pos - self.pos)
        # 公式 (2): 风险调整 + 成本内生
        reward = pnl - trade_cost - 0.5 * self.risk_lambda * pnl ** 2
        self.equity.append(self.equity[-1] * (1 + pnl - trade_cost))
        self.pos = new_pos
        self.t += 1
        done = self.t >= len(self.ret)
        return (None if done else self._obs()), float(reward), done

    # ---- 回测指标 ----
    def stats(self, freq=252):
        eq = np.array(self.equity)
        r = np.diff(eq) / eq[:-1]
        ann_ret = (eq[-1] / eq[0]) ** (freq / max(len(r), 1)) - 1
        sharpe = r.mean() / (r.std() + 1e-12) * np.sqrt(freq)
        dd = 1 - eq / np.maximum.accumulate(eq)
        return {"final_equity": eq[-1], "ann_return": ann_ret,
                "sharpe": sharpe, "max_drawdown": dd.max()}
