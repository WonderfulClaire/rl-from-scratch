# 03 · 蒙特卡洛与时序差分：无模型的策略评估

> 从本章起，$(P, R)$ 未知，只能通过与环境交互采样。核心问题：如何从样本估计 $V^\pi$？两条路线——蒙特卡洛（MC）用完整回报，时序差分（TD）用自举（bootstrapping）——它们的对比是理解一切现代 RL 的钥匙。
>
> 代码：[`mc_td.py`](mc_td.py) —— 在 5 状态 Random Walk 上对比 MC 与 TD(0) 的收敛行为（复现 Sutton & Barto 图 6.2 的结论）。

---

## 1. 蒙特卡洛估计

按 $\pi$ 采样完整回合，对每个状态取实际回报的平均：

$$
V(s_t) \leftarrow V(s_t) + \alpha\big[\underbrace{G_t}_{\text{目标}} - V(s_t)\big], \qquad G_t = \sum_{k=0}^{T-t-1}\gamma^k r_{t+k+1}. \tag{1}
$$

- **无偏**：$\mathbb{E}_\pi[G_t \mid s_t = s] = V^\pi(s)$，这是定义本身。
- **高方差**：$G_t$ 累积了整条轨迹上所有动作选择、状态转移的随机性。轨迹越长、$\gamma$ 越大，方差越大。
- 必须等回合结束才能更新，不适用于持续型任务。

首次访问（first-visit）MC 每回合只在状态第一次出现处更新，是对 i.i.d. 样本求平均，大数定律直接给出收敛性。

## 2. TD(0)

把 (1) 中的目标 $G_t$ 换成**自举目标** $r_{t+1} + \gamma V(s_{t+1})$：

$$
V(s_t) \leftarrow V(s_t) + \alpha\big[\underbrace{r_{t+1} + \gamma V(s_{t+1})}_{\text{TD 目标}} - V(s_t)\big]. \tag{2}
$$

方括号里的量就是 **TD 误差** $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$。

- **有偏**：目标里含当前估计 $V(s_{t+1})$，估计不准时目标就不准（偏差随训练减小）。
- **低方差**：随机性只来自一步的 $(r_{t+1}, s_{t+1})$。
- 每步都能更新，天然在线。

**TD(0) 在做什么**：它不是在最小化对 $V^\pi$ 的均方误差，而是在解**经验版本的贝尔曼方程**——收敛点是"确定性等价"（certainty-equivalence）估计：先用样本隐式估计出 $\hat P, \hat R$，再精确解 $\hat V$。这就是为什么小数据下 TD 通常比 MC 更准：它利用了马尔可夫结构，MC 没有。

## 3. 偏差–方差权衡：一张表

| | MC | TD(0) |
|---|---|---|
| 目标 | $G_t$ | $r_{t+1}+\gamma V(s_{t+1})$ |
| 偏差 | 无 | 有（自举） |
| 方差 | 高 | 低 |
| 利用马尔可夫性 | 否 | 是 |
| 非马尔可夫环境 | 更稳 | 可能更差 |
| 函数逼近 + off-policy | 安全 | 可能发散（致命三要素，见第 05 章） |

## 4. n-step TD 与 TD(λ)：两个极端之间的连续谱

**n-step 回报**：

$$
G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n}).
$$

$n=1$ 是 TD(0)，$n=\infty$（到回合结束）是 MC。$n$ 越大偏差越小、方差越大。

**λ-回报**把所有 n-step 回报按几何权重混合：

$$
G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}, \qquad \lambda \in [0,1].
$$

权重 $(1-\lambda)\lambda^{n-1}$ 归一化（几何级数和为 1）。$\lambda=0$ 退化为 TD(0)，$\lambda=1$ 退化为 MC。

> 第 06 章的 **GAE（广义优势估计）** 就是把同样的 λ-混合思想用在优势函数上：$\hat A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l\ge 0} (\gamma\lambda)^l \delta_{t+l}$。在这里先把 TD(λ) 吃透，GAE 就是免费的。

**资格迹（eligibility traces）**是 λ-回报的高效在线实现：维护向量 $e_t = \gamma\lambda e_{t-1} + \nabla V(s_t)$（表格情形即对访问过的状态记一笔按 $\gamma\lambda$ 衰减的"痕迹"），每步用 $\delta_t$ 按痕迹强度更新所有状态。前向视角（λ-回报）与后向视角（资格迹）在离线情形严格等价。

## 5. 收敛性说明

表格型 TD(0) 在 Robbins-Monro 步长条件

$$
\sum_t \alpha_t = \infty, \qquad \sum_t \alpha_t^2 < \infty
$$

且所有状态被无限次访问时，以概率 1 收敛到 $V^\pi$（随机逼近理论；本质是因为期望更新算子 $\mathcal{T}^\pi$ 是压缩的）。固定步长 $\alpha$ 则收敛到 $V^\pi$ 邻域内的振荡。

## 6. 运行 demo

```bash
python 03_monte_carlo_td/mc_td.py
```

demo 在 5 状态 Random Walk（A-B-C-D-E，两端终止，右端奖励 +1）上：
1. 解析算出真实 $V^\pi$（线性方程组）；
2. 分别用 first-visit MC 和 TD(0) 跑多个 episode，多种子平均，打印均方根误差随回合数的下降曲线；
3. 复现经典结论：**同等回合数下 TD(0) 的 RMSE 低于 MC**（低方差 + 利用马尔可夫结构）。

## 参考

- Sutton & Barto (2nd ed.), Ch. 5-7, 12.
- Sutton (1988), *Learning to Predict by the Methods of Temporal Differences*.
