# 02 · 动态规划：策略迭代与值迭代

> 前提：已知完整的环境模型 $(P, R)$。动态规划（DP）是所有 RL 算法的"理想型"——后面每个无模型算法都可以看成 DP 在"模型未知、只能采样"约束下的某种近似。
>
> 代码：[`dp_solvers.py`](dp_solvers.py) —— 策略迭代与值迭代求解 GridWorld，验证两者收敛到同一个 $V^*$。

---

## 1. 策略评估（Policy Evaluation）

目标：给定 $\pi$，求 $V^\pi$。第 01 章给了两种方法：

- 解析：$V^\pi = (I-\gamma P^\pi)^{-1}R^\pi$，复杂度 $O(|\mathcal{S}|^3)$；
- 迭代：反复应用 $\mathcal{T}^\pi$，每轮 $O(|\mathcal{S}|^2|\mathcal{A}|)$，误差按 $\gamma^k$ 收缩。

实践中用迭代法，且**不需要算到收敛**——这是策略迭代能加速的关键观察。

## 2. 策略改进（Policy Improvement）

**策略改进定理**：给定 $V^\pi$，定义贪心策略

$$
\pi'(s) = \arg\max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'\mid s,a) V^\pi(s') \right] = \arg\max_a Q^\pi(s,a),
$$

则 $V^{\pi'}(s) \ge V^\pi(s)$ 对所有 $s$ 成立；且若在某个状态严格大于，则 $\pi'$ 严格更好。

**证明**：由贪心定义，$Q^\pi(s, \pi'(s)) = \max_a Q^\pi(s,a) \ge \sum_a \pi(a\mid s) Q^\pi(s,a) = V^\pi(s)$。然后反复展开：

$$
\begin{aligned}
V^\pi(s) &\le Q^\pi(s, \pi'(s)) = \mathbb{E}\big[r_{t+1} + \gamma V^\pi(s_{t+1}) \,\big|\, s_t = s, a_t = \pi'(s)\big] \\
&\le \mathbb{E}_{\pi'}\big[r_{t+1} + \gamma Q^\pi(s_{t+1}, \pi'(s_{t+1})) \mid s_t = s\big] \\
&\le \mathbb{E}_{\pi'}\big[r_{t+1} + \gamma r_{t+2} + \gamma^2 Q^\pi(s_{t+2}, \pi'(s_{t+2})) \mid s_t = s\big] \\
&\le \cdots \le \mathbb{E}_{\pi'}\big[r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots \mid s_t = s\big] = V^{\pi'}(s). \qquad\blacksquare
\end{aligned}
$$

每一步都是把"跟一步 $\pi'$ 再回到 $\pi$"换成"多跟一步 $\pi'$"，由第一行的不等式，每换一次都不会变差。

**这个定理是半数 RL 算法的正当性来源**：SARSA、Q-learning、DQN、甚至 AlphaZero 的本质都是"评估当前策略 → 对评估结果贪心 → 得到更好的策略"这个循环。

## 3. 策略迭代（Policy Iteration）

$$
\pi_0 \xrightarrow{\text{评估}} V^{\pi_0} \xrightarrow{\text{贪心}} \pi_1 \xrightarrow{\text{评估}} V^{\pi_1} \xrightarrow{\text{贪心}} \pi_2 \to \cdots
$$

- 有限 MDP 中确定性策略只有 $|\mathcal{A}|^{|\mathcal{S}|}$ 个，每轮严格改进（或已最优），所以**有限步内收敛到 $\pi^*$**。
- 停机条件：贪心策略不再变化 $\iff$ $V^\pi$ 满足贝尔曼最优方程 $\iff$ $\pi$ 已最优。
- 实践中策略迭代往往只需**个位数轮**就收敛——策略空间的收敛比价值空间快得多。

## 4. 值迭代（Value Iteration）

直接迭代贝尔曼最优算子：

$$
V_{k+1}(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'\mid s,a) V_k(s') \right].
$$

由第 01 章的压缩性，$\|V_k - V^*\|_\infty \le \gamma^k \|V_0 - V^*\|_\infty$。

**两个实用的界**（demo 中会验证第一个）：

1. **停机准则**：若 $\|V_{k+1} - V_k\|_\infty < \epsilon$，则 $\|V_{k+1} - V^*\|_\infty < \dfrac{\gamma\epsilon}{1-\gamma}$。
   （由三角不等式 + 压缩性：$\|V_{k+1}-V^*\| \le \|V_{k+1}-\mathcal T^*V_{k+1}\| + \|\mathcal T^*V_{k+1} - V^*\| \le \gamma\|V_k - V_{k+1}\|\cdot\frac{1}{1-\gamma}$ 的标准推导。）
2. **贪心损失界**：对 $V$ 贪心得到的策略 $\pi_V$ 满足 $\|V^{\pi_V} - V^*\|_\infty \le \dfrac{2\gamma}{1-\gamma}\|V - V^*\|_\infty$。
   即：价值估计差一点，贪心策略的性能差距会被 $\frac{2\gamma}{1-\gamma}$ 放大——$\gamma \to 1$ 时这个放大系数爆炸，这是深度 RL 不稳定的理论根源之一。

## 5. 策略迭代 vs 值迭代

| | 策略迭代 | 值迭代 |
|---|---|---|
| 每轮成本 | 高（完整策略评估） | 低（一次备份） |
| 轮数 | 少（有限步收敛） | 多（几何收敛） |
| 统一视角 | 广义策略迭代（GPI）的两个极端：评估做到底 vs 只做一步 | |

**广义策略迭代（GPI）**：评估和改进交替进行、各自做"不完全"的一步，只要两者都持续推进就收敛。几乎所有 RL 算法（含 Actor-Critic）都是 GPI 的实例：Critic 做不完全评估，Actor 做不完全改进。

## 6. 运行 demo

```bash
python 02_dynamic_programming/dp_solvers.py
```

输出：
1. 策略迭代：打印每轮的策略变化数，观察个位数轮收敛；
2. 值迭代：打印 $\|V_{k+1}-V_k\|_\infty$，观察几何收敛，并验证停机准则给出的误差界确实成立；
3. 验证两种算法得到的 $V^*$ 与贪心策略一致，并可视化最优策略箭头图。

## 参考

- Sutton & Barto (2nd ed.), Ch. 4.
- Bertsekas, *Dynamic Programming and Optimal Control*, Vol. II（误差界的严格推导）.
