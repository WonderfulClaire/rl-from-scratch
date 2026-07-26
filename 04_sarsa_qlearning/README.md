# 04 · SARSA 与 Q-learning：无模型控制

> 从"评估"走向"控制"：不仅估计价值，还要找最优策略。本章是表格型 RL 的顶点，也是 DQN 的直接前身。核心概念：on-policy vs off-policy、探索-利用、最大化偏差。
>
> 代码：[`sarsa_qlearning.py`](sarsa_qlearning.py) —— CliffWalking 上对比 SARSA / Q-learning / Double Q-learning，复现"Q-learning 走悬崖边、SARSA 绕远路"的经典现象。

---

## 1. 从 V 到 Q：为什么控制要学 Q

无模型时，即使知道 $V^*$ 也做不了贪心——$\arg\max_a\big[R(s,a) + \gamma\sum_{s'} P(s'\mid s,a)V^*(s')\big]$ 需要模型 $(P,R)$。而 $Q^*$ 直接给出 $\pi^*(s) = \arg\max_a Q^*(s,a)$，不需要模型。**所以无模型控制学 Q 不学 V。**

## 2. 探索：ε-greedy 与 GLIE

$$
\pi_\varepsilon(a\mid s) = \begin{cases} 1-\varepsilon + \varepsilon/|\mathcal{A}| & a = \arg\max_{a'} Q(s,a') \\ \varepsilon/|\mathcal{A}| & \text{其他} \end{cases}
$$

**GLIE**（Greedy in the Limit with Infinite Exploration）条件：每个 $(s,a)$ 被访问无限次，且 $\varepsilon_t \to 0$（如 $\varepsilon_t = 1/t$）。GLIE + Robbins-Monro 步长 ⟹ on-policy 控制（SARSA）收敛到 $Q^*$。

## 3. SARSA（on-policy TD 控制）

用五元组 $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$ 更新——名字由此而来：

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\big[r_{t+1} + \gamma\, Q(s_{t+1}, a_{t+1}) - Q(s_t,a_t)\big]. 
$$

关键：$a_{t+1}$ 是**行为策略实际采出的动作**（含探索）。SARSA 学的是 $Q^{\pi_\varepsilon}$——"带着探索噪声行动的自己"的价值。它是贝尔曼**期望**方程（对 $\pi_\varepsilon$）的随机逼近。

## 4. Q-learning（off-policy TD 控制）

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\big[r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t,a_t)\big]. 
$$

目标里的 $\max_{a'}$ 与实际执行的下一动作无关——Q-learning 直接逼近 $Q^*$（贝尔曼**最优**方程的随机逼近），**无论行为策略是什么**（只要保持充分探索）。这就是 off-policy：学习的目标策略（贪心）≠ 采数据的行为策略（ε-greedy）。off-policy 能力是经验回放（第 05 章 DQN）的前提：回放池里的旧数据来自旧策略，on-policy 方法不能直接用。

**收敛定理**（Watkins & Dayan, 1992）：所有 $(s,a)$ 被无限次更新 + Robbins-Monro 步长 ⟹ $Q \to Q^*$ 以概率 1。证明思路：更新的期望方向由压缩算子 $\mathcal{T}^*$ 给出，随机逼近理论保证带零均值噪声的小步长迭代跟随期望方向，最终落入 $\mathcal{T}^*$ 的唯一不动点。

### SARSA vs Q-learning：CliffWalking 现象

悬崖行走（4×12 网格，下边缘是悬崖，掉下去 -100 并回到起点，每步 -1）中：

- **Q-learning** 学到贴着悬崖的最短路径——它学的是最优贪心策略的 Q，不在乎探索时偶尔掉崖；代价是**训练期每回合平均回报更低**（真的会掉）。
- **SARSA** 学到离悬崖远一格的安全路径——它诚实地把"我有 ε 概率乱走"计入价值，悬崖边格子的 Q 被压低，于是绕开。
- 教训：**on-policy 优化"你实际的行为"，off-policy 优化"理想化的目标行为"**。部署时若仍带探索或环境有执行噪声，SARSA 式保守可能正是想要的。对交易场景（第 11 章）：执行有滑点时，"理论最优但贴着风险边界"的策略未必是好策略。

## 5. 最大化偏差与 Double Q-learning

**问题**：(2) 中 $\max_{a'} Q(s',a')$ 用同一套带噪声的估计既**选**动作又**评**动作。对任意随机变量族 $\{X_i\}$，由 max 的凸性（Jensen 不等式）：

$$
\mathbb{E}\big[\max_i X_i\big] \;\ge\; \max_i \mathbb{E}[X_i],
$$

所以即使真实 Q 全相等、估计噪声零均值，$\max$ 的期望也系统性偏高——**过估计（maximization bias）**。噪声越大、动作数越多，偏得越狠。

**Double Q-learning**（van Hasselt, 2010）：维护两套独立估计 $Q_1, Q_2$，每次等概率更新其一，用一套**选**动作、另一套**评**动作：

$$
Q_1(s,a) \leftarrow Q_1(s,a) + \alpha\Big[r + \gamma\, Q_2\big(s',\, \arg\max_{a'} Q_1(s',a')\big) - Q_1(s,a)\Big], 
$$

（更新 $Q_2$ 时对称交换）。因为 $Q_2$ 的估计噪声与 $Q_1$ 的 argmax 选择近似独立，条件期望 $\mathbb{E}\big[Q_2(s',a_1^*)\big] \approx Q^{\text{真}}(s',a_1^*)$，不再有系统性向上的偏。代价：$Q_2$ 评估的是"$Q_1$ 认为最好的动作"，可能引入轻微**低估**——但低估通常无害（不会像高估那样自我放大，见第 05 章 3.2 节）。

行动时用 $Q_1 + Q_2$ 贪心即可。

## 6. 三种算法一览

| | 目标 | on/off-policy | 学的是谁的价值 | 过估计 |
|---|---|---|---|---|
| SARSA | $r + \gamma Q(s',a')$ | on | 当前 ε-greedy 策略 | 无 |
| Q-learning | $r + \gamma \max_{a'}Q(s',a')$ | off | 最优贪心策略 | 有 |
| Double Q | $r + \gamma Q_2(s', \arg\max Q_1)$ | off | 最优贪心策略 | 基本消除 |

## 7. 运行 demo

```bash
python 04_sarsa_qlearning/sarsa_qlearning.py
```

输出：
1. 三种算法在 CliffWalking 上的训练回报曲线（滑动平均）；
2. 各自学到的贪心路径可视化——SARSA 走安全路线、Q-learning 走悬崖边；
3. 训练期平均回报：SARSA > Q-learning（复现 Sutton & Barto 图 6.4 结论）。

## 参考

- Sutton & Barto (2nd ed.), Ch. 6.
- Watkins & Dayan (1992), *Q-learning*.
- van Hasselt (2010), *Double Q-learning*.
