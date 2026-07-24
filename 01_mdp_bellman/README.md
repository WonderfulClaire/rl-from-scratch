# 01 · MDP 与 Bellman 方程

> 本章建立整个强化学习的数学地基：马尔可夫决策过程（MDP）、价值函数、贝尔曼期望方程与贝尔曼最优方程，并证明它们为什么有唯一解。
>
> 代码：[`gridworld.py`](gridworld.py) —— 一个 4×4 GridWorld 环境 + 用线性代数直接解出 $V^\pi$ 的 demo。

---

## 1. 马尔可夫决策过程

一个（无限时域、折扣）MDP 是五元组

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

- $\mathcal{S}$：状态空间；$\mathcal{A}$：动作空间（本章都取有限集）；
- $P(s'\mid s,a)$：转移核，满足 $\sum_{s'} P(s'\mid s,a) = 1$；
- $R(s,a)$：期望即时奖励，假设有界：$|R(s,a)| \le R_{\max}$；
- $\gamma \in [0,1)$：折扣因子。

**马尔可夫性**是核心假设：

$$
\Pr(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots) = \Pr(s_{t+1} \mid s_t, a_t).
$$

直觉：状态是"对历史的充分统计量"。如果你的观测不满足马尔可夫性（例如只看到股价的当前值而看不到持仓），正确做法是**扩充状态**（把持仓、历史窗口塞进状态里），而不是假装它是马尔可夫的——第 11 章的交易环境会回到这个问题。

## 2. 回报与价值函数

折扣回报：

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}. \tag{1}
$$

由 $|r|\le R_{\max}$ 与 $\gamma<1$，级数绝对收敛且 $|G_t| \le \frac{R_{\max}}{1-\gamma}$，一切期望都良定义。

**状态价值函数**与**动作价值函数**：

$$
V^\pi(s) = \mathbb{E}_\pi\!\left[G_t \mid s_t = s\right], \qquad
Q^\pi(s,a) = \mathbb{E}_\pi\!\left[G_t \mid s_t = s, a_t = a\right]. \tag{2}
$$

两者的互相表示（第一组恒等式，后面反复用）：

$$
V^\pi(s) = \sum_a \pi(a\mid s)\, Q^\pi(s,a), \qquad
Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'\mid s,a)\, V^\pi(s'). \tag{3}
$$

## 3. 贝尔曼期望方程

把 (3) 的两条互相代入，得到 $V^\pi$ 关于自身的方程：

$$
\boxed{\,V^\pi(s) = \sum_a \pi(a\mid s)\left[R(s,a) + \gamma \sum_{s'} P(s'\mid s,a)\, V^\pi(s')\right]\,} \tag{4}
$$

**推导**（只用了回报的递归结构 $G_t = r_{t+1} + \gamma G_{t+1}$ 和全期望公式）：

$$
\begin{aligned}
V^\pi(s) &= \mathbb{E}_\pi[r_{t+1} + \gamma G_{t+1} \mid s_t = s] \\
&= \mathbb{E}_\pi[r_{t+1}\mid s_t=s] + \gamma\, \mathbb{E}_\pi\big[\mathbb{E}_\pi[G_{t+1} \mid s_{t+1}] \,\big|\, s_t = s\big] \\
&= \sum_a \pi(a\mid s) R(s,a) + \gamma \sum_a \pi(a\mid s)\sum_{s'} P(s'\mid s,a)\, V^\pi(s').
\end{aligned}
$$

第二个等号用了迭代期望律（tower property），马尔可夫性保证了内层期望只依赖 $s_{t+1}$。

### 3.1 矩阵形式与解析解

有限状态下，记 $V^\pi \in \mathbb{R}^{|\mathcal{S}|}$，定义策略诱导的转移矩阵与奖励向量：

$$
P^\pi_{ss'} = \sum_a \pi(a\mid s) P(s'\mid s,a), \qquad
R^\pi_s = \sum_a \pi(a\mid s) R(s,a),
$$

则 (4) 写成 $V^\pi = R^\pi + \gamma P^\pi V^\pi$，从而

$$
\boxed{\,V^\pi = (I - \gamma P^\pi)^{-1} R^\pi\,} \tag{5}
$$

**可逆性证明**：$P^\pi$ 是随机矩阵，谱半径 $\rho(P^\pi) \le \|P^\pi\|_\infty = 1$，故 $\gamma P^\pi$ 的谱半径 $\le \gamma < 1$，$I - \gamma P^\pi$ 的特征值都不为 0，可逆。这就是说：**给定策略，价值函数是唯一确定的**——它就是解一个线性方程组。`gridworld.py` 的 demo 会直接用 (5) 算出精确的 $V^\pi$。

## 4. 贝尔曼最优方程

最优价值函数定义为 $V^*(s) = \max_\pi V^\pi(s)$（逐状态取最大；可以证明存在一个策略同时在所有状态达到最大）。它满足：

$$
\boxed{\,V^*(s) = \max_a \left[R(s,a) + \gamma \sum_{s'} P(s'\mid s,a)\, V^*(s')\right]\,} \tag{6}
$$

$$
Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'\mid s,a) \max_{a'} Q^*(s',a'). \tag{7}
$$

与 (4) 的唯一区别：对动作的**加权平均**换成了**取最大**。这一个 max 让方程从线性变成非线性——不能再用矩阵求逆，必须迭代求解（第 02 章）。

给定 $Q^*$，最优策略直接贪心读出：$\pi^*(s) \in \arg\max_a Q^*(s,a)$。**这就是为什么价值方法（DQN 等）只学 $Q$ 不学 $\pi$**。

## 5. 为什么 (6) 有唯一解：压缩映射定理

定义**贝尔曼最优算子** $\mathcal{T}^*: \mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}^{|\mathcal{S}|}$：

$$
(\mathcal{T}^* V)(s) = \max_a \left[R(s,a) + \gamma \sum_{s'} P(s'\mid s,a) V(s')\right].
$$

**定理**：$\mathcal{T}^*$ 在无穷范数下是 $\gamma$-压缩：对任意 $U, V$，

$$
\|\mathcal{T}^* U - \mathcal{T}^* V\|_\infty \le \gamma \|U - V\|_\infty. \tag{8}
$$

**证明**：对任意 $s$，不妨设 $(\mathcal{T}^*U)(s) \ge (\mathcal{T}^*V)(s)$，取 $a_U = \arg\max_a [R(s,a) + \gamma\sum_{s'}P(s'\mid s,a)U(s')]$，则

$$
\begin{aligned}
0 \le (\mathcal{T}^* U)(s) - (\mathcal{T}^* V)(s)
&\le \left[R(s,a_U) + \gamma{\textstyle\sum_{s'}}P U(s')\right] - \left[R(s,a_U) + \gamma{\textstyle\sum_{s'}}P V(s')\right] \\
&= \gamma \sum_{s'} P(s'\mid s,a_U)\,[U(s') - V(s')] \;\le\; \gamma \|U-V\|_\infty,
\end{aligned}
$$

第二个不等号成立是因为 $a_U$ 对 $V$ 未必最优（max 换成任意固定动作只会变小）。对称地处理另一侧即得 (8)。∎

由 **Banach 不动点定理**：完备度量空间上的压缩映射存在唯一不动点，且从任意初值迭代 $V \leftarrow \mathcal{T}^* V$ 以几何速率收敛：

$$
\|V_k - V^*\|_\infty \le \gamma^k \|V_0 - V^*\|_\infty.
$$

这**同时给出了**：(a) $V^*$ 存在且唯一；(b) 求它的算法——值迭代（第 02 章）；(c) 收敛速率——$\gamma$ 越接近 1 收敛越慢，这解释了为什么长时域问题（稀疏奖励、金融里的长期收益）本质上更难。

同样的论证对贝尔曼期望算子 $\mathcal{T}^\pi$ 也成立（把 max 换成对 $\pi$ 加权平均，证明更简单），所以策略评估的迭代解法也收敛。

## 6. 直觉小结

| 概念 | 一句话直觉 |
|---|---|
| 马尔可夫性 | 状态包含决策所需的全部信息，历史不再重要 |
| $\gamma$ | 未来的钱打折；也是"有效时域 $\approx \frac{1}{1-\gamma}$"的旋钮 |
| 贝尔曼期望方程 | "现在的价值 = 即时奖励 + 折扣后的未来价值"，自洽性约束 |
| 贝尔曼最优方程 | 同上，但每步都做最优选择 |
| 压缩映射 | 每迭代一次，误差至少缩小到 $\gamma$ 倍 → 必然收敛到唯一答案 |

## 7. 运行 demo

```bash
python 01_mdp_bellman/gridworld.py
```

demo 做两件事：
1. 构造 4×4 GridWorld 的 $P, R$ 张量，对均匀随机策略用公式 (5) **解析求解** $V^\pi$；
2. 用迭代 $V \leftarrow \mathcal{T}^\pi V$ 数值求解，验证两者一致，并打印每轮误差以观察几何收敛（斜率正是 $\gamma$）。

## 参考

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Ch. 3.
- Puterman, *Markov Decision Processes*, Ch. 6（压缩映射论证的严格版本）.
