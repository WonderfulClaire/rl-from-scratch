# 05 · DQN 家族：深度价值学习

> 表格装不下 $|\mathcal{S}| = 10^{9000}$（Atari 像素）的世界。用神经网络 $Q_\phi(s,a)$ 逼近 Q 函数，就得到 DQN——但朴素的"Q-learning + 神经网络"会发散，DQN 的全部智慧在于**怎么让它不发散**。
>
> 代码：[`dqn.py`](dqn.py) —— 单文件实现 DQN，可选开关 `--double`（Double DQN）、`--dueling`（Dueling 结构）、`--per`（优先经验回放），CartPole-v1 验证。

---

## 1. 从表格到函数逼近

参数化 $Q_\phi$，把 Q-learning 更新改写成对损失

$$
\mathcal{L}(\phi) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\Big[\big(\underbrace{r + \gamma \max_{a'} Q_{\phi^-}(s',a')}_{\text{TD 目标 } y} - Q_\phi(s,a)\big)^2\Big] \tag{1}
$$

的随机梯度下降。注意目标 $y$ 里用的是**目标网络** $\phi^-$ 且**不回传梯度**（semi-gradient）：TD 目标被当作常数。

### 1.1 致命三要素（The Deadly Triad）

以下三者同时出现时，价值迭代可能**发散**（Baird 反例）：

1. **函数逼近**（不同状态共享参数，一处更新处处牵动）；
2. **自举**（目标里含自己的估计）；
3. **off-policy**（更新分布 ≠ 目标策略分布）。

DQN 三者全占。它不是从理论上消除了发散，而是用两个工程支柱把训练"稳"住。

## 2. DQN 的两大支柱

### 2.1 经验回放（Experience Replay）

把转移 $(s,a,r,s')$ 存进环形缓冲 $\mathcal{D}$，训练时均匀抽 minibatch。作用：

- **打破时间相关性**：连续帧高度相关，直接 SGD 违背 i.i.d. 假设，梯度方向系统性偏斜；
- **提高样本效率**：一条经验被多次复用（这依赖 Q-learning 的 off-policy 性质，第 04 章）。

### 2.2 目标网络（Target Network）

每 $C$ 步才把 $\phi$ 复制给 $\phi^-$（或软更新 $\phi^- \leftarrow \tau\phi + (1-\tau)\phi^-$）。若没有它，(1) 中目标 $y$ 随 $\phi$ 每步移动——"追逐自己的尾巴"，自举误差正反馈放大。目标网络把移动目标冻结一段时间，把不动点迭代 $Q \leftarrow \mathcal{T}^* Q$ 的结构近似恢复出来：每个冻结周期内，训练是朝**固定目标**的标准回归。

## 3. Double DQN

### 3.1 过估计在深度里更凶

第 04 章证明了 $\mathbb{E}[\max_i X_i] \ge \max_i \mathbb{E}[X_i]$。函数逼近的泛化误差充当了持续注入的噪声源，且过估计通过自举**沿贝尔曼方程向后传播、复利累积**。

### 3.2 修复：解耦"选择"与"评估"

Double DQN（van Hasselt et al., 2016）复用现成的两套网络——在线网络选动作、目标网络评动作：

$$
y^{\text{DDQN}} = r + \gamma\, Q_{\phi^-}\big(s',\, \arg\max_{a'} Q_\phi(s',a')\big). \tag{2}
$$

对比原始 DQN 的 $y = r + \gamma\, Q_{\phi^-}(s', \arg\max_{a'} Q_{\phi^-}(s',a'))$：只改了 argmax 的下标，一行代码，显著降低过估计。

## 4. Dueling DQN

把 Q 分解为**状态价值 + 优势**两条流：

$$
Q_\phi(s,a) = V_\eta(s) + \Big(A_\psi(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A_\psi(s,a')\Big). \tag{3}
$$

**为什么减去均值**：不减的话 $(V, A)$ 不可辨识——$V+c, A-c$ 给出同一个 Q，训练目标欠定。减去均值强制 $\sum_a A = 0$，分解唯一。（原文用 max 归一化，均值版更稳定，是通行实现。）

**为什么有效**：多数状态下动作选择无关紧要（价值主要由状态决定）。V 流让"这个状态好不好"的信息**一次更新、全动作共享**，而不是每个动作各学一遍。动作越多收益越大。

## 5. 优先经验回放（PER）

均匀采样对"惊讶"的转移（$|\delta|$ 大）不公平。PER 按优先级采样：

$$
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \qquad p_i = |\delta_i| + \epsilon. \tag{4}
$$

**代价**：改变了采样分布，(1) 的期望被扭曲。**修正**：重要性采样权重

$$
w_i = \left(\frac{1}{N\, P(i)}\right)^\beta \Big/ \max_j w_j, \tag{5}
$$

$\beta$ 从初值线性退火到 1（训练后期需要无偏；前期允许些许偏差换取速度）。损失变为 $\frac{1}{B}\sum_i w_i\, \delta_i^2$。

## 6. 其余 Rainbow 组件（简述）

| 组件 | 一句话 |
|---|---|
| Noisy Nets | 用可学习的参数噪声替代 ε-greedy，探索状态相关 |
| Distributional (C51) | 学回报的**分布**而非期望，投影贝尔曼算子 |
| n-step | 目标用 n 步回报（第 03 章），加速奖励传播 |
| Rainbow | 以上全加，消融显示 PER 与 n-step 贡献最大 |

## 7. 实现要点（坑清单）

- TD 目标**必须 `detach()`**：忘了就是在做全梯度 TD，通常更不稳。
- 终止状态的目标是 $y = r$（没有 bootstrap 项）；但 gymnasium 的 `truncated`（超时截断）**不是**真终止，超时时仍应 bootstrap——混淆两者是 CartPole 训不好的头号原因。
- Huber 损失比 MSE 更稳（大 TD 误差时梯度有界）。
- ε 线性退火比指数退火更好调。

## 8. 运行 demo

```bash
python 05_dqn_family/dqn.py                    # 原始 DQN
python 05_dqn_family/dqn.py --double --dueling # Double + Dueling
python 05_dqn_family/dqn.py --double --per     # Double + PER
```

CartPole-v1 判定：最近 20 回合平均 ≥ 475 即解决（满分 500）。CPU 上几分钟内完成。

## 参考

- Mnih et al. (2015), *Human-level control through deep RL* (Nature DQN).
- van Hasselt et al. (2016), *Deep RL with Double Q-learning*.
- Wang et al. (2016), *Dueling Network Architectures*.
- Schaul et al. (2016), *Prioritized Experience Replay*.
- Hessel et al. (2018), *Rainbow*.
