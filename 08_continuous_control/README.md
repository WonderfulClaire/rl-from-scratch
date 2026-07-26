# 08 · 连续控制：DDPG、TD3 与 SAC

> 机械臂的关节扭矩、无人机的桨速、投资组合的仓位权重——动作是连续向量时，$\max_a Q(s,a)$ 无法枚举。两条出路：把 argmax 用一个网络"记住"（DDPG/TD3），或干脆学随机策略并鼓励熵（SAC）。
>
> 代码：[`ddpg.py`](ddpg.py)、[`td3.py`](td3.py)、[`sac.py`](sac.py) —— Pendulum-v1 验证。

---

## 1. 确定性策略梯度（DPG）定理

确定性策略 $\mu_\theta: \mathcal{S} \to \mathcal{A}$，目标 $J(\theta) = \mathbb{E}_{s\sim d^{\mu}}[Q^\mu(s, \mu_\theta(s))]$。

**DPG 定理**（Silver et al., 2014）：

$$
\nabla_\theta J = \mathbb{E}_{s\sim d^{\mu}}\Big[\nabla_\theta \mu_\theta(s)\, \nabla_a Q^\mu(s,a)\big|_{a=\mu_\theta(s)}\Big]. 
$$

直觉：链式法则——"动作往哪边挪，Q 会变大？就把策略往那边挪"。与随机策略梯度（第 06 章）相比：
- 不需要对动作空间积分（期望只在状态上），**方差低得多**；
- DPG 是随机策略梯度在策略方差 → 0 时的极限（Silver et al. 证明了这一点）；
- 代价：策略自身不探索，必须外加噪声（且是 off-policy 学习）。

## 2. DDPG = DPG + DQN 全家桶

四个网络：Actor $\mu_\theta$、Critic $Q_\phi$ 及各自的目标网络 $\mu_{\theta^-}, Q_{\phi^-}$。

**Critic**（就是连续动作版 DQN，公式对照第 05 章 (1)）：

$$
\mathcal{L}(\phi) = \mathbb{E}_{\mathcal{D}}\Big[\big(r + \gamma\, Q_{\phi^-}(s', \mu_{\theta^-}(s')) - Q_\phi(s,a)\big)^2\Big]. 
$$

**Actor**（公式 (1) 的采样版，PyTorch 里就是 `-Q(s, mu(s)).mean()` 反向传播）：

$$
\nabla_\theta J \approx \frac{1}{B}\sum_i \nabla_\theta\, Q_\phi\big(s_i, \mu_\theta(s_i)\big).
$$

**软更新**：$\phi^- \leftarrow \tau\phi + (1-\tau)\phi^-$，$\tau \sim 10^{-3}$。**探索**：$a = \mu_\theta(s) + \mathcal{N}(0,\sigma)$（高斯噪声足够，OU 噪声无必要）。

**DDPG 的病**：Critic 对动作可导 ⟹ Actor 会**精确地爬到 Q 函数的（错误）峰值上**——函数逼近误差被 Actor 主动放大，比离散 max 更凶。训练后期常见崩盘。

## 3. TD3：给 DDPG 治病的三板斧

**① Clipped Double Q**：两个 Critic，目标取小：

$$
y = r + \gamma \min_{j=1,2} Q_{\phi_j^-}\big(s',\, \tilde a'\big). 
$$

与 Double DQN 的"解耦选择/评估"不同，取 min 直接制造**悲观**估计——宁可低估不要高估（低估不会被 Actor 放大，高估会）。

**② 目标策略平滑**：目标动作加截断噪声：

$$
\tilde a' = \text{clip}\big(\mu_{\theta^-}(s') + \text{clip}(\epsilon, -c, c),\; a_{\min},\, a_{\max}\big), \quad \epsilon\sim\mathcal{N}(0,\tilde\sigma).
$$

强制 Q 目标在动作邻域内平滑——正则化掉"针尖峰值"，Actor 没有尖峰可爬。

**③ 延迟策略更新**：Critic 每更新 $d$ 次（$d=2$），Actor 和目标网络才更新一次。让 Critic 先"想清楚"，减少用烂估计更新策略的次数。

## 4. SAC：最大熵 RL

### 4.1 换目标函数

$$
J(\pi) = \mathbb{E}\Big[\sum_t \gamma^t\big(r_{t+1} + \alpha\,\mathcal{H}(\pi(\cdot\mid s_t))\big)\Big]. 
$$

不是"顺便加个熵正则"，而是**改了优化的问题本身**：在拿奖励的同时保持尽可能随机。好处：探索内生、对超参鲁棒、学到多模态解（两条路一样好就都保留，环境变化时切换成本低）。

### 4.2 软贝尔曼方程

$$
Q^\pi(s,a) = r + \gamma\,\mathbb{E}_{s'}\big[V^\pi(s')\big], \qquad
V^\pi(s) = \mathbb{E}_{a\sim\pi}\big[Q^\pi(s,a) - \alpha\log\pi(a\mid s)\big]. 
$$

（把熵 $\mathcal{H} = -\mathbb{E}\log\pi$ 摊进 V 的定义即可。）软策略评估算子仍是 $\gamma$-压缩——第 01 章的机器完全复用。

**策略改进**：固定 Q，最小化

$$
\pi_{\text{new}} = \arg\min_{\pi'} D_{\mathrm{KL}}\!\left(\pi'(\cdot\mid s)\,\Big\|\,\frac{\exp(Q(s,\cdot)/\alpha)}{Z(s)}\right), 
$$

即把策略往"Q 的 Boltzmann 分布"上投影。可证明软策略迭代单调改进并收敛到最大熵最优策略（表格情形）。

> 这个"最优策略 ∝ exp(Q/α)"的结构，与第 10 章 RLHF 的 KL 约束闭式解 $\pi^* \propto \pi_{\text{ref}}\exp(r/\beta)$ 是同一个数学——记住它，DPO 推导会突然变得显然。

### 4.3 重参数化技巧

策略是 squashed Gaussian：$a = \tanh(m_\theta(s) + \sigma_\theta(s)\odot\xi)$，$\xi\sim\mathcal{N}(0,I)$。目标 $\mathbb{E}_{a\sim\pi_\theta}[\cdot]$ 的梯度不走 log-derivative（高方差），而是把随机性外置到 $\xi$，梯度直接穿过 $a$（低方差、有偏差小）。

tanh 压缩的雅可比修正（换元公式，实现时的关键一行）：

$$
\log\pi(a\mid s) = \log\rho(u) - \sum_i \log\big(1 - \tanh^2(u_i)\big), \qquad a = \tanh(u).
$$

**自动温度调节**：把"平均熵 ≥ 目标熵 $\bar{\mathcal{H}}$"（惯例取 $-\dim(\mathcal{A})$）作为约束，对偶上升学 $\alpha$：$\mathcal{L}(\alpha) = \mathbb{E}\big[-\alpha(\log\pi(a\mid s) + \bar{\mathcal{H}})\big]$。熵太低时 $\alpha$ 自动升高。

## 5. 三算法对比

| | DDPG | TD3 | SAC |
|---|---|---|---|
| 策略 | 确定性 | 确定性 | 随机（squashed Gaussian） |
| Critic 数 | 1 | 2（取 min） | 2（取 min） |
| 探索 | 外加噪声 | 外加噪声 | 内生（熵） |
| 过估计对策 | 无 | 三板斧 | min + 熵 |
| 超参敏感度 | 高 | 中 | 低 |
| 实践地位 | 教学基线 | 简单任务够用 | 连续控制默认选择 |

## 6. 运行 demo

```bash
python 08_continuous_control/ddpg.py
python 08_continuous_control/td3.py
python 08_continuous_control/sac.py
```

Pendulum-v1（回报范围约 $[-1600, 0]$），判定：最近 20 回合平均 ≥ -200。CPU 数分钟。

## 参考

- Silver et al. (2014), *Deterministic Policy Gradient Algorithms*.
- Lillicrap et al. (2016), *Continuous Control with Deep RL* (DDPG).
- Fujimoto et al. (2018), *Addressing Function Approximation Error* (TD3).
- Haarnoja et al. (2018), *Soft Actor-Critic* (+ 2019 自动温度版).
