# 统一符号表（Notation）

全库所有章节共用这套符号。与 Sutton & Barto（第二版）基本一致，深度 RL 部分补充了参数化记号。

## MDP 基本要素

| 符号 | 含义 |
|---|---|
| $\mathcal{S}$ | 状态空间 |
| $\mathcal{A}$ | 动作空间 |
| $s, s'$ | 当前状态、下一状态 |
| $a, a'$ | 当前动作、下一动作 |
| $r$ | 即时奖励（标量） |
| $P(s'\mid s,a)$ | 状态转移概率 |
| $R(s,a)$ 或 $r(s,a)$ | 期望即时奖励 |
| $\gamma \in [0,1)$ | 折扣因子 |
| $\mu_0(s)$ | 初始状态分布 |
| $\tau = (s_0,a_0,r_1,s_1,\dots)$ | 一条轨迹（trajectory） |

## 策略与价值

| 符号 | 含义 |
|---|---|
| $\pi(a\mid s)$ | 随机策略：状态 $s$ 下选动作 $a$ 的概率 |
| $\mu(s)$ | 确定性策略（DDPG/TD3 中使用） |
| $G_t = \sum_{k=0}^{\infty}\gamma^k r_{t+k+1}$ | 从时刻 $t$ 起的折扣回报 |
| $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$ | 状态价值函数 |
| $Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$ | 动作价值函数 |
| $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ | 优势函数 |
| $V^*, Q^*$ | 最优价值函数 |
| $\pi^*$ | 最优策略 |
| $d^\pi(s)$ | 策略 $\pi$ 诱导的（折扣）状态访问分布 |
| $J(\pi) = \mathbb{E}_{s_0\sim\mu_0}[V^\pi(s_0)]$ | 策略的期望回报（优化目标） |

## 参数化与深度 RL

| 符号 | 含义 |
|---|---|
| $\theta$ | 策略网络参数，策略记为 $\pi_\theta$ |
| $\phi$ 或 $w$ | 价值网络参数，如 $Q_\phi, V_\phi$ |
| $\theta^-, \phi^-$ | 目标网络（target network）参数 |
| $\alpha$ | 学习率（或 SAC 中的温度系数，随上下文说明） |
| $\mathcal{D}$ | 经验回放池（replay buffer） |
| $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ | TD 误差 |
| $\hat{A}_t$ | 优势函数的估计值（如 GAE） |
| $\rho_t(\theta) = \dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}$ | 重要性采样比 |
| $D_\text{KL}(p \,\|\, q)$ | KL 散度 |
| $\mathcal{H}(\pi(\cdot\mid s))$ | 策略熵 |

## RLHF / 偏好学习（第 10 章）

| 符号 | 含义 |
|---|---|
| $x$ | prompt（上下文） |
| $y, y_w, y_l$ | 回答；偏好对中被选中/被拒绝的回答 |
| $r_\psi(x,y)$ | 奖励模型 |
| $\pi_\text{ref}$ | 参考策略（通常是 SFT 模型） |
| $\beta$ | KL 惩罚系数 / DPO 温度 |

## 约定

- 时间下标：$s_t$ 时刻 $t$ 的状态，奖励 $r_{t+1}$ 是执行 $a_t$ 之后收到的（Sutton & Barto 约定）。
- 期望 $\mathbb{E}_\pi[\cdot]$ 表示轨迹按 $\pi$ 与环境动态采样。
- 代码中 `gamma, alpha, eps` 等变量名与上述数学符号一一对应，并在注释中标注公式编号。
