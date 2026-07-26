# 06 · 策略梯度：REINFORCE 与 A2C

> 价值方法绕道 Q 再贪心；策略方法直接对 $J(\pi_\theta)$ 做梯度上升。本章完整证明策略梯度定理——它是 REINFORCE、A2C、PPO、DDPG、SAC、RLHF 全家的宪法。
>
> 代码：[`reinforce.py`](reinforce.py)、[`a2c.py`](a2c.py) —— CartPole-v1 验证。

---

## 1. 为什么需要策略方法

- 连续/高维动作空间：$\max_a Q(s,a)$ 本身就是个优化问题，价值方法做不动；
- 需要随机策略的场合（部分可观测、博弈的混合策略均衡）；
- 策略常比价值函数**简单**（"看到球往左就往左"容易学，精确的 Q 值很难学）；
- 策略参数连续变化 ⟹ 动作分布连续变化，没有价值方法"argmax 突变"的不稳定。

代价：**朴素策略梯度方差极大、样本效率低**，本章后半和第 07 章都在治这个病。

## 2. 策略梯度定理：完整推导

目标（回合制、从初始分布出发）：

$$
J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)], \qquad R(\tau) = \sum_{t=0}^{T-1}\gamma^t r_{t+1}.
$$

轨迹概率分解（环境项与策略项分离）：

$$
p_\theta(\tau) = \mu_0(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t\mid s_t)\,P(s_{t+1}\mid s_t,a_t).
$$

**第一步：log-derivative trick。** 由 $\nabla_\theta p_\theta = p_\theta \nabla_\theta \log p_\theta$：

$$
\nabla_\theta J = \nabla_\theta \int p_\theta(\tau) R(\tau)\, d\tau = \int p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau)\, R(\tau)\, d\tau = \mathbb{E}_\tau\big[\nabla_\theta \log p_\theta(\tau)\, R(\tau)\big].
$$

**第二步：环境项消失。** $\log p_\theta(\tau) = \log\mu_0 + \sum_t \log\pi_\theta(a_t\mid s_t) + \sum_t \log P(s_{t+1}\mid s_t,a_t)$，只有中间一项含 $\theta$：

$$
\nabla_\theta J = \mathbb{E}_\tau\Big[\Big(\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\Big) R(\tau)\Big]. 
$$

**不需要环境模型可微**——这是 RL 策略梯度与控制论方法的分水岭。

**第三步：因果性（reward-to-go）。** 时刻 $t$ 的动作不影响 $t$ 之前的奖励。严格论证：对 $t' < t$，$\mathbb{E}\big[\nabla\log\pi_\theta(a_t\mid s_t)\, r_{t'+1}\big] = \mathbb{E}\big[r_{t'+1}\,\mathbb{E}[\nabla\log\pi_\theta(a_t\mid s_t)\mid s_t]\big] = 0$，因为对任意 $s$：

$$
\mathbb{E}_{a\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a\mid s)\big] = \sum_a \pi_\theta \frac{\nabla\pi_\theta}{\pi_\theta} = \nabla_\theta \sum_a \pi_\theta(a\mid s) = \nabla_\theta 1 = 0. 
$$

于是 $R(\tau)$ 可换成 reward-to-go $G_t$：

$$
\nabla_\theta J = \mathbb{E}\Big[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\, G_t\Big]. 
$$

**第四步：基线。** 由 (2)，任何只依赖状态的 $b(s_t)$ 满足 $\mathbb{E}[\nabla\log\pi_\theta(a_t\mid s_t)\, b(s_t)] = 0$，可以随意减去而**不引入偏差**：

$$
\boxed{\;\nabla_\theta J = \mathbb{E}\Big[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\big(G_t - b(s_t)\big)\Big]\;} 
$$

方差分析表明近似最优的基线是 $b(s) \approx V^\pi(s)$，此时 $G_t - V^\pi(s_t)$ 正是**优势** $A^\pi$ 的估计——"这个动作比平均好多少"。直觉：如果所有回报都是正的，(3) 会"提升一切动作的概率、只是幅度不同"，数值上噪声极大；减掉基线后，比平均差的动作概率被明确压低。

**通用形式**（policy gradient 的"万能模板"）：

$$
\nabla_\theta J = \mathbb{E}\Big[\sum_t \nabla_\theta \log\pi_\theta(a_t\mid s_t)\, \Psi_t\Big],
$$

$\Psi_t$ 可取：$R(\tau)$（REINFORCE 原始版）、$G_t$（因果版）、$G_t - b(s_t)$（带基线）、$Q^\pi(s_t,a_t)$、$A^\pi(s_t,a_t)$（方差最小的理想选择）、$\delta_t$（A2C 实际用的一步估计）、GAE（下一节）。**从 REINFORCE 到 PPO 的进化史 = $\Psi_t$ 的方差递减史。**

## 3. REINFORCE

蒙特卡洛策略梯度：采整回合，按 (4) 更新（$b$ 用回报的滑动平均或不用）：

$$
\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,(G_t - b).
$$

极简、无偏、方差巨大（MC 的老毛病，第 03 章）。CartPole 能解决，再难就不行了。

## 4. A2C（Advantage Actor-Critic）

用 Critic $V_w$ 提供低方差的优势估计，Actor 按策略梯度更新：

- **TD 优势估计**：$\hat A_t = \delta_t = r_{t+1} + \gamma V_w(s_{t+1}) - V_w(s_t)$。注意 $\mathbb{E}[\delta_t\mid s_t,a_t] = Q^\pi - V^\pi = A^\pi$，所以 $\delta_t$ 是 $A^\pi$ 的**无偏估计——如果 $V_w = V^\pi$**；$V_w$ 不准时有偏（自举偏差换方差，第 03 章的交易再现）。
- **Actor 损失**：$-\log\pi_\theta(a_t\mid s_t)\,\hat A_t$（$\hat A_t$ 必须 `detach`）；
- **Critic 损失**：$\big(r_{t+1} + \gamma V_w(s_{t+1}) - V_w(s_t)\big)^2$（目标同样 detach）；
- **熵正则**：加 $-c\,\mathcal{H}(\pi_\theta(\cdot\mid s_t))$ 防止策略过早坍缩成确定性。

这就是 GPI（第 02 章）：Critic 做不完全评估，Actor 做不完全改进。

### 4.1 GAE：广义优势估计

n-step 优势估计的 λ-混合（与第 03 章 TD(λ) 完全同构）：

$$
\hat A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l\, \delta_{t+l}. 
$$

$\lambda = 0$ 退化为 $\delta_t$（低方差高偏差），$\lambda = 1$ 时 $\sum_l \gamma^l \delta_{t+l} = G_t - V(s_t)$（无偏高方差，望远镜求和可验证）。实践常取 $\lambda \in [0.95, 0.99]$。反向递推一行实现：$\hat A_t = \delta_t + \gamma\lambda \hat A_{t+1}$。PPO（第 07 章）标配 GAE。

## 5. 运行 demo

```bash
python 06_policy_gradient/reinforce.py   # REINFORCE + 滑动平均基线
python 06_policy_gradient/a2c.py         # A2C (含 GAE 与熵正则)
```

两者都在 CartPole-v1 上达到 avg20 ≥ 475。对比观察：REINFORCE 曲线抖动明显更大——这就是方差的具象化。

## 参考

- Williams (1992), *REINFORCE*.
- Sutton et al. (2000), *Policy Gradient Methods for RL with Function Approximation*.
- Schulman et al. (2016), *High-Dimensional Continuous Control Using GAE*.
