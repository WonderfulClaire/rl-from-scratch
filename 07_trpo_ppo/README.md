# 07 · TRPO 与 PPO：可信步长的策略优化

> 策略梯度的死穴：步长。一步太大，策略崩了，采出来的数据也跟着崩——恶性循环，且没有"重来"按钮（数据分布随策略移动）。TRPO 用严格的数学回答"最多能走多大步"，PPO 用一个 clip 把答案工程化。PPO 至今仍是 RLHF 的核心引擎（第 10 章）。
>
> 代码：[`ppo.py`](ppo.py) —— 单文件 PPO-clip（GAE + minibatch 多 epoch + 熵正则），CartPole-v1 验证。

---

## 1. 性能差引理（Performance Difference Lemma）

一切的起点。对任意两个策略 $\pi', \pi$：

$$
J(\pi') - J(\pi) = \frac{1}{1-\gamma}\,\mathbb{E}_{s\sim d^{\pi'}}\mathbb{E}_{a\sim\pi'}\big[A^\pi(s,a)\big]. \tag{1}
$$

**证明**：由 $A^\pi(s,a) = \mathbb{E}[r + \gamma V^\pi(s') \mid s,a] - V^\pi(s)$，对 $\pi'$ 的轨迹逐时刻求和：

$$
\mathbb{E}_{\tau\sim\pi'}\Big[\sum_t \gamma^t A^\pi(s_t,a_t)\Big]
= \mathbb{E}_{\tau\sim\pi'}\Big[\sum_t \gamma^t\big(r_{t+1} + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)\big)\Big].
$$

后两项望远镜相消，剩 $\mathbb{E}_{\tau\sim\pi'}\big[\sum_t \gamma^t r_{t+1}\big] - \mathbb{E}_{s_0}[V^\pi(s_0)] = J(\pi') - J(\pi)$。左边按 $d^{\pi'}$ 归一化即得 (1)。∎

**含义**：新策略的提升 = 在**新策略访问的状态上**、按新策略选动作的旧优势的期望。问题：$d^{\pi'}$ 依赖 $\pi'$——还没更新就要知道更新后去哪，鸡生蛋。

## 2. 替代目标与 TRPO 下界

**近似**：用旧分布 $d^{\pi}$ 替换 $d^{\pi'}$（这是本章唯一的近似！），再用重要性采样把动作期望改写到旧策略上：

$$
L_\pi(\pi') = \frac{1}{1-\gamma}\,\mathbb{E}_{s\sim d^{\pi}}\mathbb{E}_{a\sim\pi}\left[\frac{\pi'(a\mid s)}{\pi(a\mid s)}A^\pi(s,a)\right]. \tag{2}
$$

**TRPO 定理**（Schulman et al., 2015；Achiam 版形式）：

$$
J(\pi') - J(\pi) \;\ge\; L_\pi(\pi') - C\,\sqrt{\mathbb{E}_{s\sim d^\pi}\big[D_{\mathrm{KL}}(\pi\|\pi')[s]\big]}, \qquad C \propto \frac{\gamma\,\max|A^\pi|}{(1-\gamma)^2}. \tag{3}
$$

**解读**：只要 KL 足够小，最大化替代目标 (2) 就**保证真实性能单调不降**（右边 ≥ 0 时）。这是"每次更新都不许变差"的数学承诺——minorize-maximize（MM）算法的 RL 版本。

**TRPO 的实际形式**：惩罚系数 $C$ 太保守，改成硬约束：

$$
\max_{\theta'}\; L_{\theta}(\theta') \quad \text{s.t.}\quad \bar D_{\mathrm{KL}}(\theta \| \theta') \le \delta.
$$

二阶展开后是自然梯度问题：$\theta' = \theta + \sqrt{\frac{2\delta}{g^\top F^{-1} g}}\, F^{-1} g$，其中 $F$ 是 Fisher 信息矩阵（KL 的 Hessian）、$g$ 是策略梯度。$F^{-1}g$ 用共轭梯度近似求解，再加回溯线搜索兜底。**自然梯度的意义**：在"策略分布空间"而非"参数空间"度量步长——参数变化多大不重要，分布变化多大才重要。

## 3. PPO：把信赖域做成一个 clip

TRPO 太重（二阶、共轭梯度、线搜索）。PPO-clip 用一阶方法达到同样效果。记 $\rho_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}$：

$$
\boxed{\;L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t \hat A_t,\;\ \text{clip}(\rho_t,\, 1-\epsilon,\, 1+\epsilon)\, \hat A_t\big)\Big]\;} \tag{4}
$$

**逐情形分析**（理解 PPO 只需要这张表）：

| 情形 | 未裁剪项 | 裁剪项 | min 取谁 | 效果 |
|---|---|---|---|---|
| $\hat A_t > 0$, $\rho_t \le 1+\epsilon$ | $\rho_t\hat A_t$ | ≥ 它 | 未裁剪 | 正常提升好动作概率 |
| $\hat A_t > 0$, $\rho_t > 1+\epsilon$ | 大 | $(1+\epsilon)\hat A_t$ | 裁剪项（常数） | **梯度为 0**：好动作概率提够了就停 |
| $\hat A_t < 0$, $\rho_t \ge 1-\epsilon$ | $\rho_t\hat A_t$ | ≥ 它（更接近 0） | 未裁剪 | 正常压低坏动作概率 |
| $\hat A_t < 0$, $\rho_t < 1-\epsilon$ | 接近 0 | $(1-\epsilon)\hat A_t$ | 裁剪项（常数） | **梯度为 0**：坏动作概率压够了就停 |

要点：$\min$（而非直接 clip）使 (4) 是 $L_\pi$ 的**悲观下界**——越界方向的改进不计入目标，但越界方向的**恶化**仍然被惩罚（比如 $\hat A_t>0$ 而 $\rho_t < 1-\epsilon$ 时梯度照常存在，把它拉回来）。这正是 (3) 的"下界最大化"精神的廉价实现。

### 3.1 完整算法

每轮：
1. 用 $\pi_{\theta_\text{old}}$ 收集 $N$ 步 rollout；
2. 用 GAE（第 06 章公式 (5)）算 $\hat A_t$，回报目标 $\hat R_t = \hat A_t + V(s_t)$；
3. 对同一批数据做 $K$ 个 epoch 的 minibatch SGD，损失 $= -L^{\text{CLIP}} + c_1\,(V_\theta - \hat R_t)^2 - c_2\,\mathcal{H}[\pi_\theta]$；
4. $\theta_\text{old} \leftarrow \theta$。

**为什么能复用数据做多个 epoch**：重要性采样比 $\rho_t$ 修正了分布差异，clip 限制了 $\rho_t$ 偏离 1 的程度——所以"轻度 off-policy"是安全的。这是 PPO 样本效率高于 A2C 的原因。

### 3.2 实现细节即性能（真正拉开差距的地方）

1. **优势标准化**（per-batch zero-mean unit-std）；
2. **正交初始化** + 策略头小增益（0.01）；
3. **价值损失裁剪**（可选，收益存疑但通行）；
4. **梯度范数裁剪** 0.5；
5. **学习率退火**；
6. **`truncated` 时仍 bootstrap**（老朋友了）；
7. 监控 **approx KL** 与 **clip fraction**：KL 爆了说明 epoch 太多或 lr 太大。

> RLHF 预告（第 10 章）：把这里的环境换成"prompt → 生成 token 序列 → 奖励模型打分"，PPO 原封不动就是 InstructGPT 的核心。第 10 章还会看到 GRPO 如何把 Critic 干掉。

## 4. 运行 demo

```bash
python 07_trpo_ppo/ppo.py
```

CartPole-v1，判定 avg20 ≥ 475。demo 同时打印 approx KL 和 clip fraction，观察 PPO 的"自我约束"行为。

## 参考

- Schulman et al. (2015), *Trust Region Policy Optimization*.
- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*.
- Achiam (2018), *Spinning Up: Extra Material on TRPO*.
- Huang et al. (2022), *The 37 Implementation Details of PPO*.
