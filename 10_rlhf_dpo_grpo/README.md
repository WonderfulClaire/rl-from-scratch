# 10 · 大模型后训练：PPO、GRPO、KL 与奖励设计

[返回全库](../README.md) · [策略梯度](../06_policy_gradient/) · [环境中的 PPO](../07_trpo_ppo/) · [数值演示](mechanism_demo.py) · [字符级实现](toy_rlhf.py)

**后训练中的 RL 改变的是模型生成回答的概率。** 模型先生成回答，奖励函数评估回答，再提高相对更好回答的概率。奖励通常不可微；梯度通过模型对已采样 token 的对数概率传播。

本章先建立环境 RL 与语言生成的对应，再解释一次更新的计算，最后联系奖励设计和 DPO。标准算法与玩具实现分开说明；这里没有大模型训练效果或算法排名的复现结论。

## 1. 从环境动作到生成 token

| 对象 | 经典环境 RL | 单轮语言模型 RL | 工具型 Agent |
|---|---|---|---|
| 状态 | 环境状态或观测历史 | 提示词与已生成前缀 | 对话、工具反馈与可见环境历史 |
| 动作 | 离散操作或连续控制量 | 下一个 token | 文本生成及工具调用 |
| 转移 | 环境决定，可随机 | 追加 token；给定动作后确定 | 工具执行与外部环境共同决定 |
| 奖励 | 即时或终局奖励 | 回答评分，也可有过程奖励 | 完成度、调用结果、成本等 |
| 数据 | 与环境交互的轨迹 | 提示词 → 采样回答 → 评分 | 多轮交互轨迹 |
| 初始化 | 可从随机策略开始 | 通常从预训练或 SFT 模型开始 | 通常从具备语言与工具能力的模型开始 |

整条回答作为一个动作，可得到上下文 bandit 表述；逐 token 建模则是有限回合 MDP。工具反馈不能用“只追加 token 的确定性转移”概括，部分可观测任务还需要历史或其他状态表示。

设提示词为 $`x`$，回答为 $`y=(y_1,\ldots,y_T)`$，长度 $`T`$ 包括实际生成的终止 token（若存在）。状态为 $`s_t=(x,y_{<t})`$，动作为 $`y_t`$，参数为 $`\theta`$：

```math
\pi_\theta(y\mid x)=\prod_{t=1}^T\pi_\theta(y_t\mid s_t),\qquad
\log\pi_\theta(y\mid x)=\sum_{t=1}^T\log\pi_\theta(y_t\mid s_t).
```

策略损失只覆盖回答的有效 token。提示词是条件，padding 不是动作；EOS 与达到最大长度的截断需要分别记录。有限长度无折扣任务可取 $`\gamma=1`$，这是对全库无限时域折扣记号的局部扩展。

```mermaid
flowchart LR
    X[训练提示词] --> B[行为策略采样回答]
    B --> R[评分与有效 token 掩码]
    R --> A[估计优势]
    A --> U[更新当前策略]
    O[冻结的采样 log probability] --> U
    F[参考策略] --> K[KL 正则]
    K --> U
    U --> B
```

## 2. 三个策略各做什么

| 记号 | 身份 | 更新期间的作用 |
|---|---|---|
| $`\pi_\theta`$ | 当前可训练策略 | 计算已采样回答的 log probability，接收梯度 |
| $`\pi_{\mathrm{old}}`$ | 本批数据的行为策略 | 提供冻结的采样概率；新一批 rollout 时刷新 |
| $`\pi_{\mathrm{ref}}`$ | 参考策略 | 提供分布偏离的参照；通常在一个训练阶段内固定 |

old 与 ref 可能初始相同，但用途不同。old 可以由已保存的 log probability 表示，不必另存完整模型。参考、奖励与价值模型的共享和卸载方式也影响显存，不能笼统说必须同时完整驻留四份模型。

若生成使用温度或截断采样，保存的行为概率必须对应实际采样分布。把变换后的样本直接当成未经变换策略的样本，会破坏重要性比值的解释。

## 3. PPO：从回报和价值预测到 token 优势

### 3.1 优势回答“比预期好多少”

设 $`r_t`$ 是生成 token 后的奖励，$`V_\phi(s_t)`$ 是价值模型预测的后续折扣回报。常见 PPO-RLHF 在终点加入标量回答奖励 $`R(x,y)`$，并以参考策略构造 token 级惩罚。在采样时冻结这些量：

```math
r_t=\mathbf 1_{t=T}R(x,y)-\beta\left[\log\pi_{\mathrm{old}}(y_t\mid s_t)-\log\pi_{\mathrm{ref}}(y_t\mid s_t)\right],\qquad\beta\ge0.
```

单个 log 比值可能为负，并不是逐点非负的 KL。InstructGPT 使用参考 SFT 策略的 KL 惩罚，还研究了混合预训练目标；此处只讲 RL 主路径。[InstructGPT，§3.1、§3.5](https://arxiv.org/html/2203.02155v1)

令真实终止状态价值为零，计算 TD 残差与广义优势估计（GAE）：

```math
\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t),\qquad
\hat A_t=\sum_{l=0}^{T-t}(\gamma\lambda)^l\delta_{t+l},\quad 0\le\lambda\le1.
```

时间限制导致的截断是否 bootstrap 取决于任务定义，不能都当成真实终止。更新 actor 时优势停止梯度；critic 拟合相应回报目标。终局奖励相同不意味着各 token 的 GAE 相同，因为前缀价值预测不同。

### 3.2 裁剪限制优化激励，不强制锁定概率

冻结旧概率，在旧样本上计算：

```math
\rho_t(\theta)=\exp\left[\log\pi_\theta(y_t\mid s_t)-\log\pi_{\mathrm{old}}(y_t\mid s_t)\right].
```

PPO 最大化采样平均（写成 loss 时取负号）：

```math
J_{\mathrm{clip}}=\widehat{\mathbb E}_t\left[\min\left(\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right],\quad\epsilon>0.
```

取 $`\epsilon=0.2`$：

| 优势 | 比值 | 未裁剪项 | 最终项 | 作用 |
|---|---|---|---|---|
| 2 | 1.4 | 2.8 | 2.4 | 不再奖励继续提高好动作概率 |
| -2 | 1.4 | -2.8 | -2.8 | 仍惩罚坏动作概率上升 |
| -2 | 0.6 | -1.2 | -1.6 | 不再奖励继续降低坏动作概率 |

clip 不是把所有概率强制锁在区间内，也不是对真实回报的单调改进保证；它是未裁剪 surrogate 的逐样本下界。[PPO，§3、§5](https://arxiv.org/pdf/1707.06347)

第一次更新前 $`\rho=1`$，**不意味着梯度为零**：分母冻结，分子仍可导。复用 rollout 做多次更新时，比值才可能进入裁剪区。监控比值、裁剪比例和相对 old 的 KL，才能知道裁剪是否实际发挥作用。

## 4. GRPO：同题回答之间比较

### 4.1 从一个组算起

给定同一 $`x`$，从 old 独立采样 $`G\ge2`$ 条回答 $`y_i`$，得奖励 $`R_i`$。本章数值演示采用总体标准差：

```math
\bar R=\frac1G\sum_iR_i,\qquad
\sigma_R=\sqrt{\frac1G\sum_i(R_i-\bar R)^2},\qquad
\hat A_i=\frac{R_i-\bar R}{\sigma_R+\eta},\quad\eta>0.
```

奖励 $`[1,0,1,0]`$ 的均值和标准差都是 $`0.5`$，优势近似 $`[1,-1,1,-1]`$。结果监督的 GRPO 将同一回答的优势分给所有有效 token；这不等于已经定位哪个推理步骤导致成功。

令 $`T_i`$ 为回答长度，$`\rho_{i,t}`$ 为当前/旧策略 token 比值，$`k_{i,t}`$ 为下一节的 KL 项。原始结果监督形式先在每条回答内平均，再对组平均：

```math
J_{\mathrm{GRPO}}=\widehat{\mathbb E}_{x,\{y_i\}\sim\pi_{\mathrm{old}}}
\left[\frac1G\sum_{i=1}^G\frac1{T_i}\sum_{t=1}^{T_i}
\left\{\min\left(\rho_{i,t}\hat A_i,\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i\right)-\beta k_{i,t}\right\}\right].
```

改成全批 token 平均会改变长短回答的相对权重，不是等价改写。去掉 critic 节省其开销，但组采样也消耗资源，不能直接推出总显存减半。GRPO 可用可验证奖励，也可用奖励模型；过程监督的优势定义不同。[DeepSeekMath，§4.1.1–4.1.3](https://arxiv.org/html/2402.03300v3#S4.SS1)

### 4.2 全对、全错与没有执行更新

奖励全相同则优势全零；二元奖励下全对、全错均如此。$`\eta`$ 只避免除零，不创造区分信号。此时奖励 surrogate 梯度为零，但 KL 或辅助项仍可能产生梯度，不能直接说整个模型不更新。

排查顺序：有效样本是否为空 → 奖励是否有组内差异 → actor 概率是否被错误 detach → loss 是否有计算图 → backward 与 optimizer.step 是否执行。重采样有差异的组可能提供信号，但会改变题目分布并增加成本，必须记录保留比例和采样预算。

### 4.3 组均值不能直接套“基线无偏”

以下是本章有限样本推导。对固定提示词，假设 $`G`$ 个回答独立同分布，奖励不显式依赖参数，采样策略就是求梯度的策略。令 $`g_i=\nabla_\theta\log\pi_\theta(y_i\mid x)`$。仅减组均值、不除标准差时：

```math
\mathbb E\left[\frac1G\sum_i(R_i-\bar R)g_i\right]
=\left(1-\frac1G\right)\mathbb E[Rg].
```

因为 $`\mathbb E[g_i]=0`$，不同样本的交叉项为零，而均值包含自身奖励 $`R_i/G`$。动作独立基线的无偏结论不能原样套用。leave-one-out 均值可去掉这里的缩放；再除随机标准差、做长度归一化或 clip 后，仍不能据此宣称整个 GRPO 估计无偏。

## 5. KL：方向、采样分布与归一化

固定前缀 $`s`$，设当前分布 $`p(a)=\pi_\theta(a\mid s)`$、参考分布 $`q(a)=\pi_{\mathrm{ref}}(a\mid s)`$，假设共同支持且概率为正：

```math
D_{\mathrm{KL}}(p\|q)=\sum_a p(a)\log\frac{p(a)}{q(a)}.
```

若 $`a\sim p`$，$`k_1=\log(p(a)/q(a))`$ 的期望为该 KL，但单点可为负。令 $`u=q(a)/p(a)`$，则

```math
k_3=u-\log u-1\ge0,\qquad\mathbb E_{a\sim p}[k_3]=D_{\mathrm{KL}}(p\|q).
```

等式来自 $`\mathbb E_p[u]=1`$；非负性来自 $`\log u\le u-1`$。演示用小型离散分布枚举核验。若样本来自不同 old 分布，未经校正的平均不再自动等于当前策略 KL。数值估计无偏也不等于在固定旧样本上直接反向，就获得原期望的完整梯度；还需分析采样分布随参数变化的项。

序列 log 比值是 token log 比值之和；序列 KL 还要对生成的前缀分布取期望。报告须注明整条回答总和、每回答 token 均值还是全批 token 均值。old 比值用于本轮更新，ref KL 用于参考分布约束；二者不能互相替代，KL 也不保证事实正确。

## 6. 奖励设计：先定义任务，再组合分数

RLHF 描述反馈来源，PPO、GRPO 描述优化方法。可验证奖励是否代表真实任务完成，取决于验证器的覆盖范围。

| 来源 | 输入 → 输出 | 主要失真 |
|---|---|---|
| 偏好奖励模型 | 提示词、回答 → 标量 | 标注偏好、长度偏好、分布外误判 |
| 答案或测试验证器 | 回答与标准答案/测试 → 得分 | 解析漏洞、测试覆盖不足、数据泄漏 |
| 格式奖励 | 回答 → 合规分 | 格式正确但任务失败 |
| 过程奖励 | 推理步骤及上下文 → 评分 | 局部合理、整体失败 |

数学任务可先规定独立校验正确率为主指标，再决定格式是解析条件还是附加小分。组合奖励必须记录分量、权重、版本和失败例子；不能只用总分上涨证明能力提高。改变奖励尺度，还要检查它与标准化、KL 系数和优化器的相互作用。

建议记录：独立评估正确率、奖励分量、组内标准差、零优势组比例、长度、截断率、KL 方向与单位、clip 比例、梯度范数和实际更新步数。比较方法还需对齐初始模型、数据划分、生成预算和评估采样方式。这些是待做实验的记录要求，不是本仓库已完成的大模型实验。

## 7. DPO：偏好学习的另一条路线

标准离线 DPO 使用固定偏好对 $`(x,y_w,y_l)`$，训练循环无需在线 rollout 或 critic；偏好数据的收集仍可能需要生成。

固定 $`x`$、固定奖励 $`R`$、$`\beta>0`$，考虑完整回答分布目标：

```math
F(\pi)=\mathbb E_{y\sim\pi}[R(x,y)]-\beta D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}}).
```

参考概率为正、配分函数有限时，定义

```math
Z(x)=\sum_y\pi_{\mathrm{ref}}(y\mid x)e^{R(x,y)/\beta},\qquad
\pi^{\star}(y\mid x)=\pi_{\mathrm{ref}}(y\mid x)e^{R(x,y)/\beta}/Z(x).
```

将 $`\log\pi^{\star}=\log\pi_{\mathrm{ref}}+R/\beta-\log Z`$ 代回，可得 $`F(\pi)=\beta\log Z-\beta D_{\mathrm{KL}}(\pi\|\pi^{\star})`$。因此最优完整分布为 $`\pi^{\star}`$；受限网络和有限数据不保证达到它。

反解 $`R=\beta\log(\pi^{\star}/\pi_{\mathrm{ref}})+\beta\log Z`$。Bradley–Terry 假设偏好概率为 $`\sigma(R_w-R_l)`$，同题的 $`\log Z`$ 在差中消去。用当前策略参数化，得到

```math
\mathcal L_{\mathrm{DPO}}=-\mathbb E_{(x,y_w,y_l)}\log\sigma\left[
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right].
```

$`\sigma`$ 是 logistic 函数。DPO 将偏好似然写成策略概率比，不等于 PPO，也不能据此判断谁在所有任务上更好。[DPO，§4、§5](https://arxiv.org/html/2305.18290v3)

## 8. 代码真正覆盖了什么

| 入口 | 实际覆盖 | 边界 |
|---|---|---|
| `mechanism_demo.py` | clip 斜率、同分组、KL 采样分布、组均值有限样本缩放 | 标准库确定性演示，不训练模型 |
| `toy_rlhf.py::run_ppo_rlhf` | RM、KL 整形、序列 REINFORCE 与批均值基线 | 旧函数名保留；没有 critic、GAE、多轮 PPO-clip，不是完整 PPO |
| `toy_rlhf.py::run_grpo` | 组内优势、token 比值、KL 与 mask | 同一 BOS 无条件生成；每个 rollout 仅更新一次，更新前比值为 1，不能验证 clip 激活后的作用 |
| `toy_rlhf.py::run_dpo` | 固定偏好对 DPO 损失 | 偏好由手写字符奖励产生，不是真实人类数据 |
| `toy_rlhf.py::evaluate` | 手写奖励均值、序列 log 比值均值 | 有限样本 KL 估计可为负，不证明大模型能力 |

旧玩具代码用 PyTorch 默认样本标准差，本章手算用总体标准差，有限组的尺度不同。旧代码注释中的历史命名和公式编号不作为完整性证据，定位以本节函数名为准。

从仓库根目录运行：

```bash
python 10_rlhf_dpo_grpo/mechanism_demo.py
```

演示枚举离散分布，并用有限差分检查斜率，成功末行是 `All mechanism checks passed.`。它不证明模型收敛。字符级实验仍可按原依赖运行 `python 10_rlhf_dpo_grpo/toy_rlhf.py`；本次专题更新未重跑其完整训练。

## 9. 自测与阅读出口

1. old 与 ref 为什么不能混用？——前者定义本批采样分布，后者定义正则参照。
2. 比值为 1 为什么还有梯度？——分母冻结，分子可导。
3. 全错组为何可能更新？——奖励优势为零不等于辅助项梯度为零。
4. GRPO 是否解决步骤级信用分配？——共享终局优势不定位步骤的因果贡献。
5. 非负 KL 项是否保证估计正确？——还须核对采样分布、方向和归一化。
6. 奖励上涨是否代表能力增强？——需要独立评估与失败案例支持。

练习：把混合奖励改成全对组，把 KL 权重从当前分布换成 old，再比较长短回答在两种平均方式下的权重。每次先预测，再运行解释。

## 原文与核验范围

- Schulman et al., *Proximal Policy Optimization Algorithms*（2017），§3、§5：[原文](https://arxiv.org/pdf/1707.06347)。
- Ouyang et al., *Training language models to follow instructions with human feedback*（2022），§3.1、§3.5：[原文](https://arxiv.org/html/2203.02155v1)。
- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*（2024），§4.1：[原文](https://arxiv.org/html/2402.03300v3#S4.SS1)。
- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*（2023），§4、§5：[原文](https://arxiv.org/html/2305.18290v3)。

原文章节核验日期：2026-09-25。有限样本基线与离散 KL 分析是本章自行推导和检查；没有新增性能数字。
