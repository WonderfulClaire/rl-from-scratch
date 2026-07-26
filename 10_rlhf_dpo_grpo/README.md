# 10 · LLM 时代的 RL：RLHF、DPO 与 GRPO

> ChatGPT 的对齐、DeepSeek-R1 的推理训练，核心都是 RL。本章把语言模型对齐问题严格形式化为 RL 问题，完整推导 RLHF-PPO、DPO（含闭式解的每一步）与 GRPO，并在一个字符级玩具语言模型上把三者跑出来——麻雀虽小，五脏俱全。
>
> 代码：[`toy_rlhf.py`](toy_rlhf.py) —— 玩具 LM 上的 PPO-RLHF / DPO / GRPO 三算法对照实验。

---

## 1. 语言生成作为 MDP

| RL 概念 | LLM 对应物 |
|---|---|
| 状态 $s_t$ | prompt $x$ + 已生成前缀 $y_{<t}$ |
| 动作 $a_t$ | 下一个 token $y_t$（动作空间 = 词表，$10^4$~$10^5$） |
| 策略 $\pi_\theta$ | 语言模型本身 |
| 转移 | 确定性：状态追加一个 token |
| 奖励 | 稀疏：整条回答生成完才有 $r(x,y)$ |
| 回合 | 一次完整生成（到 EOS） |

特殊性：转移确定 + 奖励只在终点 ⟹ 这是一个"树搜索"型 MDP；且我们有一个强大的初始策略（预训练+SFT 模型），RL 只做**微调**而非从零学。

## 2. RLHF 三部曲（InstructGPT 配方）

### 2.1 奖励模型：Bradley-Terry

人类不擅长打绝对分，擅长**二选一**。对偏好对 $(x, y_w \succ y_l)$，Bradley-Terry 模型假设

$$
\Pr(y_w \succ y_l \mid x) = \frac{e^{r(x,y_w)}}{e^{r(x,y_w)} + e^{r(x,y_l)}} = \sigma\big(r(x,y_w) - r(x,y_l)\big). 
$$

用最大似然训练奖励模型 $r_\psi$：

$$
\mathcal{L}_{\text{RM}}(\psi) = -\mathbb{E}_{(x,y_w,y_l)}\Big[\log\sigma\big(r_\psi(x,y_w) - r_\psi(x,y_l)\big)\Big]. 
$$

注意 (1) 只依赖**奖励差**——BT 模型对 $r \to r + c(x)$ 不变，这个规范自由度后面 DPO 会用到。

### 2.2 KL 约束的 RL 目标

直接最大化 $r_\psi$ 会**reward hacking**：策略钻奖励模型的漏洞（重复、空话、格式攻击），生成分布跑出 $r_\psi$ 的训练分布后其打分毫无意义。因此加 KL 锚：

$$
\max_{\pi_\theta}\; \mathbb{E}_{x\sim\mathcal{D},\, y\sim\pi_\theta(\cdot\mid x)}\Big[r_\psi(x,y)\Big] - \beta\, D_{\mathrm{KL}}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big). 
$$

### 2.3 用 PPO 优化 (3)

把序列级 KL 摊到每个 token 上作为奖励整形：$r_t = -\beta\log\frac{\pi_\theta(y_t\mid \cdot)}{\pi_{\text{ref}}(y_t\mid \cdot)}$，末 token 加 $r_\psi(x,y)$，然后就是标准 PPO（第 07 章）+ 价值网络 + GAE。工程上需要同时在显存里放 4 个模型（策略/参考/奖励/价值），这是 RLHF 出了名难伺候的原因。

## 3. DPO：把 RL 问题变成分类问题

### 3.1 关键洞察：(3) 有闭式解

**定理**：目标 (3) 的最优策略为

$$
\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\, e^{r(x,y)/\beta}, \qquad Z(x) = \sum_y \pi_{\text{ref}}(y\mid x)\, e^{r(x,y)/\beta}. 
$$

**证明**：把 (3) 对固定 $x$ 展开并除以 $\beta$：

$$
\max_\pi \mathbb{E}_{y\sim\pi}\Big[\frac{r(x,y)}{\beta} - \log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\Big]
= \min_\pi\; \mathbb{E}_{y\sim\pi}\Big[\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)\,e^{r(x,y)/\beta}}\Big].
$$

右边配上归一化常数 $Z(x)$ 后正是 $D_{\mathrm{KL}}\big(\pi \,\|\, \pi^*\big) - \log Z(x)$，KL ≥ 0 且当且仅当 $\pi = \pi^*$ 时取 0。∎

（这与第 08 章 SAC 的"最优策略 ∝ exp(Q/α)"是同一个数学——软最优的通用形态。）

### 3.2 反解奖励，代入 BT

(4) 两边取 log 反解出 $r$：

$$
r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x). 
$$

代入 BT 模型 (1)——**配分函数 $Z(x)$ 在奖励差中相消**（就是 2.1 说的规范自由度）：

$$
\Pr(y_w \succ y_l\mid x) = \sigma\Big(\beta\log\frac{\pi^*(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi^*(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big).
$$

于是**用策略自己参数化奖励**，直接对偏好数据做最大似然：

$$
\boxed{\;\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\Big[\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\tfrac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)\Big]\;} 
$$

**你的语言模型秘密地是个奖励模型**（DPO 论文标题的含义）。不需要奖励模型、不需要采样、不需要价值网络——RL 问题变成了一个 logistic 回归。

### 3.3 梯度的语义

$$
\nabla_\theta\mathcal{L}_{\text{DPO}} = -\beta\,\mathbb{E}\Big[\underbrace{\sigma(\hat r_l - \hat r_w)}_{\text{奖励差估错得越多权重越大}}\big[\nabla\log\pi_\theta(y_w\mid x) - \nabla\log\pi_\theta(y_l\mid x)\big]\Big],
$$

其中 $\hat r = \beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ 是隐式奖励。提升 $y_w$、压低 $y_l$，且**难样本权重大**——形式上像带自适应权重的对比学习。

**DPO vs PPO-RLHF**：DPO 简单、稳定、离线；代价是只能学偏好数据覆盖到的分布（无探索）、隐式奖励在分布外无意义、易过拟合偏好噪声。业界现状：对齐类任务 DPO 家族流行，推理类任务（数学/代码，有可验证奖励）on-policy RL（PPO/GRPO）效果更好。

## 4. GRPO：干掉价值网络

（DeepSeekMath 提出，DeepSeek-R1 用它训练出推理能力。）

PPO 需要价值网络提供基线（第 06 章公式 (4)）。GRPO 的观察：LLM 场景下**对同一个 prompt 采多条回答很便宜**——那基线不用学，直接用**组内均值**：

对 prompt $x$ 采 $G$ 条回答 $\{y_i\}$，各得奖励 $\{r_i\}$，组内标准化优势：

$$
\hat A_i = \frac{r_i - \text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}. 
$$

目标（PPO-clip 的形式 + 显式 KL 正则，序列内所有 token 共享 $\hat A_i$）：

$$
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\Big[\frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|}\sum_t \min\big(\rho_{i,t}\hat A_i,\; \text{clip}(\rho_{i,t}, 1\pm\epsilon)\hat A_i\big) - \beta\, D_{\mathrm{KL}}\big[\pi_\theta \| \pi_{\text{ref}}\big]\Big]. 
$$

**为什么合理**：组内均值正是 $V^{\pi}(x)$ 的蒙特卡洛估计（同一状态采多动作），(7) 就是第 06 章"基线不引入偏差"的直接应用（严格说除以 std 引入轻微偏差，实践无碍）。**收益**：少一个价值网络（显存减半）、没有价值函数学不准的问题。**适用前提**：能对同一 prompt 廉价多次采样 + 奖励可自动计算（数学答案对错、代码过不过测试）——这正是推理训练的设定。

### RLHF 家族速查

| | 奖励来源 | 需要 RM | 需要价值网络 | on/off-policy | 代表 |
|---|---|---|---|---|---|
| PPO-RLHF | 学习的 RM | ✓ | ✓ | on | InstructGPT |
| DPO | 隐式（策略即奖励） | ✗ | ✗ | off | Zephyr, Llama-3 对齐 |
| GRPO | 可验证奖励 / RM | 可选 | ✗ | on | DeepSeek-R1 |

## 5. 玩具实验设计

`toy_rlhf.py` 构造一个完全可控的小世界：

- **词表**：`a-e` 5 个字符 + EOS；**"LM"**：2 层小 Transformer 不必要——用一个 GRU 即可，生成长度 ≤ 8 的串；
- **真实奖励**（上帝视角，仅用于造偏好数据和评估）：串中 `a` 越多越好、相邻重复扣分——模拟"人类偏好"；
- **SFT/参考模型**：在均匀随机串上训练的基线 LM；
- 三条路线从同一参考模型出发：
  1. **PPO-RLHF**：先用偏好对训 RM（公式 2），再 PPO 优化（公式 3）；
  2. **DPO**：直接在偏好对上最小化 (6)；
  3. **GRPO**：假设奖励可验证（直接用真实奖励），组内基线 + clip（公式 7-8）。
- **评估**：各自训练后采样 500 条，比较真实奖励均值与对参考模型的 KL——验证三者都能提升奖励，并观察 KL 代价差异。

## 参考

- Ouyang et al. (2022), *Training LMs to follow instructions with human feedback*.
- Rafailov et al. (2023), *Direct Preference Optimization*.
- Shao et al. (2024), *DeepSeekMath* (GRPO); DeepSeek-AI (2025), *DeepSeek-R1*.
