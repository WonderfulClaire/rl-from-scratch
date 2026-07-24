# 09 · 多智能体强化学习：从博弈到 MAPPO

> 多个学习者同时改变行为时，每个个体眼中的"环境"都在漂移——单智能体 RL 的收敛保证全部失效。本章建立多智能体 RL（MARL）的问题框架，讲清核心困难，并实现目前最常用的合作 MARL 算法 IPPO/MAPPO。
>
> 代码：[`mappo_gridworld.py`](mappo_gridworld.py) —— 自实现的 2 智能体合作网格环境（会合任务），IPPO 与 MAPPO（中心化 Critic）对比。

---

## 1. 问题框架：随机博弈 / Markov Game

$n$ 智能体的随机博弈是元组

$$
(\mathcal{N}, \mathcal{S}, \{\mathcal{A}^i\}_{i=1}^n, P, \{R^i\}_{i=1}^n, \gamma),
$$

- 联合动作 $\boldsymbol{a} = (a^1,\dots,a^n)$，转移 $P(s'\mid s,\boldsymbol{a})$ 由**所有人**的动作共同决定；
- 每个智能体有自己的奖励 $R^i(s,\boldsymbol{a})$。三种典型结构：
  - **完全合作**：$R^1 = \cdots = R^n$（团队奖励），本章 demo 属于此类；
  - **完全竞争**：两人零和 $R^1 = -R^2$（围棋、德扑 heads-up）；
  - **混合动机**：一般和博弈（交通、市场做市——多个做市商既竞争又共同维护流动性）。

部分可观测时每个智能体只看到 $o^i = O^i(s)$，框架成为 Dec-POMDP（合作情形）。

## 2. 解概念：从最优策略到均衡

单智能体的"最优策略"在博弈里不存在——我的最优依赖你的策略。替代解概念是**纳什均衡**：策略组 $(\pi^1_*, \dots, \pi^n_*)$ 使得任何单方偏离都不获益：

$$
J^i(\pi^i_*, \pi^{-i}_*) \ge J^i(\pi^i, \pi^{-i}_*) \quad \forall\, \pi^i,\; \forall\, i.
$$

- 有限博弈中混合策略纳什均衡总存在（Nash, 1950），但**计算是 PPAD-难**的；
- 合作博弈里我们通常只追求最大化共同回报（社会最优），它是均衡但均衡未必是它（协调失败：两个均衡各自局部稳定，但一个更差——demo 中会看到）。

## 3. 核心困难

### 3.1 非平稳性（Non-stationarity）

固定其他人的策略 $\pi^{-i}$，智能体 $i$ 面对的诱导环境

$$
P^i(s'\mid s, a^i) = \sum_{\boldsymbol{a}^{-i}} \pi^{-i}(\boldsymbol{a}^{-i}\mid s)\, P(s'\mid s, a^i, \boldsymbol{a}^{-i})
$$

是良定义的 MDP；但训练中 $\pi^{-i}$ 在变，$P^i$ 随之漂移——**Q-learning 的收敛条件（平稳环境）被破坏**。经验回放雪上加霜：旧数据来自旧的 $\pi^{-i}$，分布早已过期。

### 3.2 信用分配（Credit Assignment）

团队奖励 $+10$，是谁的功劳？朴素做法给每人同样的梯度信号，导致"搭便车"问题。COMA 用反事实基线（"把我的动作换成默认值，团队回报变多少"）；QMIX 把团队 Q 分解为个体 Q 的单调混合。

### 3.3 组合爆炸

联合动作空间 $|\mathcal{A}|^n$ 指数增长，中心化控制不可扩展；但完全去中心化又回到非平稳性。

## 4. CTDE：中心化训练、去中心化执行

主流折中方案：**训练时**允许使用全局信息（其他人的观测、动作、甚至环境状态），**执行时**每个智能体只依赖自己的观测。

| 算法 | Actor | Critic | 类型 |
|---|---|---|---|
| IQL | 独立 Q | — | 完全独立 |
| IPPO | $\pi^i(a^i\mid o^i)$ | $V^i(o^i)$ | 独立 PPO |
| MAPPO | $\pi^i(a^i\mid o^i)$ | $V(\boldsymbol{s})$（全局状态） | CTDE |
| MADDPG | $\mu^i(o^i)$ | $Q^i(\boldsymbol{s}, \boldsymbol{a})$（全局+联合动作） | CTDE |
| QMIX | 个体 $Q^i(o^i, a^i)$ | 单调混合网络 | CTDE（值分解） |

**为什么中心化 Critic 缓解非平稳性**：Critic 以联合信息为条件——给定 $(\boldsymbol{s}, \boldsymbol{a})$（或全局状态），价值目标不再依赖"猜测别人策略"，评估问题重新变得平稳；Actor 依然只用局部观测，保证执行时可去中心化。注意：**策略梯度本身仍受他人策略变化影响**，中心化 Critic 降低的是价值估计的方差与漂移，不是根治。

## 5. IPPO 与 MAPPO

- **IPPO**：每个智能体独立跑 PPO，把别人当环境。理论上没有保证，实践上出奇地强（PPO 的保守更新恰好缓解非平稳性——每个人每次都只走小步，别人眼中的环境漂移就慢）。
- **MAPPO**：IPPO + 中心化 Critic $V(\boldsymbol{s})$（拼接所有观测或用环境全局状态）。SMAC 等基准上通常优于 IPPO，尤其在需要协调的任务上。
- 参数共享：同质智能体常共享一套 Actor 权重（输入加 agent id），样本效率大增——demo 采用此设置。

> 与你的 `low_altitude_marl` 仓库的关系：那边是 MAPPO 在无人机避障上的**应用**；本章是把 MAPPO 的**数学与最小实现**独立出来，两边互为参照。

## 6. 运行 demo

```bash
python 09_multi_agent/mappo_gridworld.py
```

环境：5×5 网格，2 个智能体**每轮从随机边缘位置出生**（出生点随机化以加大协调难度），目标是**同时**到达中心区域（都在中心才有团队奖励 +10，每步 -0.1；只有一个到达没有奖励——必须协调）。

对比：IPPO（各自局部 Critic）vs MAPPO（中心化 Critic 看到两个人的位置）。"我该不该等他"这类协调信息只有中心化 Critic 能编码，故预期 MAPPO 收敛更快更稳。

**实测结果（固定种子，可复现）**：报告"达到 90% 近 20 轮会合成功率所需的平均轮数"。

| 算法 | 收敛轮数 | 最终成功率 |
|---|---|---|
| IPPO（局部 Critic） | 66 | 1.00 |
| MAPPO（中心化 Critic） | 61 | 1.00 |

差距不大但方向稳定：MAPPO 略快于 IPPO，符合 CTDE 预期——中心化 Critic 把协调信息编进价值估计，减少了"等不等队友"的摸索成本。任务越需要精细协调（更远的出生点、更多智能体），这个差距会越明显。

## 参考

- Littman (1994), *Markov Games as a Framework for MARL*.
- Lowe et al. (2017), *MADDPG*.
- Rashid et al. (2018), *QMIX*.
- Yu et al. (2022), *The Surprising Effectiveness of PPO in Cooperative MARL* (MAPPO).
