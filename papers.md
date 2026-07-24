# 论文与扩展阅读清单（Papers & Further Reading）

> 本库每一章对应的**必读原始论文**与**关键扩展**。读顺序建议：先啃对应章的 README 推导，再回到这里的原论文看"作者当初怎么想出来的"。带 ★ 的是该方向的奠基性/必读论文。

通用教材与博客（反复参考）：
- ★ Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018) — 全库符号与框架的源头。
- OpenAI, *Spinning Up in Deep RL* (2018) — 策略梯度/PPO 的官方教学实现，与本库互补。
- Lilian Weng, *A (Long) Peek into Reinforcement Learning* (2018) 与 *Policy Gradient Algorithms* (2018) — 极好的中文友好综述。
- Hugging Face, *Deep RL Course* — 带练手 Notebook，适合边读边跑。

---

## 01 · MDP 与 Bellman 方程
- ★ Bellman, R. (1957). *Dynamic Programming*. Princeton University Press. — "最优性 = 局部最优 + 剩余最优"，Bellman 方程的思想源头。
- Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. — MDP 的严格数学处理（压缩映射、收敛性）。

## 02 · 动态规划
- Howard, R. A. (1960). *Dynamic Programming and Markov Processes*. — 策略迭代最早的系统表述。
- ★ Sutton & Barto (2018), Ch. 4 — 策略评估 / 策略改进 / GPI 的最清晰讲解。

## 03 · 蒙特卡洛与时序差分
- ★ Sutton, R. S. (1988). *Learning to predict by the method of temporal differences*. — TD 的开山作。
- Jaakkola, T., Jordan, M., Singh, S. (1994). *Convergence of stochastic iterative dynamic programming algorithms*. — MC/TD 收敛性分析。
- ★ Tesauro, G. (1995). *Temporal Difference Learning and TD-Gammon*. — TD 第一次在真实任务上打到专家级，RL 的里程碑。
- Barto, A., Sutton, R., Watkins, C. (1989/1990). *Learning to predict by TD* 系列与 *Connectionist learning…* — 偏差-方差权衡的直觉来源。

## 04 · SARSA 与 Q-learning
- ★ Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards* (博士论文). — Q-learning 的诞生。
- Rummery, G., Niranjan, M. (1994). *On-line Q-learning using connectionist systems*. — SARSA 原始提出（原名 "Modified Q-learning"）。- ★ van Hasselt, H. (2010). *Double Q-learning*. — 最大化偏差的首次识别与修正。
- van Hasselt, H., Guez, A., Silver, D. (2016). *Deep Reinforcement Learning with Double Q-learning* (Double DQN). — 把 Double Q 思想搬进深度 RL。

## 05 · DQN 家族
- ★ Mnih et al. (2013). *Playing Atari with Deep Reinforcement Learning*; Mnih et al. (2015). *Human-level control through deep RL* (Nature). — DQN 与经验回放/目标网络。
- Wang, Z. et al. (2016). *Dueling Network Architectures for Deep RL*. — 优势/价值分解。
- ★ Schaul, T. et al. (2016). *Prioritized Experience Replay*. — PER 与重要性采样校正。
- Hessel, M. et al. (2018). *Rainbow: Combining Improvements in Deep RL*. — 把六路改进统一，是 DQN 家族的集大成。

## 06 · 策略梯度：REINFORCE 与 A2C
- ★ Williams, R. J. (1992). *Simple statistical gradient-following algorithms for connectionist RL (REINFORCE)*. — 策略梯度的奠基。
- Sutton, R. S. et al. (2000). *Policy Gradient Methods for RL with Function Approximation*. — 策略梯度定理的完整形式化。
- ★ Schulman, J. et al. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE/A2C)*. — 优势估计的标准做法。
- Mnih et al. (2016). *Asynchronous Methods for Deep RL (A3C)*. — A2C 的异步前身。

## 07 · TRPO 与 PPO
- ★ Schulman, J. et al. (2015). *Trust Region Policy Optimization (TRPO)*. — 性能差引理 + 单调改进下界。
- ★ Schulman, J. et al. (2017). *Proximal Policy Optimization (PPO)*. — 用 clip 近似信赖域，工程上最常用；本库第 7 章逐情形分析的来源。
- Kakade, S., Langford, J. (2002). *Approximately Optimal Approximate Reinforcement Learning* — 性能差引理（PDL）的原始形式。

## 08 · 连续控制：DDPG / TD3 / SAC
- ★ Silver, D. et al. (2014). *Deterministic Policy Gradient Algorithms (DPG)*. — 确定性策略梯度的理论基石。
- ★ Lillicrap, T. et al. (2015). *Continuous control with deep reinforcement learning (DDPG)*. — DPG + DQN 技巧。
- ★ Fujimoto, S. et al. (2018). *Addressing Function Approximation Error in Actor-Critic (TD3)*. — 三板斧（Clipped Double-Q / 目标平滑 / 延迟更新）。
- ★ Haarnoja, T. et al. (2018). *Soft Actor-Critic (SAC)*. — 最大熵 RL 与自动温度；连续控制最稳的基线之一。

## 09 · 多智能体 RL（MAPPO）
- ★ Littman, M. L. (1994). *Markov Games as a Framework for Multi-Agent RL*. — 随机博弈框架。
- Lowe, R. et al. (2017). *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)*. — 中心化 Critic 范式。
- Rashid, T. et al. (2018). *QMIX: Monotonic Value Function Factorisation (QMIX)*. — 值分解路线代表作。
- Foerster, J. et al. (2018). *Counterfactual Multi-Agent Policy Gradients (COMA)*. — 反事实信用分配。
- ★ Yu, C. et al. (2022). *The Surprising Effectiveness of PPO in Cooperative MARL (MAPPO)*. — 证明简单 IPPO/MAPPO 在合作任务上极强；本库第 9 章的对照依据。

## 10 · LLM 时代的 RL：RLHF / DPO / GRPO
- ★ Christiano, P. et al. (2017). *Deep RL from Human Preferences*. — RLHF 范式奠基。
- Ziegler, D. et al. (2019). *Fine-Tuning Language Models from Human Preferences*. — 把 RLHF 用到 LM。
- ★ Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback (InstructGPT)*. — RLHF 工程化范式，PPO-RLHF 的心脏。
- ★ Rafailov, R. et al. (2023). *Direct Preference Optimization (DPO)*. — 跳过奖励模型，把 RLHF 变成简单的分类损失；本库闭式解推导的来源。
- ★ Shao, Z. et al. (2024). *DeepSeekMath: GRPO*. — 组内基线替代价值函数的做法，训练更省显存。
- Bradley, R. A., Terry, M. E. (1952). *Rank Analysis of Incomplete Block Designs (Bradley-Terry)*. — 偏好模型的统计源头。

## 11 · RL 与量化交易
- Moody, J., Saffell, M. (2001). *Learning to Trade via Direct Reinforcement*. — 把交易建模成 RL 的早期系统工作。
- ★ Nevmyvaka, Y. et al. (2006). *Reinforcement Learning for Optimized Trade Execution*. — 订单执行 RL 的开山作（Almgren-Chriss 的 RL 化）。
- Ritter, G. (2017). *Machine Learning for Trading (mean-variance RL)*. — 本库风险调整奖励（公式 2）的来源。
- ★ de Prado, M. L. (2018). *Advances in Financial Machine Learning*. — 回测纪律的"圣经"：前视偏差、标签工程、回测陷阱。
- Harris, L. (2003). *Trading and Exchanges* — 市场微观结构常识，理解成本/冲击/非平稳的必读。

---

> 阅读建议：每章 README 的推导是"有人嚼过的"，这里的原论文是"作者原味"。先 README 建立直觉，再论文看动机与实验，最后回到本库代码对照实现——三步闭环，基本能啃下任意一篇。
