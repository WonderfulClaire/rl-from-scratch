# RL From Scratch：强化学习算法全景库

> 数学推导 + 从零实现 + 可运行验证，三位一体的强化学习学习库。
>
> 每一章 = 一份**完整数学推导**（README.md）+ 一份**从零实现的代码**（不依赖 RL 框架，只用 numpy / PyTorch）+ 一个**可直接运行的 demo**（附实际运行结果）。

## 这个库是什么

市面上的 RL 资料通常有两个极端：要么只讲直觉不给推导（"PPO 就是裁剪一下"），要么只给公式不给能跑的代码。这个库的目标是把三件事钉在一起：

1. **数学**：每个算法的目标函数从哪来、每一步推导为什么成立、近似发生在哪里。
2. **代码**：每行代码能对应到推导中的某个公式，全部从零实现，不用 Stable-Baselines3 之类的黑盒。
3. **验证**：每个实现都在标准环境上跑通并给出学习曲线/收敛结果，杜绝"看起来对但跑不起来"的伪代码。

## 内容地图

### 第一部分：表格型方法（numpy 实现）

| 章节 | 主题 | 核心数学 | 环境 |
|---|---|---|---|
| [01](01_mdp_bellman/) | MDP 与 Bellman 方程 | 马尔可夫决策过程、贝尔曼期望/最优方程、压缩映射与不动点 | GridWorld（自实现） |
| [02](02_dynamic_programming/) | 动态规划 | 策略评估、策略迭代、值迭代及其收敛性证明 | GridWorld |
| [03](03_monte_carlo_td/) | 蒙特卡洛与时序差分 | 首次访问 MC、TD(0)、偏差-方差权衡、n-step / TD(λ) | Random Walk |
| [04](04_sarsa_qlearning/) | SARSA 与 Q-learning | on-policy vs off-policy、Q-learning 收敛条件、最大化偏差与 Double Q | CliffWalking（自实现） |

### 第二部分：深度价值方法（PyTorch 实现）

| 章节 | 主题 | 核心数学 | 环境 |
|---|---|---|---|
| [05](05_dqn_family/) | DQN 家族 | 致命三要素、目标网络、Double DQN 的过估计分析、Dueling 分解、优先经验回放的重要性采样 | CartPole-v1 |

### 第三部分：策略优化（PyTorch 实现）

| 章节 | 主题 | 核心数学 | 环境 |
|---|---|---|---|
| [06](06_policy_gradient/) | 策略梯度：REINFORCE 与 A2C | 策略梯度定理完整证明、基线不改变期望的证明、GAE | CartPole-v1 |
| [07](07_trpo_ppo/) | TRPO 与 PPO | 性能差引理、单调改进下界、信赖域与自然梯度、PPO-clip 的下界性质 | CartPole-v1 |
| [08](08_continuous_control/) | 连续控制：DDPG / TD3 / SAC | 确定性策略梯度定理、TD3 三板斧、最大熵 RL 与软贝尔曼方程、重参数化技巧 | Pendulum-v1 |

### 第四部分：前沿与应用

| 章节 | 主题 | 核心数学 | 环境 |
|---|---|---|---|
| [09](09_multi_agent/) | 多智能体 RL | 随机博弈、非平稳性问题、CTDE 范式、IPPO/MAPPO | 合作矩阵博弈 + 多智能体网格 |
| [10](10_rlhf_dpo_grpo/) | LLM 时代的 RL：RLHF / DPO / GRPO | Bradley-Terry 模型、KL 约束下的最优策略闭式解、DPO 完整推导、GRPO 的组内基线 | 玩具语言模型（字符级） |
| [11](11_rl_for_trading/) | RL 与量化交易 | 交易 MDP 建模、仓位管理、含交易成本的奖励设计、回测陷阱 | 自实现交易环境（合成+真实风格数据） |

## 使用方式

```bash
# 建议 Python >= 3.10
pip install -r requirements.txt

# 每一章的 demo 都可以直接运行，例如：
python 02_dynamic_programming/dp_solvers.py
python 05_dqn_family/dqn.py
python 07_trpo_ppo/ppo.py
python 11_rl_for_trading/dqn_trader.py
```

所有 demo 均在 CPU 上验证过（分钟级即可跑完），不需要 GPU。

## 统一符号表

全库使用统一的数学符号，见 [notation.md](notation.md)。读任何一章之前建议先扫一眼。

## 学习路线建议

- **零基础入门**：01 → 02 → 03 → 04 → 05 → 06 → 07，这条线走完就掌握了现代 RL 的主干。
- **面向 LLM / RLHF 岗位**：确保 06、07 的推导完全吃透（PPO 是 RLHF 的心脏），然后精读 10。
- **面向量化交易**：01-05 打基础，然后精读 11，重点关注奖励设计与回测陷阱一节。
- **只想查公式**：每章 README 的推导部分都是自包含的，可独立阅读。

## 设计原则

1. **代码只为教学服务**：单文件、无框架、注释对应公式编号；不追求工程上的最优性能。
2. **不跳步**：推导中的每个等号要么显然，要么给出理由。近似的地方明确标出"这里是近似"。
3. **诚实的实验**：所有学习曲线来自真实运行，随机种子固定，可复现。

## License

MIT
