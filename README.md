<div align="center">

# RL From Scratch · 强化学习算法全景库

**数学推导 + 从零实现 + 可运行验证 —— 三位一体的强化学习学习库**

*Rigorous math derivations, from-scratch code (numpy / PyTorch only), and honest reproducible experiments — from tabular methods all the way to RLHF and quant trading.*

[![CI](https://github.com/WonderfulClaire/rl-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/WonderfulClaire/rl-from-scratch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Stars](https://img.shields.io/github/stars/WonderfulClaire/rl-from-scratch?style=social)](https://github.com/WonderfulClaire/rl-from-scratch/stargazers)

**简体中文** · [English](README_en.md)

[内容地图](#-内容地图) · [效果一览](#-效果一览real-runs) · [快速开始](#-快速开始) · [学习路线](#-学习路线建议) · [论文清单](papers.md) · [贡献](CONTRIBUTING.md)

</div>

---

## 💡 为什么做这个库？

市面上的强化学习资料通常走两个极端：

> **要么只讲直觉不给推导** —— "PPO 就是裁剪一下"，看完公式从哪来的还是一头雾水；
>
> **要么只给公式不给能跑的代码** —— 伪代码写得很漂亮，一实现就踩坑，学习曲线根本对不上。

这个库把三件事钉在一起：**每一行代码都能对应到推导中的某个公式，每一个公式都能在实验里看到真实效果。**

11 章，从表格方法走到 RLHF 与量化交易，一条线走完现代强化学习的完整主干。

---

## ✨ 核心特色

### 🔢 数学推导完整可追溯
每个算法从目标函数出发，一步步推导到更新公式，标注每一步近似发生在哪里、为什么成立。拒绝黑盒。

### 🛠️ 全部从零实现，不依赖黑盒库
表格方法用纯 NumPy 手写，深度 RL 仅用 PyTorch 基础算子——**不封装、不继承、不用 Stable-Baselines3 之类的黑盒**。每一行代码都能在推导里找到对应位置。

### 🧪 真实可复现的实验验证
每个实现都在标准环境（Gymnasium）上跑通，学习曲线全部来自真实运行，固定种子可复现。**失败案例也如实呈现**（如第 11 章 DQN 在测试段跑输买入持有，正是回测陷阱的活体演示）。

### 📚 体系化的 11 章内容
从 MDP 基础到 RLHF 对齐，从单智能体到多智能体，从离散控制到量化交易应用——不是零散算法堆砌，是一条完整的学习路径。

---

## 🗺️ 内容地图

### 第一部分：表格型方法（NumPy 实现）

| 章节 | 主题 | 核心数学 | 环境 |
|:---:|---|---|---|
| [01](01_mdp_bellman/) | MDP 与 Bellman 方程 | 马尔可夫决策过程、贝尔曼期望/最优方程、压缩映射与不动点 | GridWorld（自实现） |
| [02](02_dynamic_programming/) | 动态规划 | 策略评估、策略迭代、值迭代及其收敛性证明 | GridWorld |
| [03](03_monte_carlo_td/) | 蒙特卡洛与时序差分 | 首次访问 MC、TD(0)、偏差-方差权衡、n-step / TD(λ) | Random Walk |
| [04](04_sarsa_qlearning/) | SARSA 与 Q-learning | on-policy vs off-policy、Q-learning 收敛条件、最大化偏差与 Double Q | CliffWalking（自实现） |

### 第二部分：深度价值方法（PyTorch 实现）

| 章节 | 主题 | 核心数学 | 环境 |
|:---:|---|---|---|
| [05](05_dqn_family/) | DQN 家族 | 致命三要素、目标网络、Double DQN 过估计分析、Dueling 分解、优先经验回放的重要性采样 | CartPole-v1 |

### 第三部分：策略优化（PyTorch 实现）

| 章节 | 主题 | 核心数学 | 环境 |
|:---:|---|---|---|
| [06](06_policy_gradient/) | 策略梯度：REINFORCE 与 A2C | 策略梯度定理完整证明、基线不改变期望的证明、GAE | CartPole-v1 |
| [07](07_trpo_ppo/) | TRPO 与 PPO | 性能差引理、单调改进下界、信赖域与自然梯度、PPO-clip 的下界性质 | CartPole-v1 |
| [08](08_continuous_control/) | 连续控制：DDPG / TD3 / SAC | 确定性策略梯度定理、TD3 三板斧、最大熵 RL 与软贝尔曼方程、重参数化技巧 | Pendulum-v1 |

### 第四部分：前沿与应用

| 章节 | 主题 | 核心数学 | 环境 |
|:---:|---|---|---|
| [09](09_multi_agent/) | 多智能体 RL | 随机博弈、非平稳性问题、CTDE 范式、IPPO / MAPPO | 合作矩阵博弈 + 多智能体网格 |
| [10](10_rlhf_dpo_grpo/) | LLM 时代的 RL：RLHF / DPO / GRPO | Bradley-Terry 模型、KL 约束下最优策略闭式解、DPO 完整推导、GRPO 组内基线 | 玩具语言模型（字符级） |
| [11](11_rl_for_trading/) | RL 与量化交易 | 交易 MDP 建模、仓位管理、含交易成本的奖励设计、回测陷阱 | 自实现交易环境（合成风格数据） |

---

## 📊 效果一览（real runs）

> 以下曲线均来自本仓库代码真实运行，由 [`tools/benchmark.py`](tools/benchmark.py) 生成，**3 个随机种子的均值 ± 标准差**，横轴为真实环境步数。完整数字见 [results/RESULTS.md](results/RESULTS.md)。

### CartPole-v1：四种主流算法同台对比

![CartPole learning curves](results/cartpole_curves.png)

*REINFORCE / A2C / PPO / DQN 在同一任务上的样本效率对比——直观展示"为什么 PPO 是默认选择"。*

### Pendulum-v1：三种连续控制算法

![Pendulum learning curves](results/pendulum_curves.png)

*DDPG / TD3 / SAC 的收敛速度与稳定性对比——TD3 与 SAC 明显比 DDPG 更稳。*

复现方式：

```bash
python tools/benchmark.py --suite cartpole --seeds 3
python tools/benchmark.py --suite pendulum --seeds 3
```

---

## 🚀 快速开始

### 环境要求
- Python 3.10+
- PyTorch 2.0+（表格章节仅需 NumPy）

### 三步跑通第一个算法

```bash
# 1. 克隆仓库
git clone https://github.com/WonderfulClaire/rl-from-scratch.git
cd rl-from-scratch

# 2. 安装依赖
pip install -r requirements.txt

# 3. 跑一个 demo（每章都能独立运行）
python 05_dqn_family/dqn.py --double --dueling   # DQN 家族（开关可叠加 --per）
```

其他 demo：

```bash
python 02_dynamic_programming/dp_solvers.py    # 策略迭代 vs 值迭代
python 07_trpo_ppo/ppo.py                      # PPO-clip
python 11_rl_for_trading/dqn_trader.py         # RL 交易 + 回测陷阱

python tests/smoke_test.py                     # 秒级冒烟测试（表格章节）
```

所有 demo 均在 **CPU** 上验证过（分钟级跑完），不需要 GPU。

---

## 📁 项目结构

```
rl-from-scratch/
├── 01_mdp_bellman/          # MDP 与 Bellman 方程
├── 02_dynamic_programming/  # 动态规划
├── 03_monte_carlo_td/       # 蒙特卡洛与时序差分
├── 04_sarsa_qlearning/      # SARSA 与 Q-Learning
├── 05_dqn_family/           # DQN 家族（Double / Dueling / PER）
├── 06_policy_gradient/      # 策略梯度（REINFORCE / A2C）
├── 07_trpo_ppo/             # TRPO 与 PPO
├── 08_continuous_control/   # 连续控制（DDPG / TD3 / SAC）
├── 09_multi_agent/          # 多智能体 RL（IPPO / MAPPO）
├── 10_rlhf_dpo_grpo/        # RLHF / DPO / GRPO
├── 11_rl_for_trading/       # RL 与量化交易
├── results/                 # 真实实验结果与曲线图
├── tests/                   # 冒烟测试
├── tools/                   # 基准测试工具
├── papers.md                # 逐章论文精读清单
├── notation.md              # 统一符号约定表
├── requirements.txt         # 依赖清单
├── CONTRIBUTING.md          # 贡献指南
└── README.md                # 你正在看的文件
```

---

## 🛤️ 学习路线建议

- **零基础入门**：01 → 02 → 03 → 04 → 05 → 06 → 07，走完就掌握了现代 RL 的主干。
- **面向 LLM / RLHF 岗位**：吃透 06、07 的推导（PPO 是 RLHF 的心脏），然后精读 10。
- **面向量化交易**：01-05 打基础，然后精读 11，重点关注奖励设计与回测陷阱一节。
- **只想查公式**：每章 README 的推导都是自包含的，可独立阅读。

先扫一眼 [notation.md](notation.md) 的统一符号表，再配合 [papers.md](papers.md)——"README 推导 → 原论文动机 → 本库代码"三步闭环。

---

## 🆚 为什么选这个库？

| 特性 | 本仓库 | Stable-Baselines3 | Spinning Up | easy-rl |
|------|:------:|:-----------------:|:-----------:|:-------:|
| 完整数学推导 | ✅ | ❌ | ⚠️ 部分 | ⚠️ 部分 |
| 从零可读实现 | ✅ | ❌ 高度封装 | ✅ | ✅ |
| 可复现实验结果 | ✅ | ✅ | ✅ | ⚠️ |
| 覆盖 RLHF / DPO / GRPO | ✅ | ❌ | ❌ | ❌ |
| 覆盖量化交易应用 | ✅ | ❌ | ❌ | ❌ |
| 中文推导与注释 | ✅ | ❌ | ❌ | ✅ |

差异化定位：**RLHF/量化应用 + 中文完整推导** 的组合，在现有 RL 学习资源里少见。

---

## 🎯 设计原则

1. **代码只为教学服务**：单文件、无框架、注释对应公式编号；不追求工程上的最优性能。
2. **不跳步**：推导中的每个等号要么显然，要么给出理由；近似的地方明确标出"这里是近似"。
3. **诚实的实验**：所有学习曲线来自真实运行，随机种子固定，可复现；失败案例也如实呈现。

---

## 🤝 贡献

欢迎修 bug、补推导、加算法、报告复现结果。请先读 [CONTRIBUTING.md](CONTRIBUTING.md)，提交 PR 即可。

<a href="https://github.com/WonderfulClaire/rl-from-scratch/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=WonderfulClaire/rl-from-scratch" />
</a>

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=WonderfulClaire/rl-from-scratch&type=Date)](https://star-history.com/#WonderfulClaire/rl-from-scratch&Date)

---

## 📄 License

[MIT](LICENSE) © WonderfulClaire

<div align="center">

**如果这个库对你有帮助，点个 ⭐ Star 支持一下吧！**

</div>
