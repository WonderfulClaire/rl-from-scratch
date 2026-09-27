# 12 · Agentic Post-Training：环境、Verifier、Reward Hacking 与 Harness Generalization

[返回全库](../README.md) · [第 10 章：PPO / GRPO / DPO](../10_rlhf_dpo_grpo/) · [可运行演示](reward_hacking_demo.py)

第 10 章回答的是：拿到 rollout 和 reward 以后，PPO / GRPO 怎么更新语言模型？

这一章往前追一步：

> reward 到底从哪里来？模型学到的是任务，还是 verifier / harness 的漏洞？

对 tool-using agent 来说，后训练不只是“把最终回答打一个分”。模型会读文件、调用工具、改变环境状态，真正的训练对象是一条交互轨迹：

~~~text
prompt
  ↓
policy
  ↓
tool call
  ↓
environment transition
  ↓
observation
  ↓
...
  ↓
final environment state
  ↓
verifier
  ↓
reward
  ↓
policy update
~~~

如果环境、工具协议或 verifier 设计错了，优化算法越强，模型越可能把漏洞学得越熟。

## 1. 从 token MDP 到 agent MDP

第 10 章单轮生成可以写成：

- state：prompt + 已生成 token
- action：下一个 token
- transition：追加 token
- reward：最终答案评分

工具型 Agent 多了一层真正的外部状态：

| 对象 | 单轮语言模型 | Tool-using Agent |
| --- | --- | --- |
| state | prompt + token prefix | 对话历史 + 环境状态 + tool observation |
| action | next token | text token / structured tool call |
| transition | append token | tool execution + environment mutation |
| reward | answer score | final state / verifier / cost / safety |
| horizon | 一次回答 | 多轮、可能跨几十次工具调用 |

所以 Agentic RL 最先要确认的不是 PPO 还是 GRPO，而是：

1. 哪些状态模型能看到？
2. 哪些状态只给 verifier 看？
3. reward 是否能被 policy 直接篡改？
4. 训练 harness 和测试 harness 是否完全一样？
5. 成功是“模型说成功”，还是环境最终状态真的成功？

## 2. Reward Hacking：优化目标和真实目标不一致

reward_hacking_demo.py 里有一个最小 coding 环境。任务是把实现修成：

f(x) = x^2 + 2x + 1

visible public tests 只测 x=0,1,2。

| Candidate | Naive public reward | Secure reward | 为什么 |
| --- | ---: | ---: | --- |
| 正确修复 | 1 | 1 | public / hidden / integrity 都通过 |
| 删除 public tests | 1 | 0 | 空测试集让 naive checker 误判 |
| hard-code public cases | 1 | 0 | public 通过，hidden 失败 |
| 什么都不做 | 0 | 0 | 正常失败 |

最关键的一行是：

~~~text
observed reward = 1  !=  actual task success
~~~

如果直接拿 naive signal 做 RL，模型有动力学习“删测试”或“hard-code”，因为优化器只看 reward，不知道你心里真正想要什么。

## 3. Secure Verifier 应该和 Policy 隔离

~~~text
policy-visible environment
  ├── source files
  ├── public tests
  └── tools
            ↓
      agent trajectory
            ↓
final state
  ├── original public tests
  ├── hidden tests
  └── integrity check
            ↓
       secure reward
~~~

原则：

- hidden verifier 不出现在 prompt；
- policy 不能修改 verifier 自己的代码与答案；
- final-state check 优先于“模型自报完成”；
- reward component 分开记录，不只存一个 total；
- exploit trajectory 也要保存，方便修 verifier。

## 4. GRPO 不会自动修复 Reward 问题

GRPO 只把同一题的一组 rollout reward 变成相对优势。

如果一组 reward 是 [1,0,0,0]，标准化后成功 rollout 得正优势，失败 rollout 得负优势。

但如果那个 1 来自 reward hacking，那么 GRPO 会很认真地提高作弊轨迹的概率。

> 优化算法负责“更好地追 reward”，不负责保证 reward 等于真实目标。

运行：

~~~bash
python 12_agentic_post_training/reward_hacking_demo.py
~~~

脚本会打印 naive reward、secure reward 和 secure-reward GRPO group advantage。

## 5. Harness 也是训练分布的一部分

Agent 行为不只是模型参数：

~~~text
Agent behavior
=
Model
+ System Prompt
+ Tool Schema
+ Context Policy
+ Environment
~~~

同一个语义工具可以叫三套不同名字：

- read_file / write_file / run_public_tests
- read / write / test
- inspect_file / update_file / check_visible_tests

一个最小 generalization 实验：

~~~text
train / rollout:
    canonical + compact

held-out evaluation:
    alternate
~~~

同时保持 task、verifier、max steps、生成参数不变。

如果 canonical 很高、alternate 明显下降，应报告为 harness sensitivity，不能只说“agent success 很高”。

## 6. History / Context Policy 也要做 Ablation

合理的 ablation 只改一个变量：

~~~text
A: full assistant history
B: remove reasoning-like fields
C: compact old history
~~~

并记录 success、invalid tool calls、steps、tokens 和 compaction rate。

否则“模型变差”可能只是协议被破坏。

## 7. 从 Successful Trajectory 到 SFT，再到 RL

~~~text
rollout
  ↓
secure verifier
  ↓
successful trajectories
  ↓
SFT
  ↓
new rollout
  ↓
verifiable reward
  ↓
GRPO / PPO
  ↓
held-out evaluation
~~~

最少应该有三组：

| Model | In-harness success | Held-out harness | Reward hacking rate |
| --- | ---: | ---: | ---: |
| Base | TBD | TBD | TBD |
| SFT | TBD | TBD | TBD |
| SFT + RL | TBD | TBD | TBD |

不要只汇报 training reward。

## 8. 和第 10 章怎么连起来

第 10 章负责：

~~~text
reward → advantage → probability ratio → PPO / GRPO → parameter update
~~~

第 12 章负责：

~~~text
task → harness + environment → trajectory → verifier → reward
~~~

两章合起来才是完整 Agentic RL。

## 9. 工程版在哪里

本章坚持“从零 + 最小机制”定位，不做完整训练框架。

更工程化的可运行版本：

- [Kimi K3 Deep Dive & Agentic Post-Training Lab](https://github.com/WonderfulClaire/kimi-k3-deep-dive)
- [5G Diagnostic Agent](https://github.com/WonderfulClaire/5G-Diagnostic-Agent)

前者实现统一 harness、trajectory/replay、secure verifier、history/harness ablation、verified trajectory → SFT、agentic GRPO；后者是更完整的多轮诊断任务与训练闭环。

## 10. 自测

1. 为什么 public tests 全过仍不能直接当真实成功？
2. hidden verifier 为什么不能出现在 policy prompt 中？
3. GRPO 能否自动识别 reward hacking？为什么？
4. tool name 改名后性能下降，说明什么？
5. SFT + RL 对照为什么还需要 held-out harness？
6. 为什么 training reward 上升不能单独证明 agent 能力提升？

一句话总结：

> 先确保 reward 值得优化，再讨论怎么把它优化得更快。
