"""
smoke_deep.py — 深度 RL 章节快跑冒烟测试

每个深度章只用极小步数调用 train()，证明代码能端到端跑通
（不要求收敛）。用于 CI 里确认 PPO/SAC/MAPPO/DQN/RLHF/交易
等章节没被改坏。任一算法崩溃即以非零退出。

运行（在仓库根目录）:
    python tests/smoke_deep.py
"""
import importlib.util
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(rel):
    path = os.path.join(REPO, rel)
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    # 让同目录的其他模块（如 trading_env）可被 import
    d = os.path.dirname(path)
    if d not in sys.path:
        sys.path.insert(0, d)
    spec.loader.exec_module(mod)
    return mod


def run(name, fn):
    t0 = time.time()
    try:
        fn()
        dt = time.time() - t0
        print(f"[PASS] {name}  ({dt:.1f}s)")
        return True
    except Exception as e:  # noqa
        dt = time.time() - t0
        print(f"[FAIL] {name}  ({dt:.1f}s): {e}")
        traceback.print_exc()
        return False


def test_dqn():
    m = _load("05_dqn_family/dqn.py")
    m.train(use_double=True, use_dueling=True, use_per=False,
            episodes=3, seed=0, quiet=True)


def test_reinforce():
    m = _load("06_policy_gradient/reinforce.py")
    m.train(episodes=3, seed=0)


def test_a2c():
    m = _load("06_policy_gradient/a2c.py")
    m.train(total_steps=2000, rollout_len=128, seed=0)


def test_ppo():
    m = _load("07_trpo_ppo/ppo.py")
    m.train(total_steps=4000, rollout_len=256, epochs=3, minibatch=64, seed=0)


def test_ddpg():
    m = _load("08_continuous_control/ddpg.py")
    m.train(total_steps=1500, start_steps=200, batch=64, seed=0)


def test_sac():
    m = _load("08_continuous_control/sac.py")
    m.train(total_steps=1500, start_steps=200, batch=64, seed=0)


def test_td3():
    m = _load("08_continuous_control/td3.py")
    m.train(total_steps=1500, start_steps=200, batch=64, seed=0)


def test_mappo():
    m = _load("09_multi_agent/mappo_gridworld.py")
    m.train(centralized=True, iters=3, episodes_per_iter=2, seed=0, quiet=True)
    m.train(centralized=False, iters=3, episodes_per_iter=2, seed=0, quiet=True)


def test_rlhf():
    m = _load("10_rlhf_dpo_grpo/toy_rlhf.py")
    m.main()


def test_trader():
    m = _load("11_rl_for_trading/dqn_trader.py")
    m.main()


def main():
    tests = [
        ("05 DQN (+Double+Dueling)", test_dqn),
        ("06 REINFORCE", test_reinforce),
        ("06 A2C", test_a2c),
        ("07 PPO", test_ppo),
        ("08 DDPG", test_ddpg),
        ("08 SAC", test_sac),
        ("08 TD3", test_td3),
        ("09 MAPPO/IPPO", test_mappo),
        ("10 Toy-RLHF (PPO/DPO/GRPO)", test_rlhf),
        ("11 DQN-Trader", test_trader),
    ]
    failed = 0
    for name, fn in tests:
        if not run(name, fn):
            failed += 1
    if failed:
        print(f"\n{failed}/{len(tests)} deep smoke tests FAILED")
        sys.exit(1)
    print(f"\nAll {len(tests)} deep smoke tests passed ✓")


if __name__ == "__main__":
    main()
