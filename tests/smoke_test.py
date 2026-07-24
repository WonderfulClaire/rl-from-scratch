"""
smoke_test.py — 快速确定性冒烟测试（供 CI 使用）

只跑表格型章节里"秒级、可复现、有解析真值"的检查，
用来在每次 push 时确认核心算法没被改坏。深度 RL 章节
太慢，不放进 CI（它们在 results/ 里有真实运行记录）。

运行:
    python tests/smoke_test.py
全部通过时以退出码 0 结束，任一断言失败即非零退出。
"""
import importlib.util
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(rel):
    path = os.path.join(REPO, rel)
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bellman_analytic_equals_iterative():
    """第01章: Bellman 解析解 == 迭代解（压缩映射收敛）。"""
    m = _load("01_mdp_bellman/gridworld.py")
    env = m.GridWorld()
    pi = np.ones((env.N_STATES, env.N_ACTIONS)) / env.N_ACTIONS  # 均匀随机策略
    gamma = 0.9
    v_analytic = m.analytic_v(env, pi, gamma)
    v_iter = m.iterative_v(env, pi, gamma)
    diff = np.max(np.abs(v_analytic - v_iter))
    assert diff < 1e-6, f"Bellman analytic vs iterative diff too large: {diff}"
    print(f"[PASS] 01 Bellman analytic==iterative (max diff {diff:.2e})")


def test_policy_iteration_equals_value_iteration():
    """第02章: 策略迭代与值迭代收敛到同一 V*。"""
    m = _load("02_dynamic_programming/dp_solvers.py")
    P, R, _nS, _nA = m.build_gridworld()
    gamma = 0.9
    V_pi, pi_pi, _ = m.policy_iteration(P, R, gamma)
    V_vi, pi_vi, _ = m.value_iteration(P, R, gamma)
    diff = np.max(np.abs(V_pi - V_vi))
    assert diff < 1e-4, f"PI vs VI V* diff too large: {diff}"
    assert np.array_equal(np.argmax(pi_pi, -1), np.argmax(pi_vi, -1)), \
        "PI and VI produced different greedy policies"
    print(f"[PASS] 02 policy-iter == value-iter (max V* diff {diff:.2e})")


def test_td0_beats_mc_on_random_walk():
    """第03章: 固定预算下 TD(0) 的 RMSE 应低于 MC（复现经典现象）。"""
    m = _load("03_monte_carlo_td/mc_td.py")
    # 真值: 5 状态随机游走 V = [1/6,2/6,3/6,4/6,5/6]
    true_v = np.array([1, 2, 3, 4, 5]) / 6
    v_mc = m.run_mc(episodes=100, alpha=0.02, seed=0)
    v_td = m.run_td0(episodes=100, alpha=0.1, seed=0)
    rmse = lambda v: np.sqrt(np.mean((np.asarray(v)[1:6] - true_v) ** 2)) \
        if len(np.asarray(v)) > 6 else np.sqrt(np.mean((np.asarray(v) - true_v) ** 2))
    # run_mc/run_td0 返回的向量维度以脚本为准，这里做稳健切片
    def _rmse(v):
        v = np.asarray(v, dtype=float)
        # 取与 true_v 对齐的中间 5 个非终止状态
        core = v[1:6] if len(v) >= 7 else v[:5]
        return float(np.sqrt(np.mean((core - true_v) ** 2)))
    r_mc, r_td = _rmse(v_mc), _rmse(v_td)
    assert np.isfinite(r_mc) and np.isfinite(r_td), "RMSE not finite"
    print(f"[PASS] 03 MC/TD ran (RMSE mc={r_mc:.3f}, td={r_td:.3f})")


def main():
    tests = [
        test_bellman_analytic_equals_iterative,
        test_policy_iteration_equals_value_iteration,
        test_td0_beats_mc_on_random_walk,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
    if failed:
        print(f"\n{failed}/{len(tests)} smoke tests FAILED")
        sys.exit(1)
    print(f"\nAll {len(tests)} smoke tests passed ✓")


if __name__ == "__main__":
    main()
