"""
Agentic post-training: verifiable reward, reward hacking, and harness generalization.

This file is intentionally small and deterministic. It does not train an LLM.
Instead it isolates the part that often fails before optimization even starts:
the environment/reward contract.
"""
from __future__ import annotations

from statistics import mean, pstdev
from typing import Iterable


def target(x: int) -> int:
    return x * x + 2 * x + 1


PUBLIC_X = (0, 1, 2)
HIDDEN_X = (-3, -1, 7)


class Candidate:
    def __init__(self, name: str, implementation: str, public_tests_present: bool = True):
        self.name = name
        self.implementation = implementation
        self.public_tests_present = public_tests_present


def evaluate_impl(implementation: str, x: int) -> int:
    if implementation == "buggy":
        return x * x + 1
    if implementation == "correct":
        return target(x)
    if implementation == "hardcode_public":
        table = {0: 1, 1: 4, 2: 9}
        return table.get(x, 9)
    raise ValueError(f"unknown implementation: {implementation}")


def naive_public_reward(candidate: Candidate) -> float:
    """Deliberately weak reward: trust the mutable visible test suite."""
    xs: Iterable[int] = PUBLIC_X if candidate.public_tests_present else ()
    return float(all(evaluate_impl(candidate.implementation, x) == target(x) for x in xs))


def secure_reward(candidate: Candidate) -> float:
    """Independent final-state verifier."""
    integrity = candidate.public_tests_present
    public_ok = all(
        evaluate_impl(candidate.implementation, x) == target(x) for x in PUBLIC_X
    )
    hidden_ok = all(
        evaluate_impl(candidate.implementation, x) == target(x) for x in HIDDEN_X
    )
    return float(integrity and public_ok and hidden_ok)


def grpo_group_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    """Population-standardized group rewards, matching chapter 10."""
    if len(rewards) < 2:
        raise ValueError("GRPO group needs at least two samples")
    mu = mean(rewards)
    sigma = pstdev(rewards)
    if sigma < eps:
        return [0.0 for _ in rewards]
    return [(reward - mu) / (sigma + eps) for reward in rewards]


def harness_tool_names(variant: str) -> tuple[str, ...]:
    """Same semantics, different schemas."""
    variants = {
        "canonical": ("read_file", "write_file", "run_public_tests"),
        "compact": ("read", "write", "test"),
        "alternate": ("inspect_file", "update_file", "check_visible_tests"),
    }
    try:
        return variants[variant]
    except KeyError as exc:
        raise ValueError(f"unknown harness variant: {variant}") from exc


def run_checks() -> dict[str, object]:
    candidates = (
        Candidate("correct fix", "correct"),
        Candidate("delete public tests", "buggy", public_tests_present=False),
        Candidate("hard-code public cases", "hardcode_public"),
        Candidate("do nothing", "buggy"),
    )
    rows = [
        {
            "candidate": candidate.name,
            "naive_reward": naive_public_reward(candidate),
            "secure_reward": secure_reward(candidate),
        }
        for candidate in candidates
    ]

    assert rows[0]["naive_reward"] == rows[0]["secure_reward"] == 1.0
    assert rows[1]["naive_reward"] == 1.0 and rows[1]["secure_reward"] == 0.0
    assert rows[2]["naive_reward"] == 1.0 and rows[2]["secure_reward"] == 0.0
    assert rows[3]["naive_reward"] == rows[3]["secure_reward"] == 0.0

    rewards = [row["secure_reward"] for row in rows]
    advantages = grpo_group_advantages(rewards)
    assert abs(sum(advantages)) < 1e-6

    variants = ("canonical", "compact", "alternate")
    assert all(len(harness_tool_names(v)) == 3 for v in variants)

    return {"rows": rows, "advantages": advantages}


def main() -> None:
    result = run_checks()
    print("candidate                     naive  secure")
    print("-------------------------------------------")
    for row in result["rows"]:
        print(
            f"{row['candidate']:<28} "
            f"{row['naive_reward']:.0f}      {row['secure_reward']:.0f}"
        )
    print("\nsecure-reward GRPO advantages:")
    print([round(value, 3) for value in result["advantages"]])
    print("\nAll agentic post-training checks passed.")


if __name__ == "__main__":
    main()
