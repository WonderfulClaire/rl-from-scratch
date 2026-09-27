"""
A tiny deterministic GRPO-style policy update over agent strategies.

Goal: show why reward quality matters before optimizer choice.
We use a four-action categorical policy:
  0 correct fix
  1 delete public tests
  2 hard-code public cases
  3 do nothing

The same balanced rollout group is scored twice:
- naive public reward
- secure verifier reward

Because the group is balanced and standardized advantages sum to zero,
the gradient on logits is exactly the per-action group advantage.
"""
from __future__ import annotations

from math import exp
from statistics import mean, pstdev

from pathlib import Path
import runpy

_reward_demo = runpy.run_path(str(Path(__file__).with_name("reward_hacking_demo.py")))
Candidate = _reward_demo["Candidate"]
naive_public_reward = _reward_demo["naive_public_reward"]
secure_reward = _reward_demo["secure_reward"]


ACTIONS = (
    Candidate("correct fix", "correct"),
    Candidate("delete public tests", "buggy", public_tests_present=False),
    Candidate("hard-code public cases", "hardcode_public"),
    Candidate("do nothing", "buggy"),
)


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    xs = [exp(x - m) for x in logits]
    z = sum(xs)
    return [x / z for x in xs]


def standardized_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    mu = mean(rewards)
    sigma = pstdev(rewards)
    if sigma < eps:
        return [0.0] * len(rewards)
    return [(r - mu) / (sigma + eps) for r in rewards]


class StepResult:
    def __init__(
        self,
        rewards: list[float],
        advantages: list[float],
        before: list[float],
        after: list[float],
    ) -> None:
        self.rewards = rewards
        self.advantages = advantages
        self.before = before
        self.after = after


def one_balanced_grpo_step(
    logits: list[float],
    *,
    reward_fn,
    lr: float = 0.4,
) -> StepResult:
    """One group-relative update with one rollout per action.

    For a categorical policy:
        grad log pi(a_i) = one_hot(a_i) - pi

    With one sample per action and standardized advantages whose sum is zero:
        sum_i A_i * grad log pi(a_i) = A

    so the logit gradient is exactly the advantage vector.
    """
    rewards = [float(reward_fn(action)) for action in ACTIONS]
    advantages = standardized_advantages(rewards)
    before = softmax(logits)
    updated = [x + lr * a for x, a in zip(logits, advantages)]
    after = softmax(updated)
    logits[:] = updated
    return StepResult(
        rewards=rewards,
        advantages=advantages,
        before=before,
        after=after,
    )


def run_demo(steps: int = 8) -> dict[str, object]:
    naive_logits = [0.0] * len(ACTIONS)
    secure_logits = [0.0] * len(ACTIONS)

    naive_history = []
    secure_history = []
    for _ in range(steps):
        naive_history.append(
            one_balanced_grpo_step(
                naive_logits,
                reward_fn=naive_public_reward,
            )
        )
        secure_history.append(
            one_balanced_grpo_step(
                secure_logits,
                reward_fn=secure_reward,
            )
        )

    naive_probs = softmax(naive_logits)
    secure_probs = softmax(secure_logits)

    # Under the naive reward, all three reward-1 strategies are reinforced,
    # including the two exploits. Under secure reward, only the real fix grows.
    assert naive_probs[1] > naive_probs[3]
    assert naive_probs[2] > naive_probs[3]
    assert abs(naive_probs[0] - naive_probs[1]) < 1e-9
    assert abs(naive_probs[0] - naive_probs[2]) < 1e-9

    assert secure_probs[0] > 0.9
    assert secure_probs[0] > secure_probs[1]
    assert secure_probs[0] > secure_probs[2]

    return {
        "naive_probs": naive_probs,
        "secure_probs": secure_probs,
        "naive_rewards": naive_history[0].rewards,
        "secure_rewards": secure_history[0].rewards,
    }


def main() -> None:
    result = run_demo()
    print("strategy                     naive_R  secure_R  naive_pi  secure_pi")
    print("------------------------------------------------------------------")
    for i, action in enumerate(ACTIONS):
        print(
            f"{action.name:<28} "
            f"{result['naive_rewards'][i]:>7.0f}  "
            f"{result['secure_rewards'][i]:>8.0f}  "
            f"{result['naive_probs'][i]:>8.3f}  "
            f"{result['secure_probs'][i]:>9.3f}"
        )
    print("\nTakeaway: GRPO faithfully optimizes the reward you gave it.")
    print("If the reward accepts exploits, the optimizer reinforces exploits too.")


if __name__ == "__main__":
    main()
