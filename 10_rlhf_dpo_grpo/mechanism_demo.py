"""Deterministic mechanism checks; no model training or external dependencies.

Run from the repository root:
    python 10_rlhf_dpo_grpo/mechanism_demo.py
"""
import itertools
import math


def surrogate(ratio, advantage, epsilon=0.2):
    clipped = min(max(ratio, 1 - epsilon), 1 + epsilon)
    return min(ratio * advantage, clipped * advantage)


def derivative(fn, x, h=1e-6):
    return (fn(x + h) - fn(x - h)) / (2 * h)


def advantages(rewards, eta=1e-8):
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    return [(r - mean) / (std + eta) for r in rewards]


def close(actual, expected, tolerance=1e-6):
    if not math.isclose(actual, expected, abs_tol=tolerance, rel_tol=tolerance):
        raise AssertionError(f'{actual} != {expected}')


def gae_terminal(rewards, values, gamma=1.0, lam=1.0):
    """One fully valid, truly terminal sequence; no padding or truncation."""
    if len(rewards) != len(values) or not rewards:
        raise ValueError('Use matching, nonempty rewards and values.')
    advantages = [0.0] * len(rewards)
    next_value, next_advantage = 0.0, 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        next_advantage = delta + gamma * lam * next_advantage
        advantages[t] = next_advantage
        next_value = values[t]
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def walkthrough_checks():
    adv, returns = gae_terminal([0, 0, 1], [0.2, 0.4, 0.6])
    for a, expected in zip(adv, [0.8, 0.6, 0.4]):
        close(a, expected)
    for value in returns:
        close(value, 1.0)
    print(f'GAE: advantages={[round(a, 4) for a in adv]}, returns={returns}')
    # Padding deliberately has a large value: it must not enter the mean.
    losses = [[1, 1, 999, 999], [3, 3, 3, 3]]
    masks = [[1, 1, 0, 0], [1, 1, 1, 1]]
    sums = [sum(v * m for v, m in zip(row, mask)) for row, mask in zip(losses, masks)]
    lengths = [sum(mask) for mask in masks]
    token_mean = sum(sums) / sum(lengths)
    sequence_mean = sum(s / n for s, n in zip(sums, lengths)) / len(sums)
    close(token_mean, 7 / 3)
    close(sequence_mean, 2.0)
    print(f'aggregation: token_mean={token_mean:.6f}, sequence_mean={sequence_mean:.6f}')
    ratios = [1.6, 0.625]
    sequence_ratio = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    close(sequence_ratio, 1.0)
    # Freeze old probabilities; varying one log-ratio changes the geometric mean.
    slope = derivative(lambda z: math.exp((z + math.log(ratios[1])) / 2), math.log(ratios[0]))
    close(slope, 0.5)
    print(f'sequence ratio={sequence_ratio:.6f}, slope for one token log-ratio={slope:.6f}')


def main():
    # Derivative is with respect to log(new probability), old held fixed.
    for advantage, ratio, expected in [(2, 1, 2), (2, 1.4, 0),
                                       (-2, 1.4, -2.8), (-2, 0.6, 0)]:
        slope = derivative(lambda z: surrogate(math.exp(z), advantage), math.log(ratio))
        close(slope, expected)
        print(f'clip: A={advantage:+}, ratio={ratio:.1f}, objective={surrogate(ratio, advantage):.3f}, slope={slope:.3f}')

    for rewards in ([1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]):
        adv = advantages(rewards)
        expected = [1, -1, 1, -1] if rewards == [1, 0, 1, 0] else [0] * 4
        for a, b in zip(adv, expected):
            close(a, b)
        print(f'group: rewards={rewards}, advantages={[round(a, 4) for a in adv]}')

    p, q, old = [0.8, 0.2], [0.5, 0.5], [0.5, 0.5]
    k3 = [qi / pi - math.log(qi / pi) - 1 for pi, qi in zip(p, q)]
    exact = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
    under_p = sum(pi * k for pi, k in zip(p, k3))
    under_old = sum(oi * k for oi, k in zip(old, k3))
    corrected = sum(oi * (pi / oi) * k for oi, pi, k in zip(old, p, k3))
    close(under_p, exact)
    close(corrected, exact)
    if math.isclose(under_old, exact, abs_tol=1e-6):
        raise AssertionError('Example must distinguish old and current sampling.')
    print(f'KL: exact={exact:.6f}, current-weighted={under_p:.6f}, old-weighted={under_old:.6f}, corrected={corrected:.6f}')

    # Bernoulli action a, reward a, logit score a-p; enumerate every group.
    probability, group = 0.3, 4
    with_mean = with_loo = 0.0
    for actions in itertools.product((0, 1), repeat=group):
        weight = math.prod(probability if a else 1 - probability for a in actions)
        mean = sum(actions) / group
        with_mean += weight * sum((a - mean) * (a - probability) for a in actions) / group
        with_loo += weight * sum((a - (sum(actions) - a) / (group - 1)) * (a - probability) for a in actions) / group
    true_gradient = probability * (1 - probability)
    close(with_mean, (1 - 1 / group) * true_gradient)
    close(with_loo, true_gradient)
    print(f'baseline: exact gradient={true_gradient:.6f}, self-included mean={with_mean:.6f}, leave-one-out={with_loo:.6f}')
    walkthrough_checks()
    print('All mechanism checks passed.')


if __name__ == '__main__':
    main()
