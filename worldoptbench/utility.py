"""Functional-utility preservation: does an optimized world model still rank policies the way the original does?
(outline section 12-D)

A world model is often used as a policy evaluator: imagine each candidate policy's rollouts and rank the policies by the
return the model predicts. Speed, visual and physics scores can all survive an optimization while this use quietly
breaks, so the check here is the use itself: evaluate the same policies under the baseline model and under each
optimized model and compare the *rankings*.

Agreement measures (policies that are tied in the reference ranking are ignored by the pairwise measure):

  spearman   rank correlation, 1 = same order, -1 = reversed
  kendall    Kendall tau, the same idea counted over pairs
  pairwise   the fraction of policy pairs ordered the same way (0.5 is what a coin flip gives)
  top1       whether the best policy is the same

With few policies these are coarse: 8 policies give 28 pairs, so a single swap moves `pairwise` by 0.036. Compare against
the same model evaluated with a different random seed (the sampling-noise floor), not against 1.0.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any


@dataclass
class RankAgreement:
    n: int
    spearman: float | None
    kendall: float | None
    pairwise: float | None
    top1: bool | None

    def summary(self) -> str:
        def show(v: float | None) -> str:
            return "n/a" if v is None else f"{v:+.2f}"

        pair = "n/a" if self.pairwise is None else f"{self.pairwise:.2f}"
        top = "n/a" if self.top1 is None else ("yes" if self.top1 else "no")
        return f"spearman {show(self.spearman)}; kendall {show(self.kendall)}; pairwise {pair}; same best policy {top} (n={self.n})"


def rank_agreement(reference: dict[str, float], other: dict[str, float]) -> RankAgreement:
    """How closely `other`'s ordering of the policies follows `reference`'s. Only policies present in both are used;
    fewer than 3 give an all-None result, and a constant score vector (no ordering) gives None for the correlations."""
    from scipy import stats  # noqa: PLC0415

    names = sorted(set(reference) & set(other))
    if len(names) < 3:
        return RankAgreement(len(names), None, None, None, None)
    a = [reference[n] for n in names]
    b = [other[n] for n in names]

    def correlation(fn: Any) -> float | None:
        if len(set(a)) < 2 or len(set(b)) < 2:
            return None
        value = fn(a, b)[0]
        return None if math.isnan(value) else float(value)

    pairs = [(i, j) for i, j in itertools.combinations(range(len(names)), 2) if a[i] != a[j]]
    pairwise = None
    if pairs:
        pairwise = sum((a[i] - a[j]) * (b[i] - b[j]) > 0 for i, j in pairs) / len(pairs)
    best_a, best_b = max(names, key=lambda n: reference[n]), max(names, key=lambda n: other[n])
    return RankAgreement(
        n=len(names), spearman=correlation(stats.spearmanr), kendall=correlation(stats.kendalltau),
        pairwise=pairwise, top1=best_a == best_b,
    )


def ranking(scores: dict[str, float]) -> list[str]:
    """Policy names from best to worst."""
    return sorted(scores, key=lambda n: scores[n], reverse=True)
