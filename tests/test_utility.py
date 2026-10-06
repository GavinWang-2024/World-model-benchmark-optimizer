"""Rank-agreement measures for the policy-ranking (functional utility) check."""

import pytest

from worldoptbench.utility import rank_agreement, ranking


def test_identical_orderings_agree_perfectly_even_if_the_scores_differ():
    a = {"p1": 10.0, "p2": 5.0, "p3": 1.0, "p4": -3.0}
    b = {"p1": 100.0, "p2": 50.0, "p3": 40.0, "p4": 0.0}  # different scale, same order
    r = rank_agreement(a, b)
    assert r.spearman == pytest.approx(1.0) and r.kendall == pytest.approx(1.0) and r.pairwise == 1.0 and r.top1 is True


def test_a_reversed_ordering_is_perfectly_anticorrelated_and_picks_the_wrong_best():
    a = {"p1": 4.0, "p2": 3.0, "p3": 2.0, "p4": 1.0}
    b = {"p1": 1.0, "p2": 2.0, "p3": 3.0, "p4": 4.0}
    r = rank_agreement(a, b)
    assert r.spearman == pytest.approx(-1.0) and r.pairwise == 0.0 and r.top1 is False


def test_one_adjacent_swap_costs_one_pair_out_of_the_total():
    a = {"p1": 4.0, "p2": 3.0, "p3": 2.0, "p4": 1.0}  # 6 pairs
    b = {"p1": 4.0, "p2": 2.0, "p3": 3.0, "p4": 1.0}  # p2 and p3 swapped
    r = rank_agreement(a, b)
    assert r.pairwise == pytest.approx(5 / 6) and r.top1 is True and 0.0 < r.spearman < 1.0


def test_only_policies_present_in_both_are_compared_and_fewer_than_three_gives_none():
    a = {"p1": 3.0, "p2": 2.0, "p3": 1.0, "only_a": 9.0}
    b = {"p1": 30.0, "p2": 20.0, "p3": 10.0, "only_b": -5.0}
    assert rank_agreement(a, b).n == 3 and rank_agreement(a, b).spearman == pytest.approx(1.0)
    short = rank_agreement({"p1": 1.0, "p2": 2.0}, {"p1": 1.0, "p2": 2.0})
    assert short.n == 2 and short.spearman is None and short.pairwise is None and short.top1 is None


def test_ties_in_the_reference_are_ignored_by_the_pairwise_measure():
    a = {"p1": 5.0, "p2": 5.0, "p3": 1.0}  # p1 and p2 tied: only 2 countable pairs
    b = {"p1": 9.0, "p2": 2.0, "p3": 1.0}
    assert rank_agreement(a, b).pairwise == 1.0


def test_a_constant_score_vector_has_no_correlation():
    r = rank_agreement({"p1": 1.0, "p2": 1.0, "p3": 1.0}, {"p1": 3.0, "p2": 2.0, "p3": 1.0})
    assert r.spearman is None and r.kendall is None and r.pairwise is None


def test_ranking_is_best_first():
    assert ranking({"a": 1.0, "b": 3.0, "c": 2.0}) == ["b", "c", "a"]


def test_summary_text_mentions_every_measure():
    text = rank_agreement({"a": 3.0, "b": 2.0, "c": 1.0}, {"a": 3.0, "b": 2.0, "c": 1.0}).summary()
    for part in ("spearman +1.00", "kendall +1.00", "pairwise 1.00", "same best policy yes", "n=3"):
        assert part in text
