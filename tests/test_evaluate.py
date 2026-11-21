"""Retrieval metrics and fusion, on hand-worked examples."""
import pytest

from rag import evaluate
from rag.retrieval import rrf


def test_recall_at_k_counts_gold_files_in_top_k():
    sources = ["a", "b", "c", "d"]
    assert evaluate.recall_at_k(sources, ["a", "d"], 3) == 0.5
    assert evaluate.recall_at_k(sources, ["a", "d"], 4) == 1.0
    assert evaluate.recall_at_k(sources, ["z"], 4) == 0.0


def test_reciprocal_rank_uses_first_gold_hit():
    assert evaluate.reciprocal_rank(["a", "b", "c"], ["c", "b"]) == pytest.approx(1 / 2)
    assert evaluate.reciprocal_rank(["a"], ["z"]) == 0.0


def test_gold_of_accepts_single_and_multiple():
    assert evaluate.gold_of({"gold_file": "x"}) == ["x"]
    assert evaluate.gold_of({"gold_files": ["x", "y"]}) == ["x", "y"]


def test_rrf_by_hand():
    # k=60: a = 1/61 + 1/63, b = 1/62 + 1/62, c = 1/63, d = 1/61.
    fused = rrf([["a", "b", "c"], ["d", "b", "a"]], k=60)
    scores = {"a": 1 / 61 + 1 / 63, "b": 2 / 62, "c": 1 / 63, "d": 1 / 61}
    assert fused == sorted(scores, key=scores.get, reverse=True)


def test_rrf_uses_positions_only():
    assert rrf([["x", "y"]]) == ["x", "y"]
