"""Judge selection, citation recall, and the benchmark's aggregation."""
import pytest

import benchmark
from rag import evaluate


class Judge:
    def __init__(self, reply="CORRECT. fine."):
        self.reply, self.prompts = reply, []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return type("M", (), {"content": self.reply})()


def test_judge_uses_reference_when_item_has_one():
    j = Judge()
    ok, _ = evaluate.judge_item({"question": "q", "reference": "REF TEXT"}, "ans", model=j)
    assert ok and "REF TEXT" in j.prompts[0]


def test_judge_uses_absent_prompt_for_expect_absent():
    j = Judge("INCORRECT. invented a figure")
    ok, _ = evaluate.judge_item({"question": "q", "expect": "absent"}, "ans", model=j)
    assert not ok and "do not contain" in j.prompts[0]


def test_judge_falls_back_to_plain_when_no_reference():
    j = Judge()
    evaluate.judge_item({"question": "q"}, "ans", model=j)
    assert "Reference answer" not in j.prompts[0] and "do not contain" not in j.prompts[0]


def test_citation_recall():
    ans = "Export controls tightened [NVDA_2024_risk_factors.txt] and again [NVDA_2025_risk_factors.txt]."
    gold = ["NVDA_2024_risk_factors.txt", "NVDA_2025_risk_factors.txt", "NVDA_2026_risk_factors.txt"]
    assert evaluate.citation_recall(ans, gold) == pytest.approx(2 / 3)
    assert evaluate.citation_recall("no citations", gold) == 0.0
    assert evaluate.citation_recall("anything", None) is None


def test_every_question_set_loads_and_is_well_formed():
    sizes = {s: len(evaluate.load_evalset(s)) for s in ("specific", "comparative", "absent")}
    assert sizes == {"specific": 14, "comparative": 10, "absent": 10}
    for item in evaluate.load_evalset("absent"):
        assert item["expect"] == "absent" and item["question"].endswith("?")
    for item in evaluate.load_evalset("specific"):
        assert item["reference"] and item["gold_file"]


def rec(agent, repeat, i, correct, set_="specific", chat=2, cit=None):
    return {"agent": agent, "repeat": repeat, "set": set_, "i": i, "question": f"q{i}",
            "correct": correct, "citation_recall": cit, "chat_calls": chat,
            "decision_calls": 1, "seconds": 3.0}


def test_summary_is_mean_and_sd_across_repeats():
    records = [rec("ladder", 0, 0, True), rec("ladder", 0, 1, True),    # repeat 0: 1.0
               rec("ladder", 1, 0, True), rec("ladder", 1, 1, False)]   # repeat 1: 0.5
    row = [r for r in benchmark.summarize(records) if r["set"] == "specific"][0]
    assert row["accuracy"] == pytest.approx(0.75)
    assert row["accuracy_sd"] == pytest.approx(0.3535, abs=1e-3)
    assert row["repeats"] == 2 and row["questions"] == 2


def test_flips_counts_per_question_changes():
    records = [rec("diagnosis-legacy", 0, 0, False), rec("diagnosis-fixed", 0, 0, True),    # improved
               rec("diagnosis-legacy", 0, 1, True), rec("diagnosis-fixed", 0, 1, False),    # worse
               rec("diagnosis-legacy", 0, 2, True), rec("diagnosis-fixed", 0, 2, True)]     # same
    assert benchmark.flips(records) == {"improved": 1, "worse": 1, "unchanged": 1}


def test_render_produces_a_table_row_per_agent_and_set():
    records = [rec("ladder", 0, 0, True, cit=1.0), rec("diagnosis-fixed", 0, 0, False)]
    text = benchmark.render(records)
    assert "| ladder | 1 | specific |" in text and "| diagnosis-fixed | 1 | all |" in text
