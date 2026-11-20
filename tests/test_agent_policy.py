"""The agent's trust in the decision model's verdicts (rag.agent.Policy)."""
import pytest

from rag import agent
from rag.agent import DEFAULT_POLICY, LEGACY, MAX_ATTEMPTS


def diagnose_with(monkeypatch, verdict, p, *, policy, attempts, strategy="filter"):
    monkeypatch.setattr(agent.decision, "diagnose", lambda q, docs: (verdict, {verdict: p}))
    state = {"question": "q", "documents": [], "attempts": attempts,
             "strategy": strategy, "policy": policy}
    return agent.diagnose(state)["verdict"]


# ----------------------------------------------------------------- absent

def test_default_never_accepts_absent_while_attempts_remain(monkeypatch):
    # Even a confident `absent` on attempt 1 is searched again.
    assert diagnose_with(monkeypatch, "absent", 0.99, policy=DEFAULT_POLICY, attempts=1) == "wrong_words"


def test_legacy_accepts_confident_absent(monkeypatch):
    assert diagnose_with(monkeypatch, "absent", 0.9, policy=LEGACY, attempts=1) == "absent"


def test_legacy_retries_unsure_absent(monkeypatch):
    assert diagnose_with(monkeypatch, "absent", 0.5, policy=LEGACY, attempts=1) == "wrong_words"


@pytest.mark.parametrize("policy", [DEFAULT_POLICY, LEGACY])
def test_absent_stands_once_attempts_are_exhausted(monkeypatch, policy):
    assert diagnose_with(monkeypatch, "absent", 0.5, policy=policy, attempts=MAX_ATTEMPTS) == "absent"


# ------------------------------------------------------------- sufficient

def test_unsure_sufficient_escalates_to_stronger_search(monkeypatch):
    v = diagnose_with(monkeypatch, "sufficient", 0.45, policy=DEFAULT_POLICY, attempts=1, strategy="filter")
    assert v == "wrong_words"


def test_unsure_sufficient_plans_when_already_strongest(monkeypatch):
    v = diagnose_with(monkeypatch, "sufficient", 0.45, policy=DEFAULT_POLICY, attempts=1,
                      strategy="filter+rrf+rerank")
    assert v == "incomplete"


def test_confident_sufficient_is_accepted(monkeypatch):
    assert diagnose_with(monkeypatch, "sufficient", 0.95, policy=DEFAULT_POLICY, attempts=1) == "sufficient"


def test_unsure_sufficient_accepted_at_attempt_cap(monkeypatch):
    v = diagnose_with(monkeypatch, "sufficient", 0.3, policy=DEFAULT_POLICY, attempts=MAX_ATTEMPTS)
    assert v == "sufficient"


def test_legacy_accepts_any_sufficient(monkeypatch):
    assert diagnose_with(monkeypatch, "sufficient", 0.3, policy=LEGACY, attempts=1) == "sufficient"


def test_sufficient_threshold_is_the_policys(monkeypatch):
    strict = agent.Policy(sufficient_min=0.9)
    assert diagnose_with(monkeypatch, "sufficient", 0.8, policy=strict, attempts=1) == "wrong_words"
    assert diagnose_with(monkeypatch, "sufficient", 0.8, policy=DEFAULT_POLICY, attempts=1) == "sufficient"


# ----------------------------------------------------------------- routing

@pytest.mark.parametrize("verdict,attempts,strategy,expected", [
    ("sufficient", 1, "filter", "answer"),
    ("absent", 1, "filter", "answer"),
    ("incomplete", 1, "filter", "plan"),
    ("incomplete", MAX_ATTEMPTS, "filter", "answer"),
    ("wrong_words", 1, "filter", "stronger"),
    ("wrong_words", MAX_ATTEMPTS, "filter", "answer"),
    ("wrong_words", 1, "filter+rrf+rerank", "answer"),   # nothing stronger left
])
def test_decide(verdict, attempts, strategy, expected):
    assert agent.decide({"verdict": verdict, "attempts": attempts, "strategy": strategy}) == expected


def test_route_falls_back_to_default_when_unsure(monkeypatch):
    monkeypatch.setattr(agent.decision, "scope", lambda q: ("specific", {"specific": agent.SURE - 0.01}))
    assert agent.route({"question": "q"})["strategy"] == agent.DEFAULT
    monkeypatch.setattr(agent.decision, "scope", lambda q: ("comparative", {"comparative": 0.9}))
    assert agent.route({"question": "q"})["strategy"] == "hybrid+rrf"


def test_figure_questions_skip_retrieval():
    assert agent.after_route({"scope": "figure"}) == "lookup"
    assert agent.after_route({"scope": "specific"}) == "retrieve"


# ------------------------------------------------------------------ answer

class FakeChat:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return type("M", (), {"content": "answer from passages"})()


def test_answer_legacy_absent_is_canned_and_free(monkeypatch):
    chat = FakeChat()
    monkeypatch.setattr(agent.config, "chat", lambda: chat)
    out = agent.answer({"verdict": "absent", "policy": LEGACY, "question": "q", "documents": []})
    assert out["chat_calls"] == 0 and chat.calls == 0
    assert "do not cover" in out["answer"]


def test_answer_default_absent_still_reads_the_passages(monkeypatch):
    chat = FakeChat()
    monkeypatch.setattr(agent.config, "chat", lambda: chat)
    out = agent.answer({"verdict": "absent", "policy": DEFAULT_POLICY, "question": "q",
                        "documents": [("NVDA_2024_business.txt", "text")]})
    assert out["chat_calls"] == 1 and chat.calls == 1
    assert out["answer"] == "answer from passages"


# --------------------------------------------------------------- execute

def test_execute_returns_only_new_passages(monkeypatch):
    held = ("A.txt", "one")
    monkeypatch.setattr(agent.retrieval, "with_filter",
                        lambda step, k: [held, ("B.txt", "two")] if step == "s1"
                        else [("B.txt", "two"), ("C.txt", "three")])
    out = agent.execute({"documents": [held], "steps": ["s1", "s2"], "attempts": 1})
    assert out["documents"] == [("B.txt", "two"), ("C.txt", "three")]
    assert out["attempts"] == 2 and out["chat_calls"] == 2


# ------------------------------------------- the whole graph, faked end to end

def run_graph(monkeypatch, policy, verdicts):
    """Specific question, strategies that each return one passage, and a
    decision model that gives the scripted verdicts in order."""
    searched = []

    def fake_search(name):
        def search(q, k):
            searched.append(name)
            return [(f"{name}.txt", f"passage from {name}")]
        return search

    for name in ("filter", "filter+rrf", "filter+rrf+rerank", "hybrid+rrf"):
        monkeypatch.setitem(agent.retrieval.STRATEGIES, name, fake_search(name))
    monkeypatch.setattr(agent.decision, "scope", lambda q: ("specific", {"specific": 0.95}))
    script = iter(verdicts)
    monkeypatch.setattr(agent.decision, "diagnose", lambda q, docs: next(script))
    monkeypatch.setattr(agent.config, "chat", lambda: FakeChat())
    state = agent.run("What are NVIDIA's target markets in fiscal 2024?", policy=policy)
    return state, searched


def test_regression_unsure_absent_on_last_attempt_still_answers_from_passages(monkeypatch):
    """The failure that motivated the policy: a gold passage was retrieved, the
    third diagnosis said `absent` at 0.5, and the legacy agent replied with a
    canned 'not covered' and no answering call."""
    verdicts = [("wrong_words", {"wrong_words": 0.6}), ("wrong_words", {"wrong_words": 0.6}),
                ("absent", {"absent": 0.5})]
    legacy, _ = run_graph(monkeypatch, LEGACY, verdicts)
    assert "do not cover" in legacy["answer"]
    assert legacy["chat_calls"] == 5           # retrieval rungs only, no answer call

    fixed, searched = run_graph(monkeypatch, DEFAULT_POLICY, verdicts)
    assert fixed["answer"] == "answer from passages"
    assert fixed["verdict"] == "absent" and fixed["attempts"] == MAX_ATTEMPTS
    assert searched == ["filter", "filter+rrf", "filter+rrf+rerank"]


def test_confident_absent_costs_more_searches_under_default_not_legacy(monkeypatch):
    verdicts = [("absent", {"absent": 0.95})] * 3
    _, searched_legacy = run_graph(monkeypatch, LEGACY, verdicts)
    _, searched_default = run_graph(monkeypatch, DEFAULT_POLICY, verdicts)
    assert len(searched_legacy) == 1
    assert len(searched_default) == MAX_ATTEMPTS


def test_unsure_sufficient_triggers_a_stronger_search(monkeypatch):
    verdicts = [("sufficient", {"sufficient": 0.45}), ("sufficient", {"sufficient": 0.9})]
    state, searched = run_graph(monkeypatch, DEFAULT_POLICY, verdicts)
    assert searched == ["filter", "filter+rrf"]
    assert state["verdict"] == "sufficient"
    _, searched_legacy = run_graph(monkeypatch, LEGACY, [("sufficient", {"sufficient": 0.45})])
    assert searched_legacy == ["filter"]
