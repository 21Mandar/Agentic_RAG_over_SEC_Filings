"""Scoring.

Every retrieval strategy has the same shape:

    search(question, k) -> [(source_file, passage_text), ...]   best first

That is the whole contract, and it is what makes the comparison fair. Two strategies that return different things cannot be compared, so
if you find yourself wanting to change the contract, change it for all of
them.

We score two things about retrieval and one thing about the answer.

  recall@k   Of the files that genuinely answer this question, how many
             turned up in the top k? This is the metric that matters most,
             because a passage the retriever never returned is a passage the
             model cannot use. Nothing downstream can fix a recall failure.

  MRR        How high up was the first genuinely relevant file? Recall says
             the evidence was in the list; MRR says whether it was at the top
             of it. Reranking is supposed to move MRR without moving recall,
             and the retrieval comparison shows exactly that.

  answer     Did the final answer actually say the right thing? Retrieval
             metrics can both look good while the answer is still wrong, so
             the agent benchmark checks the end of the pipeline too.
"""
import json
import os
import re

from . import config
from .util import retry, text_of


def load_evalset(which="specific"):
    """which is "specific", "comparative" or "absent". specific and comparative
    disagree about which retrieval strategy is best, so they are kept apart.
    absent holds questions the corpus cannot answer: the right reply is to say
    so, and it is where an agent that never gives up looks bad."""
    name = {"specific": "evalset.json", "comparative": "evalset_hard.json",
            "absent": "evalset_absent.json"}[which]
    with open(os.path.join(config.EVAL_DIR, name)) as f:
        return json.load(f)


def gold_of(item):
    """Both sets are allowed to label more than one file as correct."""
    gold = item.get("gold_files") or item.get("gold_file")
    return gold if isinstance(gold, list) else [gold]


def recall_at_k(sources, gold, k):
    """Fraction of the gold files that appear in the first k results."""
    top = sources[:k]
    return len([g for g in gold if g in top]) / len(gold)


def reciprocal_rank(sources, gold):
    """1/rank of the first gold file, or 0 if none of them came back."""
    ranks = [sources.index(g) + 1 for g in gold if g in sources]
    return 1 / min(ranks) if ranks else 0.0


def score_retrieval(search, items, k=config.K, pause=3.0, verbose=False):
    """Run one strategy over a question set and average the two metrics."""
    import time

    recall, rr = 0.0, 0.0
    for item in items:
        results = search(item["question"], k)
        sources = [source for source, _ in results]
        gold = gold_of(item)
        recall += recall_at_k(sources, gold, k)
        rr += reciprocal_rank(sources, gold)
        if verbose:
            got = "yes" if any(g in sources[:k] for g in gold) else "NO "
            print(f"  [{got}] {item['question'][:66]}")
        time.sleep(pause)
    n = len(items)
    return {"recall@k": recall / n, "mrr": rr / n, "n": n, "k": k}


def compare(strategies, items, k=config.K, pause=3.0):
    """Score several strategies over the same questions and print the table.

    strategies is an ordered dict of name -> search function. Order it so each
    row adds one thing to the row above it, because that is the only way to
    read a column and say what caused a change.
    """
    rows = []
    print(f"  {'strategy':24} {'recall@' + str(k):>9} {'MRR':>7}")
    print("  " + "-" * 42)
    for name, search in strategies.items():
        s = score_retrieval(search, items, k=k, pause=pause)
        rows.append({"strategy": name, **s})
        print(f"  {name:24} {s['recall@k']:>9.2f} {s['mrr']:>7.3f}", flush=True)
    return rows


JUDGE_PROMPT = """You are grading one answer from a question answering system
that reads SEC filings.

Question: {question}

Answer given: {answer}

The answer is correct if it addresses the question using the filings and does
not contradict them. It is incorrect if it is about the wrong company, the
wrong fiscal year, says it cannot find something that the question is clearly
about, or states a figure that the filings do not support.

Reply with one word, CORRECT or INCORRECT, then one sentence of reason."""


def judge_answer(question, answer, model=None):
    """A model grading another model. Cheap, and good enough to catch the
    cases where retrieval looked fine and the answer still went wrong. Read a
    handful of its verdicts yourself before you trust the average."""
    model = model or config.chat()
    verdict = retry(model.invoke, JUDGE_PROMPT.format(question=question, answer=answer))
    text = text_of(verdict).strip()
    return text.upper().startswith("CORRECT"), text


REFERENCE_JUDGE_PROMPT = """You are grading one answer from a question answering
system that reads SEC filings, against a reference answer written from the
filing itself.

Question: {question}

Reference answer: {reference}

Answer given: {answer}

The answer is correct if it states the same facts as the reference (wording and
extra detail do not matter). It is incorrect if it contradicts the reference,
leaves out the key fact, or says it cannot find what the reference states.

Reply with one word, CORRECT or INCORRECT, then one sentence of reason."""


ABSENT_JUDGE_PROMPT = """You are grading one answer from a question answering
system that reads only the 10-K filings of NVIDIA, AMD and Micron. The question
below asks for something those filings do not contain.

Question: {question}

Answer given: {answer}

The answer is correct if it says the filings do not provide this information,
even if it adds what the filings do say about related topics. It is incorrect
if it states a specific figure, date or fact as the answer, since that would be
invented or taken from the wrong place.

Reply with one word, CORRECT or INCORRECT, then one sentence of reason."""


def judge_item(item, answer, model=None):
    """Grade one answer with the strictest judge the item supports: against its
    reference answer when it has one, as a should-be-refused question when it is
    marked expect=absent, and on the question alone otherwise."""
    if item.get("expect") == "absent":
        prompt = ABSENT_JUDGE_PROMPT.format(question=item["question"], answer=answer)
    elif item.get("reference"):
        prompt = REFERENCE_JUDGE_PROMPT.format(
            question=item["question"], reference=item["reference"], answer=answer)
    else:
        return judge_answer(item["question"], answer, model=model)
    model = model or config.chat()
    text = text_of(retry(model.invoke, prompt)).strip()
    return text.upper().startswith("CORRECT"), text


def citation_recall(answer, gold):
    """Fraction of the gold files the answer cites by filename. Deterministic,
    so it needs no judge: the answer prompt asks for filenames in brackets."""
    if not gold:
        return None
    cited = set(re.findall(r"[A-Za-z]+_\d{4}_[a-z_]+\.txt", answer))
    return len([g for g in gold if g in cited]) / len(gold)


def score_answers(ask, items, model=None, pause=4.0):
    """End to end: ask() takes a question and returns the final answer text."""
    import time

    model = model or config.chat()
    correct, notes = 0, []
    for item in items:
        answer = ask(item["question"])
        ok, reason = judge_answer(item["question"], answer, model=model)
        correct += ok
        notes.append({"question": item["question"], "answer": answer,
                      "correct": ok, "reason": reason})
        time.sleep(pause)
    return {"accuracy": correct / len(items), "n": len(items)}, notes


# ------------------------------------------------- the whole agent, costed

def score_agent(run, items, model=None, pause=4.0):
    """End to end, with the bill.

    run(question) must return the agent's final state: a dict with "answer",
    "chat_calls" and "decision_calls". Both agents do.

    Accuracy comes from the judge. The other three columns are what the
    answer cost: chat-model calls (the ones that count against your Gemini
    quota), decision-model calls, and seconds.
    """
    import time

    model = model or config.chat()
    correct, chat, dec, secs, notes = 0, 0, 0, 0.0, []
    for item in items:
        t = time.time()
        state = run(item["question"])
        secs += time.time() - t
        ok, reason = judge_answer(item["question"], state["answer"], model=model)
        correct += ok
        chat += state.get("chat_calls", 0)
        dec += state.get("decision_calls", 0)
        notes.append({"question": item["question"], "answer": state["answer"],
                      "correct": ok, "reason": reason,
                      "chat_calls": state.get("chat_calls", 0),
                      "decision_calls": state.get("decision_calls", 0)})
        time.sleep(pause)
    n = len(items)
    return {"accuracy": correct / n, "chat_calls": chat / n,
            "decision_calls": dec / n, "seconds": secs / n, "n": n}, notes


def compare_agents(agents, items, pause=4.0):
    """agents is {name: run}. Prints one row per agent and returns the rows."""
    rows = []
    print(f"  {'agent':12} {'accuracy':>9} {'chat/q':>7} {'decision/q':>11} {'s/q':>6}")
    print("  " + "-" * 50)
    for name, run in agents.items():
        s, _ = score_agent(run, items, pause=pause)
        rows.append({"agent": name, **s})
        print(f"  {name:12} {s['accuracy']:>9.2f} {s['chat_calls']:>7.1f} "
              f"{s['decision_calls']:>11.1f} {s['seconds']:>6.1f}", flush=True)
    return rows
