"""Benchmark the three agents on the labeled questions, with repeats.

    python benchmark.py run --repeats 3 --budget 400
    python benchmark.py report

Agents
    ladder             chat-model grader, escalates one rung per failed grade
    diagnosis-legacy   decision-model agent, original verdict handling
    diagnosis-fixed    decision-model agent, default Policy (see rag/agent.py)

Question sets (eval/): specific (14), comparative (10), absent (10, the corpus
cannot answer them and the right reply is to say so).

Every (repeat, agent, set, question) is written to results/runs.jsonl as soon
as it finishes, and `run` skips what is already there, so a run can be split
across days. --budget caps chat-model calls for one invocation, because the
Gemini free tier allows 500 a day. The judge's calls count against it.

`report` reads the file and writes results/SUMMARY.md: accuracy as mean and
standard deviation across repeats, citation recall, and cost per question.
"""
import argparse
import json
import os
import statistics
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results", "runs.jsonl")
SETS = ("specific", "comparative", "absent")
AGENTS = ("ladder", "diagnosis-legacy", "diagnosis-fixed")


def agent_runners():
    from rag import agent, ladder
    return {"ladder": ladder.run,
            "diagnosis-legacy": lambda q: agent.run(q, policy=agent.LEGACY),
            "diagnosis-fixed": lambda q: agent.run(q, policy=agent.DEFAULT_POLICY)}


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def key(rec):
    return (rec["repeat"], rec["agent"], rec["set"], rec["i"])


# ------------------------------------------------------------------- running

def run(args):
    from rag import evaluate
    from rag.util import retry  # noqa: F401  (imported to fail early if keys are missing)

    runners = agent_runners()
    agents = [a for a in args.agents.split(",")]
    sets = {s: evaluate.load_evalset(s) for s in args.sets.split(",")}
    done = {key(r) for r in load(args.out)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    spent, wrote = 0, 0

    for repeat in range(args.repeats):
        for agent_name in agents:
            for set_name, items in sets.items():
                for i, item in enumerate(items):
                    rec_key = (repeat, agent_name, set_name, i)
                    if rec_key in done:
                        continue
                    if spent >= args.budget:
                        return _stop(f"budget of {args.budget} chat calls reached", wrote, args)
                    try:
                        state = runners[agent_name](item["question"])
                        ok, reason = evaluate.judge_item(item, state["answer"])
                    except RuntimeError as e:      # daily quota, most likely
                        return _stop(str(e), wrote, args)
                    chat = state.get("chat_calls", 0)
                    spent += chat + 1               # +1 for the judge
                    gold = evaluate.gold_of(item) if ("gold_file" in item or "gold_files" in item) else None
                    rec = {"repeat": repeat, "agent": agent_name, "set": set_name, "i": i,
                           "question": item["question"], "answer": state["answer"],
                           "correct": bool(ok), "reason": reason,
                           "citation_recall": evaluate.citation_recall(state["answer"], gold),
                           "chat_calls": chat, "decision_calls": state.get("decision_calls", 0),
                           "seconds": state.get("seconds"), "verdict": state.get("verdict"),
                           "attempts": state.get("attempts")}
                    with open(args.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    wrote += 1
                    print(f"[{repeat}] {agent_name:17} {set_name:11} {i:2} "
                          f"{'ok ' if ok else 'BAD'} {chat} chat  {item['question'][:50]}", flush=True)
                    time.sleep(args.pause)
    _stop("all requested runs are done", wrote, args)


def _stop(why, wrote, args):
    print(f"\nStopped: {why}. Wrote {wrote} new records to {args.out}.")
    print("Run the same command again to resume; then `python benchmark.py report`.")


# ------------------------------------------------------------------ reporting

def _mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def summarize(records):
    """One row per (agent, set) plus an 'all' row per agent.

    Accuracy is computed per repeat (share of questions judged correct in that
    repeat) and reported as mean and standard deviation across repeats, so the
    spread is the run-to-run noise of the whole agent, not a sampling guess.
    """
    rows = []
    for agent_name in AGENTS:
        mine = [r for r in records if r["agent"] == agent_name]
        if not mine:
            continue
        for set_name in (*SETS, "all"):
            sel = [r for r in mine if set_name == "all" or r["set"] == set_name]
            if not sel:
                continue
            by_repeat = defaultdict(list)
            for r in sel:
                by_repeat[r["repeat"]].append(r["correct"])
            acc_mean, acc_sd = _mean_sd([sum(v) / len(v) for v in by_repeat.values()])
            cit = [r["citation_recall"] for r in sel if r["citation_recall"] is not None]
            secs = [r["seconds"] for r in sel if r.get("seconds") is not None]
            rows.append({
                "agent": agent_name, "set": set_name,
                "questions": len({(r["set"], r["i"]) for r in sel}),
                "repeats": len(by_repeat), "accuracy": acc_mean, "accuracy_sd": acc_sd,
                "citation_recall": statistics.fmean(cit) if cit else None,
                "chat_per_q": statistics.fmean(r["chat_calls"] for r in sel),
                "decision_per_q": statistics.fmean(r["decision_calls"] for r in sel),
                "seconds_per_q": statistics.fmean(secs) if secs else None})
    return rows


def flips(records, a="diagnosis-legacy", b="diagnosis-fixed"):
    """Questions whose correct-rate across repeats went up or down from a to b."""
    rate = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["agent"] in (a, b):
            rate[r["agent"]][(r["set"], r["i"], r["question"])].append(r["correct"])
    up = down = same = 0
    for q, vals in rate[b].items():
        if q not in rate[a]:
            continue
        ra, rb = statistics.fmean(rate[a][q]), statistics.fmean(vals)
        up, down, same = up + (rb > ra), down + (rb < ra), same + (rb == ra)
    return {"improved": up, "worse": down, "unchanged": same}


def _fmt(v, spec):
    return "–" if v is None else format(v, spec)


def render(records):
    rows = summarize(records)
    lines = ["| Agent | Questions | Set | Repeats | Accuracy (mean ± sd) | Citation recall | Chat calls/q | Decision calls/q | s/q |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        acc = f"{r['accuracy']:.2f} ± {r['accuracy_sd']:.2f}"
        lines.append(f"| {r['agent']} | {r['questions']} | {r['set']} | {r['repeats']} | {acc} | "
                     f"{_fmt(r['citation_recall'], '.2f')} | {r['chat_per_q']:.1f} | "
                     f"{r['decision_per_q']:.1f} | {_fmt(r['seconds_per_q'], '.1f')} |")
    f = flips(records)
    lines += ["", f"Per-question change from diagnosis-legacy to diagnosis-fixed "
                  f"(correct-rate across repeats): {f['improved']} improved, "
                  f"{f['worse']} worse, {f['unchanged']} unchanged."]
    return "\n".join(lines)


def report(args):
    records = load(args.out)
    if not records:
        raise SystemExit(f"No records in {args.out}. Run `python benchmark.py run` first.")
    text = render(records)
    path = os.path.join(os.path.dirname(args.out), "SUMMARY.md")
    with open(path, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nWrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--agents", default=",".join(AGENTS))
    r.add_argument("--sets", default=",".join(SETS))
    r.add_argument("--budget", type=int, default=400, help="max chat-model calls this invocation")
    r.add_argument("--pause", type=float, default=2.0)
    r.add_argument("--out", default=DEFAULT_OUT)
    r.set_defaults(fn=run)
    q = sub.add_parser("report")
    q.add_argument("--out", default=DEFAULT_OUT)
    q.set_defaults(fn=report)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
