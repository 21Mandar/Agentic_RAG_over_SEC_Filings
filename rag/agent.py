"""The agent. A decision model routes and judges, a chat model writes.

    route (decision) -+-> lookup (figure questions, the financials file) -> answer
                      +-> retrieve -> diagnose (decision) -> decide
                                                             |-- sufficient / absent -> answer (chat)
                                                             |-- wrong_words        -> stronger search, back to retrieve
                                                             |-- incomplete         -> plan (chat), execute, back to diagnose

How much to trust the decision model's verdicts is a `Policy`. The default
policy distrusts two verdicts it used to take at face value (see Policy);
`LEGACY` reproduces the original behaviour so the two can be benchmarked on
the same code (eval/benchmark.py).
"""
import json
import operator
import os
import time
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from . import config, decide as decision, retrieval
from .util import retry, text_of

MAX_ATTEMPTS = 3

# What each kind of question starts with. Measured on the ladder agent: the filter wins on specific questions and loses on
# comparative ones, where hybrid wins.
START_WITH = {"specific": "filter", "comparative": "hybrid+rrf", "figure": "filter"}
DEFAULT = "filter+rrf"            # when the router is not sure
SURE = 0.6                        # below this, use DEFAULT

# wrong_words means the search was aimed right and missed, so search harder.
STRONGER = {"filter": "filter+rrf", "hybrid+rrf": "filter+rrf+rerank",
            "filter+rrf": "filter+rrf+rerank"}

# absent is a judgment, and the model can be unsure of it. Under LEGACY, below
# this we give it one more search before taking its word.
ABSENT_SURE = 0.8


@dataclass(frozen=True)
class Policy:
    """How far to trust the decision model's verdicts.

    absent_shortcut  True: an `absent` verdict (once past ABSENT_SURE, or out of
                     attempts) ends the run and returns a canned "not covered"
                     reply with no chat call. False: `absent` is first treated
                     as a search miss and escalated like `wrong_words`, and if
                     the attempts run out the chat model still reads the
                     passages and says what they do and do not contain.
    sufficient_min   A `sufficient` verdict below this probability is not
                     accepted while attempts remain: the agent searches harder
                     (or plans, when there is no stronger search) and
                     diagnoses again. 0 accepts any `sufficient`.
    """
    absent_shortcut: bool = False
    sufficient_min: float = 0.7


LEGACY = Policy(absent_shortcut=True, sufficient_min=0.0)
DEFAULT_POLICY = Policy()


# ---------------------------------------------------------------- the tools

def financial_fact(ticker: str, concept: str) -> str:
    """Look up a reported figure. ticker is NVDA, AMD or MU. concept is an
    XBRL name such as Revenues or ResearchAndDevelopmentExpense."""
    path = os.path.join(config.FINANCIALS, f"{ticker.upper()}_financials.json")
    if not os.path.exists(path):
        return f"No financials for {ticker}. Use NVDA, AMD or MU."
    facts = json.load(open(path))["facts"]
    if concept not in facts:
        return (f"{ticker} has no concept named {concept}. Available: "
                + ", ".join(sorted(facts)[:25]))
    rows = sorted(facts[concept].items())
    return f"{ticker} {concept}:\n" + "\n".join(
        f"  period ending {d}: {v:,}" for d, v in rows)


def format_passages(hits):
    return "\n\n".join(f"[{source}] {text}" for source, text in hits)


# ---------------------------------------------------------------- the state

class RetrievalState(TypedDict, total=False):
    question: str
    scope: str                 # specific, comparative, figure
    strategy: str              # which retrieval to run next
    documents: Annotated[list, operator.add]
    verdict: str               # sufficient, absent, wrong_words, incomplete
    probabilities: dict
    steps: list
    attempts: int
    chat_calls: Annotated[int, operator.add]
    decision_calls: Annotated[int, operator.add]
    answer: str
    policy: Policy
    seconds: float


# ---------------------------------------------------------------- the nodes

def route(state: RetrievalState) -> dict:
    """One decision-model call. Which kind of question, and so which search.

    Use START_WITH for the choice, unless the model is not sure (under SURE),
    in which case fall back to DEFAULT.
    """
    scope, probs = decision.scope(state["question"])
    strategy = START_WITH[scope] if probs[scope] >= SURE else DEFAULT
    return {"scope": scope, "strategy": strategy, "decision_calls": 1}

def after_route(state: RetrievalState) -> Literal["lookup", "retrieve"]:
    """Figure questions do not go to retrieval at all."""
    return "lookup" if state["scope"] == "figure" else "retrieve"

class Fact(BaseModel):
    """Which figure a question asks for."""
    ticker: Literal["NVDA", "AMD", "MU"]
    concept: Literal["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                     "ResearchAndDevelopmentExpense", "GrossProfit", "NetIncomeLoss",
                     "OperatingIncomeLoss"] = Field(
        description="the XBRL concept. Revenue questions: Revenues, or "
                    "RevenueFromContractWithCustomerExcludingAssessedTax for AMD. "
                    "R&D spend: ResearchAndDevelopmentExpense.")


FACT_PROMPT = """Which reported figure does this question ask for?

  research and development spend  -> ResearchAndDevelopmentExpense
  revenue, sales, top line         -> Revenues (AMD reports it as
                                      RevenueFromContractWithCustomerExcludingAssessedTax)
  gross profit, gross margin       -> GrossProfit
  net income, profit, earnings     -> NetIncomeLoss
  operating income                 -> OperatingIncomeLoss

ticker is the company's symbol: NVIDIA -> NVDA, AMD -> AMD, Micron -> MU.

Example: "What did Micron spend on R&D in fiscal 2025?"
         -> ticker=MU, concept=ResearchAndDevelopmentExpense

Question: {question}"""


def lookup(state: RetrievalState) -> dict:
    """A number is in the financials file, not in the prose. One chat call
    to name the figure, then a plain dictionary lookup. The table is the one
    document and the verdict is already sufficient, so the graph goes straight
    to answer."""
    namer = config.chat().with_structured_output(Fact)
    fact = retry(namer.invoke, FACT_PROMPT.format(question=state["question"]))
    table = financial_fact(fact.ticker, fact.concept)
    return {"documents": [(f"{fact.ticker}_financials.json", table)],
            "verdict": "sufficient", "attempts": 1, "chat_calls": 1}

def retrieve(state: RetrievalState) -> dict:
    search = retrieval.STRATEGIES[state["strategy"]]
    hits = search(state["question"], config.K)
    seen = set(state.get("documents") or [])
    new = [h for h in hits if h not in seen]
    chat = {"filter": 1, "hybrid+rrf": 1, "filter+rrf": 2, "filter+rrf+rerank": 2}
    return {"documents": new, "attempts": state.get("attempts", 0) + 1,
            "chat_calls": chat[state["strategy"]]}


def diagnose(state: RetrievalState) -> dict:
    """One decision-model call: enough, or why not. The verdict is then
    filtered through the policy, because two of them are judgments the model
    can be unsure of.

    absent: a search that missed looks the same as a topic the filings do not
    cover, so (unless the policy has the legacy shortcut) it is treated as a
    miss and searched again while attempts remain.

    sufficient: below policy.sufficient_min it is not accepted while attempts
    remain. A stronger search is the cheap way to get better evidence; when
    the strategy is already the strongest, plan instead.
    """
    policy = state.get("policy") or DEFAULT_POLICY
    verdict, probs = decision.diagnose(state["question"], state.get("documents") or [])
    attempts_left = state.get("attempts", 0) < MAX_ATTEMPTS
    if verdict == "absent" and attempts_left:
        if not policy.absent_shortcut or probs[verdict] < ABSENT_SURE:
            verdict = "wrong_words"
    elif verdict == "sufficient" and attempts_left and probs[verdict] < policy.sufficient_min:
        verdict = "wrong_words" if state.get("strategy") in STRONGER else "incomplete"
    return {"verdict": verdict, "probabilities": probs, "decision_calls": 1}

def decide(state: RetrievalState) -> Literal["answer", "stronger", "plan"]:
    """Every verdict has one place to go. sufficient and absent both stop
    (diagnose has already sent the ones it does not trust back for more).
    incomplete needs a plan. wrong_words needs a stronger search, unless there
    is none left. And the attempt cap stops everything."""
    verdict = state["verdict"]
    if verdict in ("sufficient", "absent"):
        return "answer"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "answer"
    if verdict == "incomplete":
        return "plan"
    # wrong_words
    return "stronger" if state.get("strategy") in STRONGER else "answer"

def stronger(state: RetrievalState) -> dict:
    """Same question, the next search up. STRONGER says which."""
    return {"strategy": STRONGER[state["strategy"]]}

class Plan(BaseModel):
    """The separate questions a comparative question breaks into."""
    steps: list[str] = Field(description="Each names one company, one year if "
                                         "relevant, and one thing to find")


PLAN_PROMPT = """A question about SEC 10-K filings needs more than one search:

{question}

Passages found so far cover: {covered}

Write the separate questions still needed. Each must name exactly one company
and ask for one thing, so that a filtered search can find it. Fewest possible."""


def plan(state: RetrievalState) -> dict:
    """One chat-model call. Break the question into per-company searches.

    The planner is told which sources the passages so far already cover, so it
    does not ask for them again."""
    planner = config.chat().with_structured_output(Plan)
    sources = sorted({source for source, _ in state.get("documents") or []})
    covered = ", ".join(sources) if sources else "nothing"
    out = retry(planner.invoke, PLAN_PROMPT.format(question=state["question"],
                                                   covered=covered))
    return {"steps": out.steps, "chat_calls": 1}

def execute(state: RetrievalState) -> dict:
    """One filtered search per step. The filter extractor is a chat call each.

    Only passages not already held are returned: documents has an add reducer,
    so the graph does the appending.
    """
    seen = set(state.get("documents") or [])
    new = []
    steps = state.get("steps") or []
    for step in steps:
        for hit in retrieval.with_filter(step, config.K):
            if hit not in seen:
                seen.add(hit)
                new.append(hit)
    return {"documents": new, "attempts": state.get("attempts", 0) + 1,
            "chat_calls": len(steps)}

ANSWER_PROMPT = """Answer the question using only these passages from SEC
filings. Cite the filing each fact comes from, by its filename in brackets.
If the passages do not cover the question, say so in one sentence.

Question: {question}

Passages:
{passages}"""


def answer(state: RetrievalState) -> dict:
    """One chat-model call. The only place prose gets written.

    Without the legacy shortcut, an `absent` verdict that survived diagnose
    (attempts exhausted) still goes to the chat model: ANSWER_PROMPT tells it
    to say so in one sentence when the passages do not cover the question, and
    to cite what they do contain."""
    policy = state.get("policy") or DEFAULT_POLICY
    if state.get("verdict") == "absent" and policy.absent_shortcut:
        return {"answer": "The filings in this corpus do not cover that.", "chat_calls": 0}
    out = retry(config.chat().invoke, ANSWER_PROMPT.format(
        question=state["question"], passages=format_passages(state["documents"])))
    return {"answer": text_of(out), "chat_calls": 1}


def build_graph():
    """Wire it as the diagram at the top of this file shows. decide and
    after_route return Literals, which declare their edges."""
    from langgraph.graph import StateGraph, START, END

    g = StateGraph(RetrievalState)
    g.add_node("route", route)
    g.add_node("lookup", lookup)
    g.add_node("retrieve", retrieve)
    g.add_node("diagnose", diagnose)
    g.add_node("stronger", stronger)
    g.add_node("plan", plan)
    g.add_node("execute", execute)
    g.add_node("answer", answer)

    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route)       # -> lookup | retrieve
    g.add_edge("lookup", "answer")
    g.add_edge("retrieve", "diagnose")
    g.add_conditional_edges("diagnose", decide)    # -> answer | stronger | plan
    g.add_edge("stronger", "retrieve")
    g.add_edge("plan", "execute")
    g.add_edge("execute", "diagnose")
    g.add_edge("answer", END)
    return g.compile()

def run(question, policy: Policy = DEFAULT_POLICY):
    """The whole final state, for tracing and cost accounting: the answer, the
    verdict and its probabilities, attempts, chat and decision calls, and
    wall-clock seconds."""
    start = time.time()
    state = build_graph().invoke({"question": question, "attempts": 0,
                                  "chat_calls": 0, "decision_calls": 0,
                                  "policy": policy})
    state["seconds"] = time.time() - start
    return state


def ask(question, verbose=False):
    state = run(question)
    if verbose:
        print(f"  {state['scope']} -> {state['strategy']}, verdict {state['verdict']}, "
              f"{state['attempts']} attempt(s), {state['chat_calls']} chat + "
              f"{state['decision_calls']} decision calls")
    return state["answer"]


# ---------------------------------------------------------------- the agent
#
# The graph above is the retrieval. The agent is what app.py serves: the same
# LangChain create_agent with two tools, the graph behind search_filings and
# the financials file behind financial_fact, and a checkpointer so the chat
# remembers the conversation. agentui.py streams it, tool calls and all.

def search_filings(question: str) -> str:
    """Answer a question from the text of the 10-K filings of NVIDIA, AMD and
    Micron: business, risk factors, management's discussion. Returns an answer
    with the filing each fact came from in brackets. Use financial_fact instead
    for a reported number such as revenue or R&D spend."""
    return run(question)["answer"]


SYSTEM_PROMPT = """You answer questions about NVIDIA, AMD and Micron from their
10-K filings. Use search_filings for anything the filings say in prose, and
financial_fact for a reported number. Relay the tool's answer, with its
citations, and do not add facts of your own. If the tools find nothing, say so."""


def build_agent():
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import MemorySaver

    return create_agent(config.chat(), tools=[search_filings, financial_fact],
                        system_prompt=SYSTEM_PROMPT, checkpointer=MemorySaver())
