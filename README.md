# DADA2 — Agentic RAG over SEC Filings

An agentic retrieval-augmented generation (RAG) system that answers questions about the annual 10-K filings of **NVIDIA, AMD and Micron**. A small decision model routes each question and checks whether the retrieved passages actually answer it. When they don't, the agent retries with a stronger search or breaks the question into sub-questions. Answers cite the filing sections they came from.

```
Deployed app: https://sec-filings-agent-pmtr.onrender.com
```

> The free Render tier sleeps when idle, so the first request can take about a minute.

## Why this is hard

The corpus holds three chip companies that write about the same topics (export controls, HBM supply, TSMC, AI demand) in similar words, year after year. A question about NVIDIA's fiscal 2026 risk factors has 26 other files full of near misses, so plain similarity search often returns the wrong company or year. The project measures which retrieval strategies fix that, and which agent design makes the best use of them.

## How it works

```
route (decision model) ─┬─> lookup (figure questions → XBRL financials) ─> answer
                        └─> retrieve ─> diagnose (decision model) ─> decide
                                                                      ├─ sufficient / absent ─> answer (chat model)
                                                                      ├─ wrong_words ────────> stronger search ─> retrieve
                                                                      └─ incomplete ─────────> plan ─> execute ─> diagnose
```

- **Chat model** (`gemini-3.5-flash-lite`) writes queries, plans and answers.
- **Decision model** (`clef-flash` on Cloudflare Workers AI) does the classification. It returns a probability for every option instead of prose, so the graph can fall back to a default when it is unsure. It decides two things: which kind of question this is, and whether the passages found are enough.
- **LangGraph** wires the nodes together, with at most 3 retrieval attempts per question.
- **A `Policy`** sets how far the agent trusts the decision model's verdicts (see [Verdict policy](#verdict-policy)).
- **Two tools** are exposed to the agent:
  - `search_filings` runs the diagnosing retrieval graph.
  - `financial_fact` reads exact figures from the XBRL financials files.

### Retrieval strategies

Each strategy adds one idea to the one before it, so the comparison stays fair. They all share the signature `search(question, k) -> [(source_file, passage_text), ...]`.

| Strategy | Adds |
|---|---|
| `baseline` | Dense vector search over Pinecone |
| `with_filter` | A metadata filter (company, year, section) extracted from the question |
| `filter_rrf` | Multi-query: 4 LLM rewrites fused with Reciprocal Rank Fusion (constant 60) |
| `filter_rrf_rerank` | Cohere `rerank-v3.5` reorders a pool of 10 down to 5 |
| `hybrid_rrf` | BM25 keyword search fused with dense search |

### The `ladder` agent (baseline for comparison)

`rag/ladder.py` is a simpler agent without a decision model. The chat model grades its own results, and each failed grade moves up one rung: `filter` → `filter+rrf` → `filter+rrf+rerank`. The write-up below compares it against the diagnosis agent.

## Corpus

- **27 section files:** Item 1 (Business), Item 1A (Risk Factors) and Item 7 (MD&A) from 9 10-K filings: NVIDIA, AMD and Micron, 3 fiscal years each.
- **3 XBRL financials files:** structured figures from EDGAR.
- **`corpus/manifest.csv`:** company, year, section and SEC source URL for every file.
- **Index:** chunks of 1200 characters with 150 overlap, embedded with `qwen3-embedding-0.6b` (1024 dimensions) in Pinecone index `dada2-1024`.

## Evaluation

`eval/` holds 34 labeled questions in three sets:

| Set | Questions | What it tests |
|---|---|---|
| `evalset.json` (specific) | 14 | One company and year, one gold file, a reference answer from the filing |
| `evalset_hard.json` (comparative) | 10 | Several companies or years, several gold files |
| `evalset_absent.json` (not covered) | 10 | Things the filings do not contain (other companies' figures, executive pay, off-topic). Competitor names appear in the corpus, their figures do not. The right answer is to say so. |

Retrieval is scored with recall@k and MRR. Answers are scored three ways: an LLM judge that sees the reference answer when there is one (and grades the "not covered" set on whether the agent declines instead of inventing), **citation recall** (the share of gold files the answer cites, which needs no judge), and cost per question (chat calls, decision calls, seconds).

### Results

The first comparison, before the verdict policy below existed: 14 questions (`specific[:8] + comparative[:6]`), one run each, with `gemini-3.5-flash-lite` as chat model and judge.

| Agent | Accuracy | Chat calls/q | Decision calls/q | s/q |
|---|---|---|---|---|
| `ladder` | 0.93 | 4.9 | 0 | 18.5 |
| diagnosis agent | 0.64 | 2.9 | 2.4 | 11.0 |

The two routers (decision model vs. chat model) labeled 20 and 19 of 24 questions correctly. The decision model was about twice as fast (0.43 s vs 0.89 s per call), and its calls do not count against the Gemini quota.

That is a small sample with one run per agent, and the judge saw neither the passages nor the reference answers. `benchmark.py` replaces it with repeated runs on all 34 questions and reference-aware judging (see [Benchmark](#benchmark)).

### What the first comparison found

The diagnosis agent was four questions behind the ladder, and none of its failures came from routing. Two came from how it trusted the decision model:

- **An unsure `absent` ended the run.** For "What target markets does NVIDIA include in its business for fiscal year 2024?", a passage from the gold file `NVDA_2024_business.txt` was retrieved, but on the third attempt the decision model returned `absent` at only 0.50. `diagnose` only overrode an unsure `absent` while attempts remained, so at the cap it stood, and `answer` returned a fixed "the filings do not cover that" without a chat call. The ladder, run on the same question, found only a passage saying the markets remain competitive, cited it, and said the passages did not list them. Both failed to name the markets, but the ladder showed what it had found and the diagnosis agent discarded it.
- **An unsure `sufficient` was never checked.** On NVIDIA's export controls over three years, the verdict was `sufficient` at 0.45 and the agent answered after one search.

A third miss was the judge's, not the agent's: the $4.5 billion H20 answer matches the reference, but the judge saw neither the passages nor the reference and rejected fiscal 2026 as hypothetical. Re-running the same agent moved the score by two questions, so a single run is not enough to separate the agents.

### Verdict policy

Both findings became an explicit, switchable `Policy` in `rag/agent.py`:

| | `LEGACY` (original) | `DEFAULT_POLICY` |
|---|---|---|
| `absent` | Ends the run once past 0.8 probability, or when attempts run out, with a canned reply and no chat call | Escalated like a search miss while attempts remain. If attempts run out, the chat model still reads the passages and says what they do and do not contain |
| `sufficient` | Accepted at any probability | Below 0.7 it is not accepted while attempts remain: search harder, or plan if already on the strongest search |

The cost of the fix is one or more extra searches on questions the filings do not cover, which is why the "not covered" set exists: it is where the fixed agent should pay, and where the legacy agent's shortcut should look best.

Unit tests pin both behaviours, including a regression test that replays the failure above through the whole graph with a scripted decision model.

A related fix: the figures `lookup` could only name NVDA, AMD or MU, so a question about another company's revenue was answered with one of the three companies' numbers. It now has an explicit `OTHER` option and reports that no figures exist.

### Benchmark

```bash
python benchmark.py run --repeats 3 --budget 400   # resumable; --budget caps chat calls per invocation
python benchmark.py report                          # writes results/SUMMARY.md
```

It runs `ladder`, `diagnosis-legacy` and `diagnosis-fixed` over all 34 questions. Each result is appended to `results/runs.jsonl` as it finishes and finished runs are skipped on the next call, so a full run can be split across days under the Gemini free tier (500 requests a day). The report gives accuracy as mean and standard deviation across repeats, citation recall, cost per question, and how many questions improved or got worse between the legacy and fixed policies.

### Limitations

- The judge is the same model family as the agent's chat model. Reading a sample of its verdicts in `results/runs.jsonl` is part of trusting a number.
- The comparative questions have gold files but no reference answers, so they are judged on the question alone, plus citation recall.
- The corpus is nine 10-K filings (three companies, three years). Results on this corpus say little about other document types.
- The thresholds (`sufficient_min = 0.7`, `ABSENT_SURE = 0.8`) were set by judgment and not tuned. `Policy` takes them as parameters so they can be swept.

## Repository layout

```
app.py              Gradio web app (Render entry point)
agentui.py          Chat UI that streams each agent step
benchmark.py        Repeated, resumable benchmark and report
rag/
  config.py         Settings: models, chunking, K, thresholds
  retrieval.py      The retrieval strategies
  agent.py          Router, diagnosis, Policy, planner and the LangGraph graph
  decide.py         Decision-model client (Cloudflare Workers AI)
  ladder.py         Baseline ladder agent
  embed.py          Cloudflare embeddings client
  evaluate.py       Metrics, judges, citation recall and cost accounting
  util.py           Rate-limit-aware retry helper
DADA2_0*.ipynb      The build-up in order: corpus and index (01), retrieval
                    strategies (02), the agent (03) and the evaluation (04)
figures/            Architecture and graph diagrams used by the notebooks
tests/              Offline test suite (faked decision model, retrieval and chat)
corpus/             Filing sections, financials and manifest
eval/               Labeled question sets
.github/workflows/  CI: runs the tests
render.yaml         Render Blueprint
```

## Getting started

### Prerequisites

- Python 3.13 (matches `render.yaml`)
- A Pinecone index already built from the corpus. The deployed app only searches the index and never builds it, and notebook `DADA2_01_Corpus_and_Index.ipynb` is what builds it.
- API keys:

| Variable | Used for |
|---|---|
| `GOOGLE_API_KEY` | Gemini chat model (free tier: 15 requests/min, 500/day) |
| `PINECONE_API_KEY` | Vector index |
| `CLOUDFLARE_API_TOKEN` | Decision model and embeddings (Workers AI permission) |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account |
| `COHERE_API_KEY` | `rerank-v3.5` reranker |

### Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# create .env with the variables listed above
python app.py              # http://localhost:7860
```

Set `EMBEDDINGS=cloudflare` (the default) so the app embeds questions the same way the index was built. `.env` is git-ignored. Never commit API keys.

### Run the tests

```bash
pip install pytest
python -m pytest -q
```

The suite is offline: the decision model, retrieval and chat model are faked, so it needs no API keys. It covers the verdict policy, routing, the graph end to end with a scripted decision model, the retrieval metrics, the judges and the benchmark's aggregation. CI runs it on every push and pull request.

### Deploy to Render

`render.yaml` is a Render Blueprint. In Render choose **New + → Blueprint**, pick this repo, and enter the five secret variables when prompted. Render runs `pip install -r requirements.txt` and then `python app.py`.

## Tech stack

Python · LangChain · LangGraph · Gemini · Cloudflare Workers AI · Pinecone · Cohere · BM25 · Gradio · Render
