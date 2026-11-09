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

`eval/` holds 24 labeled questions:

- `evalset.json`: 14 **specific** questions, one gold file each
- `evalset_hard.json`: 10 **comparative** questions, several gold files each

Retrieval is scored with recall@k and MRR. Answers are scored by an LLM judge. The full results and analysis are in the [write-up](#write-up) below.

## Repository layout

```
app.py              Gradio web app (Render entry point)
agentui.py          Chat UI that streams each agent step
rag/
  config.py         Settings: models, chunking, K, thresholds
  retrieval.py      The retrieval strategies
  agent.py          Router, diagnosis, planner and the LangGraph graph
  decide.py         Decision-model client (Cloudflare Workers AI)
  ladder.py         Baseline ladder agent
  embed.py          Cloudflare embeddings client
  evaluate.py       Metrics and comparison tables
  util.py           Rate-limit-aware retry helper
corpus/             Filing sections, financials and manifest
eval/               Labeled question sets
render.yaml         Render Blueprint
```

## Getting started

### Prerequisites

- Python 3.13 (matches `render.yaml`)
- A Pinecone index already built from the corpus. The deployed app only searches the index and never builds it, and the notebooks that build it are not in this repo.
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

### Deploy to Render

`render.yaml` is a Render Blueprint. In Render choose **New + → Blueprint**, pick this repo, and enter the five secret variables when prompted. Render runs `pip install -r requirements.txt` and then `python app.py`.

## Tech stack

Python · LangChain · LangGraph · Gemini · Cloudflare Workers AI · Pinecone · Cohere · BM25 · Gradio · Render

---

## Submission

### Write-up

**Conditions.** Chat model `gemini-3.5-flash-lite` (it also served as the judge); decision model `clef-flash` on Cloudflare Workers AI. Embeddings: `qwen3-embedding-0.6b` via Cloudflare, 1024 dimensions, Pinecone index `dada2-1024`. Chunks of 1200 characters with 150 overlap. K = 5 passages to the model; the reranker is Cohere `rerank-v3.5`, reordering a pool of 10 down to 5. Multi-query uses 4 rewrites fused with RRF (constant 60). Agent thresholds were left at their defaults: SURE 0.6, ABSENT_SURE 0.8, MAX_ATTEMPTS 3. The routers were compared on all 24 labeled questions (14 specific, 10 comparative); the agents on 14 (`specific[:8] + comparative[:6]`), each run once.

**The routers.** The decision model labeled 20 of 24 questions correctly and the chat model 19 of 24, so accuracy differs by a single question. The decision model was about twice as fast (0.43 s against 0.89 s per call). The bigger difference is cost: its calls do not count against the 500-a-day Gemini quota, and it returns a probability for every option, which is what lets the graph fall back to a default strategy when it is unsure. In my traced run, the Micron memory-pricing question, which is labeled comparative, was routed as specific and still found both gold files, because the extractor named the company and left the year empty. Classifying questions is what the decision model is for: the same accuracy, faster, and free of the chat quota.

**The agents.** The ladder scored 0.93 with 4.9 chat calls and 18.5 s per question; the diagnosis agent scored 0.64 with 2.9 chat calls, 2.4 decision calls and 11.0 s. The gap is four questions out of fourteen. In my traced run, none of its failures came from routing. The clearest failure was "What target markets does NVIDIA include in its business for fiscal year 2024?": a passage from the gold file `NVDA_2024_business.txt` was retrieved, but on the third attempt the decision model returned `absent` at only 0.50. My `diagnose` overrides an unsure `absent` only while attempts remain, so at the cap it was accepted, and `answer` returned a fixed "the filings do not cover that" without a chat call. Run separately, the ladder did not name the markets either; it found only a passage saying its target markets remain competitive, cited it, and said the passages did not list them. So on this question a gold-file hit did not mean the answering passage was retrieved, and the two agents differed in how they failed: the ladder showed what it had found, while mine discarded it. A second failure, on NVIDIA's export controls over three years, went the other way: the verdict was `sufficient` at 0.45, nothing checks the confidence of a `sufficient` verdict, and the agent answered after one search. A third was the judge's mistake rather than the agent's: the $4.5 billion H20 answer matches the reference, but the judge sees neither the passages nor the reference, and it rejected fiscal 2026 as hypothetical. In the inspection cell the agent answered 4 of 4 correctly, and in my traced run of the other ten it answered 7 of 10, so 11 of 14 across those two runs against 9 of 14 in the table: the score moved by two questions between runs.

**What did not help.** The `absent` shortcut in `answer`. It saves one chat call when it fires, and in my traced run it fired only on a question the corpus answers, even though a passage from the gold file was already in `documents`. The ABSENT_SURE guard does not protect the last attempt, which is exactly where an unsure `absent` is most likely to appear after two failed searches. Removing the shortcut costs one chat call on those questions, and the answer prompt already tells the model to say in one sentence when the passages do not cover the question. The ladder's run on the same question shows what the alternative looks like: a cited answer that says what the passages do and do not contain.

**What I would ship.** On these numbers, the ladder: it was right on four more questions, and its extra cost (2 more chat calls and 7.5 s per question) is affordable at this scale. The diagnosis agent's speed and quota savings are real, but I would only ship it after removing the `absent` shortcut and adding a minimum confidence for `sufficient`, and then measuring it again. Those fixes are not shown in any table here. Before choosing, I would want to know what share of users' questions are comparative, since both agent failures I traced came from a verdict on retrieved passages; how often they ask about things the filings truly do not cover, which is the only case the `absent` exit is good for; and I would give the judge the reference answers, since it misgraded a correct answer in this run.
