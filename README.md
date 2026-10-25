# DADA2 — Agentic RAG over SEC filings

**Fill in both sections at the bottom before you submit.** Keep the deployed
app line exactly in this form; it is how your app gets found and graded.

## Getting started

There is no separate spec. The instructions are in the notebooks, and
`DADA2_00_Start_Here.ipynb` is the one to open first: it explains the
assignment, the setup, the grading, and what each of the four notebooks after
it asks for. Work through them in order.

Get the files the same way as every class day: pull, then work in a copy:

```bash
cd lecture-files
git pull
cd ..
cp -r lecture-files/DADA2 DADA2     # work in the copy, not in lecture-files/
cd DADA2
cp example.env .env                  # then fill in your keys
```

Then open `DADA2_00_Start_Here.ipynb` from your `DADA2` copy and run its
setup check.

Deploying is `DEPLOY.md`.

---

## Submission

```
Deployed app: https://sec-filings-agent-pmtr.onrender.com
```

ADD YOUR NAME TO YOUR RENDER PROJECT SO THE LINK CAN BE PERSONALIZED.

PLEASE ENSURE YOUR URL WORKS BY TESTING IT IN INCOGNITO MODE.

### Write-up

**Conditions.** Chat model `gemini-3.5-flash-lite` (it also served as the judge); decision model `clef-flash` on Cloudflare Workers AI. Embeddings: `qwen3-embedding-0.6b` via Cloudflare, 1024 dimensions, Pinecone index `dada2-1024`. Chunks of 1200 characters with 150 overlap. K = 5 passages to the model; the reranker is Cohere `rerank-v3.5`, reordering a pool of 10 down to 5. Multi-query uses 4 rewrites fused with RRF (constant 60). Agent thresholds were left at their defaults: SURE 0.6, ABSENT_SURE 0.8, MAX_ATTEMPTS 3. The routers were compared on all 24 labeled questions (14 specific, 10 comparative); the agents on 14 (`specific[:8] + comparative[:6]`), each run once.

**The routers.** The decision model labeled 20 of 24 questions correctly and the chat model 19 of 24, so accuracy differs by a single question. The decision model was about twice as fast (0.43 s against 0.89 s per call). The bigger difference is cost: its calls do not count against the 500-a-day Gemini quota, and it returns a probability for every option, which is what lets the graph fall back to a default strategy when it is unsure. In my traced run, the Micron memory-pricing question, which is labeled comparative, was routed as specific and still found both gold files, because the extractor named the company and left the year empty. Classifying questions is what the decision model is for: the same accuracy, faster, and free of the chat quota.

**The agents.** The ladder scored 0.93 with 4.9 chat calls and 18.5 s per question; the diagnosis agent scored 0.64 with 2.9 chat calls, 2.4 decision calls and 11.0 s. The gap is four questions out of fourteen. In my traced run, none of its failures came from routing. The clearest failure was "What target markets does NVIDIA include in its business for fiscal year 2024?": a passage from the gold file `NVDA_2024_business.txt` was retrieved, but on the third attempt the decision model returned `absent` at only 0.50. My `diagnose` overrides an unsure `absent` only while attempts remain, so at the cap it was accepted, and `answer` returned a fixed "the filings do not cover that" without a chat call. Run separately, the ladder did not name the markets either; it found only a passage saying its target markets remain competitive, cited it, and said the passages did not list them. So on this question a gold-file hit did not mean the answering passage was retrieved, and the two agents differed in how they failed: the ladder showed what it had found, while mine discarded it. A second failure, on NVIDIA's export controls over three years, went the other way: the verdict was `sufficient` at 0.45, nothing checks the confidence of a `sufficient` verdict, and the agent answered after one search. A third was the judge's mistake rather than the agent's: the $4.5 billion H20 answer matches the reference, but the judge sees neither the passages nor the reference, and it rejected fiscal 2026 as hypothetical. In the inspection cell the agent answered 4 of 4 correctly, and in my traced run of the other ten it answered 7 of 10, so 11 of 14 across those two runs against 9 of 14 in the table: the score moved by two questions between runs.

**What did not help.** The `absent` shortcut in `answer`. It saves one chat call when it fires, and in my traced run it fired only on a question the corpus answers, even though a passage from the gold file was already in `documents`. The ABSENT_SURE guard does not protect the last attempt, which is exactly where an unsure `absent` is most likely to appear after two failed searches. Removing the shortcut costs one chat call on those questions, and the answer prompt already tells the model to say in one sentence when the passages do not cover the question. The ladder's run on the same question shows what the alternative looks like: a cited answer that says what the passages do and do not contain.

**What I would ship.** On these numbers, the ladder: it was right on four more questions, and its extra cost (2 more chat calls and 7.5 s per question) is affordable at this scale. The diagnosis agent's speed and quota savings are real, but I would only ship it after removing the `absent` shortcut and adding a minimum confidence for `sufficient`, and then measuring it again. Those fixes are not shown in any table here. Before choosing, I would want to know what share of users' questions are comparative, since both agent failures I traced came from a verdict on retrieved passages; how often they ask about things the filings truly do not cover, which is the only case the `absent` exit is good for; and I would give the judge the reference answers, since it misgraded a correct answer in this run.