# Eval methodology: known limits and where to take it

A note to revisit. Captures the discussion that happened around ADR-009's
reranker sweep, where the LLM-judged Overall metric was flat-to-noisy
while objective retrieval metrics moved cleanly. The headline question
was: how trustworthy is the eval, and what would it take to defend the
numbers in the memoria.

## What the harness measures today

`eval/etsi_eval/runner.py` produces, per row:

- **Retrieval metrics (objective):** `recall_at_5`, `recall_at_10`,
  `mrr`. Set-membership of `golden_urls` against the agent's retrieved
  URL list. No LLM involved.
- **Answer-quality scores (LLM-as-judge):** three sub-dimensions —
  `faithfulness`, `correctness`, `completeness` — each a 0–100 int with
  a one-sentence rationale. `overall = min(...)`.
- **Per-row aggregates** rolled up by category, difficulty, and globally.

Schema in `eval/etsi_eval/report.py`. The judge call is made via the
OpenAI SDK with `response_format=json_schema` (commit `e8db0c9`), so the
verdict shape is enforced.

## Three structural problems with the LLM-judged half

1. **Judge bias.** GPT-class judges score their own family's outputs
   higher (self-preference, ~10–30 points in published studies); reward
   fluent answers over factually weaker ones; conflate confidence with
   correctness. See Zheng et al., *Judging LLM-as-a-Judge with MT-Bench
   and Chatbot Arena* (NeurIPS 2023).
2. **Reasoning instability.** Same judge, same input, different seed →
   different score. Even at temperature=0, output is non-deterministic
   across vLLM/llama.cpp versions. Per-row variance of ±5–15 points is
   typical and not noted in our current report.
3. **Score arbitrariness.** "Faithfulness=88" is a judge-internal scalar
   with no calibration to any human-meaningful unit. Two reports of
   "88.1 vs 85.2" are not actually comparable across judges, prompts,
   or corpus snapshots.

The retrieval-tier metrics are not affected by any of this.

## Concrete demonstration from ADR-009 row 34

Question: "¿Quién es el Director de la ETSI Informática…?"
Golden URL: `…/info/10608/direccion`. **Neither the e4b nor the 26b
agent retrieved the gold doc** — top-5 was nearly identical across both
runs and the gold wasn't there. Both agents fabricated different
director names from the same off-topic candidate set.

| Agent | Answer | Score |
|---|---|---|
| e4b | "Manuel Nicolás Enciso García-Oliveros" + multiple phones | 72 |
| 26b-a4b | "Ernesto Pimentel Sánchez" + one phone | 5 |

Both wrong against the gold; the 67-point gap is judge artefact (the
e4b answer happened to contain phone-number tokens that partially
matched the reference). This row's −67 weight in the aggregate
attributed to "the bigger agent regressed" was a pure measurement
error. Per-row variance like this is the rule, not the exception.

## Five upgrade paths, ordered by defensibility-per-effort

### 1. Lean on retrieval metrics (already have)

`recall_at_k`, `MRR` are objective set-membership against gold URLs.
Decades of IR literature, well-defined statistical tests, judge-
independent. Across the four ADR-009 rerun configurations these moved
unambiguously and are the load-bearing comparison.

**Action:** in the memoria, lead the eval section with retrieval
metrics. Frame LLM-judged Overall as a secondary trend signal with
explicit acknowledgement of its limits (one paragraph citing Zheng
et al.).

### 2. Pairwise preference between configs (~half a day)

Instead of "score this answer 0–100," ask the judge "given two answers
to the same question, which better answers it? A / B / tie." Output is
a single bit per row. Translates to a defensible win-rate.

**Why it helps:** eliminates absolute-score anchoring; reduces self-
preference bias; standard in LMSYS Chatbot Arena, MT-Bench, AlpacaEval.

**Concrete shape:** new judge prompt template; new runner that takes
two result JSONs (e.g. `hybrid_topk10_ctx4092.json` vs
`20260504_202044.json`), pairs answers per row, randomises A/B order,
and reports win/loss/tie totals. Maybe 200 LOC of Python.

**Headline statement gain:** "bge-m3 is preferred 47%, hybrid 35%, tied
18% (n=83, judge=gpt-5.4-mini)" defends much better than "Overall mean
80.5 vs 80.8."

### 3. Human calibration on a sub-sample (~half a day)

Pick 20–30 rows. Hand-annotate gold answers with sub-scores using the
same rubric the judge uses. Compute:

- **Cohen's κ** — inter-rater agreement between you and the judge.
  κ > 0.4 is "moderate," > 0.6 is "substantial."
- **Pearson's r** — correlation between your scores and judge scores.

This is the cheapest credibility move. **Without this step, you cannot
honestly defend the absolute scores.** With it, the memoria gets one
defensible sentence: "On a held-out sample of 20 rows, the judge's
per-row Overall correlated with author annotation at r=0.X (κ=0.Y),
justifying its use as a relative-trend signal."

**Concrete shape:** stratified random sample (a few rows per
category/difficulty), hand-fill a CSV with author scores, small Python
script computing κ and r.

### 4. RAGAS / faithfulness via NLI (~one day)

[RAGAS](https://github.com/explodinggradients/ragas) is the de facto
standard for RAG eval. It decomposes answer-quality into:

- **Faithfulness:** does each claim follow from the retrieved context?
  (NLI model checks each sentence-level claim against context.)
- **Answer relevance:** does the answer address the question?
- **Context precision / recall:** what fraction of retrieved context is
  actually relevant?

Internally still uses an LLM for some judgements but the decomposition
is more defensible than holistic 0–100 grading. **Citable framework
adds credibility per page of the memoria.**

**Concrete shape:** add `ragas` to `eval/pyproject.toml`, write a small
adapter that takes a `results/TIMESTAMP.json` and emits RAGAS scores on
the same rows. Report alongside our judge scores; agreement is mutual
validation.

### 5. Multi-judge ensemble

Same eval against 2–3 different judges (gpt-5.4-mini + Claude Sonnet
+ Gemini Flash, say); report median or majority score; disagreement
between judges quantifies the noise floor. HELM and large leaderboards
do this.

**Cost:** ~3× judge bill per run. Probably overkill for the memoria
unless you want to write specifically about judge-disagreement
analysis. Pairwise preference is a better marginal-effort upgrade.

## Recommended memoria structure

1. Lead with retrieval metrics. Multi-config table. State that these
   are the objective measurement.
2. Report LLM-judged Overall with the limitations paragraph. Cite
   Zheng et al.
3. Include the calibration-study sentence (κ, r) — even on n=20.
4. Optional: pairwise win-rate comparison of the two final configs.
5. Optional: RAGAS scores as a citable comparison signal.
6. Limitations section explicitly listing: judge bias, per-row variance,
   reference-document narrowness (single golden URL when corpus may
   contain equivalent content elsewhere), no human calibration on full
   set.

ADR-009 already has a Limitations section in this shape — reuse the
text in the memoria methodology chapter.

## Open follow-ups (none blocking)

- Calibration study on 20–30 rows.
- Pairwise-preference judge prompt + runner.
- RAGAS faithfulness on the final config.
- Per-row variance estimate: rerun the same eval 3× at temperature=0
  and compute SD of `overall` per row. Quantifies the noise floor with
  one number.

When you come back to this, the cheapest first move is the calibration
study — it's the one item that converts the absolute-score numbers from
"indicative" to "defensible." Pairwise preference is the second.

## References

- Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and
  Chatbot Arena.* NeurIPS 2023.
- Es, S. et al. (2023). *RAGAS: Automated Evaluation of Retrieval
  Augmented Generation.* arXiv:2309.15217.
- Liu, Y. et al. (2023). *G-Eval: NLG Evaluation using GPT-4 with
  Better Human Alignment.* EMNLP 2023.
- Chiang, C. & Lee, H. (2023). *Can Large Language Models Be an
  Alternative to Human Evaluations?* ACL 2023.
- Cohen, J. (1960). *A Coefficient of Agreement for Nominal Scales.*
  Educational and Psychological Measurement.
