# Eval methodology: known limits and where to take it

A note to revisit. Captures the discussion that happened around ADR-009's
reranker sweep, where the LLM-judged Overall metric was flat-to-noisy
while objective retrieval metrics moved cleanly. The headline question
was: how trustworthy is the eval, and what would it take to defend the
numbers in the memoria.

## What the harness measures today

> **Update 2026-05-06:** per-row LLM-as-judge grading was retired from the
> regular runner. Pairwise (§2 below) is now the only judge-graded
> surface. The runner records evidence; pairwise decides direction.

`eval/etsi_eval/runner.py` produces, per row:

- **Retrieval metrics (objective):** `recall_at_5`, `recall_at_10`,
  `mrr`. Set-membership of `golden_urls` against the agent's retrieved
  URL list. No LLM involved.
- **Agent answer text** — the artifact that pairwise compares.
- **Tool calls** — name, args, error, duration_ms per invocation. Lets
  us attribute lifts to specific tools (e.g. did `read_document` fire
  on the rows where we saw improvement?).
- **Errored rows count** and per-stage error messages.

Schema in `eval/etsi_eval/report.py`. Output JSON schema_version 3.

Per-dimension scores (`faithfulness`, `correctness`, `completeness`,
`overall`) and per-category / per-difficulty breakdowns are no longer
emitted — they were within-noise for the kind of comparisons we run.

## Three structural problems with the LLM-judged half

1. **Judge bias.** GPT-class judges score their own family's outputs
   higher (self-preference, ~10–30 points in published studies); reward
   fluent answers over factually weaker ones; conflate confidence with
   correctness. See Zheng et al., _Judging LLM-as-a-Judge with MT-Bench
   and Chatbot Arena_ (NeurIPS 2023).
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

| Agent   | Answer                                                    | Score |
| ------- | --------------------------------------------------------- | ----- |
| e4b     | "Manuel Nicolás Enciso García-Oliveros" + multiple phones | 72    |
| 26b-a4b | "Ernesto Pimentel Sánchez" + one phone                    | 5     |

Both wrong against the gold; the 67-point gap is judge artefact (the
e4b answer happened to contain phone-number tokens that partially
matched the reference). This row's −67 weight in the aggregate
attributed to "the bigger agent regressed" was a pure measurement
error. Per-row variance like this is the rule, not the exception.

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
- ~~Pairwise-preference judge prompt + runner.~~ Shipped 2026-05-06.
- RAGAS faithfulness on the final config.
- Per-row variance estimate: rerun the same eval 3× at temperature=0
  and compute SD of `overall` per row. Quantifies the noise floor with
  one number.

The remaining cheap-and-defensible move is the calibration study (κ, r
on 20–30 hand-annotated rows) — it's the one item that converts the
absolute-score numbers from "indicative" to "defensible." Pairwise is
already in place; use it for every retrieval/agent change worth a
memoria mention.

## References

- Zheng, L. et al. (2023). _Judging LLM-as-a-Judge with MT-Bench and
  Chatbot Arena._ NeurIPS 2023.
- Es, S. et al. (2023). _RAGAS: Automated Evaluation of Retrieval
  Augmented Generation._ arXiv:2309.15217.
- Liu, Y. et al. (2023). _G-Eval: NLG Evaluation using GPT-4 with
  Better Human Alignment._ EMNLP 2023.
- Chiang, C. & Lee, H. (2023). _Can Large Language Models Be an
  Alternative to Human Evaluations?_ ACL 2023.
- Cohen, J. (1960). _A Coefficient of Agreement for Nominal Scales._
  Educational and Psychological Measurement.
