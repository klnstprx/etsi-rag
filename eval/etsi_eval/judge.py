"""LLM-as-judge: score an agent answer along three dimensions.

The judge evaluates each answer on three independent sub-dimensions:

- faithfulness: every claim in the answer is entailed by the reference doc(s).
- correctness:  claims that are made are factually accurate (vs. the reference).
- completeness: the answer addresses every part of the question with the key
                facts present.

`overall = min(faithfulness, correctness, completeness)`. Using min rather than
mean is the policy: an unfaithful answer is worthless even if complete; the
weakest dimension caps quality.

The shape is enforced server-side via `response_format=json_schema` (OpenAI
structured outputs) and re-validated client-side by Pydantic. There is no
free-text JSON parsing fallback: malformed output from the model is
unreachable when both ends honor the schema.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError, field_validator

from .config import BackendConfig
from .db import ReferenceDocument


JUDGE_SYSTEM_ANSWER = """You are an expert evaluator for a Spanish-language \
university question-answering system (Universidad de Málaga, ETSI Informática). \
You receive:

1. A student-style question in Spanish.
2. The system's answer (may be in Spanish or mixed).
3. The reference document(s) that the answer should be grounded on.

Score the answer on THREE independent dimensions, each on an integer 0-100 scale.
Reason briefly before each score (one sentence is enough).

Dimensions:
- "faithfulness": Is every claim in the answer entailed by the reference document(s)?
  100 = no claim is fabricated, every assertion can be checked against the reference.
  0   = the answer is largely fabricated or contradicts the reference.
- "correctness": Of the claims that are made, are they factually accurate
  relative to the reference?
  100 = every fact that appears is correct.
  0   = the named facts (names, numbers, dates, codes) are wrong.
- "completeness": Does the answer address every part of the question with the
  key facts that the reference provides?
  100 = covers the full scope of the question with the relevant facts.
  0   = empty, refusal, or misses the question entirely.

If the reference document does not actually contain the information needed,
score what you can and note it in the rationale. Refusals or empty outputs
should score low on completeness."""

JUDGE_SYSTEM_REFUSE = """You are an expert evaluator for a Spanish-language \
university question-answering system (Universidad de Málaga, ETSI Informática). \
You receive:

1. A student-style question in Spanish whose answer is NOT in the system's
   knowledge base. The system SHOULD acknowledge that it doesn't know.
2. The system's answer (may be in Spanish or mixed).

There is no reference document for this row — the correct behaviour is for
the system to refuse, indicate lack of information, or clearly state it can
only answer questions about the ETSI Informática corpus.

Score the answer on THREE independent dimensions, each on an integer 0-100 scale.

Dimensions (refusal-mode rubric):
- "faithfulness": Did the answer avoid fabricating content?
  100 = no concrete claims were made, OR only claims a reasonable assistant
        could make without the corpus (e.g. apologising, suggesting where
        else to look).
  0   = the answer fabricated facts that look like they came from the corpus.
- "correctness": Did the system correctly recognise that it should NOT answer?
  100 = explicit refusal / acknowledgement that it doesn't have the
        information / clear hand-off.
  0   = answered as if it had the information (which is a hallucination,
        because it doesn't).
- "completeness": Did the system clearly communicate the limitation to the user?
  100 = the user can tell from the answer that the system doesn't know.
  0   = empty, vague hedging, or generic non-answer that leaves the user
        unsure whether the system knows or not."""


# ---- result types --------------------------------------------------------


@dataclass(frozen=True)
class DimScore:
    score: int
    rationale: str


@dataclass(frozen=True)
class Usage:
    """Token counts reported by the judge backend on a single call.

    Mirrors the OpenAI `usage` block. Backends that don't return usage (rare,
    but possible behind a proxy that strips it) leave all fields at zero.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class Verdict:
    faithfulness: DimScore
    correctness: DimScore
    completeness: DimScore
    raw: str  # full judge response for debugging
    usage: Usage = Usage()

    @property
    def overall(self) -> int:
        """Policy: overall = min across dimensions. See module docstring."""
        return min(
            self.faithfulness.score,
            self.correctness.score,
            self.completeness.score,
        )


class JudgeError(RuntimeError):
    pass


# ---- structured-output schema --------------------------------------------


class _DimScoreModel(BaseModel):
    score: int
    rationale: str

    @field_validator("score")
    @classmethod
    def _score_range(cls, v: int) -> int:
        if not 0 <= v <= 100:
            raise ValueError(f"score must be 0-100, got {v}")
        return v


class _VerdictModel(BaseModel):
    faithfulness: _DimScoreModel
    correctness: _DimScoreModel
    completeness: _DimScoreModel


# ---- helpers -------------------------------------------------------------


def _build_user_message(
    question: str,
    agent_answer: str,
    refs: list[ReferenceDocument],
) -> str:
    answer_block = (
        f"Question:\n{question}\n\n"
        f"System's answer:\n{agent_answer or '(empty)'}\n\n"
    )
    if not refs:
        # Refuse-mode: no reference documents; the system should not have
        # answered the question with content.
        return (
            answer_block
            + "There is no reference document — the system should have refused or "
            "indicated lack of information.\n\n"
            "Rate the system's answer on faithfulness, correctness, and completeness."
        )
    refs_block = "\n\n".join(
        f"Reference document {i + 1} of {len(refs)}:\n"
        f"Title: {ref.title}\nURL: {ref.url}\n---\n{ref.content}\n---"
        for i, ref in enumerate(refs)
    )
    return (
        answer_block
        + refs_block
        + "\n\nRate the system's answer on faithfulness, correctness, and completeness."
    )


def _usage_from(completion) -> Usage:
    u = getattr(completion, "usage", None)
    if u is None:
        return Usage()
    return Usage(
        prompt_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(u, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(u, "total_tokens", 0) or 0),
    )


def _to_verdict(parsed: _VerdictModel, raw: str, usage: Usage) -> Verdict:
    return Verdict(
        faithfulness=DimScore(
            score=parsed.faithfulness.score,
            rationale=parsed.faithfulness.rationale.strip(),
        ),
        correctness=DimScore(
            score=parsed.correctness.score,
            rationale=parsed.correctness.rationale.strip(),
        ),
        completeness=DimScore(
            score=parsed.completeness.score,
            rationale=parsed.completeness.rationale.strip(),
        ),
        raw=raw,
        usage=usage,
    )


# ---- public API ----------------------------------------------------------


def grade(
    backend: BackendConfig,
    model: str,
    question: str,
    agent_answer: str,
    refs: list[ReferenceDocument],
    *,
    expected_behavior: str = "answer",
    timeout: float = 120.0,
    temperature: float = 0.0,
    client: OpenAI | None = None,
) -> Verdict:
    if expected_behavior == "refuse":
        system_prompt = JUDGE_SYSTEM_REFUSE
    elif expected_behavior == "answer":
        system_prompt = JUDGE_SYSTEM_ANSWER
    else:
        raise ValueError(f"unknown expected_behavior: {expected_behavior!r}")

    owns_client = client is None
    if owns_client:
        # Some local backends (llama-server) accept any non-empty key.
        client = OpenAI(
            base_url=backend.base_url,
            api_key=backend.api_key or "sk-no-key-required",
            timeout=timeout,
        )
    try:
        try:
            completion = client.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": _build_user_message(question, agent_answer, refs),
                    },
                ],
                temperature=temperature,
                response_format=_VerdictModel,
            )
        except ValidationError as e:
            # Server returned JSON that didn't match the schema (e.g. score
            # out of range). Surface as JudgeError so the runner records it
            # like any other judge failure rather than crashing the run.
            raise JudgeError(f"judge response failed schema validation: {e}") from e
        except OpenAIError as e:
            raise JudgeError(f"judge call failed: {e}") from e
    finally:
        if owns_client:
            client.close()

    msg = completion.choices[0].message
    if getattr(msg, "refusal", None):
        raise JudgeError(f"judge refused to grade: {msg.refusal}")
    if msg.parsed is None:
        raise JudgeError(f"judge returned no parsed content: raw={msg.content!r}")

    return _to_verdict(msg.parsed, msg.content or "", _usage_from(completion))
