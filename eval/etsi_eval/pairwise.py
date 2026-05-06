"""LLM-as-judge: pairwise preference between two candidate answers.

Given the same question, the same reference document(s), and two candidate
answers (typically from two configurations of the same agent), the judge
picks which answer better addresses the question. The verdict is a single
enum: ``A`` / ``B`` / ``tie``.

To control for positional bias the runner does TWO passes per pair: one
with answers in original order, one with answers swapped. ``grade_pair``
returns a ``PairVerdict`` whose ``winner`` is only ``A`` or ``B`` when both
passes agree on the same underlying answer; if the judge flips on swap the
verdict is ``unstable`` and excluded from the win-rate.

The shape is enforced server-side via ``response_format=json_schema``
(OpenAI structured outputs) and re-validated client-side by Pydantic. The
``winner`` field is enum-constrained to ``A``/``B``/``tie`` at both layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from .config import BackendConfig
from .db import ReferenceDocument


# ---- prompts -------------------------------------------------------------

JUDGE_SYSTEM_ANSWER = """You are an expert evaluator for a Spanish-language \
university question-answering system (Universidad de Málaga, ETSI Informática). \
You receive:

1. A student-style question in Spanish.
2. The reference document(s) that the answer should be grounded on.
3. Two candidate answers, labelled "Answer A" and "Answer B".

Pick which answer better addresses the question, taking into account
faithfulness to the reference, factual correctness, and completeness.

Reply with exactly one of:
- "A": Answer A is clearly better.
- "B": Answer B is clearly better.
- "tie": Both answers are equivalent in quality, OR neither answers the
  question, OR the difference is too small to call.

Reason briefly (one or two sentences) before committing to a winner.
The label A or B carries no information about which configuration produced
the answer — judge purely on the content."""

JUDGE_SYSTEM_REFUSE = """You are an expert evaluator for a Spanish-language \
university question-answering system (Universidad de Málaga, ETSI Informática). \
You receive:

1. A student-style question in Spanish whose answer is NOT in the system's
   knowledge base. The system SHOULD acknowledge that it doesn't know.
2. Two candidate answers, labelled "Answer A" and "Answer B".

There is no reference document for this row — the correct behaviour is for
the system to refuse, indicate lack of information, or clearly state it can
only answer questions about the ETSI Informática corpus.

Pick which answer better refused / acknowledged the limitation, penalising
fabrication and rewarding clear hand-off.

Reply with exactly one of:
- "A": Answer A handled the refusal better.
- "B": Answer B handled the refusal better.
- "tie": Both handled it equivalently, OR both fabricated, OR the
  difference is too small to call."""


# ---- result types --------------------------------------------------------


Winner = Literal["A", "B", "tie"]
PairWinner = Literal["A", "B", "tie", "unstable"]


@dataclass(frozen=True)
class Usage:
    """Token counts reported by the judge backend.

    Mirrors the OpenAI ``usage`` block. Backends that strip the block (rare,
    but possible behind a proxy) leave all fields at zero.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass(frozen=True)
class SingleVerdict:
    """Result of one judge call. ``winner`` is in the position frame: it
    refers to whichever answer was presented as ``Answer A`` in the prompt."""

    winner: Winner
    rationale: str
    raw: str
    usage: Usage = Usage()


@dataclass(frozen=True)
class PairVerdict:
    """Result of two judge calls (forward + swapped) on the same pair.

    ``winner`` is in the ORIGINAL frame (relative to the answers as passed
    to ``grade_pair``), not the position frame. ``A`` means the first
    argument; ``B`` means the second. ``unstable`` means the two passes
    disagreed about which underlying answer is better — the judge flipped
    on swap, which is the canonical positional-bias signature.
    """

    forward: SingleVerdict
    swapped: SingleVerdict

    @property
    def usage(self) -> Usage:
        return self.forward.usage + self.swapped.usage

    @property
    def consistent(self) -> bool:
        return self.winner != "unstable"

    @property
    def winner(self) -> PairWinner:
        # Re-map the swapped pass back into the original frame:
        # in the swapped pass, the position-A slot held the ORIGINAL B,
        # and vice versa, so a position-winner of "A" means original B won.
        forward = self.forward.winner
        swapped_remapped: Winner
        if self.swapped.winner == "A":
            swapped_remapped = "B"
        elif self.swapped.winner == "B":
            swapped_remapped = "A"
        else:
            swapped_remapped = "tie"

        if forward == swapped_remapped:
            return forward  # both agree: "A", "B", or "tie"
        return "unstable"


class JudgeError(RuntimeError):
    pass


# ---- structured-output schema --------------------------------------------


class _VerdictModel(BaseModel):
    winner: Winner
    rationale: str


# ---- helpers -------------------------------------------------------------


def _build_user_message(
    question: str,
    answer_a: str,
    answer_b: str,
    refs: list[ReferenceDocument],
) -> str:
    head = (
        f"Question:\n{question}\n\n"
        f"Answer A:\n{answer_a or '(empty)'}\n\n"
        f"Answer B:\n{answer_b or '(empty)'}\n\n"
    )
    if not refs:
        return (
            head
            + "There is no reference document — the system should have refused or "
            "indicated lack of information.\n\n"
            "Pick the better answer (A / B / tie)."
        )
    refs_block = "\n\n".join(
        f"Reference document {i + 1} of {len(refs)}:\n"
        f"Title: {ref.title}\nURL: {ref.url}\n---\n{ref.content}\n---"
        for i, ref in enumerate(refs)
    )
    return head + refs_block + "\n\nPick the better answer (A / B / tie)."


def _usage_from(completion) -> Usage:
    u = getattr(completion, "usage", None)
    if u is None:
        return Usage()
    return Usage(
        prompt_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(u, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(u, "total_tokens", 0) or 0),
    )


def _system_prompt(expected_behavior: str) -> str:
    if expected_behavior == "answer":
        return JUDGE_SYSTEM_ANSWER
    if expected_behavior == "refuse":
        return JUDGE_SYSTEM_REFUSE
    raise ValueError(f"unknown expected_behavior: {expected_behavior!r}")


# ---- public API ----------------------------------------------------------


def grade_single(
    backend: BackendConfig,
    model: str,
    question: str,
    answer_a: str,
    answer_b: str,
    refs: list[ReferenceDocument],
    *,
    expected_behavior: str = "answer",
    timeout: float = 120.0,
    temperature: float = 0.0,
    client: OpenAI | None = None,
) -> SingleVerdict:
    """One judge call. ``winner`` is in the position frame (A / B as presented)."""
    system_prompt = _system_prompt(expected_behavior)

    owns_client = client is None
    if owns_client:
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
                        "content": _build_user_message(question, answer_a, answer_b, refs),
                    },
                ],
                temperature=temperature,
                response_format=_VerdictModel,
            )
        except ValidationError as e:
            raise JudgeError(f"judge response failed schema validation (winner enum): {e}") from e
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

    return SingleVerdict(
        winner=msg.parsed.winner,
        rationale=msg.parsed.rationale.strip(),
        raw=msg.content or "",
        usage=_usage_from(completion),
    )


def grade_pair(
    backend: BackendConfig,
    model: str,
    question: str,
    answer_a: str,
    answer_b: str,
    refs: list[ReferenceDocument],
    *,
    expected_behavior: str = "answer",
    timeout: float = 120.0,
    temperature: float = 0.0,
    client: OpenAI | None = None,
) -> PairVerdict:
    """Two judge calls (forward + swapped). ``winner`` is in the original frame."""
    forward = grade_single(
        backend,
        model,
        question,
        answer_a,
        answer_b,
        refs,
        expected_behavior=expected_behavior,
        timeout=timeout,
        temperature=temperature,
        client=client,
    )
    swapped = grade_single(
        backend,
        model,
        question,
        answer_b,  # swapped
        answer_a,
        refs,
        expected_behavior=expected_behavior,
        timeout=timeout,
        temperature=temperature,
        client=client,
    )
    return PairVerdict(forward=forward, swapped=swapped)
