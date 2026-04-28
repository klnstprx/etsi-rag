"""LLM-as-judge: score an agent answer along three dimensions.

The judge evaluates each answer on three independent sub-dimensions:

- faithfulness: every claim in the answer is entailed by the reference doc(s).
- correctness:  claims that are made are factually accurate (vs. the reference).
- completeness: the answer addresses every part of the question with the key
                facts present.

`overall = min(faithfulness, correctness, completeness)`. Using min rather than
mean is the policy: an unfaithful answer is worthless even if complete; the
weakest dimension caps quality.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import httpx

from .config import BackendConfig
from .db import ReferenceDocument

_RESPONSE_SHAPE = """\
Respond with ONE JSON object and nothing else, in this exact shape:

{
  "faithfulness":  {"score": <int 0-100>, "rationale": "<one short sentence>"},
  "correctness":   {"score": <int 0-100>, "rationale": "<one short sentence>"},
  "completeness":  {"score": <int 0-100>, "rationale": "<one short sentence>"}
}
"""

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
should score low on completeness.

""" + _RESPONSE_SHAPE

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
        unsure whether the system knows or not.

""" + _RESPONSE_SHAPE


@dataclass(frozen=True)
class DimScore:
    score: int
    rationale: str


@dataclass(frozen=True)
class Verdict:
    faithfulness: DimScore
    correctness: DimScore
    completeness: DimScore
    raw: str  # full judge response for debugging malformed JSON

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


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

# Some models (observed with gemma4-e4b) emit Python-style escapes like \' inside
# JSON strings. \' is never a valid JSON escape, so rewriting it to a bare quote
# is safe and can't corrupt well-formed output.
_INVALID_ESCAPES = {
    r"\'": "'",
}

_DIMENSIONS = ("faithfulness", "correctness", "completeness")


def _sanitize_json(text: str) -> str:
    for bad, good in _INVALID_ESCAPES.items():
        text = text.replace(bad, good)
    return text


def _parse_dim(name: str, obj: dict) -> DimScore:
    sub = obj.get(name)
    if not isinstance(sub, dict):
        raise JudgeError(f"missing or malformed dimension {name!r}: {obj!r}")
    score = sub.get("score")
    if not isinstance(score, int) or not 0 <= score <= 100:
        raise JudgeError(
            f"dimension {name!r} score out of range or wrong type: {sub!r}"
        )
    rationale = str(sub.get("rationale", "")).strip()
    return DimScore(score=score, rationale=rationale)


def _parse_verdict(raw: str) -> Verdict:
    """Extract three-dim verdict even when the model wraps it in prose or fences."""
    text = raw.strip()
    # Strip ``` fences if present
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    m = _JSON_BLOCK.search(text)
    if not m:
        raise JudgeError(f"no JSON object in judge output: {raw!r}")
    candidate = _sanitize_json(m.group(0))
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise JudgeError(f"invalid JSON in judge output: {e}; raw={raw!r}") from e

    dims = {name: _parse_dim(name, obj) for name in _DIMENSIONS}
    return Verdict(
        faithfulness=dims["faithfulness"],
        correctness=dims["correctness"],
        completeness=dims["completeness"],
        raw=raw,
    )


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
) -> Verdict:
    if expected_behavior == "refuse":
        system_prompt = JUDGE_SYSTEM_REFUSE
    elif expected_behavior == "answer":
        system_prompt = JUDGE_SYSTEM_ANSWER
    else:
        raise ValueError(f"unknown expected_behavior: {expected_behavior!r}")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _build_user_message(question, agent_answer, refs)},
        ],
        "temperature": temperature,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if backend.api_key:
        headers["Authorization"] = f"Bearer {backend.api_key}"

    url = f"{backend.base_url}/chat/completions"
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        body = r.json()

    choices = body.get("choices") or []
    if not choices:
        raise JudgeError(f"judge returned no choices: {body}")
    content = choices[0].get("message", {}).get("content", "") or ""
    return _parse_verdict(content)
