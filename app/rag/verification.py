"""Claim-level answer verification against the exact cited evidence passages.

This module has no dependency on the chat orchestrator. The generation callable
is injected so verification can be unit-tested without starting Ollama.
"""
from __future__ import annotations

import re

from app.core.config import settings

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\\s+")


def verify_claims(
    question: str,
    evidence: str,
    draft: str,
    diagnostics: dict | None = None,
    *,
    generate_fn,
    enabled: bool | None = None,
    max_tokens: int | None = None,
) -> bool:
    """Verify that every factual claim is answered by its own cited passages."""
    if enabled is None:
        enabled = settings.verify_answers
    if not enabled:
        return True

    blocks = re.split(r"\\n\\n(?=\\[\\d+\\])", evidence or "")
    by_number = {}
    for block in blocks:
        match = re.match(r"\\[(\\d+)\\]", block)
        if match:
            by_number[int(match.group(1))] = block

    claim_pairs = []
    for sentence in _SENTENCE_SPLIT.split(draft or ""):
        sentence = sentence.strip()
        if not sentence:
            continue
        claim_text = re.sub(r"\\[[^\\]\\n]*\\]", " ", sentence)
        if len(re.findall(r"[A-Za-z]{2,}", claim_text)) < 2:
            continue
        refs = {
            int(number)
            for group in re.findall(
                r"\\[\\s*(\\d{1,3}(?:\\s*,\\s*\\d{1,3})*)\\s*\\]", sentence
            )
            for number in group.split(",")
        }
        if not refs or any(number not in by_number for number in refs):
            if diagnostics is not None:
                diagnostics["verdict"] = "MISSING_OR_INVALID_CLAIM_CITATION"
            return False
        cited_blocks = [by_number[number] for number in sorted(refs)]
        claim_pairs.append(
            "CLAIM: " + sentence + "\\nONLY EVIDENCE CITED BY THIS CLAIM:\\n"
            + "\\n\\n".join(cited_blocks)
        )

    if not claim_pairs:
        if diagnostics is not None:
            diagnostics["verdict"] = "NO_CITED_FACTUAL_CLAIMS"
        return False

    verdict_prompt = (
        "Evaluate the QUESTION and every CLAIM-EVIDENCE PAIR below.\\n"
        "For EACH claim, use ONLY the passages listed under that claim. Do not borrow "
        "support from another claim's passages or general knowledge.\\n"
        "A claim is supported only when its cited passage(s) directly state it or clearly entail it. "
        "Faithful paraphrasing and synthesis across that claim's cited passages are allowed. Shared "
        "topic words, names, or chapter references alone are NOT support.\\n"
        "Also decide whether the claims answer the QUESTION rather than a related question. "
        "For questions asking for an explicit list, steps, or set, check that every distinct item in "
        "the cited passage is included; an incomplete list is not a complete answer. "
        "If any factual claim is unsupported, any citation points to a passage that does not support "
        "its claim, the requested list is incomplete, or the answer misses the question, reject it.\\n"
        "Reply with one of: SUPPORTED ANSWERED, SUPPORTED NOT ANSWERED, or UNSUPPORTED ANSWERED.\\n"
        f"QUESTION: {question}\\n\\n" + "\\n\\n".join(claim_pairs)
    )
    token_limit = settings.verify_max_tokens if max_tokens is None else max_tokens
    verdict = generate_fn(verdict_prompt, num_predict=token_limit).strip().upper()
    if diagnostics is not None:
        diagnostics["verdict"] = verdict[:120]
        diagnostics["claims_checked"] = len(claim_pairs)
    if "UNSUPPORTED" in verdict or "NOT SUPPORTED" in verdict or "NOT SUFFICIENT" in verdict:
        return False
    if "NOT ANSWERED" in verdict or "UNANSWERED" in verdict or "DOES NOT ANSWER" in verdict:
        return False
    return "SUPPORTED" in verdict and "ANSWERED" in verdict
