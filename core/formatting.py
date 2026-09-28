"""Centralized prompt and label formatting contracts for RLCD and GLiClass.

Guarantees identical formatting across training, inference, ONNX export,
and browser WebGPU/WASM runtimes.
"""

from __future__ import annotations

from typing import Sequence

from core.primitives import (
    Choice,
    INSUFFICIENT_EVIDENCE_DESC,
    INSUFFICIENT_EVIDENCE_ID,
    Noul,
    Query,
    Score,
)

LABEL_MARKER = "<<LABEL>>"
SEP_MARKER = "<<SEP>>"
LABEL_TOKEN_ID = 50368
SEP_TOKEN_ID = 50369

MAX_SUPPORTED_CANDIDATES = 25
MAX_SUBSTANTIVE_CANDIDATES = 24


class CapacityError(ValueError):
    """Raised when a query exceeds the maximum supported candidate capacity."""

    def __init__(self, count: int, max_allowed: int = MAX_SUPPORTED_CANDIDATES):
        super().__init__(
            f"Candidate count {count} exceeds maximum model capacity of {max_allowed}."
        )
        self.count = count
        self.max_allowed = max_allowed


INPUT_TEMPLATE: str = "Question: {question}\n\nContext:\n{context}"


def format_prompt(text: str, label_descriptions: Sequence[str]) -> str:
    """Format prompt with label prefix and separator according to GLiClass ModernBERT contract.

    Formula: <<LABEL>>desc1<<LABEL>>desc2<<SEP>>text
    """
    label_prefix = "".join(f"{LABEL_MARKER}{label}" for label in label_descriptions)
    return f"{label_prefix}{SEP_MARKER}{text}"


def build_model_input(
    question: str,
    context: str,
    label_descriptions: Sequence[str],
) -> str:
    """Build the single canonical model input string across Python and JS runtimes.

    Formula: <<LABEL>>desc1<<LABEL>>desc2<<SEP>>Question: {question}\n\nContext:\n{context}
    """
    if question:
        text = INPUT_TEMPLATE.format(question=question, context=context)
    else:
        text = context
    return format_prompt(text, label_descriptions)


CLAUSE_STARTERS: tuple[str, ...] = (
    "The",
    "This",
    "That",
    "These",
    "Those",
    "There",
    "No",
    "Level",
    "A",
    "An",
)


def frame_hypothesis(desc: str) -> str:
    """Frame an option description into a natural hypothesis for GLiClass / NLI.

    Deterministic rules:
    1. Descriptions starting with (The, This, That, These, Those, There, No, Level, A, An)
       with 4+ words pass through bare (they are already complete clauses).
    2. Descriptions whose first word is a capitalized 3rd-person present verb (ends in 's':
       Asks, Wants, States, etc., excluding clause starters) template as
       "The request " + lowercased description.
    3. Everything else defaults to "It is " + description.
    """
    desc = desc.strip()
    words = desc.split()
    if not words:
        return desc

    first_word = words[0]

    # Rule 1: Clause starters with 4+ words pass through bare
    if first_word in CLAUSE_STARTERS:
        if len(words) >= 4:
            return desc
        return f"It is {desc}"

    # Rule 2: 3rd-person present verb (capitalized, ends in 's', alpha, len >= 3)
    if (
        first_word[0].isupper()
        and first_word.endswith("s")
        and len(first_word) >= 3
        and first_word.isalpha()
    ):
        lowered = desc[0].lower() + desc[1:]
        return f"The request {lowered}"

    # Rule 3: Default prefix
    return f"It is {desc}"


def format_query(
    context: str,
    query: Query,
    smart_framing: bool = True,
) -> tuple[str, list[str], list[str]]:
    """Format query context and candidate labels.

    Returns:
        formatted_text: The input text combining context and question/proposition.
        candidate_labels: Natural-language label descriptions for GLiClass.
        candidate_ids: Corresponding stable option/level/noul IDs.
    """
    import logging

    if query.kind == "choice":
        options = list(query.options)
        if len(options) > MAX_SUBSTANTIVE_CANDIDATES:
            logging.warning(
                "Query options count %d exceeds MAX_SUBSTANTIVE_CANDIDATES %d; degrading by truncating.",
                len(options),
                MAX_SUBSTANTIVE_CANDIDATES,
            )
            options = options[:MAX_SUBSTANTIVE_CANDIDATES]
        formatted_text = f"Question: {query.question}\n\nContext:\n{context}"
        if smart_framing:
            labels = [frame_hypothesis(opt.description) for opt in options] + [
                INSUFFICIENT_EVIDENCE_DESC
            ]
        else:
            labels = [f"It is {opt.description}" for opt in options] + [
                INSUFFICIENT_EVIDENCE_DESC
            ]
        ids = [opt.id for opt in options] + [INSUFFICIENT_EVIDENCE_ID]
        return formatted_text, labels, ids

    if query.kind == "score":
        levels = list(query.levels)
        if len(levels) > MAX_SUBSTANTIVE_CANDIDATES:
            logging.warning(
                "Query levels count %d exceeds MAX_SUBSTANTIVE_CANDIDATES %d; degrading by truncating.",
                len(levels),
                MAX_SUBSTANTIVE_CANDIDATES,
            )
            levels = levels[:MAX_SUBSTANTIVE_CANDIDATES]
        formatted_text = f"Question: {query.question}\n\nContext:\n{context}"
        labels = [
            f"{lvl.description} (Value: {lvl.value})" for lvl in levels
        ] + [INSUFFICIENT_EVIDENCE_DESC]
        ids = [lvl.id for lvl in levels] + [INSUFFICIENT_EVIDENCE_ID]
        return formatted_text, labels, ids

    if query.kind == "noul":
        formatted_text = (
            f"Context:\n{context}\n\nEvaluate proposition: {query.proposition}"
        )
        labels = [
            f"true: {query.proposition}",
            f"false: not {query.proposition}",
            INSUFFICIENT_EVIDENCE_DESC,
        ]
        ids = ["true", "false", INSUFFICIENT_EVIDENCE_ID]
        return formatted_text, labels, ids

    raise TypeError(f"Unknown query kind: {query!r}")
