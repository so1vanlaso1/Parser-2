from __future__ import annotations

import re
from dataclasses import dataclass, field

try:
    from .logic_skeleton import SkeletonKind
except ImportError:  # pragma: no cover - supports direct script execution.
    from logic_skeleton import SkeletonKind


@dataclass
class OperatorMatch:
    kind: SkeletonKind
    confidence: float
    risk_flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    matched_rule: str | None = None


IFF_CUES = (
    "if and only if",
    "iff",
    "exactly when",
    "precisely when",
)

ONLY_IF_CUES = (
    "only if",
    "only when",
    "depends on",
    "is required for",
    "is necessary for",
    "requires",
)

EXISTS_START_RE = re.compile(
    r"^(some|at\s+least\s+one|there\s+exists|there\s+is|one\s+or\s+more|"
    r"a\s+certain|someone)\b",
    re.IGNORECASE,
)

FORALL_START_RE = re.compile(r"^(all|every|each|any|everyone)\b", re.IGNORECASE)

IF_RULE_START_RE = re.compile(
    r"^(if|when|whenever|provided\s+that|assuming\s+that|in\s+case)\b",
    re.IGNORECASE,
)

NON_IF_CONNECTORS = (
    "causes revocation of",
    "causes removal of",
    "disqualifies from",
    "causes loss of",
    "results in",
    "leads to",
    "guarantees",
    "prevents",
    "ensures",
    "enables",
    "allows",
    "blocks",
    "causes",
    "grants",
)

OBLIGATION_CUES = (
    "not allowed to",
    "required to",
    "obligated to",
    "prohibited",
    "forbidden",
    "mandatory",
    "permitted",
    "allowed to",
    "have to",
    "must",
    "shall",
)

MODAL_CUES = (
    "not necessarily",
    "can sometimes",
    "possibly",
    "probably",
    "likely",
    "might",
    "could",
    "may",
)

FACT_VERB_RE = re.compile(
    r"\b(is|are|was|were|has|have|had|does|do|did|achieves?|receives?|gets?|"
    r"contains?|includes?|belongs?|owns?|knows?)\b",
    re.IGNORECASE,
)


def classify_operator(text: str) -> OperatorMatch:
    normalized = _normalize_for_match(text)

    if not normalized:
        return OperatorMatch(
            kind="UNKNOWN",
            confidence=0.0,
            risk_flags=["empty_premise", "needs_review"],
            notes=["empty premise"],
            matched_rule="empty",
        )

    if _looks_structurally_invalid(normalized):
        return OperatorMatch(
            kind="UNKNOWN",
            confidence=0.25,
            risk_flags=["needs_review"],
            notes=["contains unsupported structural marker"],
            matched_rule="structural_invalid",
        )

    if is_meta_like(normalized):
        return OperatorMatch(
            kind="META",
            confidence=0.88,
            risk_flags=["meta_formula"],
            notes=["nested or formula-level logic cue detected"],
            matched_rule="meta",
        )

    if _contains_any(normalized, IFF_CUES):
        return OperatorMatch(
            kind="IFF",
            confidence=0.95,
            matched_rule="iff",
        )

    only_if_rule = _match_only_if(normalized)
    if only_if_rule:
        return OperatorMatch(
            kind="ONLY_IF_RULE",
            confidence=0.92,
            risk_flags=["only_if_direction"],
            matched_rule=only_if_rule,
        )

    if _is_existential(normalized):
        return OperatorMatch(
            kind="EXISTS",
            confidence=0.9,
            matched_rule="existential",
        )

    if _is_universal(normalized):
        return OperatorMatch(
            kind="FORALL",
            confidence=0.88,
            matched_rule="universal",
        )

    if _is_rule(normalized):
        return OperatorMatch(
            kind="RULE",
            confidence=0.86,
            matched_rule="conditional_or_relative_rule",
        )

    non_if = _match_non_if_connector(normalized)
    if non_if:
        return OperatorMatch(
            kind="NON_IF_RULE",
            confidence=0.84,
            risk_flags=["non_if_rule"],
            matched_rule=non_if,
        )

    obligation = _match_obligation(normalized)
    if obligation:
        return OperatorMatch(
            kind="OBLIGATION_RULE",
            confidence=0.82,
            risk_flags=[_deontic_flag(normalized)],
            matched_rule=obligation,
        )

    modal = _match_modal(normalized)
    if modal:
        flags = ["modal_uncertainty"]
        if "not necessarily" in normalized.lower():
            flags.insert(0, "modal_not_necessarily")
        if modal == "may":
            flags.append("modal_may_ambiguous")
        return OperatorMatch(
            kind="MODAL",
            confidence=0.78,
            risk_flags=flags,
            notes=["modal cue is not classical negation"],
            matched_rule=modal,
        )

    if _is_safe_fact(normalized):
        return OperatorMatch(
            kind="FACT",
            confidence=0.74,
            matched_rule="simple_assertion",
        )

    return OperatorMatch(
        kind="UNKNOWN",
        confidence=0.35,
        risk_flags=["needs_review"],
        notes=["no safe logical operator cue matched"],
        matched_rule="unknown",
    )


def is_meta_like(text: str) -> bool:
    normalized = _normalize_for_match(text)
    lower = normalized.lower()

    if any(re.search(rf"\b{re.escape(cue)}\b", lower) for cue in IFF_CUES):
        return False

    if re.search(r"\bit\s+is\s+not\s+true\s+that\s+if\b", lower):
        return True

    if re.search(
        r"\b(the\s+)?(rule|claim|statement)\s+that\b|"
        r"\bthe\s+previous\s+statement\b|"
        r"\bthe\s+above\s+implication\b|"
        r"\bholds\s+true\b",
        lower,
    ):
        return True

    if re.search(r"\bthen\s+if\b", lower):
        return True

    if len(re.findall(r"\bif\b", lower)) >= 2:
        return True

    if lower.startswith("if "):
        parts = re.split(r"\bthen\b", lower, maxsplit=1)
        if len(parts) == 2:
            antecedent = parts[0]
            consequent = parts[1]
            formula_connectors = (
                " implies ",
                " leads to ",
                " requires ",
                " depends on ",
                " results in ",
                " causes ",
                " grants ",
                " allows ",
                " enables ",
                " ensures ",
                " guarantees ",
                " prevents ",
                " blocks ",
                " disqualifies from ",
            )
            if any(connector in antecedent for connector in formula_connectors):
                return True
            if consequent.strip().startswith("if "):
                return True

    if re.search(
        r"^if\s+(there\s+exists|there\s+is|at\s+least\s+one|some|all|every)\b"
        r".+\bthen\s+if\b",
        lower,
    ):
        return True

    return False


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _looks_structurally_invalid(text: str) -> bool:
    if "@@@" in text:
        return True
    return text.count("(") != text.count(")")


def _contains_any(text: str, cues: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(re.search(rf"\b{re.escape(cue)}\b", lower) for cue in cues)


def _match_only_if(text: str) -> str | None:
    lower = text.lower()
    for cue in ONLY_IF_CUES:
        if re.search(rf"\b{re.escape(cue)}\b", lower):
            return cue
    return None


def _is_existential(text: str) -> bool:
    if _has_conditional_cue(text):
        return False
    return bool(EXISTS_START_RE.search(text))


def _is_universal(text: str) -> bool:
    if not FORALL_START_RE.search(text):
        return False
    if _match_modal(text) or _match_obligation(text):
        return False
    if _has_relative_clause_condition(text):
        return False
    if _has_conditional_cue(text):
        return False
    return True


def _is_rule(text: str) -> bool:
    if IF_RULE_START_RE.search(text):
        return True
    if re.search(r"\s+if\s+", text, re.IGNORECASE):
        lower = text.lower()
        if "only if" not in lower and "if and only if" not in lower:
            return True
    return _has_relative_clause_condition(text)


def _match_non_if_connector(text: str) -> str | None:
    lower = text.lower()
    for connector in NON_IF_CONNECTORS:
        if re.search(rf"\b{re.escape(connector)}\b", lower):
            return connector
    return None


def _match_obligation(text: str) -> str | None:
    lower = text.lower()
    for cue in OBLIGATION_CUES:
        if re.search(rf"\b{re.escape(cue)}\b", lower):
            return cue
    return None


def _match_modal(text: str) -> str | None:
    lower = text.lower()
    for cue in MODAL_CUES:
        if re.search(rf"\b{re.escape(cue)}\b", lower):
            return cue
    return None


def _has_conditional_cue(text: str) -> bool:
    lower = text.lower()
    if lower.startswith(("if ", "when ", "whenever ")):
        return True
    return bool(re.search(r"\b(provided\s+that|assuming\s+that|in\s+case)\b", lower))


def _has_relative_clause_condition(text: str) -> bool:
    lower = text.lower()
    if re.search(r"\b(who|that|which|whose|with|without)\b", lower):
        return True
    return bool(
        re.search(
            r"^[a-z][a-z-]*s\s+"
            r"(trained|given|having|receiving|using|requiring|lacking|failing)\b",
            lower,
        )
    )


def _deontic_flag(text: str) -> str:
    lower = text.lower()
    if any(cue in lower for cue in ("forbidden", "prohibited", "not allowed to")):
        return "deontic_prohibition"
    if any(cue in lower for cue in ("permitted", "allowed to")):
        return "deontic_permission"
    return "deontic_obligation"


def _is_safe_fact(text: str) -> bool:
    lower = text.lower()
    blocked_starts = (
        "all ",
        "every ",
        "each ",
        "some ",
        "at least one ",
        "there exists ",
        "there is ",
        "if ",
        "when ",
        "whenever ",
    )
    if lower.startswith(blocked_starts):
        return False
    if _match_modal(text) or _match_obligation(text):
        return False
    return bool(FACT_VERB_RE.search(text))


def needs_review_notes(note: str) -> list[str]:
    return [note]
