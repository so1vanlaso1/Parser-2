from __future__ import annotations

"""Stage 2: generic logical-operator router.

The router chooses a skeleton kind and records evidence. It does not split text
into final predicates and it does not produce AST/solver output.
"""

import re
from dataclasses import dataclass, field
from typing import Any

try:
    from .connector_registry import DEFAULT_REGISTRY, ConnectorMatch
    from .logic_skeleton import SkeletonKind
except Exception:  # pragma: no cover
    from connector_registry import DEFAULT_REGISTRY, ConnectorMatch
    from logic_skeleton import SkeletonKind


@dataclass
class OperatorMatch:
    kind: SkeletonKind
    confidence: float
    risk_flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    matched_rule: str | None = None
    matched_evidence: dict[str, Any] | None = None


EXISTS_START_RE = re.compile(
    r"^(some|at\s+least\s+one|there\s+exists|there\s+is|one\s+or\s+more|a\s+certain|someone)\b",
    re.IGNORECASE,
)
FORALL_START_RE = re.compile(r"^(all|every|each|any|everyone|everybody|no|none\s+of|no\s+one|nobody)\b", re.IGNORECASE)
IF_RULE_START_RE = re.compile(r"^(if|when|whenever|provided\s+that|assuming\s+that|in\s+case)\b", re.IGNORECASE)

# Grammatical/event verbs used only for finding boundaries in generic relative
# clauses. This is not a domain-specific predicate list.
VERB_BOUNDARY_RE = re.compile(
    r"\b(is|are|was|were|be|being|been|has|have|had|do|does|did|can|could|may|might|"
    r"must|shall|should|will|would|receive|receives|get|gets|contain|contains|include|"
    r"includes|require|requires|need|needs|achieve|achieves|submit|submits|pass|passes|"
    r"graduate|graduates|complete|completes|attend|attends|ask|asks|provide|provides|"
    r"give|gives|lose|loses|gain|gains|enter|enters|obtain|obtains|qualify|qualifies|"
    r"allow|allows|enable|enables|lead|leads|cause|causes|succeed|succeeds|practice|"
    r"practices|access|accesses|return|returns|email|emails|motivate|motivates|support|supports)\b",
    re.IGNORECASE,
)
COPULAR_OR_HAVE_RE = re.compile(r"\b(is|are|was|were|has|have|had)\b", re.IGNORECASE)
PAST_TENSE_OR_EVENT_RE = re.compile(
    r"\b([A-Za-z]+ed|sent|paid|won|lost|made|gave|took|went|came|met|ran|ate|read|wrote|saw|left|kept)\b",
    re.IGNORECASE,
)


def classify_operator(text: str) -> OperatorMatch:
    normalized = _normalize_for_match(text)

    if not normalized:
        return _match("UNKNOWN", 0.0, rule="empty", flags=["empty_premise", "needs_review"], notes=["empty premise"])

    if _looks_structurally_invalid(normalized):
        return _match(
            "UNKNOWN",
            0.25,
            rule="structural_invalid",
            flags=["needs_review"],
            notes=["contains unsupported structural marker"],
            evidence={"rule_id": "structural_invalid", "details": {"reason": "unbalanced marker or parentheses"}},
        )

    # META first, but direct IFF remains IFF inside is_meta_like.
    meta_match = is_meta_like(normalized, return_match=True)
    if meta_match:
        evidence = _evidence(meta_match) if isinstance(meta_match, ConnectorMatch) else meta_match
        return _match("META", float(evidence.get("confidence", 0.88)), rule=str(evidence.get("rule_id", "meta")), flags=["meta_formula"], notes=["nested or formula-level logic cue detected"], evidence=evidence)

    iff = DEFAULT_REGISTRY.find("IFF", normalized)
    if iff:
        return _from_connector("IFF", iff)

    only_if = DEFAULT_REGISTRY.find("ONLY_IF_RULE", normalized)
    if only_if:
        return _from_connector("ONLY_IF_RULE", only_if)

    arrow = DEFAULT_REGISTRY.find("ARROW_RULE", normalized)
    if arrow:
        return _from_connector("RULE", arrow, flags=list(arrow.entry.risk_flags))

    # Modal before universal/rule so "not necessarily" is never classical NOT.
    modal = DEFAULT_REGISTRY.find("MODAL", normalized)
    if modal:
        return _from_connector("MODAL", modal)

    if _is_existential(normalized):
        return _match("EXISTS", 0.90, rule="quantifier.existential_start", evidence=_simple_evidence("quantifier.existential_start", normalized, 0, _first_word_end(normalized), 0.90))

    if _is_universal(normalized):
        flags = ["negative_quantifier"] if _starts_negative_universal(normalized) else []
        return _match("FORALL", 0.88, rule="quantifier.universal_start", flags=flags, evidence=_simple_evidence("quantifier.universal_start", normalized, 0, _first_word_end(normalized), 0.88))

    if _is_rule(normalized):
        evidence = _rule_evidence(normalized)
        return _match("RULE", 0.86, rule=str(evidence.get("rule_id", "conditional_or_relative_rule")), flags=list(evidence.get("risk_flags", [])), evidence=evidence)

    non_if = DEFAULT_REGISTRY.find("NON_IF_RULE", normalized)
    if non_if:
        return _from_connector("NON_IF_RULE", non_if)

    obligation = DEFAULT_REGISTRY.find("OBLIGATION_RULE", normalized)
    if obligation:
        return _from_connector("OBLIGATION_RULE", obligation, flags=[_deontic_flag(normalized, obligation.cue)])

    fact_evidence = _fact_evidence(normalized)
    if fact_evidence:
        return _match("FACT", float(fact_evidence.get("confidence", 0.76)), rule=str(fact_evidence.get("rule_id", "fact.named_or_specific_assertion")), evidence=fact_evidence)

    return _match("UNKNOWN", 0.35, rule="unknown.no_safe_match", flags=["needs_review"], notes=["no safe logical operator cue matched"], evidence={"rule_id": "unknown.no_safe_match", "details": {"reason": "no registry connector or safe assertion pattern matched"}})


def is_meta_like(text: str, *, return_match: bool = False) -> bool | ConnectorMatch | dict[str, Any]:
    normalized = _normalize_for_match(text)
    lower = normalized.lower()

    # Direct IFF is its own operator.
    if DEFAULT_REGISTRY.find("IFF", normalized):
        return False

    if re.search(r"\bit\s+is\s+not\s+true\s+that\s+if\b", lower):
        evidence = _simple_evidence("meta.negated_if_statement", normalized, 0, min(len(normalized), 32), 0.88)
        return evidence if return_match else True

    if re.search(r"\b(if|whether)\s+.+\s+(is\s+true|is\s+false|holds\s+true)\b", lower):
        evidence = _simple_evidence("meta.truth_value_of_formula", normalized, 0, len(normalized), 0.86)
        return evidence if return_match else True

    meta_cue = DEFAULT_REGISTRY.find("META_CUE", normalized)
    formula_cue = DEFAULT_REGISTRY.find("FORMULA_CONNECTOR", normalized)

    if "'" in normalized and meta_cue:
        evidence = _evidence(meta_cue)
        evidence["rule_id"] = "meta.quoted_formula_statement"
        evidence.setdefault("details", {})["quoted"] = True
        return evidence if return_match else True

    if meta_cue and meta_cue.cue in {"previous statement", "above implication", "holds true", "is true", "is false"}:
        evidence = _evidence(meta_cue)
        evidence["rule_id"] = "meta.explicit_statement_reference"
        return evidence if return_match else True

    if meta_cue and formula_cue:
        evidence = _evidence(meta_cue)
        evidence["rule_id"] = "meta.cue_plus_formula_connector"
        evidence.setdefault("details", {})["formula_connector"] = formula_cue.entry.id
        return evidence if return_match else True

    if re.search(r"\bthen\s+if\b", lower):
        pos = lower.find("then if")
        evidence = _simple_evidence("meta.then_if_nested", normalized, pos, pos + 7, 0.88)
        return evidence if return_match else True

    if len(re.findall(r"\bif\b", lower)) >= 2:
        evidence = _simple_evidence("meta.multiple_if_cues", normalized, 0, len(normalized), 0.86)
        return evidence if return_match else True

    if lower.startswith("if "):
        parts = re.split(r"\bthen\b", lower, maxsplit=1)
        if len(parts) == 2:
            antecedent = parts[0].removeprefix("if ").strip(" ,")
            consequent = parts[1].strip(" ,")
            if _fragment_is_formula_like(antecedent) or _fragment_is_formula_like(consequent):
                evidence = _simple_evidence("meta.if_contains_formula_fragment", normalized, 0, len(normalized), 0.88)
                evidence.setdefault("details", {})["antecedent_formula_like"] = _fragment_is_formula_like(antecedent)
                evidence.setdefault("details", {})["consequent_formula_like"] = _fragment_is_formula_like(consequent)
                return evidence if return_match else True

    return False


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _looks_structurally_invalid(text: str) -> bool:
    if "@@@" in text:
        return True
    return text.count("(") != text.count(")")


def _starts_negative_universal(text: str) -> bool:
    return bool(re.match(r"^(no|none\s+of|no\s+one|nobody)\b", text, flags=re.I))


def _is_existential(text: str) -> bool:
    return bool(EXISTS_START_RE.search(text)) and not _has_conditional_cue(text)


def _is_universal(text: str) -> bool:
    if not FORALL_START_RE.search(text):
        return False
    if DEFAULT_REGISTRY.find("OBLIGATION_RULE", text):
        return False
    if _has_conditional_cue(text):
        return False
    # Universal + relative clause should be represented as RULE.
    if _has_relative_clause_condition(text):
        return False
    return True


def _is_rule(text: str) -> bool:
    lower = text.lower()
    if IF_RULE_START_RE.search(text):
        return True
    if DEFAULT_REGISTRY.find("UNLESS", text):
        return True
    if DEFAULT_REGISTRY.find("ARROW_RULE", text):
        return True
    if re.search(r"\s+if\s+", text, re.IGNORECASE) and "only if" not in lower and "if and only if" not in lower:
        return True
    if re.search(r"\b(provided\s+that|assuming\s+that|in\s+case)\b", lower):
        return True
    return _has_relative_clause_condition(text)


def _has_conditional_cue(text: str) -> bool:
    lower = text.lower()
    if lower.startswith(("if ", "when ", "whenever ", "provided that ", "assuming that ", "in case ")):
        return True
    return bool(re.search(r"\b(provided\s+that|assuming\s+that|in\s+case|unless)\b", lower))


def _has_relative_clause_condition(text: str) -> bool:
    lower = text.lower()
    if re.search(r"\b(who|that|which|whose|with|without)\b", lower):
        return True
    return bool(re.search(r"^[a-z][a-z-]*s\s+(trained|given|having|receiving|using|requiring|lacking|failing)\b", lower))


def _fragment_is_formula_like(fragment: str) -> bool:
    lower = fragment.lower().strip()
    if lower.startswith(("if ", "when ", "whenever ", "provided that ", "assuming that ", "in case ")):
        return True
    if re.match(r"^(all|every|each|any|no|there\s+exists|there\s+is|at\s+least\s+one|some)\b", lower):
        return True
    if DEFAULT_REGISTRY.find("FORMULA_CONNECTOR", fragment):
        return True
    if _has_relative_clause_condition(fragment):
        return True
    return False


def _deontic_flag(text: str, cue: str | None = None) -> str:
    lower = text.lower()
    cue_lower = (cue or "").lower()
    if any(term in lower for term in ("forbidden", "prohibited", "not allowed to")) or cue_lower in {"forbidden", "prohibited", "not allowed to"}:
        return "deontic_prohibition"
    if any(term in lower for term in ("permitted", "allowed to")) or cue_lower in {"permitted", "allowed to"}:
        return "deontic_permission"
    return "deontic_obligation"


def _fact_evidence(text: str) -> dict[str, Any] | None:
    lower = text.lower()
    blocked_starts = (
        "all ", "every ", "each ", "some ", "at least one ", "there exists ", "there is ",
        "no ", "none of ", "if ", "when ", "whenever ", "provided that ", "assuming that ", "in case ",
        "lack of ", "failure to ",
    )
    if lower.startswith(blocked_starts):
        return None
    if DEFAULT_REGISTRY.find_any(("MODAL", "OBLIGATION_RULE", "NON_IF_RULE", "ONLY_IF_RULE", "IFF", "UNLESS", "ARROW_RULE"), text):
        return None
    if _has_conditional_cue(text) or _has_relative_clause_condition(text):
        return None

    words = text.rstrip(".!?").split()
    if len(words) < 2:
        return None

    # Named/specific assertion: Laura emailed..., John is certified.
    if _looks_like_named_subject(words):
        verb_idx = _first_event_or_copular_index(words, start=1, max_index=5)
        if verb_idx is not None:
            span = words[verb_idx]
            confidence = 0.80 if PAST_TENSE_OR_EVENT_RE.fullmatch(span.strip(",.;:")) else 0.76
            return _simple_evidence("fact.named_event_or_attribute_assertion", text, text.find(span), text.find(span) + len(span), confidence)

    # Definite specific past event: The committee approved the proposal.
    if words[0].lower() == "the":
        verb_idx = _first_event_index(words, start=1, max_index=6)
        if verb_idx is not None:
            span = words[verb_idx]
            return _simple_evidence("fact.definite_specific_event_assertion", text, text.find(span), text.find(span) + len(span), 0.72)

    return None


def _looks_like_named_subject(words: list[str]) -> bool:
    first = words[0].strip(",.;:")
    if not first or not first[0].isupper():
        return False
    if first.lower() in {"all", "every", "each", "some", "there", "if", "when", "whenever", "provided", "assuming", "in", "no"}:
        return False
    return True


def _first_event_or_copular_index(words: list[str], *, start: int, max_index: int) -> int | None:
    for idx in range(start, min(len(words), max_index + 1)):
        word = words[idx].strip(",.;:")
        if COPULAR_OR_HAVE_RE.fullmatch(word) or PAST_TENSE_OR_EVENT_RE.fullmatch(word):
            return idx
    return None


def _first_event_index(words: list[str], *, start: int, max_index: int) -> int | None:
    for idx in range(start, min(len(words), max_index + 1)):
        word = words[idx].strip(",.;:")
        if PAST_TENSE_OR_EVENT_RE.fullmatch(word):
            return idx
    return None


def _rule_evidence(text: str) -> dict[str, Any]:
    lower = text.lower()
    for cue in ("if", "when", "whenever", "provided that", "assuming that", "in case", "unless"):
        pos = lower.find(cue)
        if pos >= 0:
            return _simple_evidence(f"rule.conditional_cue.{cue.replace(' ', '_')}", text, pos, pos + len(cue), 0.86)
    arrow = DEFAULT_REGISTRY.find("ARROW_RULE", text)
    if arrow:
        return _evidence(arrow)
    rel_match = re.search(r"\b(who|that|which|whose|with|without)\b", lower)
    if rel_match:
        return _simple_evidence("rule.relative_clause_condition", text, rel_match.start(), rel_match.end(), 0.84)
    return {"rule_id": "rule.conditional_or_relative_rule", "confidence": 0.82, "details": {}}


def _first_word_end(text: str) -> int:
    match = re.search(r"\s", text)
    return match.start() if match else len(text)


def _from_connector(kind: SkeletonKind, conn: ConnectorMatch, *, flags: list[str] | None = None) -> OperatorMatch:
    entry = conn.entry
    risk_flags = list(flags) if flags is not None else list(entry.risk_flags)
    return _match(kind, entry.confidence, rule=entry.id, flags=risk_flags, notes=list(entry.notes), evidence=_evidence(conn))


def _match(kind: SkeletonKind, confidence: float, *, rule: str | None = None, flags: list[str] | None = None, notes: list[str] | None = None, evidence: dict[str, Any] | None = None) -> OperatorMatch:
    return OperatorMatch(kind=kind, confidence=confidence, risk_flags=flags or [], notes=notes or [], matched_rule=rule, matched_evidence=evidence)


def _evidence(conn: ConnectorMatch) -> dict[str, Any]:
    evidence = conn.evidence()
    evidence.setdefault("details", {})
    return evidence


def _simple_evidence(rule_id: str, text: str, start: int, end: int, confidence: float) -> dict[str, Any]:
    start = max(0, start)
    end = min(len(text), max(start, end))
    return {
        "rule_id": rule_id,
        "cue": text[start:end] or None,
        "span": text[start:end] or None,
        "start": start,
        "end": end,
        "confidence": confidence,
        "direction": None,
        "consequent_negation": None,
        "details": {},
    }


def needs_review_notes(note: str) -> list[str]:
    return [note]


__all__ = ["OperatorMatch", "classify_operator", "is_meta_like", "needs_review_notes"]
