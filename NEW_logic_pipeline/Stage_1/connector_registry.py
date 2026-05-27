from __future__ import annotations

"""Generic lexical connector registry for Stage 1.

All reusable connector cues live here so router/builder do not maintain
separate hardcoded cue lists. These are logical/grammatical cues, not dataset
specific premise meanings.
"""

import re
from dataclasses import dataclass, field
from typing import Literal


ConnectorKind = Literal[
    "IFF",
    "ONLY_IF_RULE",
    "NON_IF_RULE",
    "OBLIGATION_RULE",
    "MODAL",
    "META_CUE",
    "FORMULA_CONNECTOR",
    "UNLESS",
    "ARROW_RULE",
]


@dataclass(frozen=True)
class ConnectorEntry:
    id: str
    kind: ConnectorKind
    cue: str
    confidence: float
    direction: str = "left_to_right"
    consequent_negation: bool = False
    antecedent_negation: bool = False
    risk_flags: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    pattern: str | None = None
    split_pattern: str | None = None


@dataclass(frozen=True)
class ConnectorMatch:
    entry: ConnectorEntry
    span: str
    start: int
    end: int
    groups: dict[str, str] = field(default_factory=dict)

    @property
    def cue(self) -> str:
        return self.entry.cue

    @property
    def rule_id(self) -> str:
        return self.entry.id

    def evidence(self) -> dict[str, object]:
        return {
            "rule_id": self.entry.id,
            "cue": self.entry.cue,
            "span": self.span,
            "start": self.start,
            "end": self.end,
            "confidence": self.entry.confidence,
            "direction": self.entry.direction,
            "consequent_negation": self.entry.consequent_negation,
            "details": {
                "groups": dict(self.groups),
                "antecedent_negation": self.entry.antecedent_negation,
            },
        }


class ConnectorRegistry:
    def __init__(self, entries: list[ConnectorEntry]):
        self.entries = sorted(entries, key=lambda item: len(item.cue), reverse=True)

    def by_kind(self, kind: ConnectorKind) -> list[ConnectorEntry]:
        return [entry for entry in self.entries if entry.kind == kind]

    def find(self, kind: ConnectorKind, text: str) -> ConnectorMatch | None:
        clean = _normalize(text)
        best: ConnectorMatch | None = None
        for entry in self.by_kind(kind):
            match = _match_entry(entry, clean)
            if not match:
                continue
            if best is None or (match.start, -len(match.span)) < (best.start, -len(best.span)):
                best = match
        return best

    def find_any(self, kinds: tuple[ConnectorKind, ...], text: str) -> ConnectorMatch | None:
        matches = [self.find(kind, text) for kind in kinds]
        matches = [m for m in matches if m is not None]
        if not matches:
            return None
        return sorted(matches, key=lambda item: (item.start, -len(item.span)))[0]

    def split_non_if(self, text: str) -> tuple[str, str, ConnectorMatch] | None:
        clean = _normalize(text).rstrip(".!?")
        for entry in self.by_kind("NON_IF_RULE"):
            pattern = entry.split_pattern or _default_split_pattern(entry.cue)
            match = re.match(pattern, clean, flags=re.IGNORECASE)
            if not match:
                continue
            groups = {k: _clean_part(v) for k, v in match.groupdict().items() if v is not None}
            antecedent = groups.get("antecedent") or groups.get("left")
            consequent = groups.get("consequent") or groups.get("right")
            if not antecedent or not consequent:
                continue
            cue_start, cue_end = _cue_span_from_match(entry, clean, match)
            conn_match = ConnectorMatch(entry=entry, span=clean[cue_start:cue_end], start=cue_start, end=cue_end, groups=groups)

            # Preserve the semantic noun in "causes loss/removal/revocation of X".
            if entry.id in {"non_if.causes_loss_of", "non_if.causes_revocation_of", "non_if.causes_removal_of"}:
                consequent = f"{entry.cue.removeprefix('causes ')} {consequent}"
            if entry.cue.lower() == "can increase":
                consequent = f"increase {consequent}"
            return _clean_part(antecedent), _clean_part(consequent), conn_match
        return None

    def split_arrow(self, text: str) -> tuple[str, str, ConnectorMatch] | None:
        clean = _normalize(text).rstrip(".!?")
        for entry in self.by_kind("ARROW_RULE"):
            match = re.match(entry.split_pattern or r"^(?P<antecedent>.+?)\s*(?:->|=>|→)\s*(?P<consequent>.+)$", clean)
            if not match:
                continue
            groups = {k: _clean_part(v) for k, v in match.groupdict().items() if v is not None}
            antecedent = groups.get("antecedent")
            consequent = groups.get("consequent")
            if not antecedent or not consequent:
                continue
            span_match = re.search(r"->|=>|→", clean)
            start, end = (span_match.start(), span_match.end()) if span_match else (0, 0)
            return antecedent, consequent, ConnectorMatch(entry=entry, span=clean[start:end], start=start, end=end, groups=groups)
        return None


def default_connector_registry() -> ConnectorRegistry:
    entries: list[ConnectorEntry] = []

    def add(
        kind: ConnectorKind,
        cue: str,
        *,
        id: str | None = None,
        confidence: float = 0.8,
        direction: str = "left_to_right",
        consequent_negation: bool = False,
        antecedent_negation: bool = False,
        risk_flags: tuple[str, ...] = (),
        notes: tuple[str, ...] = (),
        pattern: str | None = None,
        split_pattern: str | None = None,
    ) -> None:
        entry_id = id or f"{kind.lower()}.{cue.replace(' ', '_')}"
        entries.append(ConnectorEntry(entry_id, kind, cue, confidence, direction, consequent_negation, antecedent_negation, risk_flags, notes, pattern, split_pattern))

    for cue in ("if and only if", "exactly when", "precisely when", "iff"):
        add("IFF", cue, confidence=0.95, risk_flags=("biconditional",))

    for cue in ("only if", "only when", "depends on", "is required for", "is necessary for", "requires"):
        add("ONLY_IF_RULE", cue, confidence=0.92, risk_flags=("only_if_direction",))

    add("ARROW_RULE", "->", confidence=0.93, risk_flags=("symbolic_rule",), pattern=r"->|=>|→")
    add("ARROW_RULE", "=>", confidence=0.93, risk_flags=("symbolic_rule",), pattern=r"->|=>|→")
    add("ARROW_RULE", "→", confidence=0.93, risk_flags=("symbolic_rule",), pattern=r"->|=>|→")

    # Negative-result connectors.
    add("NON_IF_RULE", "causes revocation of", id="non_if.causes_revocation_of", confidence=0.86, consequent_negation=True, risk_flags=("non_if_rule", "negative_consequence"))
    add("NON_IF_RULE", "causes removal of", id="non_if.causes_removal_of", confidence=0.86, consequent_negation=True, risk_flags=("non_if_rule", "negative_consequence"))
    add("NON_IF_RULE", "causes loss of", id="non_if.causes_loss_of", confidence=0.86, consequent_negation=True, risk_flags=("non_if_rule", "negative_consequence"))
    add("NON_IF_RULE", "disqualifies from", id="non_if.disqualifies_from", confidence=0.86, consequent_negation=True, risk_flags=("non_if_rule", "negative_consequence"), pattern=r"\bdisqualifies\b.+?\bfrom\b", split_pattern=r"^(?P<antecedent>.+?)\s+disqualifies\s+.+?\s+from\s+(?P<consequent>.+)$")
    for cue in ("prevents", "blocks", "forbids", "prohibits"):
        add("NON_IF_RULE", cue, confidence=0.85, consequent_negation=True, risk_flags=("non_if_rule", "negative_consequence"))

    # Positive-result connectors. This is a generic registry. Additions here are
    # logical relation verbs, not topic-specific shortcuts.
    for cue in (
        "results in", "leads to", "guarantees", "enables", "allows", "causes", "grants",
        "grant", "ensures", "fosters", "helps", "help", "encourages", "encourage",
        "can increase", "promotes", "produces", "yields", "triggers",
    ):
        add("NON_IF_RULE", cue, confidence=0.82, risk_flags=("non_if_rule",))

    for cue in ("unless",):
        add("UNLESS", cue, confidence=0.80, antecedent_negation=True, risk_flags=("unless_rule",))

    for cue in (
        "not necessarily", "can potentially", "could potentially", "can sometimes", "possibly",
        "probably", "likely", "might", "could", "may",
    ):
        flags = ("modal_uncertainty",)
        if cue == "not necessarily":
            flags = ("modal_not_necessarily", "modal_uncertainty")
        if cue == "may":
            flags = (*flags, "modal_may_ambiguous")
        add("MODAL", cue, confidence=0.76, risk_flags=flags, notes=("modal cue is not classical negation",))

    for cue in (
        "not allowed to", "required to", "obligated to", "prohibited", "forbidden", "mandatory",
        "permitted", "allowed to", "have to", "must", "shall",
    ):
        add("OBLIGATION_RULE", cue, confidence=0.82, risk_flags=("deontic_obligation",))

    for cue in (
        "rule", "claim", "statement", "policy", "principle", "compliance", "previous statement",
        "above implication", "holds true", "is true", "is false",
    ):
        add("META_CUE", cue, confidence=0.86, risk_flags=("meta_formula",))

    for cue in (
        "implies", "requires", "depends on", "results in", "leads to", "causes", "grants",
        "grant", "allows", "enables", "ensures", "guarantees", "prevents", "blocks",
        "disqualifies from", "fosters", "helps", "help", "encourages", "encourage",
        "can increase", "enforces", "promotes", "produces", "yields", "triggers",
    ):
        add("FORMULA_CONNECTOR", cue, confidence=0.82)

    return ConnectorRegistry(entries)


DEFAULT_REGISTRY = default_connector_registry()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _clean_part(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip(" \t\r\n,;:.!?")


def _match_entry(entry: ConnectorEntry, text: str) -> ConnectorMatch | None:
    pattern = entry.pattern or _cue_pattern(entry.cue)
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    groups = {key: _clean_part(value) for key, value in match.groupdict().items() if value is not None}
    return ConnectorMatch(entry=entry, span=match.group(0), start=match.start(), end=match.end(), groups=groups)


def _cue_pattern(cue: str) -> str:
    # Symbolic cues are not word-boundary tokens.
    if cue in {"->", "=>", "→"}:
        return r"->|=>|→"
    parts = [re.escape(part) for part in cue.split()]
    return r"\b" + r"\s+".join(parts) + r"\b"


def _default_split_pattern(cue: str) -> str:
    cue_pattern = r"\s+".join(re.escape(part) for part in cue.split())
    return rf"^(?P<antecedent>.+?)\s+{cue_pattern}\s+(?P<consequent>.+)$"


def _cue_span_from_match(entry: ConnectorEntry, text: str, split_match: re.Match[str]) -> tuple[int, int]:
    pattern = entry.pattern or _cue_pattern(entry.cue)
    cue_match = re.search(pattern, text, flags=re.IGNORECASE)
    if cue_match:
        return cue_match.start(), cue_match.end()
    return split_match.start(), split_match.end()


__all__ = [
    "ConnectorKind",
    "ConnectorEntry",
    "ConnectorMatch",
    "ConnectorRegistry",
    "default_connector_registry",
    "DEFAULT_REGISTRY",
]
