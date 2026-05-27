from __future__ import annotations

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
]


@dataclass(frozen=True)
class ConnectorEntry:
    """Generic connector metadata used by both routing and skeleton splitting.

    The registry is intentionally the only place where lexical cues live. The
    router and builder consume these entries; they should not maintain separate
    cue lists.
    """

    id: str
    kind: ConnectorKind
    cue: str
    confidence: float
    direction: str = "left_to_right"
    consequent_negation: bool = False
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
            "details": {"groups": dict(self.groups)},
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
            if best is None:
                best = match
                continue
            # Prefer the earliest cue; if tied, prefer the longer cue.
            if (match.start, -len(match.span)) < (best.start, -len(best.span)):
                best = match
        return best

    def find_any(self, kinds: tuple[ConnectorKind, ...], text: str) -> ConnectorMatch | None:
        matches = [self.find(kind, text) for kind in kinds]
        matches = [match for match in matches if match is not None]
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
            groups = {key: _clean_part(value) for key, value in match.groupdict().items() if value is not None}
            antecedent = groups.get("antecedent") or groups.get("left")
            consequent = groups.get("consequent") or groups.get("right")
            if not antecedent or not consequent:
                continue
            cue_start, cue_end = _cue_span_from_match(entry, clean, match)
            conn_match = ConnectorMatch(
                entry=entry,
                span=clean[cue_start:cue_end],
                start=cue_start,
                end=cue_end,
                groups=groups,
            )
            if entry.id in {"non_if.causes_loss_of", "non_if.causes_revocation_of", "non_if.causes_removal_of"}:
                consequent = f"{entry.cue.removeprefix('causes ')} {consequent}"
            if entry.cue.lower() == "can increase":
                consequent = f"increase {consequent}"
            return _clean_part(antecedent), _clean_part(consequent), conn_match
        return None


def default_connector_registry() -> ConnectorRegistry:
    entries: list[ConnectorEntry] = []

    def add(kind: ConnectorKind, cue: str, *, id: str | None = None, confidence: float = 0.8,
            direction: str = "left_to_right", consequent_negation: bool = False,
            risk_flags: tuple[str, ...] = (), notes: tuple[str, ...] = (),
            pattern: str | None = None, split_pattern: str | None = None) -> None:
        entry_id = id or f"{kind.lower()}.{cue.replace(' ', '_')}"
        entries.append(
            ConnectorEntry(
                id=entry_id,
                kind=kind,
                cue=cue,
                confidence=confidence,
                direction=direction,
                consequent_negation=consequent_negation,
                risk_flags=risk_flags,
                notes=notes,
                pattern=pattern,
                split_pattern=split_pattern,
            )
        )

    for cue in ("if and only if", "exactly when", "precisely when", "iff"):
        add("IFF", cue, confidence=0.95)

    for cue in ("only if", "only when", "depends on", "is required for", "is necessary for", "requires"):
        add("ONLY_IF_RULE", cue, confidence=0.92, risk_flags=("only_if_direction",))

    add("NON_IF_RULE", "causes revocation of", id="non_if.causes_revocation_of", confidence=0.86,
        consequent_negation=True, risk_flags=("non_if_rule", "negative_consequence"))
    add("NON_IF_RULE", "causes removal of", id="non_if.causes_removal_of", confidence=0.86,
        consequent_negation=True, risk_flags=("non_if_rule", "negative_consequence"))
    add("NON_IF_RULE", "causes loss of", id="non_if.causes_loss_of", confidence=0.86,
        consequent_negation=True, risk_flags=("non_if_rule", "negative_consequence"))
    add("NON_IF_RULE", "disqualifies from", id="non_if.disqualifies_from", confidence=0.86,
        consequent_negation=True, risk_flags=("non_if_rule", "negative_consequence"),
        pattern=r"\bdisqualifies\b.+?\bfrom\b",
        split_pattern=r"^(?P<antecedent>.+?)\s+disqualifies\s+.+?\s+from\s+(?P<consequent>.+)$")
    for cue in ("prevents", "blocks"):
        add("NON_IF_RULE", cue, confidence=0.86, consequent_negation=True,
            risk_flags=("non_if_rule", "negative_consequence"))

    for cue in (
        "results in", "leads to", "guarantees", "enables", "allows", "causes",
        "grants", "grant", "ensures", "fosters", "helps", "help", "encourages", "encourage",
        "can increase",
    ):
        add("NON_IF_RULE", cue, confidence=0.84, risk_flags=("non_if_rule",))

    for cue in ("not necessarily", "can potentially", "could potentially", "can sometimes", "possibly",
                "probably", "likely", "might", "could", "may"):
        flags = ("modal_uncertainty",)
        if cue == "not necessarily":
            flags = ("modal_not_necessarily", "modal_uncertainty")
        if cue == "may":
            flags = (*flags, "modal_may_ambiguous")
        add("MODAL", cue, confidence=0.78, risk_flags=flags,
            notes=("modal cue is not classical negation",))

    for cue in (
        "not allowed to", "required to", "obligated to", "prohibited", "forbidden", "mandatory",
        "permitted", "allowed to", "have to", "must", "shall",
    ):
        add("OBLIGATION_RULE", cue, confidence=0.82, risk_flags=("deontic_obligation",))

    for cue in (
        "rule", "claim", "statement", "policy", "principle", "compliance", "previous statement",
        "above implication", "holds true",
    ):
        add("META_CUE", cue, confidence=0.88, risk_flags=("meta_formula",))

    for cue in ("implies", "requires", "depends on", "results in", "leads to", "causes", "grants",
                "grant", "allows", "enables", "ensures", "guarantees", "prevents", "blocks",
                "disqualifies from", "fosters", "helps", "help", "encourages", "encourage",
                "can increase", "enforces"):
        add("FORMULA_CONNECTOR", cue, confidence=0.82)

    return ConnectorRegistry(entries)


DEFAULT_REGISTRY = default_connector_registry()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _clean_part(text: str) -> str:
    return _normalize(text).strip(" ,;:")


def _match_entry(entry: ConnectorEntry, text: str) -> ConnectorMatch | None:
    pattern = entry.pattern or _cue_pattern(entry.cue)
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    groups = {key: _clean_part(value) for key, value in match.groupdict().items() if value is not None}
    return ConnectorMatch(
        entry=entry,
        span=match.group(0),
        start=match.start(),
        end=match.end(),
        groups=groups,
    )


def _cue_pattern(cue: str) -> str:
    escaped = re.escape(cue)
    escaped = escaped.replace(r"\ ", r"\s+")
    return rf"(?<!\w){escaped}(?!\w)"


def _default_split_pattern(cue: str) -> str:
    escaped = re.escape(cue).replace(r"\ ", r"\s+")
    return rf"^(?P<antecedent>.+?)\s+{escaped}\s+(?P<consequent>.+)$"


def _cue_span_from_match(entry: ConnectorEntry, text: str, split_match: re.Match[str]) -> tuple[int, int]:
    # The split pattern may capture more than the literal cue for regex connectors
    # like "disqualifies ... from". Use the routing pattern to recover a precise
    # evidence span when possible.
    route_match = _match_entry(entry, text)
    if route_match:
        return route_match.start, route_match.end
    cue_match = re.search(_cue_pattern(entry.cue), text, flags=re.IGNORECASE)
    if cue_match:
        return cue_match.start(), cue_match.end()
    return split_match.start(), split_match.end()
