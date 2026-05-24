from __future__ import annotations

import re

try:
    from .logic_skeleton import FormulaSkeleton, LogicSkeleton, SkeletonKind, TextSpan
    from .operator_router import OperatorMatch, classify_operator, is_meta_like
except ImportError:  # pragma: no cover - supports direct script execution.
    from logic_skeleton import FormulaSkeleton, LogicSkeleton, SkeletonKind, TextSpan
    from operator_router import OperatorMatch, classify_operator, is_meta_like


IF_STARTERS = (
    "if",
    "when",
    "whenever",
    "provided that",
    "assuming that",
    "in case",
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

NEGATIVE_CONNECTORS = {"prevents", "blocks", "disqualifies from"}

VERB_BOUNDARIES = (
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "receive",
    "receives",
    "get",
    "gets",
    "contain",
    "contains",
    "include",
    "includes",
    "require",
    "requires",
    "need",
    "needs",
    "achieve",
    "achieves",
    "submit",
    "submits",
    "pass",
    "passes",
    "graduate",
    "graduates",
    "complete",
    "completes",
    "attend",
    "attends",
    "ask",
    "asks",
    "provide",
    "provides",
    "give",
    "gives",
    "lose",
    "loses",
    "gain",
    "gains",
    "enter",
    "enters",
    "obtain",
    "obtains",
    "qualify",
    "qualifies",
    "allow",
    "allows",
    "enable",
    "enables",
    "lead",
    "leads",
    "cause",
    "causes",
)

IRREGULAR_PLURALS = {
    "people": "person",
    "children": "child",
    "men": "man",
    "women": "woman",
}


def build_skeleton(premise_id: str, text: str) -> LogicSkeleton:
    original = normalize_whitespace(text)
    clean = strip_terminal_punctuation(original)
    match = classify_operator(original)

    if match.kind == "FACT":
        return _base_skeleton(
            premise_id,
            original,
            match,
            body=_span("body", clean),
        )

    if match.kind == "EXISTS":
        body = _strip_exists_cue(clean)
        return _base_skeleton(
            premise_id,
            original,
            match,
            body=_span("body", body),
            quantifier="exists",
        )

    if match.kind == "FORALL":
        antecedent, consequent = split_universal(clean)
        return _base_skeleton(
            premise_id,
            original,
            match,
            antecedent=_span("restrictor", antecedent),
            consequent=_span("property", consequent),
            quantifier="forall",
        )

    if match.kind == "RULE":
        parts = _split_rule(clean)
        if parts is None:
            return _unknown_skeleton(
                premise_id,
                original,
                match,
                "rule_split_failed",
                "classified as RULE but no safe split was found",
            )
        antecedent, consequent = parts
        return _base_skeleton(
            premise_id,
            original,
            match,
            antecedent=_span("antecedent", antecedent),
            consequent=_span("consequent", consequent),
            quantifier="forall",
        )

    if match.kind == "ONLY_IF_RULE":
        parts = split_only_if(clean)
        if parts is None:
            return _unknown_skeleton(
                premise_id,
                original,
                match,
                "only_if_split_failed",
                "classified as ONLY_IF_RULE but no safe split was found",
            )
        antecedent, consequent = parts
        return _base_skeleton(
            premise_id,
            original,
            match,
            antecedent=_span("antecedent", antecedent),
            consequent=_span("consequent", consequent),
            quantifier="forall",
        )

    if match.kind == "IFF":
        parts = split_iff(clean)
        if parts is None:
            return _unknown_skeleton(
                premise_id,
                original,
                match,
                "iff_split_failed",
                "classified as IFF but no safe split was found",
            )
        left, right = parts
        return _base_skeleton(
            premise_id,
            original,
            match,
            left=_span("left", left),
            right=_span("right", right),
            quantifier="forall",
        )

    if match.kind == "NON_IF_RULE":
        parts = split_non_if_connector(clean)
        if parts is None:
            return _unknown_skeleton(
                premise_id,
                original,
                match,
                "non_if_split_failed",
                "classified as NON_IF_RULE but no safe split was found",
            )
        antecedent, consequent, connector = parts
        consequent_span = _span("consequent", consequent)
        if connector in NEGATIVE_CONNECTORS:
            consequent_span.negation_hint = True
        return _base_skeleton(
            premise_id,
            original,
            match,
            antecedent=_span("antecedent", antecedent),
            consequent=consequent_span,
            quantifier="forall",
        )

    if match.kind == "OBLIGATION_RULE":
        return _base_skeleton(
            premise_id,
            original,
            match,
            body=_span("body", clean),
            needs_review=True,
        )

    if match.kind == "MODAL":
        return _base_skeleton(
            premise_id,
            original,
            match,
            body=_span("body", clean),
            needs_review=True,
        )

    if match.kind == "META":
        try:
            formula_tree = build_meta_formula_tree(clean)
            return _base_skeleton(
                premise_id,
                original,
                match,
                formula_tree=formula_tree,
                needs_review=False,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback.
            skeleton = _base_skeleton(
                premise_id,
                original,
                match,
                body=_span("body", clean),
                needs_review=True,
            )
            skeleton.risk_flags = _append_unique(skeleton.risk_flags, "meta_parse_failed")
            skeleton.notes.append(f"meta parse failed: {exc}")
            return skeleton

    return _unknown_skeleton(
        premise_id,
        original,
        match,
        "needs_review",
        "premise could not be safely classified",
    )


def build_skeletons(premises: list[str]) -> list[LogicSkeleton]:
    return [build_skeleton(f"P{idx}", premise) for idx, premise in enumerate(premises, start=1)]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_terminal_punctuation(text: str) -> str:
    return normalize_whitespace(text).rstrip(" \t\r\n.!?")


def singularize_subject(subject: str) -> str:
    cleaned = normalize_whitespace(subject)
    cleaned = re.sub(r"^(all|every|each|any|some|at\s+least\s+one)\s+", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^(a|an|the)\s+", "", cleaned, flags=re.I)
    if not cleaned:
        return cleaned

    words = cleaned.split()
    last = words[-1]
    lower_last = last.lower()

    if lower_last in IRREGULAR_PLURALS:
        words[-1] = _preserve_case(last, IRREGULAR_PLURALS[lower_last])
    elif lower_last.endswith("ies") and len(lower_last) > 3:
        words[-1] = last[:-3] + "y"
    elif lower_last.endswith(("sses", "ches", "shes", "xes", "zes")):
        words[-1] = last[:-2]
    elif lower_last.endswith("s") and not lower_last.endswith("ss"):
        words[-1] = last[:-1]

    return " ".join(words)


def split_if_then(text: str) -> tuple[str, str] | None:
    clean = strip_terminal_punctuation(text)
    match = re.match(r"^if\s+(.+?)\s*,?\s*then\s+(.+)$", clean, flags=re.I)
    if not match:
        return None
    return _clean_part(match.group(1)), _clean_part(match.group(2))


def split_if_comma(text: str) -> tuple[str, str] | None:
    clean = strip_terminal_punctuation(text)
    match = re.match(r"^if\s+(.+?),\s*(.+)$", clean, flags=re.I)
    if not match:
        return None
    return _clean_part(match.group(1)), _clean_part(match.group(2))


def split_a_if_b(text: str) -> tuple[str, str] | None:
    clean = strip_terminal_punctuation(text)
    lower = clean.lower()
    if lower.startswith("if ") or "only if" in lower or "if and only if" in lower:
        return None
    match = re.match(r"^(.+?)\s+if\s+(.+)$", clean, flags=re.I)
    if not match:
        return None
    return _clean_part(match.group(2)), _clean_part(match.group(1))


def split_only_if(text: str) -> tuple[str, str] | None:
    clean = strip_terminal_punctuation(text)

    for cue in ("only if", "only when"):
        match = re.match(rf"^(.+?)\s+{re.escape(cue)}\s+(.+)$", clean, flags=re.I)
        if match:
            return _clean_part(match.group(1)), _clean_part(match.group(2))

    match = re.match(r"^(.+?)\s+requires\s+(.+)$", clean, flags=re.I)
    if match:
        return _clean_part(match.group(1)), _clean_part(match.group(2))

    match = re.match(r"^(.+?)\s+depends\s+on\s+(.+)$", clean, flags=re.I)
    if match:
        return _clean_part(match.group(1)), _clean_part(match.group(2))

    match = re.match(r"^(.+?)\s+is\s+required\s+for\s+(.+)$", clean, flags=re.I)
    if match:
        return _clean_part(match.group(2)), _clean_part(match.group(1))

    match = re.match(r"^(.+?)\s+is\s+necessary\s+for\s+(.+)$", clean, flags=re.I)
    if match:
        return _clean_part(match.group(2)), _clean_part(match.group(1))

    return None


def split_iff(text: str) -> tuple[str, str] | None:
    clean = strip_terminal_punctuation(text)
    for cue in ("if and only if", "exactly when", "precisely when", "iff"):
        match = re.match(rf"^(.+?)\s+{re.escape(cue)}\s+(.+)$", clean, flags=re.I)
        if match:
            return _clean_part(match.group(1)), _clean_part(match.group(2))
    return None


def split_non_if_connector(text: str) -> tuple[str, str, str] | None:
    clean = strip_terminal_punctuation(text)
    for connector in NON_IF_CONNECTORS:
        match = re.match(rf"^(.+?)\s+{re.escape(connector)}\s+(.+)$", clean, flags=re.I)
        if match:
            consequent = _clean_part(match.group(2))
            if connector in {"causes loss of", "causes revocation of", "causes removal of"}:
                consequent = f"{connector.removeprefix('causes ')} {consequent}"
            return _clean_part(match.group(1)), consequent, connector
    return None


def detect_negation_hint(text: str) -> bool:
    lower = normalize_whitespace(text).lower()
    return bool(
        re.search(
            r"\b(no|not|never|without|cannot|can't|does\s+not|do\s+not|did\s+not|"
            r"fails?\s+to|lack\s+of|lacks?)\b",
            lower,
        )
    )


def detect_modality_hint(text: str) -> str | None:
    lower = normalize_whitespace(text).lower()
    if "not necessarily" in lower:
        return "not_necessarily"
    for cue in ("possibly", "probably", "likely", "might", "may", "could", "can sometimes"):
        if re.search(rf"\b{re.escape(cue)}\b", lower):
            return cue.replace(" ", "_")
    return None


def build_meta_formula_tree(text: str) -> FormulaSkeleton:
    clean = strip_terminal_punctuation(text)

    negated_if = re.match(r"^it\s+is\s+not\s+true\s+that\s+(.+)$", clean, flags=re.I)
    if negated_if:
        return FormulaSkeleton(
            type="not",
            children=[parse_formula_part(negated_if.group(1), "x")],
        )

    parts = split_if_then(clean) or split_if_comma(clean)
    if parts:
        antecedent, consequent = parts
        return FormulaSkeleton(
            type="implies",
            children=[
                parse_formula_part(antecedent, "x"),
                parse_formula_part(consequent, "y"),
            ],
        )

    return parse_formula_part(clean, "x")


def parse_formula_part(text: str, variable: str) -> FormulaSkeleton:
    clean = strip_terminal_punctuation(text)

    parts = split_if_then(clean) or split_if_comma(clean)
    if parts:
        antecedent, consequent = parts
        return _quantified_implies(
            variable,
            parse_formula_part(antecedent, variable),
            parse_formula_part(consequent, variable),
        )

    connector_parts = _split_formula_connector(clean)
    if connector_parts:
        antecedent, consequent = connector_parts
        return _quantified_implies(
            variable,
            _leaf(antecedent, variable),
            _leaf(consequent, variable),
        )

    if _starts_existential(clean):
        return FormulaSkeleton(
            type="exists",
            variable=variable,
            children=[_leaf(_strip_exists_cue(clean), variable)],
        )

    if _starts_universal(clean):
        antecedent, consequent = split_universal(clean)
        return _quantified_implies(
            variable,
            _leaf(antecedent, variable),
            _leaf(consequent, variable),
        )

    relative_parts = split_relative_clause_rule(clean)
    if relative_parts:
        antecedent, consequent = relative_parts
        return _quantified_implies(
            variable,
            _leaf(antecedent, variable),
            _leaf(consequent, variable),
        )

    return _leaf(clean, variable)


def split_universal(text: str) -> tuple[str, str]:
    clean = strip_terminal_punctuation(text)
    rest = re.sub(r"^(all|every|each|any|everyone)\s+", "", clean, count=1, flags=re.I)
    subject, predicate = _split_subject_predicate(rest)
    return _indefinite_subject(subject), _clean_part(predicate)


def split_relative_clause_rule(text: str) -> tuple[str, str] | None:
    clean = strip_terminal_punctuation(text)
    match = re.match(
        r"^(?P<subject>.+?)\s+(?P<rel>who|that|which)\s+"
        r"(?P<condition>.+?)\s+(?P<verb>"
        + "|".join(re.escape(verb) for verb in VERB_BOUNDARIES)
        + r")\s+(?P<rest>.+)$",
        clean,
        flags=re.I,
    )
    if match:
        subject = singularize_subject(match.group("subject"))
        antecedent = f"{_indefinite_subject(subject)} {_normalize_relative_condition(match.group('condition'))}"
        consequent = _subject_predicate(subject, match.group("verb"), match.group("rest"))
        return _clean_part(antecedent), _clean_part(consequent)

    match = re.match(
        r"^(?P<subject>[A-Za-z][A-Za-z-]*s|people|children|men|women)\s+"
        r"(?P<condition>(?:trained|given|having|receiving|using|requiring|lacking|failing)"
        r"\b.+?)\s+(?P<verb>"
        + "|".join(re.escape(verb) for verb in VERB_BOUNDARIES)
        + r")\s+(?P<rest>.+)$",
        clean,
        flags=re.I,
    )
    if match:
        subject = singularize_subject(match.group("subject"))
        antecedent = f"{_indefinite_subject(subject)} {_normalize_relative_condition(match.group('condition'))}"
        consequent = _subject_predicate(subject, match.group("verb"), match.group("rest"))
        return _clean_part(antecedent), _clean_part(consequent)

    return None


def _split_rule(text: str) -> tuple[str, str] | None:
    for splitter in (
        split_if_then,
        split_if_comma,
        _split_when_rule,
        split_a_if_b,
        split_relative_clause_rule,
    ):
        parts = splitter(text)
        if parts:
            return parts
    return None


def _split_when_rule(text: str) -> tuple[str, str] | None:
    clean = strip_terminal_punctuation(text)
    match = re.match(
        r"^(when|whenever|provided\s+that|assuming\s+that|in\s+case)\s+(.+?),\s*(.+)$",
        clean,
        flags=re.I,
    )
    if match:
        return _clean_part(match.group(2)), _clean_part(match.group(3))

    match = re.match(
        r"^(when|whenever|provided\s+that|assuming\s+that|in\s+case)\s+(.+?)\s+(.+)$",
        clean,
        flags=re.I,
    )
    if match:
        return _clean_part(match.group(2)), _clean_part(match.group(3))
    return None


def _split_formula_connector(text: str) -> tuple[str, str] | None:
    for connector in ("implies", "requires", "depends on", *NON_IF_CONNECTORS):
        match = re.match(rf"^(.+?)\s+{re.escape(connector)}\s+(.+)$", text, flags=re.I)
        if match:
            consequent = _clean_part(match.group(2))
            if connector in {"causes loss of", "causes revocation of", "causes removal of"}:
                consequent = f"{connector.removeprefix('causes ')} {consequent}"
            if connector in NEGATIVE_CONNECTORS:
                consequent = f"not {consequent}"
            return _clean_part(match.group(1)), consequent
    return None


def _split_subject_predicate(text: str) -> tuple[str, str]:
    words = normalize_whitespace(text).split()
    if len(words) <= 1:
        return text, ""

    for idx, word in enumerate(words):
        if idx == 0:
            continue
        if word.lower().strip(",") in VERB_BOUNDARIES:
            return " ".join(words[:idx]), " ".join(words[idx:])

    return words[0], " ".join(words[1:])


def _strip_exists_cue(text: str) -> str:
    clean = strip_terminal_punctuation(text)
    replacements = (
        (r"^there\s+exists\s+at\s+least\s+one\s+", "a "),
        (r"^there\s+exists\s+", ""),
        (r"^there\s+is\s+at\s+least\s+one\s+", "a "),
        (r"^there\s+is\s+", ""),
        (r"^at\s+least\s+one\s+", "a "),
        (r"^one\s+or\s+more\s+", "a "),
        (r"^a\s+certain\s+", "a "),
        (r"^some\s+", "a "),
        (r"^someone\s+", "a person "),
    )
    for pattern, replacement in replacements:
        if re.match(pattern, clean, flags=re.I):
            return _clean_part(re.sub(pattern, replacement, clean, count=1, flags=re.I))
    return clean


def _starts_existential(text: str) -> bool:
    return bool(
        re.match(
            r"^(some|at\s+least\s+one|there\s+exists|there\s+is|one\s+or\s+more|"
            r"a\s+certain|someone)\b",
            text,
            flags=re.I,
        )
    )


def _starts_universal(text: str) -> bool:
    return bool(re.match(r"^(all|every|each|any|everyone)\b", text, flags=re.I))


def _indefinite_subject(subject: str) -> str:
    clean = singularize_subject(subject)
    if not clean:
        return clean
    if re.match(r"^(a|an|the)\s+", clean, flags=re.I):
        return clean
    article = "an" if clean[0].lower() in "aeiou" else "a"
    return f"{article} {clean}"


def _span(role: str, text: str, variable: str = "x") -> TextSpan:
    clean = _clean_part(text)
    return TextSpan(
        role=role,
        text=clean,
        variable=variable,
        negation_hint=detect_negation_hint(clean),
        modality_hint=detect_modality_hint(clean),
    )


def _leaf(text: str, variable: str) -> FormulaSkeleton:
    return FormulaSkeleton(type="leaf", text=_clean_part(text), variable=variable)


def _quantified_implies(
    variable: str,
    antecedent: FormulaSkeleton,
    consequent: FormulaSkeleton,
) -> FormulaSkeleton:
    return FormulaSkeleton(
        type="forall",
        variable=variable,
        children=[
            FormulaSkeleton(
                type="implies",
                children=[antecedent, consequent],
            )
        ],
    )


def _base_skeleton(
    premise_id: str,
    original: str,
    match: OperatorMatch,
    *,
    body: TextSpan | None = None,
    antecedent: TextSpan | None = None,
    consequent: TextSpan | None = None,
    left: TextSpan | None = None,
    right: TextSpan | None = None,
    formula_tree: FormulaSkeleton | None = None,
    quantifier: str | None = None,
    needs_review: bool = False,
) -> LogicSkeleton:
    return LogicSkeleton(
        premise_id=premise_id,
        original=original,
        kind=match.kind,
        body=body,
        antecedent=antecedent,
        consequent=consequent,
        left=left,
        right=right,
        formula_tree=formula_tree,
        quantifier=quantifier,
        risk_flags=list(match.risk_flags),
        confidence=match.confidence,
        needs_review=needs_review or match.kind == "UNKNOWN",
        notes=list(match.notes),
    )


def _unknown_skeleton(
    premise_id: str,
    original: str,
    match: OperatorMatch,
    flag: str,
    note: str,
) -> LogicSkeleton:
    risk_flags = _append_unique(list(match.risk_flags), flag)
    if "needs_review" not in risk_flags:
        risk_flags.append("needs_review")
    return LogicSkeleton(
        premise_id=premise_id,
        original=original,
        kind="UNKNOWN",
        body=_span("body", strip_terminal_punctuation(original)),
        risk_flags=risk_flags,
        confidence=min(match.confidence, 0.4),
        needs_review=True,
        notes=_append_unique(list(match.notes), note),
    )


def _append_unique(values: list[str], value: str) -> list[str]:
    if value not in values:
        values.append(value)
    return values


def _clean_part(text: str) -> str:
    cleaned = strip_terminal_punctuation(text)
    return re.sub(r"\s+", " ", cleaned).strip(" ,;:")


def _preserve_case(original: str, replacement: str) -> str:
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement


def _normalize_relative_condition(condition: str) -> str:
    words = normalize_whitespace(condition).split()
    if not words:
        return ""
    words[0] = _third_person_singular_verb(words[0])
    return " ".join(words)


def _subject_predicate(subject: str, verb: str, rest: str) -> str:
    return _clean_part(f"the {subject} {_third_person_singular_verb(verb)} {rest}")


def _third_person_singular_verb(verb: str) -> str:
    lower = verb.lower()
    irregular = {
        "are": "is",
        "were": "was",
        "have": "has",
        "do": "does",
    }
    stable = {
        "is",
        "was",
        "has",
        "does",
        "did",
        "can",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "will",
        "would",
    }
    if lower in irregular:
        replacement = irregular[lower]
    elif lower in stable:
        replacement = lower
    elif lower.endswith("ss"):
        replacement = lower + "es"
    elif lower.endswith("s"):
        replacement = lower
    elif lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
        replacement = lower[:-1] + "ies"
    elif lower.endswith(("ch", "sh", "x", "z", "o")):
        replacement = lower + "es"
    else:
        replacement = lower + "s"
    return _preserve_case(verb, replacement)
