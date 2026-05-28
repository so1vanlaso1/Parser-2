from __future__ import annotations

"""Stage 2: collect leaf phrase atomization requests.

Input: LogicSkeleton / FormulaSkeleton from Stage 1.
Output: AtomizationRequest objects for Stage 3.

This module does not call an LLM, canonicalize predicates, build AST, or export
to a solver.
"""

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

try:
    from ..Stage_1.logic_skeleton import FormulaSkeleton, LogicSkeleton, TextSpan
except Exception:  # pragma: no cover
    from NEW_logic_pipeline.Stage_1.logic_skeleton import FormulaSkeleton, LogicSkeleton, TextSpan


class PredicateAtom(BaseModel):
    name: str
    arguments: list[Any]
    negated: bool = False
    source_phrase: str | None = None
    evidence_links: list[str] = Field(default_factory=list)
    confidence: float = 1.0

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value


class AtomizationRequest(BaseModel):
    request_id: str
    premise_id: str
    role: str
    phrase: str
    variable: str = "x"
    known_predicates: list[str] = Field(default_factory=list)
    negation_hint: bool = False
    modality_hint: str | None = None
    logical_cues: list[str] = Field(default_factory=list)
    required_domain_atoms: list[dict[str, Any]] = Field(default_factory=list)
    source_mentions: list[dict[str, Any]] = Field(default_factory=list)
    resolved_references: dict[str, str] = Field(default_factory=dict)
    skeleton_kind: str | None = None
    formula_path: list[int] = Field(default_factory=list)
    original_premise: str | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("phrase", "request_id", "premise_id", "role", "variable", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _space_norm(value).strip()
        return value


class AtomizationResult(BaseModel):
    request_id: str
    premise_id: str
    role: str
    phrase: str
    variable: str
    atoms: list[PredicateAtom] = Field(default_factory=list)
    source_mentions: list[dict[str, Any]] = Field(default_factory=list)
    resolved_references: dict[str, str] = Field(default_factory=dict)
    domain_restriction_added_elsewhere: bool = False
    needs_review: bool = False
    unsupported_reason: str | None = None
    notes: list[str] = Field(default_factory=list)
    formula_path: list[int] = Field(default_factory=list)


FORMULA_LIKE_LEAF_PATTERNS = (
    r"^if\b.+,",
    r"\bif\b.+\bthen\b",
    r"\bthen\b.+\bif\b",
    r"\bimplies\b",
    r"\bonly\s+if\b",
    r"\biff\b",
    r"\bif\s+and\s+only\s+if\b",
    r"^(every|all|each|any)\b",
    r"^(every|all|each|any)\b.+\b(then|implies|if)\b",
    r"\bthere\s+exists\b",
    r"\bthere\s+exists\b.+\bthen\b",
    r"\bthere\s+is\s+at\s+least\s+one\b.+\bthen\b",
    r"\bfor\s+all\b.+\bthen\b",
    r"\bexists\b",
    r"\bexists\b.+\bthen\b",
    r"\bmust\s+also\s+be\s+true\b",
    r"\bit\s+is\s+(?:true|false)\s+that\b",
    r"∀|∃|->|=>|→",
)


def collect_atomization_requests(
    skeleton: LogicSkeleton,
    known_predicates: list[str] | None = None,
) -> list[AtomizationRequest]:
    known = list(known_predicates or [])
    kind = _kind(skeleton)

    if kind == "META" and _get(skeleton, "formula_tree") is not None:
        return collect_formula_leaf_requests(
            premise_id=str(_get(skeleton, "premise_id", "P?")),
            formula=_get(skeleton, "formula_tree"),
            role_prefix="meta_leaf",
            original_premise=str(_get(skeleton, "original", "")),
            known_predicates=known,
        )

    requests: list[AtomizationRequest] = []

    def add(role: str, span: Any) -> None:
        request = _request_from_span(skeleton, role, span, known)
        if request is not None:
            requests.append(request)

    if kind in {"FACT", "EXISTS"}:
        add("body", _get(skeleton, "body"))
    elif kind == "FORALL":
        add("restrictor", _get(skeleton, "antecedent"))
        add("property", _get(skeleton, "consequent"))
    elif kind in {"RULE", "ONLY_IF_RULE", "NON_IF_RULE"}:
        add("antecedent", _get(skeleton, "antecedent"))
        add("consequent", _get(skeleton, "consequent"))
    elif kind == "IFF":
        add("left", _get(skeleton, "left"))
        add("right", _get(skeleton, "right"))
    elif kind in {"OBLIGATION_RULE", "MODAL", "UNKNOWN"}:
        add("body", _get(skeleton, "body"))
    else:
        add("body", _get(skeleton, "body"))

    return requests


def collect_batch_atomization_requests(
    skeletons: list[LogicSkeleton],
    known_predicates: list[str] | None = None,
) -> list[AtomizationRequest]:
    requests: list[AtomizationRequest] = []
    for skeleton in skeletons:
        requests.extend(collect_atomization_requests(skeleton, known_predicates))
    return requests


def collect_formula_leaf_requests(
    premise_id: str,
    formula: FormulaSkeleton,
    role_prefix: str,
    original_premise: str,
    known_predicates: list[str] | None = None,
    path: list[int] | None = None,
) -> list[AtomizationRequest]:
    """Walk FormulaSkeleton recursively and collect leaf nodes only.

    Structural `not` nodes are propagated as classical negation hints. Modal
    phrase negation such as "not necessarily" is not propagated as classical NOT.
    """

    known = list(known_predicates or [])
    start_path = list(path or [])
    requests: list[AtomizationRequest] = []

    def walk(node: Any, current_path: list[int], inherited_variable: str, inherited_negation: bool = False, inherited_modality: str | None = None) -> None:
        node_type = _node_type(node)
        variable = str(_get(node, "variable") or inherited_variable or "x")
        node_modality = _normalize_modality_hint(_get(node, "modality_hint") or inherited_modality)

        if node_type == "not":
            for idx, child in enumerate(list(_get(node, "children", []) or [])):
                walk(child, [*current_path, idx], variable, not inherited_negation, node_modality)
            return

        if node_type in {"forall", "exists"}:
            for idx, child in enumerate(list(_get(node, "children", []) or [])):
                walk(child, [*current_path, idx], variable, inherited_negation, node_modality)
            return

        if node_type == "leaf":
            phrase = _clean_phrase(_get(node, "text"))
            if not phrase:
                return
            phrase_modality = _normalize_modality_hint(node_modality or _detect_modality_hint(phrase))
            if is_formula_like_leaf(phrase):
                reparsed = _reparse_formula_like_leaf(phrase, variable)
                if reparsed is not None:
                    walk(reparsed, current_path, variable, inherited_negation, phrase_modality)
                    return
            classical_negation = bool(inherited_negation or _detect_negation_hint(phrase))
            if _is_modal_not_necessarily(phrase_modality, phrase):
                classical_negation = False
            notes = []
            if is_formula_like_leaf(phrase):
                phrase_modality = phrase_modality or "unknown_structure"
                notes.append("formula_like_leaf_detected")
            metadata = _source_metadata(phrase, variable, classical_negation, original_premise)
            requests.append(
                AtomizationRequest(
                    request_id=_formula_request_id(premise_id, role_prefix, current_path),
                    premise_id=premise_id,
                    role=role_prefix,
                    phrase=phrase,
                    variable=variable,
                    known_predicates=list(known),
                    negation_hint=classical_negation,
                    modality_hint=phrase_modality,
                    logical_cues=metadata["logical_cues"],
                    required_domain_atoms=metadata["required_domain_atoms"],
                    source_mentions=metadata["source_mentions"],
                    resolved_references=metadata["resolved_references"],
                    skeleton_kind="META",
                    formula_path=list(current_path),
                    original_premise=original_premise,
                    notes=notes,
                )
            )
            return

        for idx, child in enumerate(list(_get(node, "children", []) or [])):
            walk(child, [*current_path, idx], variable, inherited_negation, node_modality)

    walk(formula, start_path, str(_get(formula, "variable") or "x"))
    return requests


def _request_from_span(skeleton: LogicSkeleton, role: str, span: TextSpan | None, known_predicates: list[str]) -> AtomizationRequest | None:
    if span is None:
        return None
    phrase = _clean_phrase(_get(span, "text"))
    if not phrase:
        return None
    kind = _kind(skeleton)
    modality_hint = _modality_hint_for_span(kind, skeleton, span, phrase)
    variable = str(_get(span, "variable") or _get(skeleton, "variable") or "x")
    classical_negation = bool(_get(span, "negation_hint", False) or _detect_negation_hint(phrase))
    if _is_modal_not_necessarily(modality_hint, phrase):
        classical_negation = False

    notes: list[str] = []
    if kind == "UNKNOWN":
        notes.append("unknown_structure")
    if kind == "MODAL":
        notes.append("modal_phrase_not_solver_ready")
    if kind == "OBLIGATION_RULE":
        notes.append("deontic_phrase_not_solver_ready")

    if is_formula_like_leaf(phrase):
        notes.append("formula_like_leaf_detected")
        modality_hint = modality_hint or "unknown_structure"

    metadata = _source_metadata(phrase, variable, classical_negation, str(_get(skeleton, "original", "")))

    # --- Property-only phrase context injection ---
    # If the phrase looks like a predicate without an explicit subject (starts
    # with copula, auxiliary, pronoun, or negation pattern), and the skeleton
    # has a restrictor/antecedent that carries domain information, inject that
    # context so the atomizer can understand the implicit subject.
    required_domain_atoms = list(metadata["required_domain_atoms"])
    if (
        _is_property_only_phrase(phrase)
        and role in {"property", "consequent"}
        and not _should_skip_implicit_domain_injection(kind, role, phrase)
    ):
        restrictor_span = _get(skeleton, "antecedent") or _get(skeleton, "body")
        if restrictor_span is not None:
            restrictor_phrase = _clean_phrase(_get(restrictor_span, "text"))
            restrictor_domain = _leading_domain_type(restrictor_phrase) if restrictor_phrase else None
            if restrictor_domain and not any(
                d.get("predicate") == restrictor_domain for d in required_domain_atoms
            ):
                required_domain_atoms.append({
                    "predicate": restrictor_domain,
                    "arguments": [variable],
                    "negated": False,
                    "source_mention_id": "implicit_subject",
                })
                notes.append(f"injected implicit subject domain atom: {restrictor_domain}({variable})")

    return AtomizationRequest(
        request_id=_request_id(str(_get(skeleton, "premise_id", "P?")), role),
        premise_id=str(_get(skeleton, "premise_id", "P?")),
        role=role,
        phrase=phrase,
        variable=variable,
        known_predicates=list(known_predicates),
        negation_hint=classical_negation,
        modality_hint=modality_hint,
        logical_cues=metadata["logical_cues"],
        required_domain_atoms=required_domain_atoms,
        source_mentions=metadata["source_mentions"],
        resolved_references=metadata["resolved_references"],
        skeleton_kind=kind,
        formula_path=[],
        original_premise=str(_get(skeleton, "original", "")),
        notes=notes,
    )


def _modality_hint_for_span(kind: str, skeleton: Any, span: Any, phrase: str) -> str | None:
    explicit = _normalize_modality_hint(_get(span, "modality_hint"))
    if explicit:
        return explicit
    risk_flags = [str(flag) for flag in list(_get(skeleton, "risk_flags", []) or [])]
    if kind == "OBLIGATION_RULE":
        return _first_flag_with_prefix(risk_flags, "deontic") or "deontic_obligation"
    if kind == "MODAL":
        return _first_flag_with_prefix(risk_flags, "modal") or _detect_modality_hint(phrase) or "modal_uncertainty"
    if kind == "UNKNOWN":
        return "unknown_structure"
    return _detect_modality_hint(phrase)


def _source_metadata(phrase: str, variable: str, negation_hint: bool, original_premise: str | None = None) -> dict[str, Any]:
    logical_cues = _logical_cues(phrase, negation_hint)
    source_mentions = _source_mentions(phrase)
    resolved_references = _resolved_references(phrase, original_premise or phrase)
    required_domain_atoms: list[dict[str, Any]] = []

    domain_mention = next(
        (mention for mention in source_mentions if mention.get("semantic_role") == "domain_type"),
        None,
    )
    if domain_mention is not None:
        required_domain_atoms.append(
            {
                "predicate": domain_mention["canonical"],
                "arguments": [variable],
                "negated": False,
                "source_mention_id": domain_mention["id"],
            }
        )

    return {
        "logical_cues": logical_cues,
        "required_domain_atoms": required_domain_atoms,
        "source_mentions": source_mentions,
        "resolved_references": resolved_references,
    }


def _logical_cues(phrase: str, negation_hint: bool) -> list[str]:
    lower = _space_norm(phrase).lower()
    cues: list[str] = []
    cue_patterns = {
        "and": r"\band\b",
        "or": r"\bor\b",
        "negation": r"\b(no|not|never|without|cannot|can't|does\s+not|do\s+not|did\s+not|is\s+not|are\s+not|was\s+not|were\s+not)\b",
        "conditional": r"\b(if|then|only\s+if|provided\s+that)\b",
    }
    for cue, pattern in cue_patterns.items():
        if re.search(pattern, lower):
            cues.append(cue)
    if negation_hint and "negation" not in cues:
        cues.append("negation")
    return cues


def _source_mentions(phrase: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    normalized = _space_norm(phrase)
    lower = normalized.lower()

    domain = _leading_domain_type(normalized)
    if domain:
        mentions.append(_mention("domain_type", domain, domain, len(mentions) + 1))

    for role, surface in _structured_object_mentions(normalized):
        canonical = _canonical_surface(surface)
        if not canonical or (domain and canonical == domain) or _mention_exists(mentions, canonical):
            continue
        mentions.append(_mention(role, surface, canonical, len(mentions) + 1))

    entity = _leading_named_entity(normalized)
    if entity:
        canonical = _canonical_surface(entity)
        mentions.append(_mention("entity", entity, canonical, len(mentions) + 1))

    for surface in re.findall(r"\b\d+(?:\.\d+)?\b", normalized):
        mentions.append(_mention("quantity", surface, surface, len(mentions) + 1))

    for surface in _grade_mentions(normalized):
        canonical = _canonical_surface(surface.replace("+", " plus"))
        if not _mention_exists(mentions, canonical):
            mentions.append(_mention("grade", surface, canonical, len(mentions) + 1))

    pronoun_match = re.search(r"\b(it|this|that|them)\b", lower)
    if pronoun_match:
        pronoun = pronoun_match.group(1)
        mentions.append(_mention("pronoun", pronoun, pronoun, len(mentions) + 1))

    if "not necessarily" in lower:
        mentions.append(_mention("modal_uncertainty", "not necessarily", "not_necessarily", len(mentions) + 1))
    elif re.search(r"\b(cannot|can't|not|no|never|without)\b", lower):
        match = re.search(r"\b(cannot|can't|not|no|never|without)\b", lower)
        surface = match.group(0) if match else "not"
        mentions.append(_mention("negation", surface, "negation", len(mentions) + 1))

    if re.search(r"\bor\b", lower):
        mentions.append(_mention("or", "or", "or", len(mentions) + 1))

    return mentions


def _structured_object_mentions(phrase: str) -> list[tuple[str, str]]:
    mentions: list[tuple[str, str]] = []
    clean = _space_norm(phrase)
    patterns = (
        (
            "subject",
            r"\b(?:failed|passed|retaken|retook|taken|took|completed|mastered)\s+(?P<object>[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*)*)\b",
        ),
        (
            "object_type",
            r"\b(?P<object>registration\s+form|liability\s+waiver|required\s+test|test|project|requirement|gpa|credits?)\b",
        ),
        ("attribute", r"\b(?P<object>core\s+course)\b"),
    )
    for role, pattern in patterns:
        for match in re.finditer(pattern, clean, flags=re.I):
            surface = match.group("object").strip(" ,.;:()")
            if surface and surface.lower() not in {"a", "an", "the"}:
                mentions.append((role, surface))

    for surface in _article_mentions(clean):
        mentions.append(("object_type", surface))

    return mentions


def _grade_mentions(phrase: str) -> list[str]:
    return [
        match.group(0)
        for match in re.finditer(r"\b(?:A\+|A|B\+|B|C\+|C|D\+|D|F)\b", phrase)
    ]


def _mention_exists(mentions: list[dict[str, Any]], canonical: str) -> bool:
    return any(mention.get("canonical") == canonical for mention in mentions)


def _mention(role: str, surface: str, canonical: str, index: int) -> dict[str, str]:
    return {
        "id": f"m{index}",
        "surface": surface,
        "semantic_role": role,
        "canonical": canonical,
    }


def _leading_domain_type(phrase: str) -> str | None:
    match = re.match(
        r"^(?:a|an|the|each|every|any|some|all)\s+(?P<term>[a-z][a-z0-9_-]*)\b",
        phrase,
        flags=re.I,
    )
    if not match:
        return None
    return _canonical_surface(match.group("term"))


def _article_mentions(phrase: str) -> list[str]:
    matches = re.finditer(
        r"\b(?:a|an|the|each|every|any|some|all)\s+(?P<term>[a-z][a-z0-9_-]*)\b",
        phrase,
        flags=re.I,
    )
    return [match.group("term") for match in matches]


def _leading_named_entity(phrase: str) -> str | None:
    match = re.match(r"^(?P<name>[A-ZÀ-Ỹ][\wÀ-ỹ.'-]*)\b", phrase)
    if not match:
        return None
    name = match.group("name").strip(" ,.;:()")
    if name.lower() in {"a", "an", "the"}:
        return None
    return name


def _canonical_surface(surface: str) -> str:
    value = _space_norm(surface).strip(" ,.;:()")
    value = re.sub(r"['’]s$", "", value)
    value = re.sub(r"[^A-Za-z0-9À-ỹ]+", "_", value).strip("_").lower()
    if "_" not in value and len(value) > 3 and value.endswith("s"):
        value = value[:-1]
    return value


def _resolved_references(phrase: str, original_premise: str) -> dict[str, str]:
    lower = _space_norm(phrase).lower()
    if not re.search(r"\b(it|this|that|them)\b", lower):
        return {}
    subject = _latest_named_subject_before_pronoun(_space_norm(original_premise))
    if not subject:
        return {}
    return {"it": subject, "this": subject, "that": subject, "them": subject}


def _latest_named_subject_before_pronoun(text: str) -> str | None:
    pronoun_match = re.search(r"\b(it|this|that|them)\b", text, flags=re.I)
    prefix = text[: pronoun_match.start()] if pronoun_match else text
    matches = list(
        re.finditer(
            r"\b(?:failed|passed|retaken|retook|taken|took|completed|mastered)\s+(?P<object>[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*)*)\b",
            prefix,
        )
    )
    if not matches:
        return None
    return _canonical_surface(matches[-1].group("object"))


def _normalize_modality_hint(value: Any) -> str | None:
    if value is None:
        return None
    hint = str(value).strip().lower().replace(" ", "_")
    if not hint:
        return None
    mapping = {
        "not_necessarily": "modal_not_necessarily",
        "uncertain": "modal_uncertainty",
        "uncertainty": "modal_uncertainty",
        "obligation": "deontic_obligation",
        "mandatory": "deontic_obligation",
        "deontic": "deontic_obligation",
    }
    return mapping.get(hint, hint)


def _detect_modality_hint(text: str) -> str | None:
    lower = _space_norm(text).lower()
    if "not necessarily" in lower:
        return "modal_not_necessarily"
    if re.search(r"\b(might|could|possibly|probably|likely|may\s+sometimes|can\s+sometimes)\b", lower):
        return "modal_uncertainty"
    if re.search(r"\bmust\s+(have|be|satisfy|meet)\b", lower):
        return None
    if re.search(r"\b(must|mandatory|required|obligated|forbidden|prohibited|not\s+allowed|permitted|allowed\s+to)\b", lower):
        return "deontic_obligation"
    return None


def _detect_negation_hint(text: str) -> bool:
    lower = _space_norm(text).lower()
    if "not necessarily" in lower:
        return False
    return bool(re.search(r"\b(no|not|never|without|cannot|can't|does\s+not|do\s+not|did\s+not|is\s+not|are\s+not|was\s+not|were\s+not|fails?\s+to|failure\s+to|lack\s+of|lacks?|loss\s+of)\b", lower))


def _is_modal_not_necessarily(modality_hint: str | None, phrase: str) -> bool:
    return modality_hint == "modal_not_necessarily" or "not necessarily" in _space_norm(phrase).lower()


def _first_flag_with_prefix(flags: list[str], prefix: str) -> str | None:
    for flag in flags:
        normalized = _normalize_modality_hint(flag) or flag
        if normalized.startswith(prefix):
            return normalized
    return None


def _kind(skeleton: Any) -> str:
    value = _get(skeleton, "kind", "UNKNOWN")
    return str(value.value) if hasattr(value, "value") else str(value)


def _node_type(node: Any) -> str:
    value = _get(node, "type", "leaf")
    return str(value.value) if hasattr(value, "value") else str(value)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _request_id(premise_id: str, role: str) -> str:
    return f"{premise_id}_{_safe_id(role)}"


def _formula_request_id(premise_id: str, role_prefix: str, path: list[int]) -> str:
    suffix = "_".join(str(part) for part in path) if path else "root"
    return f"{premise_id}_{_safe_id(role_prefix)}_{suffix}"


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()
    return safe or "leaf"


def _clean_phrase(text: Any) -> str:
    return _space_norm(str(text or "")).strip(" \t\r\n,;:.!?")


def _space_norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def is_formula_like_leaf(text: str) -> bool:
    clean = _space_norm(text).strip("() ")
    if not clean:
        return False
    lower = clean.lower()
    return any(re.search(pattern, lower, flags=re.I) for pattern in FORMULA_LIKE_LEAF_PATTERNS)


def _reparse_formula_like_leaf(phrase: str, variable: str) -> FormulaSkeleton | None:
    try:
        from ..Stage_1.skeleton_builder import parse_formula_part
    except Exception:  # pragma: no cover
        try:
            from NEW_logic_pipeline.Stage_1.skeleton_builder import parse_formula_part
        except Exception:
            return None

    reparsed = parse_formula_part(phrase, variable)
    if _node_type(reparsed) == "leaf" and _clean_phrase(_get(reparsed, "text")) == _clean_phrase(phrase):
        return None
    return reparsed


def _is_property_only_phrase(phrase: str) -> bool:
    """Detect phrases that describe a property without an explicit subject.

    General linguistic check — looks for leading copulae, auxiliaries, pronouns,
    negation patterns, and bare gerund/participial forms.  Does not use
    domain-specific vocabulary.

    Examples that should return True:
        "is preparing for an exam"
        "does not attend tutorials"
        "they are not understanding the material"
        "has completed a course"
        "preparing for an exam"
    """
    lower = _space_norm(phrase).lower()
    # Starts with copula / auxiliary
    if re.match(r"^(is|are|was|were|has|have|had|does|do|did|can|could|may|might|must|shall|should|will|would)\s+", lower):
        return True
    # Starts with subject pronoun + verb pattern
    if re.match(r"^(he|she|it|they|we|you|one)\s+(is|are|was|were|has|have|had|does|do|did|can|could)\s+", lower):
        return True
    # Starts with negation + verb
    if re.match(r"^(not|cannot|can't|does not|do not|did not|is not|are not|was not|were not)\s+", lower):
        return True
    # Bare gerund/participial (no article/determiner before it)
    if re.match(r"^[a-z]+ing\s+", lower) and not re.match(r"^(a|an|the|this|that|some|any|each|every|all)\s+", lower):
        return True
    return False


def _should_skip_implicit_domain_injection(kind: str, role: str, phrase: str) -> bool:
    if role != "consequent" or kind not in {"RULE", "ONLY_IF_RULE", "NON_IF_RULE"}:
        return False
    return bool(re.match(r"^(he|she|it|they|we|you|one)\b", _space_norm(phrase).lower()))


__all__ = [
    "PredicateAtom",
    "AtomizationRequest",
    "AtomizationResult",
    "collect_atomization_requests",
    "collect_batch_atomization_requests",
    "collect_formula_leaf_requests",
    "is_formula_like_leaf",
]
