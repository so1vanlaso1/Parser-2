from __future__ import annotations

"""Stage 4: collect leaf phrase atomization requests.

Input: LogicSkeleton / FormulaSkeleton from Stage 1–3.
Output: AtomizationRequest objects for Stage 5.

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
    domain_restriction_added_elsewhere: bool = False
    needs_review: bool = False
    unsupported_reason: str | None = None
    notes: list[str] = Field(default_factory=list)
    formula_path: list[int] = Field(default_factory=list)


FORMULA_LIKE_LEAF_PATTERNS = (
    r"\bif\b",
    r"\bthen\b",
    r"\bimplies\b",
    r"\bonly\s+if\b",
    r"\bif\s+and\s+only\s+if\b",
    r"\bevery\b",
    r"\ball\b",
    r"\beach\b",
    r"\bany\b",
    r"\bthere\s+exists\b",
    r"\bthere\s+is\s+at\s+least\s+one\b",
    r"\bfor\s+all\b",
    r"\bexists\b",
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
            metadata = _source_metadata(phrase, variable, classical_negation)
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

    metadata = _source_metadata(phrase, variable, classical_negation)
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
        required_domain_atoms=metadata["required_domain_atoms"],
        source_mentions=metadata["source_mentions"],
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


def _source_metadata(phrase: str, variable: str, negation_hint: bool) -> dict[str, Any]:
    logical_cues = _logical_cues(phrase, negation_hint)
    source_mentions = _source_mentions(phrase)
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

    for surface in _article_mentions(normalized):
        canonical = _canonical_surface(surface)
        if not canonical or (domain and canonical == domain):
            continue
        mentions.append(_mention("object_type", surface, canonical, len(mentions) + 1))

    entity = _leading_named_entity(normalized)
    if entity:
        canonical = _canonical_surface(entity)
        mentions.append(_mention("entity", entity, canonical, len(mentions) + 1))

    for surface in re.findall(r"\b\d+(?:\.\d+)?\b", normalized):
        mentions.append(_mention("quantity", surface, surface, len(mentions) + 1))

    if "not necessarily" in lower:
        mentions.append(_mention("modal_uncertainty", "not necessarily", "not_necessarily", len(mentions) + 1))
    elif re.search(r"\b(cannot|can't|not|no|never|without)\b", lower):
        match = re.search(r"\b(cannot|can't|not|no|never|without)\b", lower)
        surface = match.group(0) if match else "not"
        mentions.append(_mention("negation", surface, "negation", len(mentions) + 1))

    if re.search(r"\bor\b", lower):
        mentions.append(_mention("or", "or", "or", len(mentions) + 1))

    return mentions


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
    if len(value) > 3 and value.endswith("s"):
        value = value[:-1]
    return value


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


__all__ = [
    "PredicateAtom",
    "AtomizationRequest",
    "AtomizationResult",
    "collect_atomization_requests",
    "collect_batch_atomization_requests",
    "collect_formula_leaf_requests",
    "is_formula_like_leaf",
]
