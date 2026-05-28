from __future__ import annotations

import re
from typing import Any

from . import issue_codes as C
from ._utils import get_value
from .validation_models import SemanticPolicy, ValidationIssue


def validate_semantics(
    parsed_record: dict,
    registry: dict,
    policy: SemanticPolicy | dict | None = None,
) -> list[ValidationIssue]:
    semantic_policy = SemanticPolicy.from_mapping(policy)
    issues: list[ValidationIssue] = []
    issues.extend(validate_or_not_became_and(parsed_record, registry, semantic_policy))
    issues.extend(validate_source_mentions_preserved(parsed_record, semantic_policy))
    issues.extend(validate_domain_restriction_not_lost(parsed_record, semantic_policy))
    issues.extend(validate_numeric_values_are_typed(parsed_record, semantic_policy))
    issues.extend(validate_modal_not_necessarily(parsed_record))
    issues.extend(validate_deontic_unresolved(parsed_record))
    issues.extend(validate_negation_not_dropped(parsed_record))
    issues.extend(validate_formula_like_leaf_not_atomized(parsed_record))
    issues.extend(validate_important_arguments_not_dropped(parsed_record))
    issues.extend(validate_domain_atoms_not_inferred_unsafely(parsed_record, registry))
    return issues


def validate_or_not_became_and(
    parsed_record: dict,
    registry: dict,
    policy: SemanticPolicy,
) -> list[ValidationIssue]:
    if not policy.disjunction_requires_or_or_group:
        return []

    issues: list[ValidationIssue] = []

    for request in get_value(parsed_record, "atomization_requests", []) or []:
        if not request_has_logical_cue(request, "or", policy):
            continue

        result = _find_result(parsed_record, get_value(request, "request_id"))
        if not result:
            continue

        atoms = list(get_value(result, "atoms", []) or [])
        has_or_structure = get_value(result, "operator") == "or" or get_value(result, "logical_operator") == "or"
        has_group_predicate = any(atom_encodes_disjunction(atom, registry) for atom in atoms)

        if len(atoms) >= 2 and not has_or_structure and not has_group_predicate:
            issues.append(
                ValidationIssue(
                    code=C.SEMANTIC_OR_BECAME_AND,
                    severity="error",
                    message="Source indicates OR but atomization produced multiple atoms without OR/group structure.",
                    premise_id=get_value(request, "premise_id"),
                    request_id=get_value(request, "request_id"),
                    evidence={
                        "logical_cues": get_value(request, "logical_cues", []),
                        "source_mentions": get_value(request, "source_mentions", []),
                        "atoms": atoms,
                    },
                    suggested_stage="Stage 5",
                )
            )

    return issues


def validate_source_mentions_preserved(
    parsed_record: dict,
    policy: SemanticPolicy,
) -> list[ValidationIssue]:
    if not policy.require_evidence_links:
        return []

    issues: list[ValidationIssue] = []

    for request in get_value(parsed_record, "atomization_requests", []) or []:
        source_mentions = get_value(request, "source_mentions", []) or []
        important_mentions = [
            mention
            for mention in source_mentions
            if get_value(mention, "semantic_role") in policy.important_source_mention_roles
            and get_value(mention, "id")
        ]

        if not important_mentions:
            continue

        result = _find_result(parsed_record, get_value(request, "request_id"))
        if not result:
            continue

        linked_ids: set[str] = set()
        for atom in get_value(result, "atoms", []) or []:
            linked_ids.update(str(item) for item in get_value(atom, "evidence_links", []) or [])

        for mention in important_mentions:
            mention_id = str(get_value(mention, "id"))
            if mention_id in linked_ids:
                continue
            issues.append(
                ValidationIssue(
                    code=C.SEMANTIC_SOURCE_MENTION_DROPPED,
                    severity="warning",
                    message="Important source mention was not linked to any atom.",
                    premise_id=get_value(request, "premise_id"),
                    request_id=get_value(request, "request_id"),
                    evidence={
                        "mention": mention,
                        "atoms": get_value(result, "atoms", []),
                    },
                    suggested_stage="Stage 3 or Stage 4",
                )
            )

    return issues


def validate_domain_restriction_not_lost(
    parsed_record: dict,
    policy: SemanticPolicy,
) -> list[ValidationIssue]:
    if not policy.require_domain_restrictions:
        return []

    issues: list[ValidationIssue] = []

    for request in get_value(parsed_record, "atomization_requests", []) or []:
        required_atoms = get_value(request, "required_domain_atoms", []) or []
        if not required_atoms:
            continue

        result = _find_result(parsed_record, get_value(request, "request_id"))
        if not result:
            continue

        actual_atoms = get_value(result, "atoms", []) or []
        added_elsewhere = bool(get_value(result, "domain_restriction_added_elsewhere", False))

        for required in required_atoms:
            if added_elsewhere:
                continue
            if atom_equivalent_present(required, actual_atoms):
                continue
            issues.append(
                ValidationIssue(
                    code=C.SEMANTIC_DOMAIN_RESTRICTION_LOST,
                    severity="warning",
                    message="Required domain restriction was not preserved.",
                    premise_id=get_value(request, "premise_id"),
                    request_id=get_value(request, "request_id"),
                    evidence={
                        "required_domain_atom": required,
                        "actual_atoms": actual_atoms,
                        "phrase": get_value(request, "phrase"),
                    },
                    suggested_stage="Stage 2, Stage 3, or Stage 5",
                )
            )

    return issues


def validate_numeric_values_are_typed(
    parsed_record: dict,
    policy: SemanticPolicy,
) -> list[ValidationIssue]:
    if not policy.numeric_arguments_must_be_strings:
        return []

    issues: list[ValidationIssue] = []

    for result in get_value(parsed_record, "atomization_results", []) or []:
        for atom in get_value(result, "atoms", []) or []:
            for arg in get_value(atom, "arguments", []) or []:
                bad_argument = _raw_numeric_argument(arg)
                if bad_argument is None:
                    continue
                issues.append(
                    ValidationIssue(
                        code=C.SEMANTIC_NUMERIC_VALUE_UNTYPED,
                        severity="warning",
                        message="Numeric predicate arguments should be typed consistently as strings.",
                        premise_id=get_value(result, "premise_id"),
                        request_id=get_value(result, "request_id"),
                        evidence={"atom": atom, "bad_argument": bad_argument},
                        suggested_stage="Stage 4",
                    )
                )

    return issues


def validate_modal_not_necessarily(parsed_record: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for result in get_value(parsed_record, "atomization_results", []) or []:
        phrase = str(get_value(result, "phrase", "")).lower()
        if "not necessarily" not in phrase:
            continue

        for atom in get_value(result, "atoms", []) or []:
            if get_value(atom, "negated") is True:
                issues.append(
                    ValidationIssue(
                        code=C.MODAL_NOT_NECESSARILY_AS_NEGATION,
                        severity="error",
                        message="'not necessarily' is modal uncertainty, not classical negation.",
                        premise_id=get_value(result, "premise_id"),
                        request_id=get_value(result, "request_id"),
                        evidence={"phrase": phrase, "atom": atom},
                        suggested_stage="Stage 1 or Stage 3",
                    )
                )

    return issues


def validate_deontic_unresolved(parsed_record: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for result in get_value(parsed_record, "atomization_results", []) or []:
        unsupported_reason = str(get_value(result, "unsupported_reason", "") or "").lower()
        notes = " ".join(str(note).lower() for note in get_value(result, "notes", []) or [])
        phrase = str(get_value(result, "phrase", "") or "").lower()
        if "deontic" not in unsupported_reason and "deontic" not in notes:
            continue
        if re.search(r"\bmust\s+(have|be|satisfy|meet)\b", phrase):
            continue
        issues.append(
            ValidationIssue(
                code=C.SEMANTIC_DEONTIC_UNRESOLVED,
                severity="error",
                message="Deontic/modal atomization requires later-stage resolution.",
                premise_id=get_value(result, "premise_id"),
                request_id=get_value(result, "request_id"),
                evidence={
                    "phrase": get_value(result, "phrase"),
                    "unsupported_reason": get_value(result, "unsupported_reason"),
                    "notes": list(get_value(result, "notes", []) or []),
                },
                suggested_stage="Stage 8",
            )
        )

    return issues


def validate_negation_not_dropped(parsed_record: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for request in get_value(parsed_record, "atomization_requests", []) or []:
        phrase = str(get_value(request, "phrase", "") or "")
        if not _has_classical_negation(phrase, bool(get_value(request, "negation_hint", False))):
            continue
        result = _find_result(parsed_record, get_value(request, "request_id"))
        if not result:
            continue
        atoms = list(get_value(result, "atoms", []) or [])
        atom_has_negation = any(bool(get_value(atom, "negated", False)) for atom in atoms)
        ast_has_negation = _premise_ast_has_not(parsed_record, str(get_value(request, "premise_id", "")))
        if atom_has_negation or ast_has_negation:
            continue
        issues.append(
            ValidationIssue(
                code=C.NEGATION_DROPPED,
                severity="error",
                message="Source text contains classical negation, but no negated atom or AST NOT was produced.",
                premise_id=get_value(request, "premise_id"),
                request_id=get_value(request, "request_id"),
                evidence={"phrase": phrase, "atoms": atoms},
                suggested_stage="Stage 3 or Stage 5",
            )
        )
    return issues


def validate_formula_like_leaf_not_atomized(parsed_record: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for request in get_value(parsed_record, "atomization_requests", []) or []:
        notes = {str(note) for note in get_value(request, "notes", []) or []}
        if "formula_like_leaf_detected" not in notes:
            continue
        issues.append(
            ValidationIssue(
                code=C.FORMULA_LIKE_LEAF_REACHED_ATOMIZER,
                severity="error",
                message="Formula-like text reached the leaf atomization boundary.",
                premise_id=get_value(request, "premise_id"),
                request_id=get_value(request, "request_id"),
                evidence={"phrase": get_value(request, "phrase")},
                suggested_stage="Stage 2",
            )
        )
    for result in get_value(parsed_record, "atomization_results", []) or []:
        reason = str(get_value(result, "unsupported_reason", "") or "")
        if reason != "formula_like_leaf_requires_recursive_parse":
            continue
        issues.append(
            ValidationIssue(
                code=C.FORMULA_LIKE_LEAF_REACHED_ATOMIZER,
                severity="error",
                message="Formula-like text was blocked at atomization.",
                premise_id=get_value(result, "premise_id"),
                request_id=get_value(result, "request_id"),
                evidence={"phrase": get_value(result, "phrase"), "unsupported_reason": reason},
                suggested_stage="Stage 2",
            )
        )
    return issues


def validate_important_arguments_not_dropped(parsed_record: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    object_roles = {"object", "object_type", "subject", "attribute"}
    for request in get_value(parsed_record, "atomization_requests", []) or []:
        result = _find_result(parsed_record, get_value(request, "request_id"))
        if not result:
            continue
        atoms = list(get_value(result, "atoms", []) or [])
        atom_text = _atom_surface_text(atoms)
        for mention in get_value(request, "source_mentions", []) or []:
            role = str(get_value(mention, "semantic_role", ""))
            canonical = str(get_value(mention, "canonical", ""))
            if not canonical:
                continue
            if role == "quantity" and canonical not in atom_text:
                issues.append(
                    ValidationIssue(
                        code=C.NUMERIC_VALUE_DROPPED,
                        severity="error",
                        message="A numeric source value was not preserved in predicate arguments or names.",
                        premise_id=get_value(request, "premise_id"),
                        request_id=get_value(request, "request_id"),
                        evidence={"mention": mention, "atoms": atoms},
                        suggested_stage="Stage 2, Stage 3, or Stage 4",
                    )
                )
            if role in object_roles and canonical not in atom_text:
                issues.append(
                    ValidationIssue(
                        code=C.OBJECT_ARGUMENT_DROPPED,
                        severity="error",
                        message="A stated source object was not preserved in predicate arguments or names.",
                        premise_id=get_value(request, "premise_id"),
                        request_id=get_value(request, "request_id"),
                        evidence={"mention": mention, "atoms": atoms},
                        suggested_stage="Stage 2, Stage 3, or Stage 4",
                    )
                )
    return issues


def validate_domain_atoms_not_inferred_unsafely(parsed_record: dict, registry: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for request in get_value(parsed_record, "atomization_requests", []) or []:
        result = _find_result(parsed_record, get_value(request, "request_id"))
        if not result:
            continue
        phrase = str(get_value(request, "phrase", "") or "").lower()
        required = {
            str(get_value(item, "predicate", ""))
            for item in get_value(request, "required_domain_atoms", []) or []
        }
        mentioned = {
            str(get_value(mention, "canonical", ""))
            for mention in get_value(request, "source_mentions", []) or []
            if str(get_value(mention, "semantic_role", "")) == "domain_type"
        }
        for atom in get_value(result, "atoms", []) or []:
            name = str(get_value(atom, "name", ""))
            args = list(get_value(atom, "arguments", []) or [])
            if name in required or name in mentioned or name in phrase:
                continue
            metadata = registry.get(name, {}) if name in registry else {}
            if len(args) == 1 and not get_value(atom, "negated", False) and get_value(metadata, "domain_atom", False):
                issues.append(
                    ValidationIssue(
                        code=C.DOMAIN_ATOM_INFERRED_UNSAFELY,
                        severity="error",
                        message="A domain/class atom was produced without source support.",
                        premise_id=get_value(request, "premise_id"),
                        request_id=get_value(request, "request_id"),
                        evidence={"atom": atom, "phrase": get_value(request, "phrase")},
                        suggested_stage="Stage 3 or Stage 4",
                    )
                )
    return issues


def atom_equivalent_present(required: dict, atoms: list[dict]) -> bool:
    required_name = get_value(required, "predicate") or get_value(required, "name")
    required_args = [_argument_value(arg) for arg in get_value(required, "arguments", []) or []]
    required_negated = bool(get_value(required, "negated", False))

    return any(
        get_value(atom, "name") == required_name
        and [_argument_value(arg) for arg in get_value(atom, "arguments", []) or []] == required_args
        and bool(get_value(atom, "negated", False)) == required_negated
        for atom in atoms
    )


def atom_encodes_disjunction(atom: dict, registry: dict) -> bool:
    name = get_value(atom, "name")
    metadata = registry.get(name, {}) if name in registry else {}
    if get_value(metadata, "encodes_disjunction", False):
        return True

    argument_values = registry.get("__argument_values__", {}) or {}
    for arg in get_value(atom, "arguments", []) or []:
        arg_metadata = argument_values.get(str(_argument_value(arg)), {})
        if get_value(arg_metadata, "encodes_disjunction", False):
            return True

    return False


def request_has_logical_cue(request: Any, cue: str, policy: SemanticPolicy) -> bool:
    cue = cue.lower()
    explicit_cues = {str(item).lower() for item in get_value(request, "logical_cues", []) or []}
    if cue in explicit_cues:
        return True

    for mention in get_value(request, "source_mentions", []) or []:
        role = str(get_value(mention, "semantic_role", "")).lower()
        canonical = str(get_value(mention, "canonical", "")).lower()
        if role == cue or canonical == cue:
            return True

    phrase = f" {str(get_value(request, 'phrase', '')).lower()} "
    for word in policy.logical_cue_words.get(cue, []):
        if re.search(rf"\b{re.escape(str(word).lower())}\b", phrase):
            return True
    return False


def _find_result(parsed_record: dict, request_id: str) -> object | None:
    for result in get_value(parsed_record, "atomization_results", []) or []:
        if get_value(result, "request_id") == request_id:
            return result
    return None


def _argument_value(arg: Any) -> str:
    if isinstance(arg, dict):
        return str(arg.get("value", ""))
    return str(arg)


def _raw_numeric_argument(arg: Any) -> int | float | None:
    value = arg.get("value") if isinstance(arg, dict) else arg
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _has_classical_negation(phrase: str, negation_hint: bool) -> bool:
    lower = re.sub(r"\s+", " ", str(phrase or "").lower()).strip()
    if "not necessarily" in lower:
        return False
    return bool(negation_hint) or bool(
        re.search(
            r"\b(no|not|never|without|cannot|can't|does\s+not|do\s+not|did\s+not|has\s+not|have\s+not|had\s+not|is\s+not|are\s+not|was\s+not|were\s+not)\b",
            lower,
        )
    )


def _premise_ast_has_not(parsed_record: dict, premise_id: str) -> bool:
    asts = get_value(parsed_record, "asts") or get_value(parsed_record, "ast") or []

    def walk(node: Any) -> bool:
        if str(get_value(node, "premise_id", "") or "") not in {"", premise_id}:
            return False
        if str(get_value(node, "type", "")).upper() == "NOT":
            return True
        return any(walk(child) for child in get_value(node, "children", []) or [])

    nodes = asts if isinstance(asts, list) else [asts]
    return any(walk(node) for node in nodes)


def _atom_surface_text(atoms: list[Any]) -> str:
    parts: list[str] = []
    for atom in atoms:
        parts.append(str(get_value(atom, "name", "")))
        parts.extend(_argument_value(arg) for arg in get_value(atom, "arguments", []) or [])
    return " ".join(parts)
