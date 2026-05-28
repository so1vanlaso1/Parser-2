from __future__ import annotations

import re
from typing import Any, Iterator

from . import issue_codes as C
from ._utils import atom_dict, get_value
from .validation_models import ValidationIssue


BAD_NEGATION_NAME_PARTS = (
    "not_",
    "_not",
    "non_",
    "without_",
    "lack_",
    "cannot_",
)
UNRESOLVED_PRONOUN_PARTS = ("it", "this", "that", "them")
VALID_DYNAMIC_PREDICATE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


def iter_atoms(parsed_record: dict) -> Iterator[tuple[Any, dict[str, Any]]]:
    for result in get_value(parsed_record, "atomization_results", []) or []:
        for atom in get_value(result, "atoms", []) or []:
            yield result, atom_dict(atom)


def build_effective_registry(parsed_record: dict, base_registry: dict) -> dict:
    """Merge Stage 4's clean canonical predicates into the validator registry.

    Stage 4 owns the open predicate vocabulary for a row.  Stage 6 should still
    reject bad names and unstable arities, but clean canonical predicates from
    that row should not fail merely because they were absent from the static
    base registry.
    """

    registry = dict(base_registry or {})
    canonicalization = get_value(parsed_record, "canonicalization", {}) or {}
    allowed_names = {
        str(name)
        for name in get_value(canonicalization, "canonical_predicates", []) or []
        if _clean_dynamic_predicate(str(name))
    }
    if not allowed_names:
        return registry

    canonical_signatures = get_value(canonicalization, "predicate_signatures", {}) or {}
    observed: dict[str, set[int]] = {}
    for _, atom in iter_atoms(parsed_record):
        name = str(atom.get("name") or "")
        if name not in allowed_names or name in registry:
            continue
        observed.setdefault(name, set()).add(len(atom.get("arguments", []) or []))

    for name, arities in observed.items():
        signature = canonical_signatures.get(name, {}) if isinstance(canonical_signatures, dict) else {}
        signature_arity = signature.get("arity") if isinstance(signature, dict) else None
        signature_roles = signature.get("roles") if isinstance(signature, dict) else None
        if isinstance(signature_arity, int) and isinstance(signature_roles, list) and len(signature_roles) == signature_arity:
            registry[name] = {
                "arity": signature_arity,
                "roles": [str(role) for role in signature_roles],
                "solver_safe": True,
                "dynamic": True,
            }
            continue
        if len(arities) != 1:
            continue
        arity = next(iter(arities))
        if arity <= 0:
            continue
        registry[name] = {
            "arity": arity,
            "roles": ["entity"] * arity,
            "solver_safe": True,
            "dynamic": True,
        }

    return registry


def validate_predicates(
    parsed_record: dict,
    registry: dict,
    *,
    allow_dynamic_predicates: bool = False,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    canonicalization = get_value(parsed_record, "canonicalization", {}) or {}
    conflict_count = int(get_value(canonicalization, "conflict_count", 0) or 0)
    if conflict_count:
        issues.append(
            ValidationIssue(
                code=C.PRED_CANONICALIZATION_CONFLICT,
                severity="error",
                message=f"Predicate canonicalization reported {conflict_count} conflict(s).",
                evidence={"conflict_count": conflict_count},
                suggested_stage="Stage 4",
            )
        )

    for result, atom in iter_atoms(parsed_record):
        name = atom.get("name")
        args = [_argument_value(arg) for arg in atom.get("arguments", [])]

        if name and _predicate_contains_unresolved_pronoun(str(name)):
            issues.append(
                ValidationIssue(
                    code=C.UNRESOLVED_PRONOUN_IN_PREDICATE,
                    severity="error",
                    message="Predicate name contains an unresolved pronoun.",
                    premise_id=get_value(result, "premise_id"),
                    request_id=get_value(result, "request_id"),
                    evidence={
                        "predicate": name,
                        "source_phrase": atom.get("source_phrase") or get_value(result, "phrase"),
                    },
                    suggested_stage="Stage 2 or Stage 4",
                )
            )

        if name not in registry:
            if allow_dynamic_predicates:
                continue
            issues.append(
                ValidationIssue(
                    code=C.PRED_UNKNOWN,
                    severity="error",
                    message=f"Unknown predicate: {name}",
                    premise_id=get_value(result, "premise_id"),
                    request_id=get_value(result, "request_id"),
                    evidence={
                        "predicate": name,
                        "arguments": args,
                        "source_phrase": atom.get("source_phrase") or get_value(result, "phrase"),
                    },
                    suggested_stage="Stage 4",
                )
            )
            continue

        expected_arity = int(registry[name]["arity"])
        actual_arity = len(args)
        if expected_arity != actual_arity:
            issues.append(
                ValidationIssue(
                    code=C.PREDICATE_ARITY_MISMATCH,
                    severity="error",
                    message=f"{name} expects {expected_arity} arguments, got {actual_arity}.",
                    premise_id=get_value(result, "premise_id"),
                    request_id=get_value(result, "request_id"),
                    evidence={
                        "predicate": name,
                        "expected_arity": expected_arity,
                        "actual_arity": actual_arity,
                        "arguments": args,
                        "source_phrase": atom.get("source_phrase") or get_value(result, "phrase"),
                    },
                    suggested_stage="Stage 4",
                )
            )

        if name and any(part in str(name) for part in BAD_NEGATION_NAME_PARTS):
            issues.append(
                ValidationIssue(
                    code=C.NEGATION_INSIDE_PREDICATE,
                    severity="error",
                    message="Negation should be outside predicate name.",
                    premise_id=get_value(result, "premise_id"),
                    request_id=get_value(result, "request_id"),
                    evidence={"atom": atom},
                    suggested_stage="Stage 3 or Stage 4",
                )
            )

    return issues


def _clean_dynamic_predicate(name: str) -> bool:
    if not VALID_DYNAMIC_PREDICATE_RE.fullmatch(name):
        return False
    return not any(part in name for part in BAD_NEGATION_NAME_PARTS) and not _predicate_contains_unresolved_pronoun(name)


def _argument_value(arg: Any) -> str:
    if isinstance(arg, dict):
        return str(arg.get("value", ""))
    return str(arg)


def _predicate_contains_unresolved_pronoun(name: str) -> bool:
    parts = str(name or "").split("_")
    return any(part in UNRESOLVED_PRONOUN_PARTS for part in parts)
