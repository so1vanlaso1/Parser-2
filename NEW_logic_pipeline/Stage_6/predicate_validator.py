from __future__ import annotations

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


def iter_atoms(parsed_record: dict) -> Iterator[tuple[Any, dict[str, Any]]]:
    for result in get_value(parsed_record, "atomization_results", []) or []:
        for atom in get_value(result, "atoms", []) or []:
            yield result, atom_dict(atom)


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
                    code=C.PRED_ARITY_MISMATCH,
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


def _argument_value(arg: Any) -> str:
    if isinstance(arg, dict):
        return str(arg.get("value", ""))
    return str(arg)
