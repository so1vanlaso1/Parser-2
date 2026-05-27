from __future__ import annotations

from typing import Any

from . import issue_codes as C
from ._utils import get_value
from .predicate_validator import iter_atoms
from .registry_schema import incompatible_role_sets
from .validation_models import ValidationIssue


VARIABLE_NAMES = {"u", "v", "w", "x", "y", "z"}


def normalize_argument(arg: Any) -> dict[str, Any]:
    if isinstance(arg, dict):
        return {
            "value": str(arg.get("value", "")),
            "kind": str(arg.get("kind") or _infer_kind(arg.get("value", ""))),
            "semantic_type": arg.get("semantic_type"),
        }
    value = str(arg)
    return {
        "value": value,
        "kind": _infer_kind(value),
        "semantic_type": None,
    }


def validate_argument_roles(parsed_record: dict, registry: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for result, atom in iter_atoms(parsed_record):
        name = atom.get("name")
        if name not in registry:
            continue

        roles = list(registry[name].get("roles", []))
        args = list(atom.get("arguments", []) or [])

        if len(args) != len(roles):
            continue

        for index, expected_role in enumerate(roles):
            arg = normalize_argument(args[index])
            actual_type = arg.get("semantic_type")
            if actual_type and actual_type != expected_role:
                issues.append(
                    ValidationIssue(
                        code=C.ARG_ROLE_MISMATCH,
                        severity="error",
                        message=f"Argument {index} of {name} should be {expected_role}, got {actual_type}.",
                        premise_id=get_value(result, "premise_id"),
                        request_id=get_value(result, "request_id"),
                        evidence={
                            "atom": atom,
                            "argument": arg,
                            "expected_role": expected_role,
                        },
                        suggested_stage="Stage 3 or Stage 4",
                    )
                )

    issues.extend(validate_type_unification(parsed_record, registry))
    return issues


def collect_type_constraints(parsed_record: dict, registry: dict) -> dict[str, set[str]]:
    constraints: dict[str, set[str]] = {}

    for _, atom in iter_atoms(parsed_record):
        name = atom.get("name")
        if name not in registry:
            continue

        roles = list(registry[name].get("roles", []))
        args = list(atom.get("arguments", []) or [])

        for arg, role in zip(args, roles):
            value = normalize_argument(arg)["value"]
            if value:
                constraints.setdefault(value, set()).add(str(role))

    return constraints


def validate_type_unification(parsed_record: dict, registry: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    constraints = collect_type_constraints(parsed_record, registry)
    bad_role_sets = incompatible_role_sets(registry)

    if not bad_role_sets:
        return issues

    for value, roles in constraints.items():
        for bad_set in bad_role_sets:
            if not bad_set.issubset(roles):
                continue
            issues.append(
                ValidationIssue(
                    code=C.ARG_ROLE_MISMATCH,
                    severity="error",
                    message=f"Argument {value} has incompatible roles: {sorted(roles)}.",
                    evidence={"argument": value, "roles": sorted(roles)},
                    suggested_stage="Stage 3 or Stage 4",
                )
            )
            break

    return issues


def _infer_kind(value: Any) -> str:
    text = str(value)
    if text in VARIABLE_NAMES:
        return "variable"
    return "constant"


__all__ = [
    "collect_type_constraints",
    "normalize_argument",
    "validate_argument_roles",
    "validate_type_unification",
]
