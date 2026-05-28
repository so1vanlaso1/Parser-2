from __future__ import annotations

import re
from typing import Any

from . import issue_codes as C
from ._utils import as_list, get_value, iter_children, node_type
from .validation_models import ValidationIssue

# Prefixes that indicate negation was encoded inside a predicate name
_NEGATION_NAME_PATTERNS = re.compile(
    r"^(not_|non_|no_|without_|lack_|cannot_)", re.IGNORECASE
)


def validate_asts(
    parsed_record: dict,
    registry: dict,
    *,
    allow_dynamic_predicates: bool = False,
) -> list[ValidationIssue]:
    asts = get_value(parsed_record, "asts") or get_value(parsed_record, "ast")

    if not asts:
        return [
            ValidationIssue(
                code=C.AST_MISSING,
                severity="warning",
                message="No Stage 5 AST found. Skipping AST validation.",
                suggested_stage="Stage 5",
            )
        ]

    issues: list[ValidationIssue] = []
    # Collect phrase→predicate mappings for consistency checking
    phrase_predicate_map: dict[str, set[str]] = {}

    for index, ast in enumerate(as_list(asts)):
        issues.extend(
            validate_ast_node(
                ast,
                registry,
                path=[index],
                allow_dynamic_predicates=allow_dynamic_predicates,
                phrase_predicate_map=phrase_predicate_map,
            )
        )

    # --- Semantic sanity: inconsistent phrase mappings ---
    issues.extend(_check_phrase_consistency(phrase_predicate_map))

    return issues


def validate_ast_node(
    node: dict,
    registry: dict,
    path: list[int],
    *,
    allow_dynamic_predicates: bool = False,
    phrase_predicate_map: dict[str, set[str]] | None = None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    current_type = node_type(node, uppercase=True)
    children = list(iter_children(node))

    if current_type == "ATOM":
        predicate = get_value(node, "predicate") or get_value(node, "name")
        if not predicate:
            issues.append(
                ValidationIssue(
                    code=C.AST_ATOM_MISSING_PREDICATE,
                    severity="error",
                    message="ATOM node missing predicate.",
                    path=path,
                    evidence={"node": _node_evidence(node)},
                    suggested_stage="Stage 5",
                )
            )
        else:
            if predicate not in registry and not allow_dynamic_predicates:
                issues.append(
                    ValidationIssue(
                        code=C.AST_ATOM_UNKNOWN_PREDICATE,
                        severity="error",
                        message=f"AST contains unknown predicate: {predicate}.",
                        path=path,
                        premise_id=get_value(node, "premise_id"),
                        evidence={
                            "predicate": predicate,
                            "arguments": list(get_value(node, "arguments", []) or []),
                            "premise_id": get_value(node, "premise_id"),
                            "source_text": get_value(node, "source_text"),
                            "registry_size": _registry_predicate_count(registry),
                            "node": _node_evidence(node),
                        },
                        suggested_stage="Stage 5",
                    )
                )

            # Check for negation encoded inside predicate name
            if _NEGATION_NAME_PATTERNS.search(str(predicate)):
                issues.append(
                    ValidationIssue(
                        code=C.AST_NEGATION_IN_PREDICATE_NAME,
                        severity="error",
                        message=f"Predicate name encodes negation: {predicate!r}. Use NOT node instead.",
                        path=path,
                        evidence={"predicate": predicate},
                        suggested_stage="Stage 3 or Stage 4",
                    )
                )

        # Track phrase→predicate for consistency checking
        if phrase_predicate_map is not None and predicate:
            source = get_value(node, "source_text") or get_value(node, "source_phrase")
            if source:
                normalized_source = _normalize_for_consistency(str(source))
                if normalized_source:
                    phrase_predicate_map.setdefault(normalized_source, set()).add(str(predicate))

    elif current_type == "NOT":
        if len(children) != 1:
            issues.append(_bad_child_count(current_type, path, 1, len(children)))

    elif current_type in {"AND", "OR"}:
        if len(children) < 2:
            issues.append(_bad_child_count(current_type, path, "at least 2", len(children)))

    elif current_type in {"IMPLIES", "IFF"}:
        if len(children) != 2:
            issues.append(_bad_child_count(current_type, path, 2, len(children)))

    elif current_type in {"FORALL", "EXISTS"}:
        if not get_value(node, "variable"):
            issues.append(
                ValidationIssue(
                    code=C.AST_BAD_CHILD_COUNT,
                    severity="error",
                    message=f"{current_type} node missing variable.",
                    path=path,
                    evidence={"node": _node_evidence(node)},
                    suggested_stage="Stage 5",
                )
            )
        if len(children) != 1:
            issues.append(_bad_child_count(current_type, path, 1, len(children)))

    elif current_type == "META":
        # META nodes are structurally valid but not solver-ready
        pass

    elif current_type == "UNSUPPORTED":
        reason = get_value(node, "reason") or "unknown"
        issues.append(
            ValidationIssue(
                code=C.AST_UNSUPPORTED_NODE,
                severity="warning",
                message=f"AST contains UNSUPPORTED node: {reason}",
                path=path,
                evidence={"reason": reason, "source_text": get_value(node, "source_text")},
                suggested_stage="Stage 5 or Stage 7",
            )
        )

    else:
        issues.append(
            ValidationIssue(
                code=C.AST_UNKNOWN_NODE_TYPE,
                severity="error",
                message=f"Unknown AST node type: {current_type}.",
                path=path,
                evidence={"node": _node_evidence(node)},
                suggested_stage="Stage 5",
            )
        )

    for index, child in enumerate(children):
        issues.extend(
            validate_ast_node(
                child,
                registry,
                path + [index],
                allow_dynamic_predicates=allow_dynamic_predicates,
                phrase_predicate_map=phrase_predicate_map,
            )
        )

    return issues


def _check_phrase_consistency(phrase_predicate_map: dict[str, set[str]]) -> list[ValidationIssue]:
    """Check that the same normalized phrase maps to the same main predicate(s)."""
    issues: list[ValidationIssue] = []
    for phrase, predicates in phrase_predicate_map.items():
        main_predicates = _main_property_predicates(predicates)
        if len(main_predicates) > 1:
            issues.append(
                ValidationIssue(
                    code=C.AST_INCONSISTENT_PHRASE_MAPPING,
                    severity="warning",
                    message=f"Same phrase maps to multiple main predicates: {sorted(main_predicates)}",
                    evidence={
                        "phrase": phrase,
                        "main_predicates": sorted(main_predicates),
                        "all_predicates": sorted(predicates),
                    },
                    suggested_stage="Stage 3 or Stage 4",
                )
            )
    return issues


def _main_property_predicates(predicates: set[str]) -> set[str]:
    """Return property-like predicates, ignoring bare class atoms when present."""

    multi_token = {predicate for predicate in predicates if "_" in predicate}
    return multi_token or set(predicates)


def _normalize_for_consistency(text: str) -> str:
    """Normalize a phrase for consistency comparison.

    General linguistic normalization: lowercase, strip articles/determiners,
    collapse whitespace.
    """
    lower = re.sub(r"\s+", " ", text.strip()).lower()
    lower = re.sub(r"^(a|an|the|this|that|some|any)\s+", "", lower)
    return lower.strip()


def _bad_child_count(node_type: str, path: list[int], expected, actual: int) -> ValidationIssue:
    return ValidationIssue(
        code=C.AST_BAD_CHILD_COUNT,
        severity="error",
        message=f"{node_type} expects {expected} children, got {actual}.",
        path=path,
        evidence={"node_type": node_type, "actual": actual, "expected": expected},
        suggested_stage="Stage 5",
    )


def _node_evidence(node: object) -> dict:
    if isinstance(node, dict):
        return node
    if hasattr(node, "model_dump"):
        return node.model_dump(mode="json")
    return {"repr": repr(node)}


def _registry_predicate_count(registry: dict) -> int:
    return sum(1 for name in registry if not str(name).startswith("__"))
