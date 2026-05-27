from __future__ import annotations

from . import issue_codes as C
from ._utils import as_list, get_value, iter_children, node_type
from .validation_models import ValidationIssue


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
    for index, ast in enumerate(as_list(asts)):
        issues.extend(
            validate_ast_node(
                ast,
                registry,
                path=[index],
                allow_dynamic_predicates=allow_dynamic_predicates,
            )
        )
    return issues


def validate_ast_node(
    node: dict,
    registry: dict,
    path: list[int],
    *,
    allow_dynamic_predicates: bool = False,
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
        elif predicate not in registry and not allow_dynamic_predicates:
            issues.append(
                ValidationIssue(
                    code=C.AST_ATOM_UNKNOWN_PREDICATE,
                    severity="error",
                    message=f"AST contains unknown predicate: {predicate}.",
                    path=path,
                    evidence={"node": _node_evidence(node)},
                    suggested_stage="Stage 5",
                )
            )

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
        pass

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
            )
        )

    return issues


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
