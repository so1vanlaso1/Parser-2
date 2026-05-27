from __future__ import annotations

from . import issue_codes as C
from ._utils import get_value, iter_children, node_type, skeleton_kind
from .validation_models import ValidationIssue


VALID_KINDS = {
    "FACT",
    "EXISTS",
    "FORALL",
    "RULE",
    "ONLY_IF_RULE",
    "IFF",
    "NON_IF_RULE",
    "OBLIGATION_RULE",
    "MODAL",
    "META",
    "UNKNOWN",
}

FORMULA_CUES = (
    " if ",
    " then ",
    " implies ",
    " only if ",
    " if and only if ",
    " there exists ",
    " every ",
    " all ",
    " forall ",
    " exists ",
)


def validate_skeletons(parsed_record: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for skeleton in get_value(parsed_record, "skeletons", []) or []:
        premise_id = get_value(skeleton, "premise_id")
        kind = skeleton_kind(skeleton)

        if kind not in VALID_KINDS:
            issues.append(
                ValidationIssue(
                    code=C.STRUCT_UNKNOWN_SKELETON_KIND,
                    severity="error",
                    message=f"Unknown skeleton kind: {kind}",
                    premise_id=premise_id,
                    evidence={"kind": kind},
                    suggested_stage="Stage 1",
                )
            )
            continue

        if kind == "UNKNOWN":
            issues.append(
                ValidationIssue(
                    code=C.STRUCT_UNKNOWN_SKELETON_KIND,
                    severity="error",
                    message="Skeleton kind is UNKNOWN and cannot be considered structurally safe.",
                    premise_id=premise_id,
                    evidence={"kind": kind, "notes": list(get_value(skeleton, "notes", []) or [])},
                    suggested_stage="Stage 1",
                )
            )

        if kind == "FORALL":
            if not get_value(skeleton, "antecedent") or not get_value(skeleton, "consequent"):
                issues.append(_missing_field(premise_id, kind, "antecedent/consequent"))

        elif kind == "EXISTS":
            if not get_value(skeleton, "body") and not get_value(skeleton, "formula_tree"):
                issues.append(_missing_field(premise_id, kind, "body/formula_tree"))

        elif kind in {"RULE", "ONLY_IF_RULE", "NON_IF_RULE", "OBLIGATION_RULE"}:
            if kind == "OBLIGATION_RULE":
                if not get_value(skeleton, "body"):
                    issues.append(_missing_field(premise_id, kind, "body"))
            elif not get_value(skeleton, "antecedent") or not get_value(skeleton, "consequent"):
                issues.append(_missing_field(premise_id, kind, "antecedent/consequent"))

        elif kind == "IFF":
            if not get_value(skeleton, "left") or not get_value(skeleton, "right"):
                issues.append(_missing_field(premise_id, kind, "left/right"))

        elif kind == "META":
            formula_tree = get_value(skeleton, "formula_tree")
            if not formula_tree:
                issues.append(
                    ValidationIssue(
                        code=C.STRUCT_META_WITHOUT_FORMULA_TREE,
                        severity="error",
                        message="META skeleton must contain formula_tree.",
                        premise_id=premise_id,
                        suggested_stage="Stage 1",
                    )
                )
            else:
                issues.extend(validate_formula_tree(skeleton, formula_tree))

    return issues


def _missing_field(premise_id: str | None, kind: str, field_name: str) -> ValidationIssue:
    return ValidationIssue(
        code=C.STRUCT_MISSING_REQUIRED_FIELD,
        severity="error",
        message=f"{kind} skeleton missing required field: {field_name}.",
        premise_id=premise_id,
        evidence={"kind": kind, "missing": field_name},
        suggested_stage="Stage 1",
    )


def validate_formula_tree(skeleton: dict, root: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    premise_id = get_value(skeleton, "premise_id")

    def walk(node: dict, path: list[int]) -> None:
        current_type = node_type(node)
        children = list(iter_children(node))

        if current_type == "leaf":
            text = f" {str(get_value(node, 'text', '')).lower()} "
            if any(cue in text for cue in FORMULA_CUES):
                issues.append(
                    ValidationIssue(
                        code=C.STRUCT_FORMULA_LIKE_LEAF,
                        severity="error",
                        message="Formula-like text reached atomization as a leaf.",
                        premise_id=premise_id,
                        path=path,
                        evidence={"leaf_text": get_value(node, "text")},
                        suggested_stage="Stage 1 or Stage 2",
                    )
                )
            return

        expected_children = {
            "implies": 2,
            "iff": 2,
            "forall": 1,
            "exists": 1,
            "not": 1,
        }

        if current_type in expected_children:
            expected = expected_children[current_type]
            actual = len(children)
            if actual != expected:
                issues.append(
                    ValidationIssue(
                        code=C.STRUCT_BAD_NODE_CHILD_COUNT,
                        severity="error",
                        message=f"Formula node {current_type} expects {expected} children, got {actual}.",
                        premise_id=premise_id,
                        path=path,
                        evidence={"node_type": current_type, "actual_children": actual},
                        suggested_stage="Stage 1",
                    )
                )

        elif current_type in {"and", "or"}:
            if len(children) < 2:
                issues.append(
                    ValidationIssue(
                        code=C.STRUCT_BAD_NODE_CHILD_COUNT,
                        severity="error",
                        message=f"Formula node {current_type} expects at least 2 children.",
                        premise_id=premise_id,
                        path=path,
                        suggested_stage="Stage 1",
                    )
                )

        elif current_type not in {"equation", "comparison", "cardinality"}:
            issues.append(
                ValidationIssue(
                    code=C.STRUCT_BAD_NODE_CHILD_COUNT,
                    severity="error",
                    message=f"Unknown formula node type: {current_type}.",
                    premise_id=premise_id,
                    path=path,
                    evidence={"node_type": current_type},
                    suggested_stage="Stage 1",
                )
            )

        for index, child in enumerate(children):
            walk(child, path + [index])

    walk(root, [])
    return issues
