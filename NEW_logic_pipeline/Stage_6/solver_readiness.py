from __future__ import annotations

from . import issue_codes as C
from ._utils import as_list, get_value, iter_children, node_type, skeleton_kind
from .predicate_validator import iter_atoms
from .validation_models import ValidationIssue


REVIEW_WARNING_CODES = {
    C.SEMANTIC_OBJECT_CHANGED,
    C.SEMANTIC_SOURCE_MENTION_DROPPED,
    C.SEMANTIC_DOMAIN_RESTRICTION_LOST,
    C.SEMANTIC_NUMERIC_VALUE_UNTYPED,
}


def classify_solver_readiness(
    parsed_record: dict,
    issues: list[ValidationIssue],
    solver_capabilities: dict,
    registry: dict | None = None,
) -> dict:
    reasons: list[str] = []
    readiness_issues: list[ValidationIssue] = []

    fatal_or_error = [issue for issue in issues if issue.severity in {"fatal", "error"}]
    review_warnings = [issue for issue in issues if issue.code in REVIEW_WARNING_CODES]
    ast_missing = any(issue.code == C.AST_MISSING for issue in issues)

    has_meta = any(
        skeleton_kind(skeleton) == "META"
        for skeleton in get_value(parsed_record, "skeletons", []) or []
    ) or ast_contains(parsed_record, "META")
    has_or = ast_contains(parsed_record, "OR") or formula_tree_contains(parsed_record, "or")
    has_exists = ast_contains(parsed_record, "EXISTS") or formula_tree_contains(parsed_record, "exists") or any(
        skeleton_kind(skeleton) == "EXISTS"
        for skeleton in get_value(parsed_record, "skeletons", []) or []
    )
    has_iff = ast_contains(parsed_record, "IFF") or formula_tree_contains(parsed_record, "iff") or any(
        skeleton_kind(skeleton) == "IFF"
        for skeleton in get_value(parsed_record, "skeletons", []) or []
    )
    has_cardinality = any_atom_requires_lowering(parsed_record, registry or {})
    has_unsupported = ast_contains(parsed_record, "UNSUPPORTED")
    unsupported_results = atomization_unsupported_results(parsed_record)

    unsupported = bool(unsupported_results)
    needs_meta_resolution = False
    needs_lowering = False
    needs_review = False

    if fatal_or_error:
        needs_review = True
        reasons.extend(issue.code for issue in fatal_or_error)

    if review_warnings:
        needs_review = True
        reasons.extend(issue.code for issue in review_warnings)

    if ast_missing:
        reasons.append("ast_missing")

    if unsupported_results:
        reasons.append("unsupported_atomization")
        readiness_issues.append(
            ValidationIssue(
                code=C.READY_UNSUPPORTED_LOGIC,
                severity="error",
                message="At least one atomization result reported unsupported logic.",
                evidence={
                    "unsupported_results": [
                        {
                            "premise_id": get_value(result, "premise_id"),
                            "request_id": get_value(result, "request_id"),
                            "unsupported_reason": get_value(result, "unsupported_reason"),
                        }
                        for result in unsupported_results
                    ]
                },
                suggested_stage="Stage 7 or Stage 8",
            )
        )
        needs_review = True

    if has_unsupported:
        needs_review = True
        unsupported = True
        reasons.append("unsupported_ast_node")
        readiness_issues.append(
            ValidationIssue(
                code=C.READY_UNSUPPORTED_LOGIC,
                severity="error",
                message="AST contains UNSUPPORTED node(s) that cannot be sent to solver.",
                suggested_stage="Stage 5 or Stage 7",
            )
        )

    if has_meta and not solver_capabilities.get("supports_meta", False):
        needs_meta_resolution = True
        reasons.append("meta_requires_resolution")
        readiness_issues.append(
            ValidationIssue(
                code=C.READY_META_REQUIRES_RESOLUTION,
                severity="info",
                message="META formulas require resolution before the current solver can consume them.",
                suggested_stage="Stage 8",
            )
        )

    if has_or and not solver_capabilities.get("supports_or", False):
        needs_lowering = True
        reasons.append("or_requires_lowering")
        readiness_issues.append(
            ValidationIssue(
                code=C.READY_OR_REQUIRES_LOWERING,
                severity="info",
                message="OR requires lowering for the current solver.",
                suggested_stage="Stage 8 or Solver Adapter",
            )
        )

    if has_exists and not solver_capabilities.get("supports_exists", False):
        needs_lowering = True
        reasons.append("exists_requires_skolemization")
        readiness_issues.append(
            ValidationIssue(
                code=C.READY_EXISTS_REQUIRES_SKOLEMIZATION,
                severity="info",
                message="EXISTS requires skolemization or existential support.",
                suggested_stage="Stage 8 or Solver Adapter",
            )
        )

    if has_iff and not solver_capabilities.get("supports_iff", False):
        needs_lowering = True
        reasons.append("iff_requires_lowering")
        readiness_issues.append(
            ValidationIssue(
                code=C.READY_IFF_REQUIRES_LOWERING,
                severity="info",
                message="IFF requires lowering for the current solver.",
                suggested_stage="Stage 8 or Solver Adapter",
            )
        )

    if has_cardinality and not solver_capabilities.get("supports_cardinality", False):
        needs_lowering = True
        reasons.append("cardinality_requires_lowering")
        readiness_issues.append(
            ValidationIssue(
                code=C.READY_CARDINALITY_REQUIRES_LOWERING,
                severity="info",
                message="Cardinality predicates require lowering for the current solver.",
                suggested_stage="Stage 8 or Solver Adapter",
            )
        )

    parse_valid = not any(issue.severity in {"fatal", "error"} for issue in [*issues, *readiness_issues])

    direct_solver_ready = (
        parse_valid
        and not ast_missing
        and not needs_review
        and not needs_meta_resolution
        and not needs_lowering
        and not unsupported
    )

    return {
        "parse_valid": parse_valid,
        "direct_solver_ready": direct_solver_ready,
        "needs_lowering": needs_lowering,
        "needs_meta_resolution": needs_meta_resolution,
        "needs_review": needs_review,
        "unsupported": unsupported,
        "reasons": sorted(set(reasons)),
        "issues": readiness_issues,
    }


def ast_contains(parsed_record: dict, target_type: str) -> bool:
    asts = get_value(parsed_record, "asts") or get_value(parsed_record, "ast")
    if not asts:
        return False

    def walk(node: dict) -> bool:
        if node_type(node, uppercase=True) == target_type:
            return True
        return any(walk(child) for child in iter_children(node))

    return any(walk(ast) for ast in as_list(asts))


def formula_tree_contains(parsed_record: dict, target_type: str) -> bool:
    def walk(node: dict) -> bool:
        if node_type(node) == target_type:
            return True
        return any(walk(child) for child in iter_children(node))

    for skeleton in get_value(parsed_record, "skeletons", []) or []:
        formula_tree = get_value(skeleton, "formula_tree")
        if formula_tree and walk(formula_tree):
            return True
    return False


def any_atom_requires_lowering(parsed_record: dict, registry: dict) -> bool:
    for _, atom in iter_atoms(parsed_record):
        name = atom.get("name")
        if name in registry and registry[name].get("requires_lowering"):
            return True
    return False


def atomization_unsupported_results(parsed_record: dict) -> list[object]:
    return [
        result
        for result in get_value(parsed_record, "atomization_results", []) or []
        if get_value(result, "unsupported_reason")
    ]
