from dataclasses import dataclass, field

from .schemas import LogicNode, Stage3Output


@dataclass
class ValidationIssue:
    premise_id: str
    severity: str  # "error", "warning"
    message: str


@dataclass
class ValidationReport:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)


class LogicValidator:
    """
    Python-side validator for Stage 3 AST output.

    Catches structural errors that would break a solver:
    - Unbound variables
    - Negation encoded in predicate names
    - Double negation
    - Incorrect arity for implies/not nodes
    - Missing implies in ONLY_IF_RULE
    - Missing iff in IFF premises
    """

    def validate_stage3(self, output: Stage3Output) -> ValidationReport:
        issues: list[ValidationIssue] = []

        for item in output.compiled:
            issues.extend(self.validate_node(item.premise_id, item.ast))

            if item.kind == "ONLY_IF_RULE":
                if not self.contains_node_type(item.ast, "implies"):
                    issues.append(
                        ValidationIssue(
                            item.premise_id,
                            "error",
                            "ONLY_IF_RULE must compile to an implies node.",
                        )
                    )

            if item.kind == "IFF":
                if not self.contains_node_type(item.ast, "iff"):
                    issues.append(
                        ValidationIssue(
                            item.premise_id,
                            "error",
                            "IFF premise must compile to an iff node.",
                        )
                    )

        return ValidationReport(
            ok=not any(i.severity == "error" for i in issues),
            issues=issues,
        )

    def validate_node(self, premise_id: str, node: LogicNode) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        bound_vars: set[str] = set()

        self._walk(node, premise_id, bound_vars, issues)
        return issues

    def _walk(
        self,
        node: LogicNode,
        premise_id: str,
        bound_vars: set[str],
        issues: list[ValidationIssue],
    ):
        if node.type in {"forall", "exists"}:
            if not node.variable:
                issues.append(
                    ValidationIssue(premise_id, "error", f"{node.type} missing variable")
                )
                return

            new_bound = set(bound_vars)
            new_bound.add(node.variable)

            for child in node.children:
                self._walk(child, premise_id, new_bound, issues)
            return

        if node.type == "atomic":
            for arg in node.arguments:
                # Constants (multi-char names like john, alphanet) are always allowed.
                # Single-letter variables (x, y, z, s, t, u, v, w) must be bound.
                if len(arg) == 1 and arg.isalpha() and arg.islower() and arg not in bound_vars:
                    issues.append(
                        ValidationIssue(
                            premise_id,
                            "error",
                            f"Unbound variable '{arg}' in atomic predicate {node.name}",
                        )
                    )

            if node.name and node.name.startswith("not_"):
                issues.append(
                    ValidationIssue(
                        premise_id,
                        "error",
                        f"Predicate name '{node.name}' contains negation. Use a NOT node instead.",
                    )
                )

        if node.type == "not":
            if len(node.children) != 1:
                issues.append(
                    ValidationIssue(premise_id, "error", "NOT node must have exactly one child.")
                )

            child = node.children[0] if node.children else None
            if child and child.type == "not":
                issues.append(
                    ValidationIssue(
                        premise_id,
                        "warning",
                        "Double negation found. Consider simplifying.",
                    )
                )

        if node.type == "implies":
            if len(node.children) != 2:
                issues.append(
                    ValidationIssue(premise_id, "error", "IMPLIES node must have exactly 2 children.")
                )

        for child in node.children:
            self._walk(child, premise_id, bound_vars, issues)

    def contains_node_type(self, node: LogicNode, target: str) -> bool:
        if node.type == target:
            return True
        return any(self.contains_node_type(child, target) for child in node.children)


def classify_solver_readiness(kind: str, ast_type: str, risk_flags: list[str]) -> str:
    """
    Classify a compiled premise into solver-readiness buckets.

    Returns: "solver_ready", "needs_review", "needs_lowering", or "unsupported"
    """
    if "modal_not_necessarily" in risk_flags:
        return "needs_review"

    if "modal_scope_ambiguous" in risk_flags:
        return "needs_review"

    if kind in {"META", "OBLIGATION_RULE"}:
        return "needs_review"

    if ast_type in {"atomic", "forall", "exists", "implies", "and", "not"}:
        return "solver_ready"

    if ast_type in {"or", "iff", "equation"}:
        return "needs_lowering"

    return "unsupported"
