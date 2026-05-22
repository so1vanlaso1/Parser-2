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
    - Incorrect arity for implies/not/and/or/iff nodes
    - Quantifier child count and missing variable
    - Atomic nodes missing name or arguments
    - Missing implies in ONLY_IF_RULE
    - Missing iff in IFF premises
    - Nested implies inside RULE (suggests META reclassification)
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

            # Nested rule detector: RULE with implies-inside-implies → suggest META.
            if item.kind == "RULE":
                if self._has_nested_implies(item.ast):
                    issues.append(
                        ValidationIssue(
                            item.premise_id,
                            "warning",
                            "RULE contains nested implies. Consider reclassifying as META.",
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
        # --- Quantifier checks ---
        if node.type in {"forall", "exists"}:
            if not node.variable:
                issues.append(
                    ValidationIssue(premise_id, "error", f"{node.type} missing variable")
                )
                return

            if len(node.children) != 1:
                issues.append(
                    ValidationIssue(
                        premise_id,
                        "error",
                        f"{node.type} node must have exactly 1 child, has {len(node.children)}.",
                    )
                )

            new_bound = set(bound_vars)
            new_bound.add(node.variable)

            for child in node.children:
                self._walk(child, premise_id, new_bound, issues)
            return

        # --- Atomic checks ---
        if node.type == "atomic":
            if not node.name:
                issues.append(
                    ValidationIssue(premise_id, "error", "atomic node missing name.")
                )

            if not node.arguments:
                issues.append(
                    ValidationIssue(premise_id, "error", f"atomic predicate '{node.name or '?'}' has no arguments.")
                )

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

        # --- NOT checks ---
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

        # --- Implies checks ---
        if node.type == "implies":
            if len(node.children) != 2:
                issues.append(
                    ValidationIssue(premise_id, "error", "IMPLIES node must have exactly 2 children.")
                )

        # --- And / Or / IFF checks ---
        if node.type in {"and", "or", "iff"}:
            if len(node.children) < 2:
                issues.append(
                    ValidationIssue(
                        premise_id,
                        "error",
                        f"{node.type.upper()} node must have at least 2 children, has {len(node.children)}.",
                    )
                )

        # --- Equation checks ---
        if node.type == "equation":
            if node.operator is None or node.left is None or node.right is None:
                issues.append(
                    ValidationIssue(
                        premise_id,
                        "error",
                        "equation node missing operator, left, or right.",
                    )
                )

        for child in node.children:
            self._walk(child, premise_id, bound_vars, issues)

    def contains_node_type(self, node: LogicNode, target: str) -> bool:
        if node.type == target:
            return True
        return any(self.contains_node_type(child, target) for child in node.children)

    def _has_nested_implies(self, node: LogicNode, depth: int = 0) -> bool:
        """Check if an implies node contains another implies node in its subtree."""
        if node.type == "implies":
            if depth > 0:
                return True
            for child in node.children:
                if self._has_nested_implies(child, depth + 1):
                    return True
        else:
            for child in node.children:
                if self._has_nested_implies(child, depth):
                    return True
        return False


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

