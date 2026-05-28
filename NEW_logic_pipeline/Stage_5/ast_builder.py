from __future__ import annotations

"""Stage 5: deterministic AST builder.

Input:  LogicSkeletons + canonical AtomizationResults from Stages 1–4.
Output: LogicNode AST trees.

This module deterministically builds typed logic AST trees from the structural
information in skeletons and the flat predicate atoms from atomization.  It does
not call an LLM, resolve META, or export to a solver.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

try:
    from ..Stage_1.logic_skeleton import FormulaSkeleton, LogicSkeleton
    from ..Stage_2.atomization_requests import AtomizationResult, PredicateAtom
except Exception:  # pragma: no cover
    from NEW_logic_pipeline.Stage_1.logic_skeleton import FormulaSkeleton, LogicSkeleton
    from NEW_logic_pipeline.Stage_2.atomization_requests import AtomizationResult, PredicateAtom


# ---------------------------------------------------------------------------
# AST node model
# ---------------------------------------------------------------------------

ASTNodeType = Literal[
    "ATOM",
    "NOT",
    "AND",
    "OR",
    "IMPLIES",
    "IFF",
    "FORALL",
    "EXISTS",
    "META",
    "UNSUPPORTED",
]


class LogicNode(BaseModel):
    """A typed logic AST node.

    Leaf nodes have ``type == "ATOM"`` and carry a predicate with arguments.
    Interior nodes have children.  ``FORALL`` / ``EXISTS`` nodes also carry
    a bound variable.  ``UNSUPPORTED`` nodes carry a human-readable reason.
    ``META`` nodes represent formulas about formulas and are never directly
    solver-ready.
    """

    type: ASTNodeType
    predicate: str | None = None
    arguments: list[Any] = Field(default_factory=list)
    negated: bool = False
    variable: str | None = None
    children: list["LogicNode"] = Field(default_factory=list)
    premise_id: str | None = None
    source_text: str | None = None
    evidence_links: list[str] = Field(default_factory=list)
    reason: str | None = None
    confidence: float = 1.0

    def pretty(self, indent: int = 0) -> str:
        """Return a human-readable multi-line representation."""
        pad = "  " * indent
        if self.type == "ATOM":
            neg = "NOT " if self.negated else ""
            args = ", ".join(str(a) for a in self.arguments)
            return f"{pad}{neg}{self.predicate}({args})"
        if self.type == "UNSUPPORTED":
            return f"{pad}UNSUPPORTED: {self.reason}"
        parts = [f"{pad}{self.type}"]
        if self.variable:
            parts[0] += f" {self.variable}"
        for child in self.children:
            parts.append(child.pretty(indent + 1))
        return "\n".join(parts)


LogicNode.model_rebuild()


# ---------------------------------------------------------------------------
# AST building from skeleton + atomization results
# ---------------------------------------------------------------------------

def build_ast(
    skeleton: LogicSkeleton,
    atomization_results: list[AtomizationResult],
) -> LogicNode:
    """Build a LogicNode AST from a skeleton and its matching atomization results.

    The builder is deterministic: it uses the skeleton ``kind`` and ``quantifier``
    to decide the AST shape, and slots canonical atoms from atomization results
    into the appropriate leaf positions.
    """
    kind = _kind(skeleton)
    premise_id = str(_get(skeleton, "premise_id", "P?"))
    original = str(_get(skeleton, "original", ""))

    # --- Index atomization results by role ---
    by_role = _index_by_role(atomization_results, premise_id)

    if kind == "FACT":
        atoms = _atoms_for_role(by_role, ("body",))
        node = _atoms_to_conjunction(atoms, premise_id)
        node.source_text = original
        return node

    if kind == "EXISTS":
        variable = str(_get(skeleton, "variable") or "x")
        body_atoms = _atoms_for_role(by_role, ("body",))
        body = _atoms_to_conjunction(body_atoms, premise_id)
        return LogicNode(
            type="EXISTS",
            variable=variable,
            children=[body],
            premise_id=premise_id,
            source_text=original,
        )

    if kind == "FORALL":
        variable = str(_get(skeleton, "variable") or "x")
        restrictor_atoms = _atoms_for_role(by_role, ("restrictor", "antecedent"))
        property_atoms = _atoms_for_role(by_role, ("property", "consequent"))
        return _forall_implies(
            variable, restrictor_atoms, property_atoms, premise_id, original,
        )

    if kind in {"RULE", "ONLY_IF_RULE", "NON_IF_RULE"}:
        variable = str(_get(skeleton, "variable") or "x")
        antecedent_atoms = _atoms_for_role(by_role, ("antecedent",))
        consequent_atoms = _atoms_for_role(by_role, ("consequent",))
        return _forall_implies(
            variable, antecedent_atoms, consequent_atoms, premise_id, original,
        )

    if kind == "IFF":
        variable = str(_get(skeleton, "variable") or "x")
        left_atoms = _atoms_for_role(by_role, ("left",))
        right_atoms = _atoms_for_role(by_role, ("right",))
        left = _atoms_to_conjunction(left_atoms, premise_id)
        right = _atoms_to_conjunction(right_atoms, premise_id)
        return LogicNode(
            type="FORALL",
            variable=variable,
            children=[
                LogicNode(
                    type="IFF",
                    children=[left, right],
                    premise_id=premise_id,
                )
            ],
            premise_id=premise_id,
            source_text=original,
        )

    if kind == "META":
        formula_tree = _get(skeleton, "formula_tree")
        if formula_tree is not None:
            # Index leaf results by formula_path for META
            by_path = _index_by_formula_path(atomization_results, premise_id)
            inner = _build_formula_ast(formula_tree, by_path, premise_id, [])
            return LogicNode(
                type="META",
                children=[inner],
                premise_id=premise_id,
                source_text=original,
            )
        # META without formula tree — fall through to UNSUPPORTED
        return LogicNode(
            type="UNSUPPORTED",
            premise_id=premise_id,
            source_text=original,
            reason="META skeleton without formula_tree",
        )

    # OBLIGATION_RULE, MODAL, UNKNOWN → UNSUPPORTED
    return LogicNode(
        type="UNSUPPORTED",
        premise_id=premise_id,
        source_text=original,
        reason=f"skeleton kind {kind} is not directly solver-ready",
    )


def build_asts(
    skeletons: list[LogicSkeleton],
    atomization_results: list[AtomizationResult],
) -> list[LogicNode]:
    """Build ASTs for a batch of skeletons."""
    return [build_ast(skeleton, atomization_results) for skeleton in skeletons]


# ---------------------------------------------------------------------------
# Formula tree → AST (for META skeletons)
# ---------------------------------------------------------------------------

def _build_formula_ast(
    node: FormulaSkeleton | dict,
    by_path: dict[str, list[PredicateAtom]],
    premise_id: str,
    current_path: list[int],
) -> LogicNode:
    """Recursively convert a FormulaSkeleton into LogicNode AST.

    Leaf nodes are replaced with ATOM nodes from matching atomization results.
    Structural nodes (implies, forall, exists, not, and, or) become the
    corresponding LogicNode types.
    """
    node_type = _formula_node_type(node)
    children = list(_get(node, "children") or [])
    variable = str(_get(node, "variable") or "x")

    if node_type == "leaf":
        path_key = _path_key(current_path)
        atoms = by_path.get(path_key, [])
        if atoms:
            return _predicate_atoms_to_node(atoms, premise_id)
        # No atoms found for this leaf — create placeholder from text
        text = str(_get(node, "text") or "")
        return LogicNode(
            type="UNSUPPORTED",
            premise_id=premise_id,
            source_text=text,
            reason=f"no atomization result for formula leaf at path {current_path}",
        )

    if node_type == "not":
        if children:
            inner = _build_formula_ast(children[0], by_path, premise_id, [*current_path, 0])
            return LogicNode(type="NOT", children=[inner], premise_id=premise_id)
        return LogicNode(type="UNSUPPORTED", premise_id=premise_id, reason="NOT with no children")

    if node_type == "implies":
        child_nodes = [
            _build_formula_ast(child, by_path, premise_id, [*current_path, i])
            for i, child in enumerate(children)
        ]
        if len(child_nodes) == 2:
            return LogicNode(type="IMPLIES", children=child_nodes, premise_id=premise_id)
        return LogicNode(type="UNSUPPORTED", premise_id=premise_id, reason=f"IMPLIES with {len(child_nodes)} children")

    if node_type == "iff":
        child_nodes = [
            _build_formula_ast(child, by_path, premise_id, [*current_path, i])
            for i, child in enumerate(children)
        ]
        if len(child_nodes) == 2:
            return LogicNode(type="IFF", children=child_nodes, premise_id=premise_id)
        return LogicNode(type="UNSUPPORTED", premise_id=premise_id, reason=f"IFF with {len(child_nodes)} children")

    if node_type in {"and", "or"}:
        child_nodes = [
            _build_formula_ast(child, by_path, premise_id, [*current_path, i])
            for i, child in enumerate(children)
        ]
        return LogicNode(
            type=node_type.upper(),
            children=child_nodes,
            premise_id=premise_id,
        )

    if node_type == "forall":
        if children:
            inner = _build_formula_ast(children[0], by_path, premise_id, [*current_path, 0])
            return LogicNode(type="FORALL", variable=variable, children=[inner], premise_id=premise_id)
        return LogicNode(type="UNSUPPORTED", premise_id=premise_id, reason="FORALL with no children")

    if node_type == "exists":
        if children:
            inner = _build_formula_ast(children[0], by_path, premise_id, [*current_path, 0])
            return LogicNode(type="EXISTS", variable=variable, children=[inner], premise_id=premise_id)
        return LogicNode(type="UNSUPPORTED", premise_id=premise_id, reason="EXISTS with no children")

    # equation, comparison, cardinality, unknown → UNSUPPORTED
    return LogicNode(
        type="UNSUPPORTED",
        premise_id=premise_id,
        reason=f"unsupported formula node type: {node_type}",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _forall_implies(
    variable: str,
    antecedent_atoms: list[PredicateAtom],
    consequent_atoms: list[PredicateAtom],
    premise_id: str,
    source_text: str,
) -> LogicNode:
    """Build FORALL variable (IMPLIES antecedent consequent)."""
    antecedent = _atoms_to_conjunction(antecedent_atoms, premise_id)
    consequent = _atoms_to_conjunction(consequent_atoms, premise_id)
    return LogicNode(
        type="FORALL",
        variable=variable,
        children=[
            LogicNode(
                type="IMPLIES",
                children=[antecedent, consequent],
                premise_id=premise_id,
            )
        ],
        premise_id=premise_id,
        source_text=source_text,
    )


def _atoms_to_conjunction(atoms: list[PredicateAtom], premise_id: str) -> LogicNode:
    """Convert a list of PredicateAtom to a single LogicNode.

    One atom → single ATOM node.
    Multiple atoms → AND node wrapping them.
    No atoms → UNSUPPORTED node.
    """
    if not atoms:
        return LogicNode(
            type="UNSUPPORTED",
            premise_id=premise_id,
            reason="no atoms available for this position",
        )

    nodes = [_atom_to_node(atom, premise_id) for atom in atoms]
    if len(nodes) == 1:
        return nodes[0]
    return LogicNode(type="AND", children=nodes, premise_id=premise_id)


def _atom_to_node(atom: PredicateAtom, premise_id: str) -> LogicNode:
    """Convert a single PredicateAtom to a LogicNode.

    If the atom is negated, wrap it in a NOT node.
    """
    inner = LogicNode(
        type="ATOM",
        predicate=atom.name,
        arguments=list(atom.arguments),
        negated=False,
        premise_id=premise_id,
        source_text=atom.source_phrase,
        evidence_links=list(atom.evidence_links),
        confidence=atom.confidence,
    )
    if atom.negated:
        return LogicNode(type="NOT", children=[inner], premise_id=premise_id)
    return inner


def _predicate_atoms_to_node(atoms: list[PredicateAtom], premise_id: str) -> LogicNode:
    """Convert a list of predicate atoms into a single node (conjunction if multiple)."""
    return _atoms_to_conjunction(atoms, premise_id)


def _index_by_role(
    results: list[AtomizationResult],
    premise_id: str,
) -> dict[str, list[PredicateAtom]]:
    """Index atomization results by their role for a given premise."""
    by_role: dict[str, list[PredicateAtom]] = {}
    for result in results:
        if str(_get(result, "premise_id")) != premise_id:
            continue
        role = str(_get(result, "role", ""))
        atoms = list(_get(result, "atoms") or [])
        by_role.setdefault(role, []).extend(atoms)
    return by_role


def _index_by_formula_path(
    results: list[AtomizationResult],
    premise_id: str,
) -> dict[str, list[PredicateAtom]]:
    """Index atomization results by their formula_path for META premises."""
    by_path: dict[str, list[PredicateAtom]] = {}
    for result in results:
        if str(_get(result, "premise_id")) != premise_id:
            continue
        path = list(_get(result, "formula_path") or [])
        key = _path_key(path)
        atoms = list(_get(result, "atoms") or [])
        by_path.setdefault(key, []).extend(atoms)
    return by_path


def _atoms_for_role(
    by_role: dict[str, list[PredicateAtom]],
    role_names: tuple[str, ...],
) -> list[PredicateAtom]:
    """Collect atoms from the first matching role."""
    for role in role_names:
        atoms = by_role.get(role, [])
        if atoms:
            return atoms
    return []


def _path_key(path: list[int]) -> str:
    """Convert a formula path to a string key."""
    return "_".join(str(p) for p in path) if path else "root"


def _kind(obj: Any) -> str:
    value = _get(obj, "kind", "UNKNOWN")
    return str(value.value) if hasattr(value, "value") else str(value)


def _formula_node_type(node: Any) -> str:
    value = _get(node, "type", "leaf")
    return str(value.value) if hasattr(value, "value") else str(value).lower()


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


__all__ = [
    "ASTNodeType",
    "LogicNode",
    "build_ast",
    "build_asts",
]
