from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from .json_utils import extract_json_object
from .llm_client import ChatModel
from .schemas import CompiledPremise, FlatAtom, LogicNode, Stage3Output


META_PATTERNS = [
    r"\bif\b.*\bthen\s+if\b",
    r"\bif\s+there exists\b.*\bthen\b",
    r"\bif\s+there is at least one\b.*\bthen\b",
    r"\bif\s+at least one\b.*\bthen\b",
    r"\bif\b.*\bimplies\b.*\bthen\b",
    r"\bif\b.*\bleads to\b.*\bthen\b",
    r"\bif\b.*\brequires\b.*\bthen\b",
    r"\bif\b.*\bevery\b.*\bthen\b.*\bif\b",
    r"\bif\b.*\ball\b.*\bthen\b.*\bif\b",
]


META_ATOMIZER_SYSTEM_PROMPT = """\
/no_think

You are an atomizer for a logic parser.

Task:
Convert one leaf phrase into flat predicate atoms.

Rules:
- Do not infer.
- Do not solve.
- Do not create implication.
- Do not create quantifiers.
- Output only atoms.
- Preserve negation.
- Use lowercase snake_case predicates.
- Use the provided variable exactly.
- Output valid JSON only.

Input:
{
  "phrase": "...",
  "variable": "x"
}

Output schema:
{
  "atoms": [
    {
      "predicate": "predicate_name",
      "arguments": ["x"],
      "negated": false
    }
  ]
}
"""


def is_higher_order_or_meta(text: str) -> bool:
    s = text.lower().strip()
    return any(re.search(pattern, s) for pattern in META_PATTERNS)


def split_meta_formula(text: str) -> dict[str, Any]:
    """Split a META/nested premise into a text-leaf formula tree.

    The splitter intentionally preserves formula structure and leaves predicate
    extraction for the leaf atomizer.
    """
    clean = _strip_sentence(text)
    outer = _split_if_then(clean)
    if outer is None:
        raise ValueError("META formula must have an outer if/then structure")

    antecedent_text, consequent_text = outer
    return {
        "type": "implies",
        "antecedent": _parse_formula_part(antecedent_text, variable=_choose_variable(antecedent_text, "y")),
        "consequent": _parse_formula_part(consequent_text, variable="x"),
    }


def process_formula_node(
    node: Any,
    llm_atomizer: Callable[[str, str], list[dict[str, Any]]],
    *,
    premise_id: str,
    flat_atoms: list[FlatAtom] | None = None,
    current_variable: str = "x",
    counter: list[int] | None = None,
) -> tuple[Any, list[FlatAtom]]:
    """Atomize formula leaves while preserving formula-level structure."""
    if flat_atoms is None:
        flat_atoms = []
    if counter is None:
        counter = [1]

    if isinstance(node, str):
        return _atomize_leaf_to_formula(
            node,
            current_variable,
            llm_atomizer,
            premise_id=premise_id,
            flat_atoms=flat_atoms,
            counter=counter,
        ), flat_atoms

    if not isinstance(node, dict):
        raise ValueError(f"Unsupported formula node: {node!r}")

    node_type = node.get("type")

    if node_type == "leaf":
        return _atomize_leaf_to_formula(
            str(node.get("text", "")),
            str(node.get("variable") or current_variable),
            llm_atomizer,
            premise_id=premise_id,
            flat_atoms=flat_atoms,
            counter=counter,
        ), flat_atoms

    if node_type == "not":
        child, _ = process_formula_node(
            node.get("child") or _single_child(node),
            llm_atomizer,
            premise_id=premise_id,
            flat_atoms=flat_atoms,
            current_variable=current_variable,
            counter=counter,
        )
        return {"type": "not", "children": [child]}, flat_atoms

    if node_type in {"and", "or"}:
        children = [
            process_formula_node(
                child,
                llm_atomizer,
                premise_id=premise_id,
                flat_atoms=flat_atoms,
                current_variable=current_variable,
                counter=counter,
            )[0]
            for child in node.get("children", [])
        ]
        return {"type": node_type, "children": children}, flat_atoms

    if node_type == "implies":
        antecedent = node.get("antecedent")
        consequent = node.get("consequent")
        if antecedent is None or consequent is None:
            children = node.get("children", [])
            if len(children) != 2:
                raise ValueError("implies node must have antecedent/consequent or 2 children")
            antecedent, consequent = children
        return {
            "type": "implies",
            "children": [
                process_formula_node(
                    antecedent,
                    llm_atomizer,
                    premise_id=premise_id,
                    flat_atoms=flat_atoms,
                    current_variable=current_variable,
                    counter=counter,
                )[0],
                process_formula_node(
                    consequent,
                    llm_atomizer,
                    premise_id=premise_id,
                    flat_atoms=flat_atoms,
                    current_variable=current_variable,
                    counter=counter,
                )[0],
            ],
        }, flat_atoms

    if node_type in {"forall", "exists"}:
        variable = str(node.get("variable") or current_variable)
        body = node.get("body")
        if body is None:
            body = _single_child(node)
        child, _ = process_formula_node(
            body,
            llm_atomizer,
            premise_id=premise_id,
            flat_atoms=flat_atoms,
            current_variable=variable,
            counter=counter,
        )
        return {"type": node_type, "variable": variable, "children": [child]}, flat_atoms

    if node_type == "forall_rule":
        variable = str(node.get("variable") or current_variable)
        return process_formula_node(
            {
                "type": "forall",
                "variable": variable,
                "body": {
                    "type": "implies",
                    "antecedent": _as_leaf(node.get("antecedent"), variable),
                    "consequent": _as_leaf(node.get("consequent"), variable),
                },
            },
            llm_atomizer,
            premise_id=premise_id,
            flat_atoms=flat_atoms,
            current_variable=variable,
            counter=counter,
        )

    raise ValueError(f"Unsupported formula node type: {node_type}")


def atomize_leaf_with_llm(
    phrase: str,
    variable: str,
    llm: ChatModel,
    *,
    max_new_tokens: int = 300,
) -> list[dict[str, Any]]:
    user_prompt = json.dumps(
        {"phrase": phrase, "variable": variable},
        ensure_ascii=False,
        indent=2,
    )
    raw_text = llm.generate(
        META_ATOMIZER_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.0,
        max_new_tokens=max_new_tokens,
    )
    data = extract_json_object(raw_text)
    atoms = data.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        raise ValueError("atomizer returned no atoms")

    normalized = []
    for atom in atoms:
        predicate = atom.get("predicate") or atom.get("name")
        arguments = atom.get("arguments")
        if not predicate or not isinstance(arguments, list) or not arguments:
            raise ValueError(f"invalid atomizer atom: {atom!r}")
        normalized.append(
            {
                "predicate": str(predicate),
                "arguments": [str(arg) for arg in arguments],
                "negated": bool(atom.get("negated", atom.get("polarity") == "negative")),
            }
        )
    return normalized


def formula_tree_to_logic_node(formula_tree: Any, flat_atoms: list[FlatAtom]) -> LogicNode:
    atom_by_id = {atom.atom_id: atom for atom in flat_atoms}

    def build(node: Any) -> LogicNode:
        if isinstance(node, str):
            atom = atom_by_id[node]
            atomic = LogicNode(
                type="atomic",
                name=atom.predicate,
                arguments=atom.arguments,
                source_premise_id=atom.source_premise_id,
            )
            if atom.negated:
                return LogicNode(type="not", children=[atomic], source_premise_id=atom.source_premise_id)
            return atomic

        if not isinstance(node, dict):
            raise ValueError(f"Unsupported processed formula node: {node!r}")

        node_type = node.get("type")
        if node_type in {"and", "or", "implies", "iff"}:
            return LogicNode(
                type=node_type,
                children=[build(child) for child in node.get("children", [])],
            )
        if node_type == "not":
            return LogicNode(type="not", children=[build(_single_child(node))])
        if node_type in {"forall", "exists"}:
            return LogicNode(
                type=node_type,
                variable=node.get("variable"),
                children=[build(_single_child(node))],
            )
        if node_type == "atomic":
            return LogicNode(
                type="atomic",
                name=node.get("name"),
                arguments=[str(arg) for arg in node.get("arguments", [])],
            )

        raise ValueError(f"Unsupported processed formula node type: {node_type}")

    return build(formula_tree)


def is_direct_solver_ready_formula(formula: Any) -> bool:
    if isinstance(formula, str):
        return True

    if isinstance(formula, LogicNode):
        return _is_direct_solver_ready_logic_node(formula)

    if not isinstance(formula, dict):
        return False

    node_type = formula.get("type")
    if node_type == "atomic":
        return True

    if node_type == "not":
        child = _single_child(formula)
        return isinstance(child, str) or (isinstance(child, dict) and child.get("type") == "atomic")

    if node_type == "and":
        return not contains_formula_level_node(formula)

    if node_type in {"exists", "forall"}:
        children = formula.get("children", [])
        if len(children) != 1:
            return False
        child = children[0]
        if isinstance(child, dict) and child.get("type") == "implies":
            implication_children = child.get("children", [])
            if len(implication_children) != 2:
                return False
            ant, cons = implication_children
            return not contains_formula_level_node(ant) and not contains_formula_level_node(cons)
        return not contains_formula_level_node(child)

    if node_type == "implies":
        children = formula.get("children", [])
        if len(children) != 2:
            return False
        ant, cons = children
        return not contains_formula_level_node(ant) and not contains_formula_level_node(cons)

    return False


def contains_formula_level_node(node: Any) -> bool:
    if isinstance(node, str):
        return False
    if isinstance(node, LogicNode):
        if node.type in {"forall", "exists", "implies"}:
            return True
        return any(contains_formula_level_node(child) for child in node.children)
    if not isinstance(node, dict):
        return False

    if node.get("type") in {"forall", "exists", "implies"}:
        return True
    return any(contains_formula_level_node(child) for child in node.get("children", []))


def resolve_meta_premises(output: Stage3Output, *, max_rounds: int = 5) -> Stage3Output:
    """Resolve META formulas against available formula signatures to a fixed point.

    META premises are never treated as direct solver premises. A resolved META
    can either be redundant because its consequent is already available, or it
    can materialize only its consequent formula. Materialized consequents are
    added to the signature index for later META rounds.
    """
    available_signatures: dict[Any, list[str]] = {}
    signature_solver_ready: dict[Any, bool] = {}

    for item in output.compiled:
        if item.kind == "META":
            continue

        signature = formula_signature(item.formula_tree or item.ast)
        available_signatures.setdefault(signature, []).append(item.premise_id)
        signature_solver_ready[signature] = signature_solver_ready.get(
            signature,
            False,
        ) or is_direct_solver_ready_formula(item.formula_tree or item.ast)

    meta_items = [item for item in output.compiled if item.kind == "META"]
    for item in meta_items:
        _reset_meta_resolution_state(item)

    for round_id in range(max_rounds):
        known_before_round = set(available_signatures)
        materialized: list[tuple[Any, str, bool]] = []

        for item in meta_items:
            if item.meta_resolved:
                continue
            materialization = _resolve_meta_item_once(
                item,
                available_signatures,
                signature_solver_ready,
                round_id=round_id,
            )
            if materialization is not None:
                materialized.append(materialization)

        for signature, source_label, ready in materialized:
            sources = available_signatures.setdefault(signature, [])
            if source_label not in sources:
                sources.append(source_label)
            signature_solver_ready[signature] = signature_solver_ready.get(signature, False) or ready

        if not any(signature not in known_before_round for signature, _, _ in materialized):
            break

    return output


def _reset_meta_resolution_state(item: CompiledPremise) -> None:
    item.direct_solver_ready = (
        is_direct_solver_ready_formula(item.formula_tree)
        if item.formula_tree
        else False
    )
    item.meta_resolvable = False
    item.meta_resolved = False
    item.solver_ready_after_meta_resolution = False
    item.add_to_solver = False
    item.resolution = None
    item.solver_export = []
    item.meta_links = []

    if item.unsupported:
        _mark_meta_unresolved(item, "META premise is unsupported")
    elif not item.formula_tree:
        _mark_meta_unresolved(item, "META premise has no formula tree")


def _resolve_meta_item_once(
    item: CompiledPremise,
    available_signatures: dict[Any, list[str]],
    signature_solver_ready: dict[Any, bool],
    *,
    round_id: int,
) -> tuple[Any, str, bool] | None:
    if item.unsupported or not item.formula_tree:
        return None

    item.meta_links = []
    item.solver_export = []
    item.add_to_solver = False
    item.resolution = None

    if item.formula_tree.get("type") != "implies":
        _mark_meta_unresolved(item, "META formula is not an outer implication", round_id=round_id)
        return None

    children = item.formula_tree.get("children", [])
    if len(children) != 2:
        _mark_meta_unresolved(
            item,
            "META implication does not have exactly two children",
            round_id=round_id,
        )
        return None

    atom_by_id = {atom.atom_id: atom for atom in item.flat_atoms}
    antecedent_sig = formula_signature(children[0], atom_by_id=atom_by_id)
    consequent_sig = formula_signature(children[1], atom_by_id=atom_by_id)
    antecedent_matches = available_signatures.get(antecedent_sig, [])
    consequent_matches = available_signatures.get(consequent_sig, [])

    for premise_id in antecedent_matches:
        item.meta_links.append(
            {
                "type": "antecedent_matches_premise",
                "from": item.premise_id,
                "to": premise_id,
                "round": round_id,
            }
        )
    for premise_id in consequent_matches:
        item.meta_links.append(
            {
                "type": "consequent_matches_premise",
                "from": item.premise_id,
                "to": premise_id,
                "round": round_id,
            }
        )

    if antecedent_matches and consequent_matches:
        item.meta_resolvable = True
        item.meta_resolved = True
        item.solver_ready_after_meta_resolution = signature_solver_ready.get(consequent_sig, False)
        item.add_to_solver = False
        item.resolution = "redundant_formula_link"
        item.meta_links.append(
            {
                "type": "resolution",
                "status": "redundant",
                "reason": "antecedent and consequent formulas already exist",
                "premises": sorted(set(antecedent_matches + consequent_matches)),
                "round": round_id,
            }
        )
        _append_note(item, "meta_resolution: redundant")
        return None

    if antecedent_matches:
        consequent_ready = is_direct_solver_ready_formula(children[1])
        item.meta_resolvable = True
        item.meta_resolved = True
        item.solver_ready_after_meta_resolution = consequent_ready
        item.resolution = "materialized_consequent"
        if consequent_ready:
            item.solver_export = [_formula_export(children[1], item)]
        item.add_to_solver = bool(item.solver_export)
        item.meta_links.append(
            {
                "type": "resolution",
                "status": "materialize_consequent",
                "reason": "antecedent formula already exists",
                "premises": antecedent_matches,
                "round": round_id,
            }
        )
        _append_note(item, "meta_resolution: materialize_consequent")
        return (
            consequent_sig,
            f"{item.premise_id}:materialized_consequent",
            consequent_ready,
        )

    _mark_meta_unresolved(
        item,
        "antecedent formula is not available as a premise",
        round_id=round_id,
    )
    return None


def _mark_meta_unresolved(
    item: CompiledPremise,
    reason: str,
    *,
    round_id: int | None = None,
) -> None:
    item.meta_resolvable = False
    item.meta_resolved = False
    item.solver_ready_after_meta_resolution = False
    item.add_to_solver = False
    item.resolution = "unresolved"
    link: dict[str, Any] = {
        "type": "resolution",
        "status": "unresolved",
        "reason": reason,
    }
    if round_id is not None:
        link["round"] = round_id
    item.meta_links.append(link)
    _append_note(item, "meta_resolution: unresolved")


def _formula_export(formula: Any, item: CompiledPremise) -> dict[str, Any]:
    return formula_tree_to_logic_node(formula, item.flat_atoms).model_dump(exclude_none=True)


def formula_signature(node: Any, *, atom_by_id: dict[str, FlatAtom] | None = None) -> Any:
    env: dict[str, str] = {}

    def normalize_arg(arg: str) -> str:
        if arg in env:
            return env[arg]
        return f"const:{arg}"

    def visit(current: Any, next_var_index: int) -> tuple[Any, int]:
        if isinstance(current, str):
            if atom_by_id is None:
                return ("atom_ref", current), next_var_index
            atom = atom_by_id[current]
            return (
                "atom",
                atom.predicate,
                tuple(normalize_arg(arg) for arg in atom.arguments),
                atom.negated,
            ), next_var_index

        if isinstance(current, LogicNode):
            if current.type == "atomic":
                return (
                    "atom",
                    current.name,
                    tuple(normalize_arg(arg) for arg in current.arguments),
                    False,
                ), next_var_index
            if current.type == "not" and len(current.children) == 1:
                child = current.children[0]
                if child.type == "atomic":
                    return (
                        "atom",
                        child.name,
                        tuple(normalize_arg(arg) for arg in child.arguments),
                        True,
                    ), next_var_index
                child_sig, next_var_index = visit(child, next_var_index)
                return ("not", child_sig), next_var_index
            if current.type in {"forall", "exists"}:
                variable = current.variable or "_"
                old = env.get(variable)
                env[variable] = f"v{next_var_index}"
                child_sig, new_next = visit(current.children[0], next_var_index + 1)
                if old is None:
                    env.pop(variable, None)
                else:
                    env[variable] = old
                return (current.type, child_sig), new_next
            child_sigs = []
            for child in current.children:
                child_sig, next_var_index = visit(child, next_var_index)
                child_sigs.append(child_sig)
            if current.type in {"and", "or", "iff"}:
                child_sigs = sorted(child_sigs, key=repr)
            return (current.type, tuple(child_sigs)), next_var_index

        if isinstance(current, dict):
            node_type = current.get("type")
            if node_type == "atomic":
                return (
                    "atom",
                    current.get("name"),
                    tuple(normalize_arg(str(arg)) for arg in current.get("arguments", [])),
                    bool(current.get("negated", False)),
                ), next_var_index
            if node_type == "not":
                child_sig, next_var_index = visit(_single_child(current), next_var_index)
                return ("not", child_sig), next_var_index
            if node_type in {"forall", "exists"}:
                variable = str(current.get("variable") or "_")
                old = env.get(variable)
                env[variable] = f"v{next_var_index}"
                child_sig, new_next = visit(_single_child(current), next_var_index + 1)
                if old is None:
                    env.pop(variable, None)
                else:
                    env[variable] = old
                return (node_type, child_sig), new_next
            child_sigs = []
            for child in current.get("children", []):
                child_sig, next_var_index = visit(child, next_var_index)
                child_sigs.append(child_sig)
            if node_type in {"and", "or", "iff"}:
                child_sigs = sorted(child_sigs, key=repr)
            return (node_type, tuple(child_sigs)), next_var_index

        return ("unknown", repr(current)), next_var_index

    return visit(node, 0)[0]


def _is_direct_solver_ready_logic_node(node: LogicNode) -> bool:
    if node.type == "atomic":
        return True
    if node.type == "not":
        return len(node.children) == 1 and node.children[0].type == "atomic"
    if node.type == "and":
        return not contains_formula_level_node(node)
    if node.type in {"exists", "forall"}:
        if len(node.children) != 1:
            return False
        child = node.children[0]
        if child.type == "implies":
            ant, cons = child.children
            return not contains_formula_level_node(ant) and not contains_formula_level_node(cons)
        return not contains_formula_level_node(child)
    if node.type == "implies":
        if len(node.children) != 2:
            return False
        ant, cons = node.children
        return not contains_formula_level_node(ant) and not contains_formula_level_node(cons)
    return False


def _atomize_leaf_to_formula(
    phrase: str,
    variable: str,
    llm_atomizer: Callable[[str, str], list[dict[str, Any]]],
    *,
    premise_id: str,
    flat_atoms: list[FlatAtom],
    counter: list[int],
) -> Any:
    clean_phrase = _strip_sentence(phrase)
    if not clean_phrase:
        raise ValueError("empty formula leaf")

    atoms = llm_atomizer(clean_phrase, variable)
    atom_ids = []
    for atom in atoms:
        atom_id = f"{premise_id}_A{counter[0]}"
        counter[0] += 1
        flat_atoms.append(
            FlatAtom(
                atom_id=atom_id,
                predicate=str(atom["predicate"]),
                arguments=[str(arg) for arg in atom["arguments"]],
                negated=bool(atom.get("negated", False)),
                source_premise_id=premise_id,
            )
        )
        atom_ids.append(atom_id)

    if len(atom_ids) == 1:
        return atom_ids[0]
    return {"type": "and", "children": atom_ids}


def _parse_formula_part(text: str, *, variable: str) -> dict[str, Any]:
    part = _strip_outer_parens(_strip_sentence(text))

    if part.lower().startswith("then "):
        part = part[5:].strip()

    if_rule = _split_if_rule(part)
    if if_rule is not None:
        antecedent, consequent = if_rule
        return {
            "type": "forall",
            "variable": variable,
            "body": {
                "type": "implies",
                "antecedent": _as_leaf(antecedent, variable),
                "consequent": _as_leaf(consequent, variable),
            },
        }

    exists_body = _parse_exists_body(part)
    if exists_body is not None:
        return {
            "type": "exists",
            "variable": _choose_variable(part, "y"),
            "body": _as_leaf(exists_body, _choose_variable(part, "y")),
        }

    universal = _parse_universal_body(part)
    if universal is not None:
        subject_phrase, predicate_phrase = universal
        return {
            "type": "forall",
            "variable": variable,
            "body": {
                "type": "implies",
                "antecedent": _as_leaf(subject_phrase, variable),
                "consequent": _as_leaf(predicate_phrase, variable),
            },
        }

    bare_plural = _parse_bare_plural_universal(part)
    if bare_plural is not None:
        subject_phrase, predicate_phrase = bare_plural
        return {
            "type": "forall",
            "variable": variable,
            "body": {
                "type": "implies",
                "antecedent": _as_leaf(subject_phrase, variable),
                "consequent": _as_leaf(predicate_phrase, variable),
            },
        }

    implication = _split_implication_phrase(part)
    if implication is not None:
        antecedent, consequent = implication
        return {
            "type": "forall",
            "variable": variable,
            "body": {
                "type": "implies",
                "antecedent": _as_leaf(antecedent, variable),
                "consequent": _as_leaf(consequent, variable),
            },
        }

    relative_rule = _parse_relative_rule(part)
    if relative_rule is not None:
        antecedent, consequent = relative_rule
        return {
            "type": "forall",
            "variable": variable,
            "body": {
                "type": "implies",
                "antecedent": _as_leaf(antecedent, variable),
                "consequent": _as_leaf(consequent, variable),
            },
        }

    return _as_leaf(part, variable)


def _parse_exists_body(text: str) -> str | None:
    patterns = [
        r"^there exists(?:\s+at least one)?\s+(?P<subject>.+?)\s+(?:who|that)\s+(?P<body>.+)$",
        r"^there is at least one\s+(?P<subject>.+?)\s+(?:who|that)\s+(?P<body>.+)$",
        r"^at least one\s+(?P<subject>.+?)\s+(?:who|that)\s+(?P<body>.+)$",
        r"^there exists\s+(?:one|a|an|at least one)\s+(?P<subject>.+?)\s+(?P<body>is|are|has|have|does|do|can|will|must|should|receives?|gets?|gains?|attends?|attending|asks?|asking|passes?|passing|prepares?|preparing|understands?|understanding|revises?|revising|studies?|studying)\b(?P<rest>.*)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            subject = _singular_subject(match.group("subject"))
            body = match.group("body").strip()
            rest = match.groupdict().get("rest")
            if rest is not None:
                body = f"{body}{rest}".strip()
            return f"a {subject} {body}"
    return None


def _parse_universal_body(text: str) -> tuple[str, str] | None:
    match = re.match(
        r"^(?:every|all|any)\s+(?P<subject>.+?)\s+(?P<body>is|are|has|have|does|do|can|will|must|should|receives?|gets?|gains?|attends?|asks?|passes?|prepares?|understands?|revises?|studies?)\b(?P<rest>.*)$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    subject = _singular_subject(match.group("subject"))
    predicate = _subject_predicate_phrase(subject, match.group("body"), match.group("rest"))
    return f"a {subject}", predicate


def _parse_bare_plural_universal(text: str) -> tuple[str, str] | None:
    match = re.match(
        r"^(?P<subject>students?|models?|persons?|people|employees?|teachers?)\s+(?P<body>is|are|has|have|does|do|can|will|must|should|receives?|gets?|gains?|attends?|asks?|passes?|prepares?|understands?|revises?|studies?)\b(?P<rest>.*)$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    subject = _singular_subject(match.group("subject"))
    predicate = _subject_predicate_phrase(subject, match.group("body"), match.group("rest"))
    return f"a {subject}", predicate


def _parse_relative_rule(text: str) -> tuple[str, str] | None:
    match = re.match(
        r"^(?P<subject>students?|models?|persons?|people|employees?|teachers?)\s+who\s+(?P<condition>.+?)\s+(?P<consequent>are|is|receive|receives|get|gets|gain|gains|can|will|must|should)\b(?P<rest>.*)$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    subject = _singular_subject(match.group("subject"))
    condition = match.group("condition").strip()
    consequent = _subject_predicate_phrase(subject, match.group("consequent"), match.group("rest"))
    return f"a {subject} {condition}", consequent


def _split_if_then(text: str) -> tuple[str, str] | None:
    clean = _strip_outer_parens(_strip_sentence(text))
    if not clean.lower().startswith("if "):
        return None

    then_start = _find_top_level_word(clean, "then", start=3)
    if then_start == -1:
        return None

    antecedent = clean[3:then_start].strip(" ,")
    consequent = clean[then_start + len("then"):].strip(" ,")
    return antecedent, consequent


def _split_if_rule(text: str) -> tuple[str, str] | None:
    clean = _strip_outer_parens(_strip_sentence(text))
    if not clean.lower().startswith("if "):
        return None

    split = _split_if_then(clean)
    if split is not None:
        return split

    comma = _find_top_level_char(clean, ",", start=3)
    if comma == -1:
        return None
    return clean[3:comma].strip(), clean[comma + 1:].strip()


def _split_implication_phrase(text: str) -> tuple[str, str] | None:
    connectors = ["implies", "leads to", "requires"]
    for connector in connectors:
        start = _find_top_level_phrase(text, connector)
        if start != -1:
            return (
                text[:start].strip(" ,"),
                text[start + len(connector):].strip(" ,"),
            )
    return None


def _choose_variable(text: str, default: str) -> str:
    lower = text.lower()
    if "there exists" in lower or "at least one" in lower:
        return "y"
    return default


def _as_leaf(text: Any, variable: str) -> dict[str, Any]:
    if isinstance(text, dict):
        return text
    return {"type": "leaf", "text": str(text).strip(), "variable": variable}


def _single_child(node: dict[str, Any]) -> Any:
    children = node.get("children", [])
    if len(children) != 1:
        raise ValueError(f"{node.get('type')} node must have exactly one child")
    return children[0]


def _append_note(item: CompiledPremise, note: str) -> None:
    if note not in item.notes:
        item.notes.append(note)


def _strip_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip(".")


def _strip_outer_parens(text: str) -> str:
    clean = text.strip()
    while clean.startswith("(") and clean.endswith(")") and _outer_parens_wrap(clean):
        clean = clean[1:-1].strip()
    return clean


def _outer_parens_wrap(text: str) -> bool:
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return False
    return depth == 0


def _find_top_level_word(text: str, word: str, *, start: int = 0) -> int:
    pattern = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
    for match in pattern.finditer(text, pos=start):
        if _depth_at(text, match.start()) == 0:
            return match.start()
    return -1


def _find_top_level_phrase(text: str, phrase: str, *, start: int = 0) -> int:
    pattern = re.compile(rf"\b{re.escape(phrase)}\b", flags=re.IGNORECASE)
    for match in pattern.finditer(text, pos=start):
        if _depth_at(text, match.start()) == 0:
            return match.start()
    return -1


def _find_top_level_char(text: str, char: str, *, start: int = 0) -> int:
    depth = 0
    for index, current in enumerate(text[start:], start=start):
        if current == "(":
            depth += 1
        elif current == ")":
            depth -= 1
        elif current == char and depth == 0:
            return index
    return -1


def _depth_at(text: str, index: int) -> int:
    depth = 0
    for char in text[:index]:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
    return depth


def _singular_subject(subject: str) -> str:
    clean = subject.strip().lower()
    clean = re.sub(r"^(?:a|an|one|the)\s+", "", clean)
    words = clean.split()
    if not words:
        return clean
    head = words[-1]
    replacements = {"students": "student", "models": "model", "employees": "employee", "teachers": "teacher"}
    words[-1] = replacements.get(head, head[:-1] if head.endswith("s") and len(head) > 3 else head)
    return " ".join(words)


def _subject_predicate_phrase(subject: str, verb: str, rest: str) -> str:
    normalized = verb.lower()
    replacements = {"are": "is", "have": "has", "do": "does", "receive": "receives", "get": "gets", "gain": "gains"}
    normalized = replacements.get(normalized, normalized)
    return f"the {subject} {normalized}{rest}".strip()
