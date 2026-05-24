from __future__ import annotations

from .schemas import LogicNode, QuestionParse, Stage3Output


CANONICAL_PREDICATES = {
    "requires_extensive_hyperparameter_tuning": "has_extensive_hyperparameter_tuning",
    "has_been_extensively_tuned": "has_extensive_hyperparameter_tuning",
    "is_extensively_tuned": "has_extensive_hyperparameter_tuning",
    "masters_subject": "mastered_subject",
    "entity_receive_scholarship": "receive_scholarship",
}


def canonicalize_predicate_name(name: str) -> str:
    return CANONICAL_PREDICATES.get(name, name)


def canonicalize_node(node: LogicNode) -> LogicNode:
    if node.type == "atomic" and node.name:
        node.name = canonicalize_predicate_name(node.name)

    for child in node.children:
        canonicalize_node(child)

    if isinstance(node.left, LogicNode):
        canonicalize_node(node.left)
    if isinstance(node.right, LogicNode):
        canonicalize_node(node.right)

    return node


def canonicalize_stage3(output: Stage3Output) -> Stage3Output:
    for premise in output.compiled:
        canonicalize_node(premise.ast)
        for atom in premise.flat_atoms:
            atom.predicate = canonicalize_predicate_name(atom.predicate)
    return output


def canonicalize_question_parse(question: QuestionParse) -> QuestionParse:
    if question.query:
        canonicalize_node(question.query)
    for choice in question.choices.values():
        canonicalize_node(choice)
    for atom in question.flat_atoms:
        atom.predicate = canonicalize_predicate_name(atom.predicate)
    return question


def collect_predicate_names(output: Stage3Output) -> list[str]:
    names: set[str] = set()

    def visit(node: LogicNode) -> None:
        if node.type == "atomic" and node.name:
            names.add(node.name)
        for child in node.children:
            visit(child)
        if isinstance(node.left, LogicNode):
            visit(node.left)
        if isinstance(node.right, LogicNode):
            visit(node.right)

    for premise in output.compiled:
        visit(premise.ast)

    return sorted(names)
