import pytest

from src.predicate_canonicalizer import (
    canonicalize_node,
    canonicalize_question_parse,
    canonicalize_stage3,
)
from src.schemas import CompiledPremise, LogicNode, QuestionParse, Stage3Output
from src.config import PipelineConfig
from src.json_utils import extract_json_object
from src.pipeline import LogicPipeline
from src.stage1_cnl import (
    has_explicit_numeric_condition,
    remove_false_numeric_flags,
    stage1_token_budget,
)
from src.stage4_validate import classify_solver_readiness, has_solver_blocking_predicate, is_direct_solver_ready
from src.schemas import CNLStatement, Stage1Output


def test_numeric_condition_requires_explicit_threshold():
    output = Stage1Output(
        statements=[
            CNLStatement(
                premise_id="P1",
                original="Models trained with large datasets achieve high accuracy.",
                kind_hint="RULE",
                cnl="If a model is trained with large datasets, then the model achieves high accuracy.",
                risk_flags=["relative_clause_rule", "numeric_condition"],
            )
        ]
    )

    cleaned = remove_false_numeric_flags(output)

    assert cleaned.statements[0].risk_flags == ["relative_clause_rule"]
    assert not has_explicit_numeric_condition(output.statements[0].original)
    assert has_explicit_numeric_condition("Students with GPA above 3.5 receive scholarships.")


def test_predicate_canonicalizer_unifies_tuning_synonyms():
    node = LogicNode(
        type="and",
        children=[
            LogicNode(
                type="atomic",
                name="requires_extensive_hyperparameter_tuning",
                arguments=["alphanet"],
            ),
            LogicNode(
                type="atomic",
                name="has_been_extensively_tuned",
                arguments=["alphanet"],
            ),
        ],
    )

    canonicalize_node(node)

    assert [child.name for child in node.children] == [
        "has_extensive_hyperparameter_tuning",
        "has_extensive_hyperparameter_tuning",
    ]


def test_stage3_canonicalizer_rewrites_p3_tuning_consequent():
    output = Stage3Output(
        compiled=[
            CompiledPremise(
                premise_id="P3",
                kind="RULE",
                cnl=(
                    "If a model achieves high accuracy and processes data quickly, "
                    "then the model requires extensive hyperparameter tuning."
                ),
                ast=LogicNode(
                    type="forall",
                    variable="x",
                    children=[
                        LogicNode(
                            type="implies",
                            children=[
                                LogicNode(
                                    type="and",
                                    children=[
                                        LogicNode(
                                            type="atomic",
                                            name="achieves_high_accuracy",
                                            arguments=["x"],
                                        ),
                                        LogicNode(
                                            type="atomic",
                                            name="processes_data_quickly",
                                            arguments=["x"],
                                        ),
                                    ],
                                ),
                                LogicNode(
                                    type="atomic",
                                    name="requires_extensive_hyperparameter_tuning",
                                    arguments=["x"],
                                ),
                            ],
                        )
                    ],
                ),
            )
        ]
    )

    canonicalize_stage3(output)

    consequent = output.compiled[0].ast.children[0].children[1]
    assert consequent.name == "has_extensive_hyperparameter_tuning"


def test_question_canonicalizer_unifies_choice_predicates():
    question = QuestionParse(
        question="Which choice follows?",
        choices={
            "B": LogicNode(
                type="atomic",
                name="has_been_extensively_tuned",
                arguments=["alphanet"],
            )
        },
    )

    canonicalize_question_parse(question)

    assert question.choices["B"].name == "has_extensive_hyperparameter_tuning"


def test_invalid_question_node_type_is_rejected():
    with pytest.raises(ValueError):
        QuestionParse.model_validate(
            {
                "question": "Which choice follows?",
                "query": None,
                "choices": {
                    "A": {
                        "type": "inference",
                        "predicate": "has_been_extensively_tuned",
                        "arguments": ["alphanet"],
                    }
                },
            }
        )


def test_stage1_token_budget_scales_with_premise_count():
    config = PipelineConfig(max_new_tokens=10_000, stage1_max_new_tokens=500)

    assert stage1_token_budget(config, 1) == 600
    assert stage1_token_budget(config, 6) == 1850


def test_json_extractor_reports_truncated_object():
    with pytest.raises(ValueError, match="truncated"):
        extract_json_object('{"statements": [{"premise_id": "P1"}')


def test_json_extractor_ignores_end_marker():
    assert extract_json_object('{"ok": true}<END_JSON>') == {"ok": True}


def test_placeholder_predicates_block_direct_solver_readiness():
    node = LogicNode(type="atomic", name="unsupported_premise", arguments=["p8"])

    assert has_solver_blocking_predicate(node)
    assert not is_direct_solver_ready(node)
    assert classify_solver_readiness("UNKNOWN", "atomic", ["needs_review"]) == "needs_review"


def test_subject_learning_record_uses_binary_predicates_without_llm_fallback():
    premises = [
        "Every subject contains knowledge.",
        "If a student has knowledge of a subject, they can explain it to their friends.",
        "If a student explains a subject to their friends and the friends understand it, the student has mastered the subject.",
        "If a student masters a subject, they can earn an A or A+.",
        "If a student earns at least five A or A+ grades, they can receive a scholarship.",
        "Tuấn has earned three A grades.",
        "Tuấn has not earned any additional A+ grades.",
        "If a student earns an A in a subject, they must have mastered the subject.",
        "If Tuấn's friends do not understand a subject, Tuấn has not mastered it.",
        "If a student cannot explain a subject, they do not have knowledge of it.",
    ]
    result = LogicPipeline(
        PipelineConfig(max_repair_attempts=0, llm_live_trace=False),
        rag_path="logic_pipeline/data/structural_examples.jsonl",
    ).parse(
        premises,
        question="Does Tuan have knowledge of his A-grade subjects?",
    )

    by_id = {premise.premise_id: premise for premise in result.premises}

    assert result.status == "success"
    assert all(not premise.unsupported for premise in result.premises)
    assert all(premise.solver_ready for premise in result.premises)
    assert all(premise.add_to_solver for premise in result.premises)

    assert _predicate_names(by_id["P2"].ast) >= {
        "student",
        "subject",
        "has_knowledge_of_subject",
        "can_explain_subject_to_friends",
    }
    assert _predicate_names(by_id["P3"].ast) >= {
        "can_explain_subject_to_friends",
        "friends_understand_subject",
        "mastered_subject",
    }
    assert _predicate_names(by_id["P4"].ast) >= {"mastered_subject", "can_earn_high_grade"}
    assert _predicate_names(by_id["P8"].ast) >= {"earned_grade", "mastered_subject"}

    p6_atom = by_id["P6"].ast
    assert p6_atom.name == "earned_grade_count"
    assert p6_atom.arguments == ["tuan", "A", "3"]
    assert p6_atom.arguments[0] != "tu_n"

    p7_atom = by_id["P7"].ast
    assert p7_atom.name == "earned_additional_grade_count"
    assert p7_atom.arguments == ["tuan", "A_plus", "0"]

    assert result.question is not None
    assert result.question.direct_solver_ready is True
    assert result.question.query is not None
    assert _predicate_names(result.question.query) == {
        "subject",
        "earned_grade",
        "has_knowledge_of_subject",
    }


def _predicate_names(node: LogicNode) -> set[str]:
    names = set()
    if node.type == "atomic" and node.name:
        names.add(node.name)
    for child in node.children:
        names.update(_predicate_names(child))
    return names
