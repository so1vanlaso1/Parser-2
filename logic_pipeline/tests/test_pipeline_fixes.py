import pytest

from src.predicate_canonicalizer import (
    canonicalize_node,
    canonicalize_question_parse,
    canonicalize_stage3,
)
from src.schemas import CompiledPremise, LogicNode, QuestionParse, Stage3Output
from src.config import PipelineConfig
from src.json_utils import extract_json_object
from src.stage1_cnl import (
    has_explicit_numeric_condition,
    remove_false_numeric_flags,
    stage1_token_budget,
)
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
