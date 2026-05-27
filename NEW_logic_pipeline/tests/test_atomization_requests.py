from NEW_logic_pipeline.Stage_1.logic_skeleton import FormulaSkeleton, LogicSkeleton, TextSpan
from NEW_logic_pipeline.Stage_1.skeleton_builder import build_skeleton
from NEW_logic_pipeline.Stage_2.atomization_requests import (
    collect_atomization_requests,
    collect_formula_leaf_requests,
)


def test_rule_skeleton_creates_two_requests():
    skeleton = LogicSkeleton(
        premise_id="P1",
        original="If a student does not maintain GPA, then the student does not have housing.",
        kind="RULE",
        antecedent=TextSpan(role="antecedent", text="a student does not maintain GPA", negation_hint=True),
        consequent=TextSpan(role="consequent", text="the student does not have housing", negation_hint=True),
    )

    requests = collect_atomization_requests(skeleton)

    assert len(requests) == 2
    assert {request.role for request in requests} == {"antecedent", "consequent"}
    assert all(request.premise_id == "P1" for request in requests)


def test_exists_skeleton_creates_body_request():
    skeleton = LogicSkeleton(
        premise_id="P1",
        original="At least one student has completed a course.",
        kind="EXISTS",
        body=TextSpan(role="body", text="a student has completed a course"),
        quantifier="exists",
    )

    requests = collect_atomization_requests(skeleton)

    assert len(requests) == 1
    assert requests[0].role == "body"
    assert "student" in requests[0].phrase
    assert "completed" in requests[0].phrase


def test_forall_skeleton_creates_restrictor_property_requests():
    skeleton = LogicSkeleton(
        premise_id="P1",
        original="All students receive training.",
        kind="FORALL",
        antecedent=TextSpan(role="restrictor", text="a student"),
        consequent=TextSpan(role="property", text="receives training"),
        quantifier="forall",
    )

    requests = collect_atomization_requests(skeleton)

    assert len(requests) == 2
    assert {request.role for request in requests} == {"restrictor", "property"}
    assert all(request.variable == "x" for request in requests)


def test_iff_skeleton_creates_left_right_requests():
    skeleton = LogicSkeleton(
        premise_id="P1",
        original="A student is eligible if and only if the student passes the exam.",
        kind="IFF",
        left=TextSpan(role="left", text="a student is eligible"),
        right=TextSpan(role="right", text="the student passes the exam"),
    )

    requests = collect_atomization_requests(skeleton)

    assert len(requests) == 2
    assert {request.role for request in requests} == {"left", "right"}


def test_non_if_rule_preserves_consequent_negation_hint():
    skeleton = LogicSkeleton(
        premise_id="P1",
        original="Lack of approval prevents laboratory access.",
        kind="NON_IF_RULE",
        antecedent=TextSpan(role="antecedent", text="Lack of approval", negation_hint=True),
        consequent=TextSpan(role="consequent", text="laboratory access", negation_hint=True),
    )

    requests = collect_atomization_requests(skeleton)

    consequent = next(request for request in requests if request.role == "consequent")
    assert consequent.negation_hint is True


def test_meta_creates_leaf_requests_with_formula_paths():
    formula = FormulaSkeleton(
        type="implies",
        children=[
            FormulaSkeleton(
                type="forall",
                variable="x",
                children=[
                    FormulaSkeleton(
                        type="implies",
                        children=[
                            FormulaSkeleton(type="leaf", text="a student passes the exam", variable="x"),
                            FormulaSkeleton(type="leaf", text="the student graduates", variable="x"),
                        ],
                    )
                ],
            ),
            FormulaSkeleton(
                type="forall",
                variable="y",
                children=[
                    FormulaSkeleton(
                        type="implies",
                        children=[
                            FormulaSkeleton(type="leaf", text="a student passes the exam", variable="y"),
                            FormulaSkeleton(type="leaf", text="the student is eligible", variable="y"),
                        ],
                    )
                ],
            ),
        ],
    )

    requests = collect_formula_leaf_requests(
        premise_id="P1",
        formula=formula,
        role_prefix="meta",
        original_premise="If passing the exam implies graduation, then students who pass are eligible.",
    )

    assert len(requests) == 4
    assert all(request.formula_path for request in requests)
    phrases = [request.phrase for request in requests]
    assert any("passes the exam" in phrase for phrase in phrases)
    assert any("graduates" in phrase for phrase in phrases)
    assert any("eligible" in phrase for phrase in phrases)
    assert requests[0].formula_path == [0, 0, 0]
    assert requests[3].formula_path == [1, 0, 1]


def test_meta_consequent_if_is_recursively_split_before_atomization():
    skeleton = build_skeleton(
        "P5",
        "If there exists at least one student who is attending tutorials, then (if a student is not asking questions, they are not attending tutorials).",
    )

    requests = collect_atomization_requests(skeleton)

    phrases = [request.phrase for request in requests]
    assert len(phrases) == 3
    assert "a student who is attending tutorials" in phrases
    assert "a student is not asking questions" in phrases
    assert "they are not attending tutorials" in phrases
    assert not any(phrase.strip().lower().startswith("(if") for phrase in phrases)
    assert not any("formula_like_leaf_detected" in request.notes for request in requests)


def test_formula_like_leaf_is_reparsed_in_request_collector():
    formula = FormulaSkeleton(
        type="leaf",
        text="(if a student is not asking questions, they are not attending tutorials)",
        variable="x",
    )

    requests = collect_formula_leaf_requests(
        premise_id="P1",
        formula=formula,
        role_prefix="meta",
        original_premise="If there exists a student, then (if a student is not asking questions, they are not attending tutorials).",
    )

    assert [request.phrase for request in requests] == [
        "a student is not asking questions",
        "they are not attending tutorials",
    ]
