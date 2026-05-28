from __future__ import annotations

from NEW_logic_pipeline.Stage_1.logic_skeleton import LogicSkeleton, FormulaSkeleton
from NEW_logic_pipeline.Stage_2.atomization_requests import AtomizationResult, PredicateAtom
from NEW_logic_pipeline.Stage_5.ast_builder import build_ast, LogicNode
from NEW_logic_pipeline.Stage_6.validator import Stage6Validator
from NEW_logic_pipeline.Stage_6.registry_schema import normalize_registry_config

REGISTRY = normalize_registry_config({
    "certified": {"arity": 1, "roles": ["person"], "solver_safe": True},
    "student": {"arity": 1, "roles": ["person"], "solver_safe": True},
    "learner": {"arity": 1, "roles": ["person"], "solver_safe": True},
    "happy": {"arity": 1, "roles": ["person"], "solver_safe": True},
})

def test_build_ast_fact():
    skeleton = LogicSkeleton(
        premise_id="P1",
        original="Laura is certified.",
        kind="FACT",
    )
    results = [
        AtomizationResult(
            premise_id="P1",
            request_id="P1_body",
            phrase="Laura is certified",
            role="body",
            variable="x",
            atoms=[
                PredicateAtom(name="certified", arguments=["laura"], negated=False, source_phrase="Laura is certified")
            ],
        )
    ]
    ast = build_ast(skeleton, results)
    assert ast.type == "ATOM"
    assert ast.predicate == "certified"
    assert ast.arguments == ["laura"]
    assert not ast.negated
    assert ast.premise_id == "P1"

def test_build_ast_exists():
    skeleton = LogicSkeleton(
        premise_id="P1",
        original="Someone is a student.",
        kind="EXISTS",
        variable="x",
    )
    results = [
        AtomizationResult(
            premise_id="P1",
            request_id="P1_body",
            phrase="Someone is a student",
            role="body",
            variable="x",
            atoms=[
                PredicateAtom(name="student", arguments=["x"], negated=False, source_phrase="student")
            ],
        )
    ]
    ast = build_ast(skeleton, results)
    assert ast.type == "EXISTS"
    assert ast.variable == "x"
    assert len(ast.children) == 1
    assert ast.children[0].type == "ATOM"
    assert ast.children[0].predicate == "student"

def test_build_ast_forall():
    skeleton = LogicSkeleton(
        premise_id="P1",
        original="All students are certified.",
        kind="FORALL",
        variable="x",
    )
    results = [
        AtomizationResult(
            premise_id="P1",
            request_id="P1_restrictor",
            phrase="student",
            role="restrictor",
            variable="x",
            atoms=[
                PredicateAtom(name="student", arguments=["x"], negated=False, source_phrase="student")
            ],
        ),
        AtomizationResult(
            premise_id="P1",
            request_id="P1_property",
            phrase="certified",
            role="property",
            variable="x",
            atoms=[
                PredicateAtom(name="certified", arguments=["x"], negated=False, source_phrase="certified")
            ],
        ),
    ]
    ast = build_ast(skeleton, results)
    assert ast.type == "FORALL"
    assert ast.variable == "x"
    assert len(ast.children) == 1
    impl = ast.children[0]
    assert impl.type == "IMPLIES"
    assert len(impl.children) == 2
    assert impl.children[0].type == "ATOM"
    assert impl.children[0].predicate == "student"
    assert impl.children[1].type == "ATOM"
    assert impl.children[1].predicate == "certified"

def test_build_ast_rule():
    skeleton = LogicSkeleton(
        premise_id="P1",
        original="If they are a student, they are certified.",
        kind="RULE",
        variable="x",
    )
    results = [
        AtomizationResult(
            premise_id="P1",
            request_id="P1_antecedent",
            phrase="student",
            role="antecedent",
            variable="x",
            atoms=[
                PredicateAtom(name="student", arguments=["x"], negated=False, source_phrase="student")
            ],
        ),
        AtomizationResult(
            premise_id="P1",
            request_id="P1_consequent",
            phrase="certified",
            role="consequent",
            variable="x",
            atoms=[
                PredicateAtom(name="certified", arguments=["x"], negated=False, source_phrase="certified")
            ],
        ),
    ]
    ast = build_ast(skeleton, results)
    assert ast.type == "FORALL"
    assert ast.variable == "x"
    impl = ast.children[0]
    assert impl.type == "IMPLIES"
    assert impl.children[0].predicate == "student"
    assert impl.children[1].predicate == "certified"

def test_build_ast_iff():
    skeleton = LogicSkeleton(
        premise_id="P1",
        original="A person is a student if and only if they are a learner.",
        kind="IFF",
        variable="x",
    )
    results = [
        AtomizationResult(
            premise_id="P1",
            request_id="P1_left",
            phrase="student",
            role="left",
            variable="x",
            atoms=[
                PredicateAtom(name="student", arguments=["x"], negated=False, source_phrase="student")
            ],
        ),
        AtomizationResult(
            premise_id="P1",
            request_id="P1_right",
            phrase="learner",
            role="right",
            variable="x",
            atoms=[
                PredicateAtom(name="learner", arguments=["x"], negated=False, source_phrase="learner")
            ],
        ),
    ]
    ast = build_ast(skeleton, results)
    assert ast.type == "FORALL"
    assert ast.variable == "x"
    iff = ast.children[0]
    assert iff.type == "IFF"
    assert iff.children[0].predicate == "student"
    assert iff.children[1].predicate == "learner"

def test_build_ast_meta():
    formula_tree = FormulaSkeleton(
        type="implies",
        children=[
            FormulaSkeleton(type="leaf", text="A"),
            FormulaSkeleton(type="leaf", text="B"),
        ]
    )
    skeleton = LogicSkeleton(
        premise_id="P1",
        original="Statement A implies Statement B.",
        kind="META",
        formula_tree=formula_tree,
    )
    results = [
        AtomizationResult(
            premise_id="P1",
            request_id="P1_f_0",
            phrase="A",
            role="leaf",
            variable="x",
            formula_path=[0],
            atoms=[
                PredicateAtom(name="student", arguments=["x"], negated=False, source_phrase="A")
            ],
        ),
        AtomizationResult(
            premise_id="P1",
            request_id="P1_f_1",
            phrase="B",
            role="leaf",
            variable="x",
            formula_path=[1],
            atoms=[
                PredicateAtom(name="certified", arguments=["x"], negated=False, source_phrase="B")
            ],
        ),
    ]
    ast = build_ast(skeleton, results)
    assert ast.type == "META"
    impl = ast.children[0]
    assert impl.type == "IMPLIES"
    assert impl.children[0].predicate == "student"
    assert impl.children[1].predicate == "certified"

def test_build_ast_unsupported():
    skeleton = LogicSkeleton(
        premise_id="P1",
        original="Laura should attend class.",
        kind="OBLIGATION_RULE",
    )
    ast = build_ast(skeleton, [])
    assert ast.type == "UNSUPPORTED"
    assert "obligation" in ast.reason or "directly solver-ready" in ast.reason

def test_validation_issues_negation_in_name():
    # If the predicate name starts with not_
    parsed = {
        "asts": [
            {
                "type": "ATOM",
                "predicate": "not_certified",
                "arguments": ["laura"],
                "negated": False,
                "source_text": "is not certified",
            }
        ],
    }
    report = Stage6Validator(predicate_registry=REGISTRY).validate(parsed)
    assert not report.parse_valid
    assert "AST_NEGATION_IN_PREDICATE_NAME" in report.summary["issue_codes"]

def test_validation_issues_inconsistent_phrase_mapping():
    # Same phrase mapping to different predicates
    parsed = {
        "asts": [
            {
                "type": "ATOM",
                "predicate": "student",
                "arguments": ["x"],
                "source_text": "attends training",
            },
            {
                "type": "ATOM",
                "predicate": "certified",
                "arguments": ["y"],
                "source_text": "attends training",
            }
        ],
    }
    report = Stage6Validator(predicate_registry=REGISTRY).validate(parsed)
    # This is a warning, so parse_valid can still be True depending on registry.
    assert "AST_INCONSISTENT_PHRASE_MAPPING" in report.summary["issue_codes"]

def test_validation_issues_unsupported_node():
    parsed = {
        "asts": [
            {
                "type": "UNSUPPORTED",
                "reason": "obligation rule not supported",
                "source_text": "Laura must attend",
            }
        ],
    }
    report = Stage6Validator(predicate_registry=REGISTRY).validate(parsed)
    assert not report.parse_valid
    assert not report.direct_solver_ready
    assert report.unsupported
    assert "AST_UNSUPPORTED_NODE" in report.summary["issue_codes"]
