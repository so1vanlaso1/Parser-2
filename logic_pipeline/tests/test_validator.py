"""Tests for the Stage 4 LogicValidator."""
from src.schemas import CompiledPremise, LogicNode, Stage3Output
from src.stage4_validate import LogicValidator, classify_solver_readiness


def _make_stage3(premises: list[CompiledPremise]) -> Stage3Output:
    return Stage3Output(compiled=premises)


def test_unbound_variable_detected():
    """Single-letter variable 'x' without a quantifier should be caught."""
    output = _make_stage3([
        CompiledPremise(
            premise_id="P1",
            kind="FACT",
            cnl="x is a student",
            ast=LogicNode(type="atomic", name="student", arguments=["x"]),
        )
    ])

    report = LogicValidator().validate_stage3(output)
    assert not report.ok
    assert any("Unbound variable" in i.message for i in report.issues)


def test_bound_variable_passes():
    """Variable 'x' bound by forall should pass."""
    output = _make_stage3([
        CompiledPremise(
            premise_id="P1",
            kind="FORALL",
            cnl="Every student is eligible",
            ast=LogicNode(
                type="forall",
                variable="x",
                children=[
                    LogicNode(
                        type="implies",
                        children=[
                            LogicNode(type="atomic", name="student", arguments=["x"]),
                            LogicNode(type="atomic", name="eligible", arguments=["x"]),
                        ],
                    )
                ],
            ),
        )
    ])

    report = LogicValidator().validate_stage3(output)
    assert report.ok


def test_constant_not_flagged_as_unbound():
    """Multi-char constants like 'john' should not be flagged."""
    output = _make_stage3([
        CompiledPremise(
            premise_id="P1",
            kind="FACT",
            cnl="John is a student",
            ast=LogicNode(type="atomic", name="student", arguments=["john"]),
        )
    ])

    report = LogicValidator().validate_stage3(output)
    assert report.ok


def test_not_prefix_predicate_caught():
    """Predicate names starting with 'not_' should be flagged."""
    output = _make_stage3([
        CompiledPremise(
            premise_id="P1",
            kind="FACT",
            cnl="John does not have housing",
            ast=LogicNode(type="atomic", name="not_has_housing", arguments=["john"]),
        )
    ])

    report = LogicValidator().validate_stage3(output)
    assert not report.ok
    assert any("contains negation" in i.message for i in report.issues)


def test_double_negation_warning():
    """Double negation should produce a warning (not error)."""
    output = _make_stage3([
        CompiledPremise(
            premise_id="P1",
            kind="FACT",
            cnl="not not eligible",
            ast=LogicNode(
                type="not",
                children=[
                    LogicNode(
                        type="not",
                        children=[
                            LogicNode(type="atomic", name="eligible", arguments=["john"]),
                        ],
                    )
                ],
            ),
        )
    ])

    report = LogicValidator().validate_stage3(output)
    # Warnings don't cause ok=False.
    assert report.ok
    assert any("Double negation" in i.message for i in report.issues)


def test_only_if_rule_without_implies():
    """ONLY_IF_RULE kind without an implies node should fail."""
    output = _make_stage3([
        CompiledPremise(
            premise_id="P1",
            kind="ONLY_IF_RULE",
            cnl="A only if B",
            ast=LogicNode(type="atomic", name="some_fact", arguments=["x"]),
        )
    ])

    report = LogicValidator().validate_stage3(output)
    assert not report.ok
    assert any("ONLY_IF_RULE" in i.message for i in report.issues)


def test_iff_kind_without_iff_node():
    """IFF kind without an iff node should fail."""
    output = _make_stage3([
        CompiledPremise(
            premise_id="P1",
            kind="IFF",
            cnl="A if and only if B",
            ast=LogicNode(
                type="implies",
                children=[
                    LogicNode(type="atomic", name="a", arguments=["x"]),
                    LogicNode(type="atomic", name="b", arguments=["x"]),
                ],
            ),
        )
    ])

    report = LogicValidator().validate_stage3(output)
    assert not report.ok
    assert any("IFF" in i.message for i in report.issues)


def test_classify_solver_ready():
    assert classify_solver_readiness("RULE", "implies", []) == "solver_ready"
    assert classify_solver_readiness("FACT", "atomic", []) == "solver_ready"
    assert classify_solver_readiness("FORALL", "forall", []) == "solver_ready"


def test_classify_needs_review():
    assert classify_solver_readiness("META", "atomic", []) == "needs_review"
    assert classify_solver_readiness("RULE", "implies", ["modal_not_necessarily"]) == "needs_review"


def test_classify_needs_lowering():
    assert classify_solver_readiness("IFF", "iff", []) == "needs_lowering"
    assert classify_solver_readiness("RULE", "or", []) == "needs_lowering"
