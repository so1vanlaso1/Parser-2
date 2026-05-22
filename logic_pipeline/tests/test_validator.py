"""Tests for the Stage 4 LogicValidator."""
from src.schemas import CompiledPremise, LogicNode, Stage3Output
from src.stage4_validate import LogicValidator, classify_solver_readiness


def _make_stage3(premises: list[CompiledPremise]) -> Stage3Output:
    return Stage3Output(compiled=premises)


# ── Existing tests (unchanged) ──────────────────────────────────────────

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


# ── New tests: transferred checks from schema ───────────────────────────

def test_atomic_missing_name_error():
    """atomic node with empty name should be caught by validator."""
    output = _make_stage3([
        CompiledPremise(
            premise_id="P1",
            kind="FACT",
            cnl="something",
            ast=LogicNode(type="atomic", arguments=["john"]),
        )
    ])

    report = LogicValidator().validate_stage3(output)
    assert not report.ok
    assert any("missing name" in i.message for i in report.issues)


def test_atomic_missing_arguments_error():
    """atomic node with no arguments should be caught by validator."""
    output = _make_stage3([
        CompiledPremise(
            premise_id="P1",
            kind="FACT",
            cnl="student",
            ast=LogicNode(type="atomic", name="student"),
        )
    ])

    report = LogicValidator().validate_stage3(output)
    assert not report.ok
    assert any("no arguments" in i.message for i in report.issues)


def test_forall_wrong_child_count_error():
    """forall with 0 children should be caught by validator."""
    output = _make_stage3([
        CompiledPremise(
            premise_id="P1",
            kind="FORALL",
            cnl="Every student",
            ast=LogicNode(type="forall", variable="x"),
        )
    ])

    report = LogicValidator().validate_stage3(output)
    assert not report.ok
    assert any("exactly 1 child" in i.message for i in report.issues)


def test_exists_wrong_child_count_error():
    """exists with 2 children should be caught by validator."""
    output = _make_stage3([
        CompiledPremise(
            premise_id="P1",
            kind="EXISTS",
            cnl="Some student",
            ast=LogicNode(
                type="exists",
                variable="x",
                children=[
                    LogicNode(type="atomic", name="a", arguments=["x"]),
                    LogicNode(type="atomic", name="b", arguments=["x"]),
                ],
            ),
        )
    ])

    report = LogicValidator().validate_stage3(output)
    assert not report.ok
    assert any("exactly 1 child" in i.message for i in report.issues)


def test_and_too_few_children_error():
    """and node with 1 child should be caught by validator."""
    output = _make_stage3([
        CompiledPremise(
            premise_id="P1",
            kind="FACT",
            cnl="a and ...",
            ast=LogicNode(
                type="and",
                children=[LogicNode(type="atomic", name="a", arguments=["john"])],
            ),
        )
    ])

    report = LogicValidator().validate_stage3(output)
    assert not report.ok
    assert any("at least 2 children" in i.message for i in report.issues)


def test_or_too_few_children_error():
    """or node with 1 child should be caught by validator."""
    output = _make_stage3([
        CompiledPremise(
            premise_id="P1",
            kind="FACT",
            cnl="a or ...",
            ast=LogicNode(
                type="or",
                children=[LogicNode(type="atomic", name="a", arguments=["john"])],
            ),
        )
    ])

    report = LogicValidator().validate_stage3(output)
    assert not report.ok
    assert any("at least 2 children" in i.message for i in report.issues)


def test_equation_missing_fields_error():
    """equation with missing operator/left/right should be caught."""
    output = _make_stage3([
        CompiledPremise(
            premise_id="P1",
            kind="FACT",
            cnl="x == 5",
            ast=LogicNode(type="equation", operator="=="),
        )
    ])

    report = LogicValidator().validate_stage3(output)
    assert not report.ok
    assert any("equation" in i.message.lower() for i in report.issues)


def test_nested_rule_detection_warning():
    """RULE with implies-inside-implies should produce a warning suggesting META."""
    output = _make_stage3([
        CompiledPremise(
            premise_id="P1",
            kind="RULE",
            cnl="If passing implies graduation, then students are eligible.",
            ast=LogicNode(
                type="forall",
                variable="x",
                children=[
                    LogicNode(
                        type="implies",
                        children=[
                            LogicNode(
                                type="implies",
                                children=[
                                    LogicNode(type="atomic", name="passes", arguments=["x"]),
                                    LogicNode(type="atomic", name="graduates", arguments=["x"]),
                                ],
                            ),
                            LogicNode(type="atomic", name="eligible", arguments=["x"]),
                        ],
                    )
                ],
            ),
        )
    ])

    report = LogicValidator().validate_stage3(output)
    # Nested implies in RULE is a warning, not an error — report should still be ok.
    assert report.ok
    assert any("nested implies" in i.message.lower() for i in report.issues)
    assert any("META" in i.message for i in report.issues)


def test_nested_rule_not_triggered_for_meta():
    """META kind with nested implies should NOT get the nested-rule warning."""
    output = _make_stage3([
        CompiledPremise(
            premise_id="P1",
            kind="META",
            cnl="If passing implies graduation, then students are eligible.",
            ast=LogicNode(
                type="forall",
                variable="x",
                children=[
                    LogicNode(
                        type="implies",
                        children=[
                            LogicNode(
                                type="implies",
                                children=[
                                    LogicNode(type="atomic", name="passes", arguments=["x"]),
                                    LogicNode(type="atomic", name="graduates", arguments=["x"]),
                                ],
                            ),
                            LogicNode(type="atomic", name="eligible", arguments=["x"]),
                        ],
                    )
                ],
            ),
        )
    ])

    report = LogicValidator().validate_stage3(output)
    # No nested-rule warning for META kind.
    assert not any("nested implies" in i.message.lower() for i in report.issues)


# ── Solver readiness classification (unchanged) ─────────────────────────

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
