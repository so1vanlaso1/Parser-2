"""Tests for Pydantic schemas — LogicNode shape validation.

After the schema relaxation (Step 1), malformed nodes are no longer rejected
by Pydantic.  Instead they are constructed with _shape_warnings so the
Stage 4 Validator / Stage 5 Repair Loop can handle them.
"""
import pytest
from src.schemas import LogicNode


# ── Valid nodes (unchanged) ──────────────────────────────────────────────

def test_atomic_node_valid():
    node = LogicNode(type="atomic", name="student", arguments=["john"])
    assert node.name == "student"
    assert node.arguments == ["john"]


def test_implies_node_valid():
    node = LogicNode(
        type="implies",
        children=[
            LogicNode(type="atomic", name="student", arguments=["x"]),
            LogicNode(type="atomic", name="eligible", arguments=["x"]),
        ],
    )
    assert node.type == "implies"
    assert len(node.children) == 2


def test_not_node_valid():
    node = LogicNode(
        type="not",
        children=[LogicNode(type="atomic", name="has_housing", arguments=["x"])],
    )
    assert node.type == "not"
    assert len(node.children) == 1


def test_forall_node_valid():
    node = LogicNode(
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
    )
    assert node.type == "forall"
    assert node.variable == "x"


def test_exists_node_valid():
    node = LogicNode(
        type="exists",
        variable="x",
        children=[
            LogicNode(
                type="and",
                children=[
                    LogicNode(type="atomic", name="student", arguments=["x"]),
                    LogicNode(type="atomic", name="has_housing", arguments=["x"]),
                ],
            )
        ],
    )
    assert node.type == "exists"


def test_iff_node_valid():
    node = LogicNode(
        type="iff",
        children=[
            LogicNode(type="atomic", name="passes", arguments=["x"]),
            LogicNode(type="atomic", name="submits_thesis", arguments=["x"]),
        ],
    )
    assert node.type == "iff"


def test_equation_node_valid():
    node = LogicNode(
        type="equation",
        operator=">=",
        left="credits",
        right=150,
    )
    assert node.type == "equation"


# ── Malformed nodes now ACCEPTED (lenient schema) ────────────────────────
# These previously raised ValueError. After schema relaxation, they must
# construct successfully with _shape_warnings populated.

def test_atomic_missing_name_accepted():
    """atomic without name — previously crashed, now accepted with warning."""
    node = LogicNode(type="atomic", arguments=["john"])
    assert node._shape_warnings
    assert any("missing name" in w for w in node._shape_warnings)


def test_atomic_missing_arguments_accepted():
    """atomic without arguments — previously crashed, now accepted with warning."""
    node = LogicNode(type="atomic", name="student")
    assert node._shape_warnings
    assert any("missing arguments" in w for w in node._shape_warnings)


def test_implies_too_few_children_accepted():
    """implies with 1 child — previously crashed, now accepted with warning."""
    node = LogicNode(
        type="implies",
        children=[LogicNode(type="atomic", name="student", arguments=["x"])],
    )
    assert node._shape_warnings
    assert any("children" in w for w in node._shape_warnings)


def test_not_wrong_child_count_accepted():
    """not with 2 children — previously crashed, now accepted with warning."""
    node = LogicNode(
        type="not",
        children=[
            LogicNode(type="atomic", name="a", arguments=["x"]),
            LogicNode(type="atomic", name="b", arguments=["x"]),
        ],
    )
    assert node._shape_warnings
    assert any("children" in w for w in node._shape_warnings)


def test_forall_missing_variable_accepted():
    """forall without variable — previously crashed, now accepted with warning."""
    node = LogicNode(
        type="forall",
        children=[LogicNode(type="atomic", name="student", arguments=["x"])],
    )
    assert node._shape_warnings
    assert any("missing variable" in w for w in node._shape_warnings)


def test_forall_wrong_child_count_accepted():
    """forall with 0 children — previously crashed, now accepted with warning."""
    node = LogicNode(type="forall", variable="x")
    assert node._shape_warnings
    assert any("children" in w for w in node._shape_warnings)


def test_exists_wrong_child_count_accepted():
    """exists with 2 children — previously crashed, now accepted with warning."""
    node = LogicNode(
        type="exists",
        variable="x",
        children=[
            LogicNode(type="atomic", name="a", arguments=["x"]),
            LogicNode(type="atomic", name="b", arguments=["x"]),
        ],
    )
    assert node._shape_warnings
    assert any("children" in w for w in node._shape_warnings)


def test_and_too_few_children_accepted():
    """and with 1 child — previously crashed, now accepted with warning."""
    node = LogicNode(
        type="and",
        children=[LogicNode(type="atomic", name="a", arguments=["x"])],
    )
    assert node._shape_warnings
    assert any("children" in w for w in node._shape_warnings)


def test_equation_missing_fields_accepted():
    """equation with missing left/right — previously crashed, now accepted with warning."""
    node = LogicNode(type="equation", operator="==")
    assert node._shape_warnings
    assert any("missing" in w for w in node._shape_warnings)


# ── Well-formed nodes should have NO warnings ────────────────────────────

def test_valid_node_no_warnings():
    """A well-formed node should have zero shape warnings."""
    node = LogicNode(type="atomic", name="student", arguments=["john"])
    assert node._shape_warnings == []
