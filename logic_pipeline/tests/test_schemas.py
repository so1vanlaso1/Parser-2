"""Tests for Pydantic schemas — LogicNode shape validation."""
import pytest
from src.schemas import LogicNode


def test_atomic_node_valid():
    node = LogicNode(type="atomic", name="student", arguments=["john"])
    assert node.name == "student"
    assert node.arguments == ["john"]


def test_atomic_node_missing_name():
    with pytest.raises(ValueError, match="atomic node requires name"):
        LogicNode(type="atomic", arguments=["john"])


def test_atomic_node_missing_arguments():
    with pytest.raises(ValueError, match="atomic node requires at least one argument"):
        LogicNode(type="atomic", name="student")


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


def test_implies_node_too_few_children():
    with pytest.raises(ValueError, match="implies node requires at least 2 children"):
        LogicNode(
            type="implies",
            children=[LogicNode(type="atomic", name="student", arguments=["x"])],
        )


def test_not_node_valid():
    node = LogicNode(
        type="not",
        children=[LogicNode(type="atomic", name="has_housing", arguments=["x"])],
    )
    assert node.type == "not"
    assert len(node.children) == 1


def test_not_node_wrong_child_count():
    with pytest.raises(ValueError, match="not node requires exactly 1 child"):
        LogicNode(
            type="not",
            children=[
                LogicNode(type="atomic", name="a", arguments=["x"]),
                LogicNode(type="atomic", name="b", arguments=["x"]),
            ],
        )


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


def test_forall_node_missing_variable():
    with pytest.raises(ValueError, match="forall node requires variable"):
        LogicNode(
            type="forall",
            children=[LogicNode(type="atomic", name="student", arguments=["x"])],
        )


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


def test_and_node_too_few_children():
    with pytest.raises(ValueError, match="and node requires at least 2 children"):
        LogicNode(
            type="and",
            children=[LogicNode(type="atomic", name="a", arguments=["x"])],
        )


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


def test_equation_node_missing_fields():
    with pytest.raises(ValueError, match="equation node requires operator, left, and right"):
        LogicNode(type="equation", operator="==")
