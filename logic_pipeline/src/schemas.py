from __future__ import annotations

from typing import Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Kind / Type literals
# ---------------------------------------------------------------------------

PremiseKind = Literal[
    "FACT",
    "EXISTS",
    "FORALL",
    "RULE",
    "ONLY_IF_RULE",
    "IFF",
    "NON_IF_RULE",
    "OBLIGATION_RULE",
    "META",
    "UNKNOWN",
]

LogicNodeType = Literal[
    "atomic",
    "and",
    "or",
    "not",
    "implies",
    "iff",
    "forall",
    "exists",
    "equation",
]


# ---------------------------------------------------------------------------
# Stage 1 models
# ---------------------------------------------------------------------------

class RiskFlag(BaseModel):
    name: str
    detail: Optional[str] = None


class CNLStatement(BaseModel):
    premise_id: str
    original: str
    kind_hint: PremiseKind
    cnl: str

    # Important for later validation.
    risk_flags: list[str] = Field(default_factory=list)

    # Optional structured slots from Stage 1.
    if_part: Optional[str] = None
    then_part: Optional[str] = None
    body: Optional[str] = None


class Stage1Output(BaseModel):
    statements: list[CNLStatement]


# ---------------------------------------------------------------------------
# Core AST node
# ---------------------------------------------------------------------------

class LogicNode(BaseModel):
    type: LogicNodeType

    # atomic predicate
    name: Optional[str] = None
    arguments: list[str] = Field(default_factory=list)

    # logical children
    children: list["LogicNode"] = Field(default_factory=list)

    # quantifier
    variable: Optional[str] = None

    # equation / comparison
    operator: Optional[Literal["==", "!=", ">", "<", ">=", "<="]] = None
    left: Optional[Union["LogicNode", str, int, float]] = None
    right: Optional[Union["LogicNode", str, int, float]] = None

    # metadata
    source_premise_id: Optional[str] = None
    confidence: float = 1.0

    @model_validator(mode="after")
    def check_shape(self):
        if self.type == "atomic":
            if not self.name:
                raise ValueError("atomic node requires name")
            if not self.arguments:
                raise ValueError("atomic node requires at least one argument")

        if self.type in {"and", "or", "implies", "iff"}:
            if len(self.children) < 2:
                raise ValueError(f"{self.type} node requires at least 2 children")

        if self.type == "not":
            if len(self.children) != 1:
                raise ValueError("not node requires exactly 1 child")

        if self.type in {"forall", "exists"}:
            if not self.variable:
                raise ValueError(f"{self.type} node requires variable")
            if len(self.children) != 1:
                raise ValueError(f"{self.type} node requires exactly 1 scoped child")

        if self.type == "equation":
            if self.operator is None or self.left is None or self.right is None:
                raise ValueError("equation node requires operator, left, and right")

        return self


LogicNode.model_rebuild()


# ---------------------------------------------------------------------------
# Stage 3 models
# ---------------------------------------------------------------------------

class CompiledPremise(BaseModel):
    premise_id: str
    kind: PremiseKind
    cnl: str
    ast: LogicNode
    solver_ready: bool = False
    needs_review: bool = False
    unsupported: bool = False
    notes: list[str] = Field(default_factory=list)


class Stage3Output(BaseModel):
    compiled: list[CompiledPremise]


# ---------------------------------------------------------------------------
# Question parse models
# ---------------------------------------------------------------------------

class QuestionParse(BaseModel):
    question: str
    choices: dict[str, LogicNode] = Field(default_factory=dict)
    query: Optional[LogicNode] = None


class FullParseResult(BaseModel):
    premises: list[CompiledPremise]
    question: Optional[QuestionParse] = None
