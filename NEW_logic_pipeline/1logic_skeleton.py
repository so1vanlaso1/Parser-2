from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SkeletonKind = Literal[
    "FACT",
    "EXISTS",
    "FORALL",
    "RULE",
    "ONLY_IF_RULE",
    "IFF",
    "NON_IF_RULE",
    "OBLIGATION_RULE",
    "MODAL",
    "META",
    "UNKNOWN",
]


FormulaNodeType = Literal[
    "leaf",
    "and",
    "or",
    "not",
    "implies",
    "iff",
    "forall",
    "exists",
    "equation",
    "comparison",
    "cardinality",
]


class TextSpan(BaseModel):
    role: str
    text: str
    variable: str = "x"
    negation_hint: bool = False
    modality_hint: str | None = None
    source: str | None = None


class FormulaSkeleton(BaseModel):
    type: FormulaNodeType
    text: str | None = None
    variable: str | None = None
    children: list["FormulaSkeleton"] = Field(default_factory=list)


class LogicSkeleton(BaseModel):
    premise_id: str
    original: str
    kind: SkeletonKind
    body: TextSpan | None = None
    antecedent: TextSpan | None = None
    consequent: TextSpan | None = None
    left: TextSpan | None = None
    right: TextSpan | None = None
    formula_tree: FormulaSkeleton | None = None
    quantifier: Literal["forall", "exists"] | None = None
    variable: str = "x"
    risk_flags: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    needs_review: bool = False
    notes: list[str] = Field(default_factory=list)


class SkeletonBuildResult(BaseModel):
    skeletons: list[LogicSkeleton]
    errors: list[str] = Field(default_factory=list)


FormulaSkeleton.model_rebuild()
