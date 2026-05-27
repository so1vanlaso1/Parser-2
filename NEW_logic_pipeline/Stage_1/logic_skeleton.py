from __future__ import annotations

"""Shared Stage 1 / Stage 4 schema contracts.

These models intentionally store *structure and text spans only*. They do not
contain final predicate atoms, solver ASTs, or solver exports.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


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
    """A raw English phrase span that later stages atomize.

    `negation_hint` is only a hint. The atomizer/validator still decides
    whether it can apply classical negation safely. Modal phrases such as
    "not necessarily" should use modality_hint instead of negation_hint.
    """

    role: str
    text: str
    variable: str = "x"
    negation_hint: bool = False
    modality_hint: str | None = None
    source: str | None = None

    @field_validator("text", "role", "variable", mode="before")
    @classmethod
    def _strip_strings(cls, value: Any) -> Any:
        if isinstance(value, str):
            return " ".join(value.split()).strip()
        return value


class FormulaSkeleton(BaseModel):
    """Recursive text-only formula skeleton for META/nested premises."""

    type: FormulaNodeType
    text: str | None = None
    variable: str | None = None
    children: list["FormulaSkeleton"] = Field(default_factory=list)

    @field_validator("text", "variable", mode="before")
    @classmethod
    def _strip_optional_strings(cls, value: Any) -> Any:
        if isinstance(value, str):
            cleaned = " ".join(value.split()).strip()
            return cleaned or None
        return value


class MatchedEvidence(BaseModel):
    """Debug evidence explaining why the router selected a skeleton kind."""

    rule_id: str
    cue: str | None = None
    span: str | None = None
    start: int | None = None
    end: int | None = None
    confidence: float | None = None
    direction: str | None = None
    consequent_negation: bool | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class LogicSkeleton(BaseModel):
    """The main Stage 1 output.

    The object is deliberately not solver-ready. It only preserves the logical
    shape and text leaves needed by later stages.
    """

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

    matched_rule: str | None = None
    matched_evidence: MatchedEvidence | None = None

    @field_validator("original", "premise_id", "variable", mode="before")
    @classmethod
    def _strip_strings(cls, value: Any) -> Any:
        if isinstance(value, str):
            return " ".join(value.split()).strip()
        return value


class SkeletonBuildResult(BaseModel):
    skeletons: list[LogicSkeleton]
    errors: list[str] = Field(default_factory=list)


FormulaSkeleton.model_rebuild()


__all__ = [
    "SkeletonKind",
    "FormulaNodeType",
    "TextSpan",
    "FormulaSkeleton",
    "MatchedEvidence",
    "LogicSkeleton",
    "SkeletonBuildResult",
]
