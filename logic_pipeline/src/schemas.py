from __future__ import annotations

from typing import Any, Literal, Optional, Union
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

PredicateGroupConnective = Literal["and", "or"]
CIRKind = Literal["fact", "exists", "forall", "rule", "meta"]
CIRConnective = Literal["and", "or", "not", "implies", "iff", "forall", "exists", "atom"]


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

    # Stage 1 structural-guide fields.  These are the CIR migration source of
    # truth where available; cnl/kind_hint remain for backward compatibility.
    mode: Literal["direct_solver", "llm_guided", "blocked_review"] = "llm_guided"
    recognized_type: Optional[str] = None
    target_kind: Optional[CIRKind] = None
    subject_type: Optional[str] = None
    subject: Optional[str] = None
    slots: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    direct_cir: Optional["CIRPremise"] = None

    # Important for later validation.
    risk_flags: list[str] = Field(default_factory=list)

    # Optional structured slots from Stage 1.
    if_part: Optional[str] = None
    then_part: Optional[str] = None
    body: Optional[str] = None


class Stage1Output(BaseModel):
    statements: list[CNLStatement]


# ---------------------------------------------------------------------------
# CIR models
# ---------------------------------------------------------------------------

class CIRAtom(BaseModel):
    name: str
    arguments: list[str] = Field(default_factory=list)
    negated: bool = False


class CIRLink(BaseModel):
    type: CIRConnective
    children: list[Any] = Field(default_factory=list)
    atom: Optional[CIRAtom] = None
    variable: Optional[str] = None


class CIRFact(BaseModel):
    kind: Literal["fact"] = "fact"
    atoms: list[CIRAtom] = Field(default_factory=list)


class CIRExists(BaseModel):
    kind: Literal["exists"] = "exists"
    variable: str = "x"
    body: list[CIRAtom] = Field(default_factory=list)


class CIRForall(BaseModel):
    kind: Literal["forall"] = "forall"
    variable: str = "x"
    antecedent: list[CIRAtom] = Field(default_factory=list)
    consequent: list[CIRAtom] = Field(default_factory=list)
    body: list[CIRAtom] = Field(default_factory=list)


class CIRRule(BaseModel):
    kind: Literal["rule"] = "rule"
    variable: str = "x"
    antecedent: list[CIRAtom] = Field(default_factory=list)
    consequent: list[CIRAtom] = Field(default_factory=list)


class CIRMeta(BaseModel):
    kind: Literal["meta"] = "meta"
    formula: dict[str, Any]
    flat_atoms: list[CIRAtom] = Field(default_factory=list)


class CIRPremise(BaseModel):
    premise_id: str
    kind: CIRKind
    cir: Union[CIRFact, CIRExists, CIRForall, CIRRule, CIRMeta]


class Stage3CIROutput(BaseModel):
    premises: list[CIRPremise]


# ---------------------------------------------------------------------------
# Stage 3-lite predicate frame models
# ---------------------------------------------------------------------------

class PredicateAtom(BaseModel):
    name: str
    arguments: list[str] = Field(default_factory=list)
    negated: bool = False


class PredicateGroup(BaseModel):
    connective: PredicateGroupConnective = "and"
    atoms: list[PredicateAtom] = Field(default_factory=list)


class PredicateFrame(BaseModel):
    premise_id: str
    kind: PremiseKind
    cnl: str

    # For row-level logic tasks, most variables are single quantified variables
    # such as x/y. Constants remain in atom arguments and do not need binding.
    variable: Optional[str] = None

    # Rules and universal statements use antecedent/consequent. Facts and
    # existential statements usually use body.
    antecedent: Optional[PredicateGroup] = None
    consequent: Optional[PredicateGroup] = None
    body: Optional[PredicateGroup] = None

    # If true, the deterministic builder will skip this frame and use the
    # existing full-AST compiler fallback for the premise.
    unsupported: bool = False
    notes: list[str] = Field(default_factory=list)


class PredicateFrameOutput(BaseModel):
    frames: list[PredicateFrame]


# ---------------------------------------------------------------------------
# Formula graph models
# ---------------------------------------------------------------------------

class FlatAtom(BaseModel):
    atom_id: str
    predicate: str
    arguments: list[str] = Field(default_factory=list)
    negated: bool = False
    source_premise_id: Optional[str] = None


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

    # Lenient: shape problems are stored as warnings instead of raising
    # ValueError.  The Stage 4 Validator surfaces them and Stage 5 Repair
    # Loop can fix them before the pipeline fails.
    _shape_warnings: list[str] = []

    @model_validator(mode="after")
    def check_shape(self):
        """Lenient shape check — collects warnings instead of raising."""
        warnings: list[str] = []

        if self.type == "atomic":
            if not self.name:
                warnings.append("atomic node missing name")
            if not self.arguments:
                warnings.append("atomic node missing arguments")

        if self.type in {"and", "or", "implies", "iff"}:
            if len(self.children) < 2:
                warnings.append(f"{self.type} node has {len(self.children)} children (need >=2)")

        if self.type == "not":
            if len(self.children) != 1:
                warnings.append(f"not node has {len(self.children)} children (need 1)")

        if self.type in {"forall", "exists"}:
            if not self.variable:
                warnings.append(f"{self.type} node missing variable")
            if len(self.children) != 1:
                warnings.append(f"{self.type} node has {len(self.children)} children (need 1)")

        if self.type == "equation":
            if self.operator is None or self.left is None or self.right is None:
                warnings.append("equation node missing operator, left, or right")

        object.__setattr__(self, "_shape_warnings", warnings)
        return self


LogicNode.model_rebuild()
CNLStatement.model_rebuild()


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
    direct_solver_ready: bool = False
    meta_resolvable: bool = False
    meta_resolved: bool = False
    solver_ready_after_meta_resolution: bool = False
    add_to_solver: bool = False
    resolution: Optional[str] = None
    flat_atoms: list[FlatAtom] = Field(default_factory=list)
    formula_tree: Optional[dict[str, Any]] = None
    solver_export: list[Any] = Field(default_factory=list)
    meta_links: list[dict[str, Any]] = Field(default_factory=list)
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
    query_type: str = "UNKNOWN"
    direct_solver_ready: bool = False
    formula_tree: Optional[dict[str, Any]] = None
    flat_atoms: list[FlatAtom] = Field(default_factory=list)
    unsupported: bool = False
    notes: list[str] = Field(default_factory=list)


class FullParseResult(BaseModel):
    premises: list[CompiledPremise]
    question: Optional[QuestionParse] = None
    status: Literal["success", "partial_success", "failed"] = "success"
    error: Optional[str] = None
    question_parse_valid: bool = True
