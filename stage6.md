# Stage 6 Validator Implementation Guide

## Goal

Stage 6 is the deterministic quality gate between your parser and your solver.

It does **not** parse new text, create predicates, repair logic, or solve anything. Its job is to inspect the outputs from Stage 0–5 and decide whether the parsed logic is safe enough to continue.

```text
Stage 0–5 output  →  Stage 6 Validator  →  readiness classification
```

Stage 6 should answer:

```text
Is the skeleton structurally valid?
Are all leaves properly atomized?
Are all predicates known and canonical?
Are predicate arguments valid?
Did the parse preserve the meaning of the original phrase?
Can this go directly to the solver?
Does it need lowering?
Does it need META resolution?
Does it need review?
Is it unsupported?
```

---

## Core Design Rule

Stage 6 should be **deterministic**.

Do not use an LLM as the main validator.

Use:

```text
✓ Python rules
✓ predicate registry
✓ arity schema
✓ argument-role schema
✓ AST schema
✓ solver capability config
✓ issue codes
✓ readiness classifier
```

Avoid:

```text
✗ LLM deciding whether a parse is valid
✗ LLM silently repairing logic
✗ LLM inventing missing facts
✗ LLM inventing new predicates inside validation
```

Optional LLM use is okay only for explanation after deterministic validation, not for the validation decision itself.

---

## Recommended Folder Structure

```text
NEW_logic_pipeline/
  Stage_6/
    __init__.py
    validator.py
    validation_models.py
    issue_codes.py
    structural_validator.py
    predicate_validator.py
    argument_validator.py
    semantic_validator.py
    ast_validator.py
    solver_readiness.py
    registry_schema.py
    tests/
      test_structural_validator.py
      test_predicate_validator.py
      test_argument_validator.py
      test_semantic_validator.py
      test_solver_readiness.py
```

---

## Input Contract

Stage 6 receives the full parsed record from Stage 0–5.

Minimum expected input:

```python
parsed_record = {
    "id": "logic-xxxx-qx",
    "premises_nl": [...],
    "question": "...",
    "skeletons": [...],
    "atomization_requests": [...],
    "atomization_results": [...],
    "canonicalization": {...},
    "asts": [...],              # Stage 5 output, recommended
    "predicate_registry": {...} # or passed separately
}
```

Stage 6 should still work even if `asts` is missing. In that case, it should skip AST validation and report:

```text
AST_MISSING
```

or classify the row as not directly solver-ready.

---

## Output Contract

Stage 6 outputs a validation report.

```python
{
    "parse_valid": bool,
    "direct_solver_ready": bool,
    "needs_lowering": bool,
    "needs_meta_resolution": bool,
    "needs_review": bool,
    "unsupported": bool,
    "issues": [...],
    "readiness_reasons": [...],
    "summary": {...}
}
```

Example:

```json
{
  "parse_valid": true,
  "direct_solver_ready": false,
  "needs_lowering": true,
  "needs_meta_resolution": false,
  "needs_review": false,
  "unsupported": false,
  "issues": [
    {
      "code": "READY_CARDINALITY_REQUIRES_LOWERING",
      "severity": "info",
      "message": "grade_count_at_least requires lowering for current solver.",
      "premise_id": "P5",
      "request_id": "P5_antecedent",
      "suggested_stage": "Stage 8 or Solver Adapter"
    }
  ],
  "readiness_reasons": [
    "cardinality_requires_lowering"
  ]
}
```

---

# 1. Validation Models

Create `validation_models.py`.

```python
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

Severity = Literal["info", "warning", "error", "fatal"]


@dataclass
class ValidationIssue:
    code: str
    severity: Severity
    message: str
    premise_id: str | None = None
    request_id: str | None = None
    path: list[int] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    suggested_stage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    parse_valid: bool
    direct_solver_ready: bool
    needs_lowering: bool
    needs_meta_resolution: bool
    needs_review: bool
    unsupported: bool
    issues: list[ValidationIssue]
    readiness_reasons: list[str]
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["issues"] = [issue.to_dict() for issue in self.issues]
        return data
```

---

# 2. Issue Codes

Create `issue_codes.py`.

```python
STRUCT_MISSING_REQUIRED_FIELD = "STRUCT_MISSING_REQUIRED_FIELD"
STRUCT_BAD_NODE_CHILD_COUNT = "STRUCT_BAD_NODE_CHILD_COUNT"
STRUCT_FORMULA_LIKE_LEAF = "STRUCT_FORMULA_LIKE_LEAF"
STRUCT_META_WITHOUT_FORMULA_TREE = "STRUCT_META_WITHOUT_FORMULA_TREE"
STRUCT_UNKNOWN_SKELETON_KIND = "STRUCT_UNKNOWN_SKELETON_KIND"

PRED_UNKNOWN = "PRED_UNKNOWN"
PRED_ARITY_MISMATCH = "PRED_ARITY_MISMATCH"
PRED_CANONICALIZATION_CONFLICT = "PRED_CANONICALIZATION_CONFLICT"

ARG_ROLE_MISMATCH = "ARG_ROLE_MISMATCH"
ARG_SAME_PERSON_AND_OBJECT = "ARG_SAME_PERSON_AND_OBJECT"
ARG_VARIABLE_SHOULD_BE_CONSTANT = "ARG_VARIABLE_SHOULD_BE_CONSTANT"
ARG_CONSTANT_SHOULD_BE_VARIABLE = "ARG_CONSTANT_SHOULD_BE_VARIABLE"

NEGATION_INSIDE_PREDICATE = "NEGATION_INSIDE_PREDICATE"
NEGATION_DOUBLE_NEGATION = "NEGATION_DOUBLE_NEGATION"
MODAL_NOT_NECESSARILY_AS_NEGATION = "MODAL_NOT_NECESSARILY_AS_NEGATION"

SEMANTIC_OR_BECAME_AND = "SEMANTIC_OR_BECAME_AND"
SEMANTIC_OBJECT_CHANGED = "SEMANTIC_OBJECT_CHANGED"
SEMANTIC_DOMAIN_RESTRICTION_LOST = "SEMANTIC_DOMAIN_RESTRICTION_LOST"
SEMANTIC_NUMERIC_VALUE_UNTYPED = "SEMANTIC_NUMERIC_VALUE_UNTYPED"
SEMANTIC_DEONTIC_UNRESOLVED = "SEMANTIC_DEONTIC_UNRESOLVED"

AST_MISSING = "AST_MISSING"
AST_UNKNOWN_NODE_TYPE = "AST_UNKNOWN_NODE_TYPE"
AST_BAD_CHILD_COUNT = "AST_BAD_CHILD_COUNT"
AST_ATOM_MISSING_PREDICATE = "AST_ATOM_MISSING_PREDICATE"
AST_ATOM_UNKNOWN_PREDICATE = "AST_ATOM_UNKNOWN_PREDICATE"

READY_META_REQUIRES_RESOLUTION = "READY_META_REQUIRES_RESOLUTION"
READY_OR_REQUIRES_LOWERING = "READY_OR_REQUIRES_LOWERING"
READY_EXISTS_REQUIRES_SKOLEMIZATION = "READY_EXISTS_REQUIRES_SKOLEMIZATION"
READY_IFF_REQUIRES_LOWERING = "READY_IFF_REQUIRES_LOWERING"
READY_CARDINALITY_REQUIRES_LOWERING = "READY_CARDINALITY_REQUIRES_LOWERING"
READY_UNSUPPORTED_LOGIC = "READY_UNSUPPORTED_LOGIC"
```

---

# 3. Predicate Registry Schema

Create `registry_schema.py`.

Your predicate registry should store:

```text
name
arity
argument roles
whether it is solver-safe
whether it needs lowering
```

Example:

```python
DEFAULT_PREDICATE_REGISTRY = {
    "student": {
        "arity": 1,
        "roles": ["person"],
        "solver_safe": True,
    },
    "subject": {
        "arity": 1,
        "roles": ["subject"],
        "solver_safe": True,
    },
    "ask_questions": {
        "arity": 1,
        "roles": ["person"],
        "solver_safe": True,
    },
    "attending_tutorials": {
        "arity": 1,
        "roles": ["person"],
        "solver_safe": True,
    },
    "preparing_for_exam": {
        "arity": 1,
        "roles": ["person"],
        "solver_safe": True,
    },
    "understand_material": {
        "arity": 1,
        "roles": ["person"],
        "solver_safe": True,
    },
    "contains_knowledge": {
        "arity": 1,
        "roles": ["subject"],
        "solver_safe": True,
    },
    "has_knowledge_of_subject": {
        "arity": 2,
        "roles": ["person", "subject"],
        "solver_safe": True,
    },
    "explain_subject": {
        "arity": 2,
        "roles": ["person", "subject"],
        "solver_safe": True,
    },
    "friends_understand_subject": {
        "arity": 2,
        "roles": ["person", "subject"],
        "solver_safe": True,
    },
    "mastered_subject": {
        "arity": 2,
        "roles": ["person", "subject"],
        "solver_safe": True,
    },
    "earned_grade": {
        "arity": 2,
        "roles": ["person", "grade"],
        "solver_safe": True,
    },
    "earned_grade_in_subject": {
        "arity": 3,
        "roles": ["person", "grade", "subject"],
        "solver_safe": True,
    },
    "grade_count": {
        "arity": 3,
        "roles": ["person", "grade", "count"],
        "solver_safe": False,
        "requires_lowering": True,
    },
    "grade_count_at_least": {
        "arity": 3,
        "roles": ["person", "grade_group", "count"],
        "solver_safe": False,
        "requires_lowering": True,
    },
    "receive_scholarship": {
        "arity": 1,
        "roles": ["person"],
        "solver_safe": True,
    },
}
```

Role values:

```python
ROLE_VALUES = {
    "person": {"x", "y", "z", "tuan", "laura", "student"},
    "subject": {"subject", "the_subject", "math", "physics", "chemistry", "biology", "it"},
    "grade": {"a", "a_plus", "b", "c", "d", "f"},
    "grade_group": {"a_or_a_plus"},
    "count": {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"},
}
```

---

# 4. Structural Validator

Create `structural_validator.py`.

Purpose:

```text
Check skeleton shape.
Check formula_tree shape.
Ensure formula-like text does not reach leaf atomization.
```

```python
from .validation_models import ValidationIssue
from . import issue_codes as C

VALID_KINDS = {
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
}

FORMULA_CUES = [
    " if ",
    " then ",
    " implies ",
    " only if ",
    " if and only if ",
    " there exists ",
    " every ",
    " all ",
]


def validate_skeletons(parsed_record: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for skeleton in parsed_record.get("skeletons", []):
        premise_id = skeleton.get("premise_id")
        kind = skeleton.get("kind")

        if kind not in VALID_KINDS:
            issues.append(ValidationIssue(
                code=C.STRUCT_UNKNOWN_SKELETON_KIND,
                severity="error",
                message=f"Unknown skeleton kind: {kind}",
                premise_id=premise_id,
                evidence={"kind": kind},
                suggested_stage="Stage 1",
            ))
            continue

        if kind == "FORALL":
            if not skeleton.get("antecedent") or not skeleton.get("consequent"):
                issues.append(_missing_field(premise_id, kind, "antecedent/consequent"))

        elif kind == "EXISTS":
            if not skeleton.get("body") and not skeleton.get("formula_tree"):
                issues.append(_missing_field(premise_id, kind, "body/formula_tree"))

        elif kind in {"RULE", "ONLY_IF_RULE", "NON_IF_RULE", "OBLIGATION_RULE"}:
            if not skeleton.get("antecedent") or not skeleton.get("consequent"):
                issues.append(_missing_field(premise_id, kind, "antecedent/consequent"))

        elif kind == "IFF":
            if not skeleton.get("left") or not skeleton.get("right"):
                issues.append(_missing_field(premise_id, kind, "left/right"))

        elif kind == "META":
            if not skeleton.get("formula_tree"):
                issues.append(ValidationIssue(
                    code=C.STRUCT_META_WITHOUT_FORMULA_TREE,
                    severity="error",
                    message="META skeleton must contain formula_tree.",
                    premise_id=premise_id,
                    suggested_stage="Stage 1",
                ))
            else:
                issues.extend(validate_formula_tree(skeleton, skeleton["formula_tree"]))

    return issues


def _missing_field(premise_id: str | None, kind: str, field_name: str) -> ValidationIssue:
    return ValidationIssue(
        code=C.STRUCT_MISSING_REQUIRED_FIELD,
        severity="error",
        message=f"{kind} skeleton missing required field: {field_name}.",
        premise_id=premise_id,
        evidence={"kind": kind, "missing": field_name},
        suggested_stage="Stage 1",
    )


def validate_formula_tree(skeleton: dict, root: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    premise_id = skeleton.get("premise_id")

    def walk(node: dict, path: list[int]) -> None:
        node_type = node.get("type")
        children = node.get("children", []) or []

        if node_type == "leaf":
            text = f" {str(node.get('text', '')).lower()} "
            if any(cue in text for cue in FORMULA_CUES):
                issues.append(ValidationIssue(
                    code=C.STRUCT_FORMULA_LIKE_LEAF,
                    severity="error",
                    message="Formula-like text reached atomization as a leaf.",
                    premise_id=premise_id,
                    path=path,
                    evidence={"leaf_text": node.get("text")},
                    suggested_stage="Stage 1 or Stage 2",
                ))
            return

        expected_children = {
            "implies": 2,
            "iff": 2,
            "forall": 1,
            "exists": 1,
            "not": 1,
        }

        if node_type in expected_children:
            expected = expected_children[node_type]
            actual = len(children)
            if actual != expected:
                issues.append(ValidationIssue(
                    code=C.STRUCT_BAD_NODE_CHILD_COUNT,
                    severity="error",
                    message=f"Formula node {node_type} expects {expected} children, got {actual}.",
                    premise_id=premise_id,
                    path=path,
                    evidence={"node_type": node_type, "actual_children": actual},
                    suggested_stage="Stage 1",
                ))

        elif node_type in {"and", "or"}:
            if len(children) < 2:
                issues.append(ValidationIssue(
                    code=C.STRUCT_BAD_NODE_CHILD_COUNT,
                    severity="error",
                    message=f"Formula node {node_type} expects at least 2 children.",
                    premise_id=premise_id,
                    path=path,
                    suggested_stage="Stage 1",
                ))

        else:
            issues.append(ValidationIssue(
                code=C.STRUCT_BAD_NODE_CHILD_COUNT,
                severity="error",
                message=f"Unknown formula node type: {node_type}.",
                premise_id=premise_id,
                path=path,
                evidence={"node_type": node_type},
                suggested_stage="Stage 1",
            ))

        for index, child in enumerate(children):
            walk(child, path + [index])

    walk(root, [])
    return issues
```

---

# 5. Predicate Validator

Create `predicate_validator.py`.

Purpose:

```text
Check unknown predicates.
Check arity.
Check negation is outside predicate name.
```

```python
from .validation_models import ValidationIssue
from . import issue_codes as C

BAD_NEGATION_NAME_PARTS = [
    "not_",
    "_not",
    "non_",
    "without_",
    "lack_",
    "cannot_",
]


def iter_atoms(parsed_record: dict):
    for result in parsed_record.get("atomization_results", []):
        for atom in result.get("atoms", []):
            yield result, atom


def validate_predicates(parsed_record: dict, registry: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for result, atom in iter_atoms(parsed_record):
        name = atom.get("name")
        args = [str(arg) for arg in atom.get("arguments", [])]

        if name not in registry:
            issues.append(ValidationIssue(
                code=C.PRED_UNKNOWN,
                severity="error",
                message=f"Unknown predicate: {name}",
                premise_id=result.get("premise_id"),
                request_id=result.get("request_id"),
                evidence={
                    "predicate": name,
                    "arguments": args,
                    "source_phrase": atom.get("source_phrase"),
                },
                suggested_stage="Stage 4",
            ))
            continue

        expected_arity = registry[name]["arity"]
        actual_arity = len(args)
        if expected_arity != actual_arity:
            issues.append(ValidationIssue(
                code=C.PRED_ARITY_MISMATCH,
                severity="error",
                message=f"{name} expects {expected_arity} arguments, got {actual_arity}.",
                premise_id=result.get("premise_id"),
                request_id=result.get("request_id"),
                evidence={
                    "predicate": name,
                    "expected_arity": expected_arity,
                    "actual_arity": actual_arity,
                    "arguments": args,
                    "source_phrase": atom.get("source_phrase"),
                },
                suggested_stage="Stage 4",
            ))

        if any(part in name for part in BAD_NEGATION_NAME_PARTS):
            issues.append(ValidationIssue(
                code=C.NEGATION_INSIDE_PREDICATE,
                severity="error",
                message="Negation should be outside predicate name.",
                premise_id=result.get("premise_id"),
                request_id=result.get("request_id"),
                evidence={"atom": atom},
                suggested_stage="Stage 3 or Stage 4",
            ))

    return issues
```

---

# 6. Argument Validator

Create `argument_validator.py`.

Purpose:

```text
Check whether predicate arguments match expected semantic roles.
Catch bugs like explain_subject(x, x).
```

```python
from .validation_models import ValidationIssue
from . import issue_codes as C
from .predicate_validator import iter_atoms


def argument_matches_role(arg: str, role: str) -> bool:
    arg = str(arg).lower()

    if role == "person":
        return arg in {"x", "y", "z", "tuan", "laura"} or arg.endswith("_person")

    if role == "subject":
        return arg in {"subject", "the_subject", "math", "physics", "chemistry", "biology", "it"}

    if role == "grade":
        return arg in {"a", "a_plus", "b", "c", "d", "f"}

    if role == "grade_group":
        return arg in {"a_or_a_plus"}

    if role == "count":
        return arg.isdigit()

    return True


def validate_argument_roles(parsed_record: dict, registry: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for result, atom in iter_atoms(parsed_record):
        name = atom.get("name")
        args = [str(arg) for arg in atom.get("arguments", [])]

        if name not in registry:
            continue

        roles = registry[name].get("roles", [])

        if len(args) != len(roles):
            continue

        for index, (arg, role) in enumerate(zip(args, roles)):
            if not argument_matches_role(arg, role):
                issues.append(ValidationIssue(
                    code=C.ARG_ROLE_MISMATCH,
                    severity="error",
                    message=f"Argument {index} of {name} should be {role}, got {arg}.",
                    premise_id=result.get("premise_id"),
                    request_id=result.get("request_id"),
                    evidence={
                        "predicate": name,
                        "arguments": args,
                        "bad_argument": arg,
                        "expected_role": role,
                        "source_phrase": atom.get("source_phrase"),
                    },
                    suggested_stage="Stage 4",
                ))

        if name.endswith("_subject") and len(args) >= 2 and args[0] == args[1]:
            issues.append(ValidationIssue(
                code=C.ARG_SAME_PERSON_AND_OBJECT,
                severity="error",
                message=f"{name} has the same person and object argument: {args}.",
                premise_id=result.get("premise_id"),
                request_id=result.get("request_id"),
                evidence={
                    "predicate": name,
                    "arguments": args,
                    "source_phrase": atom.get("source_phrase"),
                },
                suggested_stage="Stage 4",
            ))

    return issues
```

---

# 7. Semantic Validator

Create `semantic_validator.py`.

Purpose:

```text
Catch meaning-changing parses.
```

This is where you catch:

```text
OR becoming AND
material becoming subject
lost domain restrictions
not necessarily becoming classical NOT
```

```python
from .validation_models import ValidationIssue
from . import issue_codes as C


def _find_result(parsed_record: dict, request_id: str) -> dict | None:
    for result in parsed_record.get("atomization_results", []):
        if result.get("request_id") == request_id:
            return result
    return None


def validate_semantics(parsed_record: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    issues.extend(validate_or_not_became_and(parsed_record))
    issues.extend(validate_object_not_changed(parsed_record))
    issues.extend(validate_domain_restriction_not_lost(parsed_record))
    issues.extend(validate_modal_not_necessarily(parsed_record))
    return issues


def validate_or_not_became_and(parsed_record: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for request in parsed_record.get("atomization_requests", []):
        phrase = str(request.get("phrase", "")).lower()
        if " or " not in phrase:
            continue

        result = _find_result(parsed_record, request.get("request_id"))
        if not result:
            continue

        atoms = result.get("atoms", [])
        has_or_structure = result.get("operator") == "or" or result.get("logical_operator") == "or"
        has_group_predicate = any(
            "_group" in atom.get("name", "") or "a_or_a_plus" in [str(a) for a in atom.get("arguments", [])]
            for atom in atoms
        )

        if len(atoms) >= 2 and not has_or_structure and not has_group_predicate:
            issues.append(ValidationIssue(
                code=C.SEMANTIC_OR_BECAME_AND,
                severity="error",
                message="Phrase contains OR but atomization produced multiple atoms without OR/group structure.",
                premise_id=request.get("premise_id"),
                request_id=request.get("request_id"),
                evidence={
                    "phrase": request.get("phrase"),
                    "atoms": atoms,
                },
                suggested_stage="Stage 5",
            ))

    return issues


def validate_object_not_changed(parsed_record: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    risky_mappings = [
        ("material", "subject"),
        ("tutorials", "subject"),
        ("exam", "subject"),
    ]

    for result in parsed_record.get("atomization_results", []):
        phrase = str(result.get("phrase", "")).lower()
        for atom in result.get("atoms", []):
            args = [str(arg).lower() for arg in atom.get("arguments", [])]
            for source_word, wrong_arg in risky_mappings:
                if source_word in phrase and wrong_arg in args:
                    issues.append(ValidationIssue(
                        code=C.SEMANTIC_OBJECT_CHANGED,
                        severity="warning",
                        message=f"Source mentions {source_word}, but atom uses {wrong_arg}.",
                        premise_id=result.get("premise_id"),
                        request_id=result.get("request_id"),
                        evidence={
                            "phrase": result.get("phrase"),
                            "atom": atom,
                        },
                        suggested_stage="Stage 4",
                    ))

    return issues


def validate_domain_restriction_not_lost(parsed_record: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for request in parsed_record.get("atomization_requests", []):
        phrase = str(request.get("phrase", "")).lower()
        role = request.get("role")

        if "student" not in phrase:
            continue

        result = _find_result(parsed_record, request.get("request_id"))
        if not result:
            continue

        atoms = result.get("atoms", [])
        has_student_atom = any(atom.get("name") == "student" for atom in atoms)

        # If your AST builder adds domain restrictions separately, set this flag in the request/result.
        domain_added_elsewhere = bool(result.get("domain_restriction_added_elsewhere"))

        if role in {"antecedent", "body", "meta_leaf"} and not has_student_atom and not domain_added_elsewhere:
            issues.append(ValidationIssue(
                code=C.SEMANTIC_DOMAIN_RESTRICTION_LOST,
                severity="warning",
                message="Phrase mentions student, but student(x) restriction is not present or marked as added elsewhere.",
                premise_id=request.get("premise_id"),
                request_id=request.get("request_id"),
                evidence={
                    "phrase": request.get("phrase"),
                    "atoms": atoms,
                },
                suggested_stage="Stage 3 or Stage 5",
            ))

    return issues


def validate_modal_not_necessarily(parsed_record: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for result in parsed_record.get("atomization_results", []):
        phrase = str(result.get("phrase", "")).lower()
        if "not necessarily" not in phrase:
            continue

        for atom in result.get("atoms", []):
            if atom.get("negated") is True:
                issues.append(ValidationIssue(
                    code=C.MODAL_NOT_NECESSARILY_AS_NEGATION,
                    severity="error",
                    message="'not necessarily' is modal uncertainty, not classical negation.",
                    premise_id=result.get("premise_id"),
                    request_id=result.get("request_id"),
                    evidence={"phrase": phrase, "atom": atom},
                    suggested_stage="Stage 1 or Stage 3",
                ))

    return issues
```

---

# 8. AST Validator

Create `ast_validator.py`.

Purpose:

```text
Check Stage 5 LogicNode AST shape.
```

Expected AST node types:

```text
ATOM
NOT
AND
OR
IMPLIES
IFF
FORALL
EXISTS
META
```

```python
from .validation_models import ValidationIssue
from . import issue_codes as C


def validate_asts(parsed_record: dict, registry: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    asts = parsed_record.get("asts") or parsed_record.get("ast")

    if not asts:
        return [ValidationIssue(
            code=C.AST_MISSING,
            severity="warning",
            message="No Stage 5 AST found. Skipping AST validation.",
            suggested_stage="Stage 5",
        )]

    if isinstance(asts, dict):
        asts = [asts]

    for index, ast in enumerate(asts):
        issues.extend(validate_ast_node(ast, registry, path=[index]))

    return issues


def validate_ast_node(node: dict, registry: dict, path: list[int]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    node_type = node.get("type")
    children = node.get("children", []) or []

    if node_type == "ATOM":
        predicate = node.get("predicate") or node.get("name")
        if not predicate:
            issues.append(ValidationIssue(
                code=C.AST_ATOM_MISSING_PREDICATE,
                severity="error",
                message="ATOM node missing predicate.",
                path=path,
                evidence={"node": node},
                suggested_stage="Stage 5",
            ))
        elif predicate not in registry:
            issues.append(ValidationIssue(
                code=C.AST_ATOM_UNKNOWN_PREDICATE,
                severity="error",
                message=f"AST contains unknown predicate: {predicate}.",
                path=path,
                evidence={"node": node},
                suggested_stage="Stage 5",
            ))

    elif node_type == "NOT":
        if len(children) != 1:
            issues.append(_bad_child_count(node_type, path, 1, len(children)))

    elif node_type in {"AND", "OR"}:
        if len(children) < 2:
            issues.append(_bad_child_count(node_type, path, "at least 2", len(children)))

    elif node_type in {"IMPLIES", "IFF"}:
        if len(children) != 2:
            issues.append(_bad_child_count(node_type, path, 2, len(children)))

    elif node_type in {"FORALL", "EXISTS"}:
        if not node.get("variable"):
            issues.append(ValidationIssue(
                code=C.AST_BAD_CHILD_COUNT,
                severity="error",
                message=f"{node_type} node missing variable.",
                path=path,
                evidence={"node": node},
                suggested_stage="Stage 5",
            ))
        if len(children) != 1:
            issues.append(_bad_child_count(node_type, path, 1, len(children)))

    elif node_type == "META":
        # META is valid structurally, but not direct solver-ready.
        pass

    else:
        issues.append(ValidationIssue(
            code=C.AST_UNKNOWN_NODE_TYPE,
            severity="error",
            message=f"Unknown AST node type: {node_type}.",
            path=path,
            evidence={"node": node},
            suggested_stage="Stage 5",
        ))

    for index, child in enumerate(children):
        issues.extend(validate_ast_node(child, registry, path + [index]))

    return issues


def _bad_child_count(node_type: str, path: list[int], expected, actual: int) -> ValidationIssue:
    return ValidationIssue(
        code=C.AST_BAD_CHILD_COUNT,
        severity="error",
        message=f"{node_type} expects {expected} children, got {actual}.",
        path=path,
        evidence={"node_type": node_type, "actual": actual, "expected": expected},
        suggested_stage="Stage 5",
    )
```

---

# 9. Solver Readiness Classifier

Create `solver_readiness.py`.

Purpose:

```text
Turn validation issues + solver capability config into final readiness flags.
```

Example solver capability config:

```python
DEFAULT_SOLVER_CAPABILITIES = {
    "supports_or": False,
    "supports_exists": False,
    "supports_iff": False,
    "supports_meta": False,
    "supports_cardinality": False,
    "supports_classical_negation": True,
}
```

Implementation:

```python
from .validation_models import ValidationIssue
from . import issue_codes as C


def classify_solver_readiness(
    parsed_record: dict,
    issues: list[ValidationIssue],
    solver_capabilities: dict,
) -> dict:
    reasons: list[str] = []

    fatal_or_error = [issue for issue in issues if issue.severity in {"fatal", "error"}]

    has_meta = any(
        skeleton.get("kind") == "META"
        for skeleton in parsed_record.get("skeletons", [])
    )

    has_or = ast_contains(parsed_record, "OR")
    has_exists = ast_contains(parsed_record, "EXISTS") or any(
        skeleton.get("kind") == "EXISTS"
        for skeleton in parsed_record.get("skeletons", [])
    )
    has_iff = ast_contains(parsed_record, "IFF") or any(
        skeleton.get("kind") == "IFF"
        for skeleton in parsed_record.get("skeletons", [])
    )
    has_cardinality = any_atom_name(parsed_record, {"grade_count", "grade_count_at_least"})

    unsupported = False
    needs_meta_resolution = False
    needs_lowering = False
    needs_review = False

    if fatal_or_error:
        needs_review = True
        reasons.extend(issue.code for issue in fatal_or_error)

    if has_meta:
        needs_meta_resolution = True
        reasons.append("meta_requires_resolution")

    if has_or and not solver_capabilities.get("supports_or", False):
        needs_lowering = True
        reasons.append("or_requires_lowering")

    if has_exists and not solver_capabilities.get("supports_exists", False):
        needs_lowering = True
        reasons.append("exists_requires_skolemization")

    if has_iff and not solver_capabilities.get("supports_iff", False):
        needs_lowering = True
        reasons.append("iff_requires_lowering")

    if has_cardinality and not solver_capabilities.get("supports_cardinality", False):
        needs_lowering = True
        reasons.append("cardinality_requires_lowering")

    parse_valid = not any(issue.severity in {"fatal", "error"} for issue in issues)

    direct_solver_ready = (
        parse_valid
        and not needs_review
        and not needs_meta_resolution
        and not needs_lowering
        and not unsupported
    )

    return {
        "parse_valid": parse_valid,
        "direct_solver_ready": direct_solver_ready,
        "needs_lowering": needs_lowering,
        "needs_meta_resolution": needs_meta_resolution,
        "needs_review": needs_review,
        "unsupported": unsupported,
        "reasons": sorted(set(reasons)),
    }


def ast_contains(parsed_record: dict, node_type: str) -> bool:
    asts = parsed_record.get("asts") or parsed_record.get("ast")
    if not asts:
        return False
    if isinstance(asts, dict):
        asts = [asts]

    def walk(node: dict) -> bool:
        if node.get("type") == node_type:
            return True
        return any(walk(child) for child in node.get("children", []) or [])

    return any(walk(ast) for ast in asts)


def any_atom_name(parsed_record: dict, names: set[str]) -> bool:
    for result in parsed_record.get("atomization_results", []):
        for atom in result.get("atoms", []):
            if atom.get("name") in names:
                return True
    return False
```

---

# 10. Main Validator Orchestrator

Create `validator.py`.

```python
from .validation_models import ValidationReport
from .structural_validator import validate_skeletons
from .predicate_validator import validate_predicates
from .argument_validator import validate_argument_roles
from .semantic_validator import validate_semantics
from .ast_validator import validate_asts
from .solver_readiness import classify_solver_readiness
from .registry_schema import DEFAULT_PREDICATE_REGISTRY

DEFAULT_SOLVER_CAPABILITIES = {
    "supports_or": False,
    "supports_exists": False,
    "supports_iff": False,
    "supports_meta": False,
    "supports_cardinality": False,
    "supports_classical_negation": True,
}


class Stage6Validator:
    def __init__(
        self,
        predicate_registry: dict | None = None,
        solver_capabilities: dict | None = None,
    ):
        self.predicate_registry = predicate_registry or DEFAULT_PREDICATE_REGISTRY
        self.solver_capabilities = solver_capabilities or DEFAULT_SOLVER_CAPABILITIES

    def validate(self, parsed_record: dict) -> ValidationReport:
        issues = []

        issues.extend(validate_skeletons(parsed_record))
        issues.extend(validate_predicates(parsed_record, self.predicate_registry))
        issues.extend(validate_argument_roles(parsed_record, self.predicate_registry))
        issues.extend(validate_semantics(parsed_record))
        issues.extend(validate_asts(parsed_record, self.predicate_registry))

        readiness = classify_solver_readiness(
            parsed_record=parsed_record,
            issues=issues,
            solver_capabilities=self.solver_capabilities,
        )

        summary = {
            "issue_count": len(issues),
            "error_count": sum(1 for issue in issues if issue.severity == "error"),
            "warning_count": sum(1 for issue in issues if issue.severity == "warning"),
            "info_count": sum(1 for issue in issues if issue.severity == "info"),
            "issue_codes": sorted(set(issue.code for issue in issues)),
        }

        return ValidationReport(
            parse_valid=readiness["parse_valid"],
            direct_solver_ready=readiness["direct_solver_ready"],
            needs_lowering=readiness["needs_lowering"],
            needs_meta_resolution=readiness["needs_meta_resolution"],
            needs_review=readiness["needs_review"],
            unsupported=readiness["unsupported"],
            issues=issues,
            readiness_reasons=readiness["reasons"],
            summary=summary,
        )
```

---

# 11. Integration Into Pipeline

After Stage 5:

```python
from NEW_logic_pipeline.Stage_6.validator import Stage6Validator

validator = Stage6Validator(
    predicate_registry=predicate_registry,
    solver_capabilities={
        "supports_or": False,
        "supports_exists": False,
        "supports_iff": False,
        "supports_meta": False,
        "supports_cardinality": False,
        "supports_classical_negation": True,
    },
)

report = validator.validate(parsed_record)
parsed_record["validation"] = report.to_dict()
parsed_record["readiness"] = {
    "parse_valid": report.parse_valid,
    "direct_solver_ready": report.direct_solver_ready,
    "needs_lowering": report.needs_lowering,
    "needs_meta_resolution": report.needs_meta_resolution,
    "needs_review": report.needs_review,
    "unsupported": report.unsupported,
    "reasons": report.readiness_reasons,
}
```

Then route:

```python
if report.unsupported:
    route = "unsupported"
elif report.needs_review:
    route = "needs_review"
elif report.needs_meta_resolution:
    route = "stage_8_meta_resolution"
elif report.needs_lowering:
    route = "stage_8_lowering"
elif report.direct_solver_ready:
    route = "solver"
else:
    route = "needs_review"
```

---

# 12. Important Validation Rules For Your Current Pipeline

## Rule A: Formula-like leaf must fail

Bad:

```text
leaf: "if a student is not asking questions, they are not attending tutorials"
```

Issue:

```text
STRUCT_FORMULA_LIKE_LEAF
```

Correct:

```text
forall y:
  implies(
    NOT ask_questions(y),
    NOT attending_tutorials(y)
  )
```

---

## Rule B: Unknown predicate must fail or be registered

Bad if not registered:

```text
preparing_for_exam(x)
```

Issue:

```text
PRED_UNKNOWN
```

Fix options:

```text
1. Add preparing_for_exam to registry.
2. Canonicalize it to an existing predicate.
3. Mark as needs_review if unsupported.
```

---

## Rule C: OR must not become AND

Bad:

```text
"A or A+"
→ earned_grade(x, a)
→ earned_grade(x, a_plus)
```

If Stage 5 combines these as AND, that changes meaning.

Correct:

```text
OR(earned_grade(x, a), earned_grade(x, a_plus))
```

Horn-safe alternative:

```text
earned_grade_group(x, a_or_a_plus)
```

---

## Rule D: Object should not be changed

Bad:

```text
"understanding the material"
→ understand_subject(x, subject)
```

Correct:

```text
understand_material(x)
```

or:

```text
understand(x, material)
```

---

## Rule E: Argument roles must be sane

Bad:

```text
explain_subject(x, x)
```

Correct:

```text
explain_subject(x, subject)
```

Issue:

```text
ARG_SAME_PERSON_AND_OBJECT
```

---

## Rule F: Named entities should remain constants

Bad:

```text
Tuấn has earned three A grades.
→ grade_count(x, a, 3)
```

Correct:

```text
grade_count(tuan, a, 3)
```

---

## Rule G: Numbers should be typed strings or numeric objects consistently

Recommended simple format:

```json
{"name": "grade_count", "arguments": ["tuan", "a", "3"]}
```

Do not randomly mix:

```json
["tuan", "a", 3]
```

unless your schema officially supports integer arguments.

---

# 13. Minimum Tests

Create tests before expanding Stage 7–9.

## Test 1: Formula-like leaf rejected

```python
def test_formula_like_leaf_rejected():
    parsed = {
        "skeletons": [{
            "premise_id": "P1",
            "kind": "META",
            "formula_tree": {
                "type": "implies",
                "children": [
                    {"type": "leaf", "text": "there exists at least one student", "children": []},
                    {"type": "leaf", "text": "if a student asks questions then they attend tutorials", "children": []},
                ],
            },
        }],
        "atomization_results": [],
    }
    report = Stage6Validator().validate(parsed)
    assert not report.parse_valid
    assert "STRUCT_FORMULA_LIKE_LEAF" in report.summary["issue_codes"]
```

## Test 2: Unknown predicate rejected

```python
def test_unknown_predicate_rejected():
    parsed = {
        "skeletons": [],
        "atomization_results": [{
            "premise_id": "P1",
            "request_id": "P1_body",
            "phrase": "is happy",
            "atoms": [{"name": "happy", "arguments": ["x"], "negated": False}],
        }],
    }
    report = Stage6Validator().validate(parsed)
    assert not report.parse_valid
    assert "PRED_UNKNOWN" in report.summary["issue_codes"]
```

## Test 3: OR became AND rejected

```python
def test_or_became_and_rejected():
    parsed = {
        "skeletons": [],
        "atomization_requests": [{
            "premise_id": "P1",
            "request_id": "P1_consequent",
            "phrase": "they can earn an A or A+",
            "role": "consequent",
        }],
        "atomization_results": [{
            "premise_id": "P1",
            "request_id": "P1_consequent",
            "phrase": "they can earn an A or A+",
            "atoms": [
                {"name": "earned_grade", "arguments": ["x", "a"], "negated": False},
                {"name": "earned_grade", "arguments": ["x", "a_plus"], "negated": False},
            ],
        }],
    }
    report = Stage6Validator().validate(parsed)
    assert not report.parse_valid
    assert "SEMANTIC_OR_BECAME_AND" in report.summary["issue_codes"]
```

## Test 4: Bad argument role rejected

```python
def test_bad_argument_role_rejected():
    parsed = {
        "skeletons": [],
        "atomization_results": [{
            "premise_id": "P1",
            "request_id": "P1_antecedent",
            "phrase": "a student cannot explain a subject",
            "atoms": [{"name": "explain_subject", "arguments": ["x", "x"], "negated": True}],
        }],
    }
    report = Stage6Validator().validate(parsed)
    assert not report.parse_valid
    assert "ARG_SAME_PERSON_AND_OBJECT" in report.summary["issue_codes"]
```

## Test 5: Cardinality needs lowering

```python
def test_cardinality_needs_lowering():
    parsed = {
        "skeletons": [],
        "atomization_results": [{
            "premise_id": "P1",
            "request_id": "P1_body",
            "phrase": "Tuấn has earned three A grades",
            "atoms": [{"name": "grade_count", "arguments": ["tuan", "a", "3"], "negated": False}],
        }],
    }
    report = Stage6Validator().validate(parsed)
    assert report.parse_valid
    assert not report.direct_solver_ready
    assert report.needs_lowering
```

---

# 14. Development Order

Implement Stage 6 in this order:

```text
1. validation_models.py
2. issue_codes.py
3. registry_schema.py
4. predicate_validator.py
5. argument_validator.py
6. structural_validator.py
7. semantic_validator.py
8. ast_validator.py
9. solver_readiness.py
10. validator.py
11. tests
12. pipeline integration
```

This order lets you test simple atom-level issues before adding formula and AST complexity.

---

# 15. Practical Readiness Policy

Use this strict policy:

```text
fatal/error issue  → parse_valid = false, needs_review = true
warning only       → parse_valid = true, but may still needs_review depending on policy
META present       → needs_meta_resolution = true
OR unsupported     → needs_lowering = true
EXISTS unsupported → needs_lowering = true
IFF unsupported    → needs_lowering = true
cardinality        → needs_lowering = true
no issues + solver-compatible AST → direct_solver_ready = true
```

Recommended warning policy:

```text
Warnings do not automatically make parse_valid false.
But warnings that indicate meaning change should become error.
```

Examples:

```text
SEMANTIC_OBJECT_CHANGED = warning or error depending on strictness
SEMANTIC_OR_BECAME_AND = error
ARG_SAME_PERSON_AND_OBJECT = error
PRED_UNKNOWN = error
STRUCT_FORMULA_LIKE_LEAF = error
```

---

# 16. Final Stage 6 Checklist

Before Stage 6 is considered complete, it should catch all of these:

```text
[ ] META without formula_tree
[ ] formula-like leaf sent to atomizer
[ ] unknown predicate
[ ] predicate arity mismatch
[ ] bad argument role
[ ] explain_subject(x, x)
[ ] negation inside predicate name
[ ] not necessarily treated as NOT
[ ] OR phrase converted into AND atoms
[ ] source object changed, e.g. material → subject
[ ] student domain restriction lost
[ ] missing AST
[ ] invalid AST child count
[ ] OR requires lowering for Horn solver
[ ] EXISTS requires Skolemization if solver does not support exists
[ ] IFF requires lowering if solver does not support iff
[ ] cardinality requires lowering if solver does not support counts
[ ] final readiness route is correct
```

---

# 17. What Stage 6 Should Not Do

Stage 6 should not:

```text
✗ create new predicates
✗ repair wrong atoms
✗ call the LLM to judge validity
✗ solve the question
✗ perform META resolution
✗ lower formulas
✗ silently change AST
```

Those belong to later stages:

```text
Stage 7 = repair/fallback
Stage 8 = META resolution and lowering
Stage 9 = question parser
Solver = reasoning
```

---

# 18. Summary

Stage 6 is your parser’s safety wall.

Your Stage 0–5 can be imperfect, because Stage 6 will identify exactly where the parse is unsafe.

The best implementation is:

```text
Deterministic validators
+ predicate registry
+ argument-role schema
+ semantic sanity checks
+ AST validation
+ solver capability classifier
```

The main output should not just be `true` or `false`. It should tell you exactly why a record is or is not solver-ready.

