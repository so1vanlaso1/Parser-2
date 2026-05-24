# Summary Guide: Non-Hardcoded Logic Premise Parser Pipeline

This guide summarizes the new implementation direction for the logic parser project. Use it to guide an agent/coding assistant to refactor the current parser into a more general, non-hardcoded, solver-safe pipeline.

---

## 1. Main Design Principle

Do **not** build a parser that recognizes exact premise sentences.

Build a **compiler-style pipeline**:

```text
Raw English premise
→ logic skeleton
→ phrase/leaf atomization
→ predicate canonicalization
→ deterministic AST building
→ validation
→ META resolution / lowering
→ solver export
```

The key division:

| Responsibility | Owner |
|---|---|
| Logical structure | Python / deterministic router |
| English phrase meaning | LLM atomizer |
| Predicate consistency | Predicate registry / canonicalizer |
| AST construction | Python |
| Validation | Python |
| Solver reasoning | Solver |
| Final answer selection | Solver / evaluator, not parser |

The LLM should understand language, but it should **not control logic**.

---

## 2. Current Problem to Fix

The old/current pipeline is too eager to produce `direct_cir` from deterministic parsing.

That creates bad behavior such as:

```text
"At least one student has completed a course."
→ wrongly parsed as FACT
```

or:

```text
"Students who fail to maintain GPA lose housing."
→ wrongly parsed as a generic sentence fact
```

The fix:

```text
Stage 1 should NOT produce final solver logic by default.
Stage 1 should produce a logic skeleton only.
```

Final predicates and AST should be produced later.

---

## 3. New Pipeline Architecture

Recommended pipeline:

```text
Stage 0: Input normalization
  - Load premises/question/choices
  - Preserve original text
  - Normalize whitespace only
  - Do not change logic

Stage 1: Logic skeleton router
  - Classify logical form:
    FACT / EXISTS / FORALL / RULE / ONLY_IF_RULE / IFF /
    NON_IF_RULE / OBLIGATION_RULE / MODAL / META / UNKNOWN
  - Split premise into text spans:
    body / antecedent / consequent / left / right
  - For META, build recursive formula skeleton
  - Do not create final predicate atoms yet

Stage 2: Leaf phrase extraction
  - Convert skeleton spans into AtomizationRequest objects
  - One request per logical leaf

Stage 3: LLM atomizer
  - Convert small English phrases into predicate atoms
  - No quantifiers
  - No implications
  - No solving
  - No full AST creation

Stage 4: Predicate canonicalizer / registry
  - Reuse known predicates
  - Merge synonyms
  - Keep negation outside predicate names

Stage 5: AST builder
  - Python builds LogicNode AST from skeleton + atoms
  - Deterministic construction of forall/exists/implies/iff/not/and/or

Stage 6: Validator
  - Structural validation
  - Semantic sanity validation
  - Solver-readiness classification

Stage 7: Repair / fallback
  - Repair only structure if needed
  - Do not invent facts
  - Unsafe premise becomes needs_review / unsupported

Stage 8: META resolution and lowering
  - META is never sent directly to solver
  - Resolve META by matching antecedent formula against existing formulas
  - Materialize consequent only if safe

Stage 9: Question parser
  - Parse question/choices using known predicate registry
  - Reuse premise predicates
```

---

## 4. Skeleton Kinds to Pre-create

Pre-create **logic-form skeletons**, not domain-specific skeletons.

Use these main kinds:

```text
FACT
EXISTS
FORALL
RULE
ONLY_IF_RULE
IFF
NON_IF_RULE
OBLIGATION_RULE
MODAL
META
UNKNOWN
```

Do **not** create domain-specific skeletons like:

```text
student_housing_rule
scholarship_rule
quantum_lab_rule
gpa_rule
```

Those overfit the dataset.

---

## 5. Core Data Models

### 5.1 TextSpan

```python
class TextSpan(BaseModel):
    role: str
    text: str
    variable: str = "x"
    negation_hint: bool = False
    modality_hint: str | None = None
    source: str | None = None
```

### 5.2 LogicSkeleton

```python
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

class LogicSkeleton(BaseModel):
    premise_id: str
    original: str
    kind: SkeletonKind

    body: TextSpan | None = None
    antecedent: TextSpan | None = None
    consequent: TextSpan | None = None

    left: TextSpan | None = None
    right: TextSpan | None = None

    formula_tree: dict[str, Any] | None = None

    quantifier: Literal["forall", "exists"] | None = None
    variable: str = "x"

    risk_flags: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    needs_review: bool = False
```

### 5.3 FormulaSkeleton

Use this for META and nested formulas.

```python
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

class FormulaSkeleton(BaseModel):
    type: FormulaNodeType
    text: str | None = None
    variable: str | None = None
    children: list["FormulaSkeleton"] = Field(default_factory=list)
```

### 5.4 AtomizationRequest

```python
class AtomizationRequest(BaseModel):
    premise_id: str
    role: str
    phrase: str
    variable: str
    known_predicates: list[str] = Field(default_factory=list)
    negation_policy: Literal[
        "preserve",
        "force_positive",
        "force_negative"
    ] = "preserve"
```

### 5.5 AtomizationResult

```python
class AtomizationResult(BaseModel):
    premise_id: str
    role: str
    atoms: list[PredicateAtom]
    unsupported_reason: str | None = None
```

### 5.6 SolverReadiness

```python
class SolverReadiness(BaseModel):
    parse_valid: bool = False
    direct_solver_ready: bool = False
    needs_lowering: bool = False
    needs_meta_resolution: bool = False
    needs_review: bool = False
    unsupported: bool = False
    reason: list[str] = Field(default_factory=list)
```

---

## 6. Skeleton Examples

### FACT

Input:

```text
John is certified.
```

Skeleton:

```text
kind = FACT
body = "John is certified"
```

Later atomization:

```text
certified(john)
```

---

### EXISTS

Input:

```text
At least one student has completed a course.
```

Skeleton:

```text
kind = EXISTS
quantifier = exists
body = "a student has completed a course"
```

Later AST:

```text
exists x: student(x) AND completed_course(x)
```

---

### FORALL

Input:

```text
All students receive training.
```

Skeleton:

```text
kind = FORALL
quantifier = forall
antecedent = "a student"
consequent = "receives training"
```

Later AST:

```text
forall x: student(x) -> receives_training(x)
```

---

### RULE

Input:

```text
If a student did not submit the final report, then they did not receive course recognition.
```

Skeleton:

```text
kind = RULE
antecedent = "a student did not submit the final report"
consequent = "the student did not receive course recognition"
```

Later AST:

```text
forall x:
  student(x) AND NOT submit_final_report(x)
  -> NOT receive_course_recognition(x)
```

---

### ONLY_IF_RULE

Input:

```text
A student graduates only if the student passes the final exam.
```

Skeleton:

```text
kind = ONLY_IF_RULE
antecedent = "a student graduates"
consequent = "the student passes the final exam"
risk_flags = ["only_if_direction"]
```

Rule:

```text
A only if B = A -> B
```

---

### IFF

Input:

```text
A student is eligible if and only if the student passes the exam.
```

Skeleton:

```text
kind = IFF
left = "a student is eligible"
right = "the student passes the exam"
```

Later AST:

```text
forall x: eligible(x) <-> passes_exam(x)
```

---

### NON_IF_RULE

Input:

```text
Passing Philosophy grants eligibility for the Quantum Physics lab.
```

Skeleton:

```text
kind = NON_IF_RULE
antecedent = "passing Philosophy"
consequent = "eligibility for the Quantum Physics lab"
risk_flags = ["non_if_rule"]
```

Later AST:

```text
forall x: passes_philosophy(x) -> eligible_for_quantum_physics_lab(x)
```

---

## 7. META and Nested Parsing

META means the premise talks about a formula, rule, implication, claim, statement, or nested logical condition.

Examples:

```text
If passing the exam implies graduation, then students who pass are eligible.
If there exists a student who attends tutorials, then if a student does not ask questions, the student does not attend tutorials.
It is not true that if a model is accurate, it is reliable.
```

META must be detected **before RULE**.

Router priority:

```text
META / nested
IFF
ONLY_IF_RULE
EXISTS
FORALL
RULE
NON_IF_RULE
OBLIGATION_RULE / MODAL
FACT
UNKNOWN
```

### META Example

Input:

```text
If passing the exam implies graduation, then students who pass are eligible.
```

Correct formula skeleton:

```text
implies(
  forall x:
    passes_exam(x) -> graduates(x),
  forall y:
    passes_exam(y) -> eligible(y)
)
```

But Stage 1 should first preserve it as leaves:

```python
{
  "kind": "META",
  "formula_tree": {
    "type": "implies",
    "children": [
      {
        "type": "forall",
        "variable": "x",
        "children": [
          {
            "type": "implies",
            "children": [
              {"type": "leaf", "text": "a student passes the exam", "variable": "x"},
              {"type": "leaf", "text": "the student graduates", "variable": "x"}
            ]
          }
        ]
      },
      {
        "type": "forall",
        "variable": "y",
        "children": [
          {
            "type": "implies",
            "children": [
              {"type": "leaf", "text": "a student passes the exam", "variable": "y"},
              {"type": "leaf", "text": "the student is eligible", "variable": "y"}
            ]
          }
        ]
      }
    ]
  }
}
```

---

## 8. Is META Solver Ready?

No.

META is **parse-ready**, but it is not directly solver-ready.

Use this rule:

```text
META.solver_ready = False always.
META.needs_meta_resolution = True unless unsupported.
META.add_to_solver = True only if META resolution materializes a safe consequent.
```

Example:

```text
P1: If a student passes the exam, then the student graduates.

P2: If passing the exam implies graduation,
    then students who pass the exam are eligible.
```

P2 is META.

Do not send this to the solver:

```text
(passes_exam -> graduates) -> eligible
```

Instead:

1. Check whether P2 antecedent matches P1.
2. If yes, materialize P2 consequent:

```text
If a student passes the exam, then the student is eligible.
```

3. Send only the materialized consequent to the solver.

---

## 9. META Resolution

You need META resolution if the solver is Horn/FOL-style and cannot reason over higher-order formulas directly.

Recommended module:

```text
meta_resolution.py
```

Core functions:

```python
def build_formula_index(non_meta_premises):
    """
    Store canonical signatures of normal formulas.
    Example:
    P1 = forall x: A(x) -> B(x)
    """

def resolve_meta_premise(meta_premise, formula_index):
    """
    If meta antecedent matches an existing formula,
    materialize the consequent.
    """

def materialize_consequent(meta_premise):
    """
    Export only the consequent formula, not the full META formula.
    """
```

Output shape:

```json
{
  "kind": "META",
  "solver_ready": false,
  "meta_resolved": true,
  "solver_ready_after_meta_resolution": true,
  "add_to_solver": true,
  "resolution": "materialized_consequent",
  "solver_export": [
    "materialized consequent AST only"
  ]
}
```

---

## 10. LLM Role

The LLM role should be small and controlled.

The LLM should **only** do:

```text
small English phrase → predicate atoms
```

Example input:

```json
{
  "phrase": "a student does not maintain GPA",
  "variable": "x",
  "known_predicates": ["student", "maintain_gpa", "has_housing"]
}
```

Expected output:

```json
{
  "atoms": [
    {"name": "student", "arguments": ["x"], "negated": false},
    {"name": "maintain_gpa", "arguments": ["x"], "negated": true}
  ]
}
```

The LLM can also help with:

```text
- predicate reuse
- ambiguous phrase detection
- META leaf atomization
- question phrase atomization
```

The LLM must **not**:

```text
- solve the problem
- infer new facts
- decide the answer
- build the full AST when Python can build it
- decide solver_ready
- flatten META into normal RULE
- put negation inside predicate names
- invent a new predicate when a known predicate fits
```

Good design:

```text
Python:
  controls logic structure

LLM:
  converts phrases to atoms

Python:
  canonicalizes predicates
  builds AST
  validates
  resolves META
  exports to solver
```

---

## 11. Predicate Registry / Canonicalizer

Build a dynamic predicate registry per problem.

Purpose:

```text
Reuse the same predicate for the same meaning.
Avoid synonyms breaking solver matching.
```

Example:

```text
"receive course recognition"
"gets course recognition"
"is granted course recognition"
→ receive_course_recognition(x)
```

Registry concept:

```python
class PredicateRegistry:
    canonical_name: str
    aliases: list[str]
    arity: int
    examples: list[str]
```

Canonicalization steps:

```text
1. Normalize text
2. Compare with existing predicate aliases
3. Use lexical/embedding similarity if available
4. Ask LLM equivalence check only if uncertain
5. Reuse existing predicate when semantically same
```

Do not use only a tiny hardcoded dictionary.

---

## 12. Validation Requirements

Validation must be both structural and semantic.

### Structural checks

```text
- atomic node has name and arguments
- not has exactly 1 child
- implies has exactly 2 children
- forall/exists has variable and exactly 1 child
- and/or/iff have at least 2 children
- no unbound variables
- no negation inside predicate name
- no sentence-like long predicate names
```

### Semantic sanity checks

Add these checks:

```text
If original has "some / at least one / there exists / there is"
but kind is FACT → reject.

If original has "only if"
but skeleton is not ONLY_IF_RULE → reject.

If original has "if"
but no antecedent/consequent split → reject.

If original has "not necessarily"
but parse uses classical NOT → reject.

If original has "who / that / with / without"
and became one atomic FACT → needs_review.

If predicate name has more than 5 words → needs_review.

If FACT uses generic variable x/y → reject.

If RULE has nested implication → reclassify as META.

If premise has numeric threshold
but number appears inside predicate name → reject.
```

---

## 13. Coverage Policy

The system does not need to make every premise solver-ready.

It needs to safely classify every premise.

Possible statuses:

```text
solver_ready
needs_lowering
needs_meta_resolution
needs_review
unsupported
```

Examples:

| Premise type | Status |
|---|---|
| Simple fact | solver_ready |
| Normal rule | solver_ready |
| Exists | solver_ready after skolemization/lowering |
| IFF | needs_lowering |
| OR/disjunction | needs_lowering |
| Numeric comparison | needs_lowering |
| META | needs_meta_resolution |
| Modal uncertainty | needs_review |
| Deontic obligation | needs_review or special deontic solver |
| Ambiguous premise | needs_review |
| Unknown structure | unsupported |

Important principle:

```text
Every premise should be parse-classified.
Only safe premises should become solver_ready.
```

---

## 14. Additional Formula Node Types

Support these formula node types:

```text
leaf
and
or
not
implies
iff
forall
exists
equation
comparison
cardinality
```

This improves coverage for:

```text
- not all
- no / none
- disjunction
- numeric thresholds
- at least N / exactly N
- equations/comparisons
```

---

## 15. Implementation Order

Recommended order for the coding agent:

### Step 1: Add new schemas

Create:

```text
logic_skeleton.py
```

Add:

```text
TextSpan
LogicSkeleton
FormulaSkeleton
AtomizationRequest
AtomizationResult
SolverReadiness
```

---

### Step 2: Build operator router

Create:

```text
operator_router.py
```

Implement generic operator detectors:

```text
MetaOperator
IffOperator
OnlyIfOperator
ExistsOperator
ForallOperator
IfThenOperator
RelativeClauseRuleOperator
NonIfRuleOperator
ObligationOperator
ModalOperator
FactOperator
UnknownOperator
```

Important:

```text
No domain-specific templates.
Only logical operator cues.
```

---

### Step 3: Build skeleton splitter

Create:

```text
skeleton_builder.py
```

Responsibilities:

```text
- Given premise text, select operator
- Produce LogicSkeleton
- Split body/antecedent/consequent/left/right
- For META, produce recursive FormulaSkeleton
```

---

### Step 4: Build atomization request collector

Create:

```text
atomization_requests.py
```

Responsibilities:

```text
- Walk LogicSkeleton / FormulaSkeleton
- Extract all leaf phrases
- Produce AtomizationRequest objects
```

---

### Step 5: Build LLM atomizer

Create:

```text
leaf_atomizer.py
```

Prompt contract:

```text
Input:
phrase, variable, known_predicates

Output:
atoms only
```

No implications, no quantifiers, no solving.

---

### Step 6: Build predicate registry

Create:

```text
predicate_registry.py
```

Responsibilities:

```text
- Store canonical predicates
- Track aliases
- Reuse predicates
- Resolve synonym conflicts
```

---

### Step 7: Build AST builder

Create:

```text
ast_builder.py
```

Responsibilities:

```text
- LogicSkeleton + atoms → LogicNode AST
- FormulaSkeleton + atoms → formula LogicNode
- Python decides forall/exists/implies/iff
```

---

### Step 8: Upgrade validator

Create or update:

```text
validator.py
```

Add:

```text
- structural checks
- semantic sanity checks
- solver-readiness classification
```

---

### Step 9: Build META resolution

Create:

```text
meta_resolution.py
```

Responsibilities:

```text
- Build formula signatures
- Match META antecedent with existing formulas
- Materialize consequent only when safe
- Never mark raw META as solver_ready
```

---

### Step 10: Update full pipeline

Create/update:

```text
pipeline.py
```

Final flow:

```text
load input
→ skeleton builder
→ atomization requests
→ LLM atomizer
→ predicate registry/canonicalizer
→ AST builder
→ validator
→ META resolution
→ solver-readiness classifier
→ question parser
→ FullParseResult
```

---

## 16. Evaluation Metrics

Do not only measure total accuracy.

Track:

```text
EXISTS classification accuracy
FORALL classification accuracy
RULE direction accuracy
ONLY_IF direction accuracy
IFF handling accuracy
negation accuracy
META detection accuracy
META resolution accuracy
predicate canonicalization accuracy
solver_ready false-positive rate
unsupported false-negative rate
question-predicate matching accuracy
```

Most important metric:

```text
solver_ready false-positive rate
```

A wrong premise marked `solver_ready` is worse than a correct premise marked `needs_review`.

---

## 17. Final Rule for the Agent

The agent should follow this rule throughout implementation:

```text
Do not hardcode dataset-specific premise meanings.
Hardcode only logical operators and safe grammar cues.
Use the LLM only for phrase-to-atom conversion.
Python owns logic structure.
META is parsed as formula tree and never directly solver_ready.
Only validated, safe ASTs are exported to the solver.
```
