# Multi-Stage Neurosymbolic Logic Parser: Full Implementation Guide

> Current migration note: the implementation is moving to a CIR-only semantic
> boundary. LLM components may produce structural guide data or CIR JSON, but
> solver-facing `LogicNode` AST is built deterministically in Python.

This guide turns your current pipeline into a complete implementation plan for a **logic parser + solver pipeline** that takes:

```text
Premise-NL + Question-NL
```

and produces:

```text
parsed predicates / AST → validated solver input → answer
```

The main design principle is:

> **LLM = semantic extractor / CNL rewriter / atomizer.  
> Python = structure builder, validator, repair controller, and solver bridge.  
> Solver = the only component that proves or answers.**

Do **not** let the LLM directly solve the problem.

---

## 1. Final Architecture

Your uploaded pipeline has four important ideas:

1. **Stage 1: CNL Rewriter**
2. **Structural RAG**
3. **AST Compiler**
4. **Python Validation + Feedback Loop**

For a logic solver project, implement it like this:

```text
[Raw Premise-NL + Question-NL]
        |
        v
[0. Input Loader + Statement Splitter]
        |
        v
[1. First Parser / CNL Rewriter]
        |
        |  Output:
        |  - clean controlled natural language
        |  - premise kind hints
        |  - risk flags
        |  - no solving
        v
[2. Structural RAG Retriever]
        |
        |  Retrieves similar examples:
        |  CNL -> AST / predicate records
        v
[3. Second Parser / AST Compiler]
        |
        |  Output:
        |  - typed LogicNode AST
        |  - predicate atoms
        |  - quantifier scope
        |  - normalized negation
        v
[4. Python Validator]
        |
        |  Checks:
        |  - schema validity
        |  - variable binding
        |  - negation consistency
        |  - rule direction
        |  - solver compatibility
        v
[5. Repair Loop]
        |
        |  If invalid:
        |  - retry Stage 1 or Stage 3 with exact error message
        |  - never guess silently
        v
[6. Solver Export]
        |
        |  Export to:
        |  - Horn clauses
        |  - Z3
        |  - Prolog/Datalog
        |  - custom forward-chaining engine
        v
[7. Solver Answer]
```

---

## 2. Recommended Repository Structure

Create this structure:

```text
logic_pipeline/
│
├── README.md
├── requirements.txt
├── .env.example
│
├── data/
│   ├── examples.jsonl
│   ├── structural_examples.jsonl
│   └── sample_inputs.jsonl
│
├── src/
│   ├── __init__.py
│   │
│   ├── config.py
│   ├── schemas.py
│   ├── json_utils.py
│   │
│   ├── stage0_input.py
│   ├── stage1_cnl.py
│   ├── stage2_rag.py
│   ├── stage3_ast.py
│   ├── stage4_validate.py
│   ├── stage5_repair.py
│   ├── stage6_solver_export.py
│   │
│   └── pipeline.py
│
├── scripts/
│   ├── run_one.py
│   ├── run_jsonl.py
│   └── build_rag_index.py
│
├── tests/
│   ├── test_schemas.py
│   ├── test_validator.py
│   ├── test_pipeline_smoke.py
│   └── test_solver_export.py
│
└── artifacts/
    ├── predictions.jsonl
    ├── validation_report.jsonl
    └── repair_log.jsonl
```

---

## 3. Install Dependencies

Create `requirements.txt`:

```txt
ollama
pydantic
sentence-transformers
numpy
python-dotenv
pytest
```

Install:

```bash
pip install -r requirements.txt
```

Make sure Ollama is running locally:

```bash
ollama serve
```

Pull a local model under 8B. Examples:

```bash
ollama pull qwen3.5:4b
```

For your project, prefer an instruction-tuned model with good JSON following.

---

## 4. Configuration

Create `src/config.py`:

```python
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    model_name: str = "qwen3.5:4b"
    ollama_host: str = "http://localhost:11434"

    temperature: float = 0.0
    seed: int = 42

    rag_top_k: int = 3
    max_repair_attempts: int = 2

    fail_on_unbound_variable: bool = True
    fail_on_unknown_node_type: bool = True
    fail_on_modal_as_negation: bool = True

    solver_target: str = "horn"  # "horn", "z3", "datalog"
```

---

## 5. Core Schemas

Create `src/schemas.py`.

This file is the most important part of the system. The LLM must fit into this schema. Python rejects anything invalid.

```python
from __future__ import annotations

from typing import Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator


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


class QuestionParse(BaseModel):
    question: str
    choices: dict[str, LogicNode] = Field(default_factory=dict)
    query: Optional[LogicNode] = None


class FullParseResult(BaseModel):
    premises: list[CompiledPremise]
    question: Optional[QuestionParse] = None
```

---

## 6. Safe JSON Utilities

Small local models often produce extra text around JSON. Add a strict parser.

Create `src/json_utils.py`:

```python
import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Extract the first valid JSON object from a model response.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response:\n{text[:500]}")

    candidate = match.group(0)
    return json.loads(candidate)


def extract_json_array_or_object(text: str) -> Any:
    """
    Extract a JSON object or array.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    obj_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    arr_match = re.search(r"\[.*\]", text, flags=re.DOTALL)

    candidates = []
    if obj_match:
        candidates.append(obj_match.group(0))
    if arr_match:
        candidates.append(arr_match.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"No valid JSON found in response:\n{text[:500]}")
```

---

## 7. Stage 0: Input Loader and Splitter

Create `src/stage0_input.py`:

```python
from pydantic import BaseModel, Field


class RawLogicProblem(BaseModel):
    id: str
    premises: list[str]
    question: str | None = None
    choices: dict[str, str] = Field(default_factory=dict)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def load_problem(raw: dict) -> RawLogicProblem:
    return RawLogicProblem(
        id=raw.get("id", "unknown"),
        premises=[normalize_whitespace(p) for p in raw.get("premises", [])],
        question=normalize_whitespace(raw["question"]) if raw.get("question") else None,
        choices={k: normalize_whitespace(v) for k, v in raw.get("choices", {}).items()},
    )
```

---

## 8. Stage 1: First Parser / CNL Rewriter

Stage 1 does **not** generate final logic. It only makes the text easier for code and Stage 3 to parse.

It should output:

```text
original text
kind hint
controlled natural language
risk flags
if/then/body slots if obvious
```

Create `src/stage1_cnl.py`:

```python
from ollama import Client

from .config import PipelineConfig
from .json_utils import extract_json_object
from .schemas import Stage1Output


STAGE1_SYSTEM_PROMPT = """
You are Stage 1 of a neurosymbolic logic parser.

Your job:
Convert raw English premises into clean Controlled Natural Language (CNL).

You must NOT solve the problem.
You must NOT infer new facts.
You must preserve logical direction, quantifiers, negation, modality, and named entities.

Allowed kind_hint values:
FACT, EXISTS, FORALL, RULE, ONLY_IF_RULE, IFF, NON_IF_RULE, OBLIGATION_RULE, META, UNKNOWN

Rules:
1. Rewrite conditionals as: If [condition], then [consequence].
2. Preserve "only if" direction. "A only if B" means A -> B.
3. Preserve "if and only if" as IFF.
4. Treat "not necessarily" as modal uncertainty, not classical NOT.
5. Normalize simple negation:
   - without housing -> NOT has housing
   - fail to maintain GPA -> NOT maintain GPA
   - cannot participate -> NOT participate
6. For nested logic, use kind_hint META and keep the nested statement literal.
7. Do not create predicate names yet.
8. Return JSON only.

Return this exact shape:
{
  "statements": [
    {
      "premise_id": "P1",
      "original": "...",
      "kind_hint": "RULE",
      "cnl": "If ..., then ...",
      "risk_flags": [],
      "if_part": "...",
      "then_part": "...",
      "body": null
    }
  ]
}
"""


class CNLRewriter:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = Client(host=config.ollama_host)

    def rewrite(self, premises: list[str]) -> Stage1Output:
        numbered = "\n".join([f"P{i+1}: {p}" for i, p in enumerate(premises)])

        response = self.client.chat(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
                {"role": "user", "content": f"Rewrite these premises into CNL:\n{numbered}"},
            ],
            options={
                "temperature": self.config.temperature,
                "seed": self.config.seed,
            },
            format="json",
        )

        raw_text = response["message"]["content"]
        data = extract_json_object(raw_text)
        return Stage1Output.model_validate(data)
```

### Stage 1 Example

Input:

```text
Students who fail to maintain GPA cannot participate in the lab.
```

Expected Stage 1 output:

```json
{
  "premise_id": "P1",
  "kind_hint": "RULE",
  "cnl": "If a student does NOT maintain GPA, then the student does NOT participate in the lab.",
  "risk_flags": ["negative_body"],
  "if_part": "a student does NOT maintain GPA",
  "then_part": "the student does NOT participate in the lab"
}
```

---

## 9. Stage 2: Structural RAG

The purpose of RAG here is not external knowledge. It is **structural memory**.

You store examples like:

```text
CNL sentence -> correct AST
```

Then retrieve similar structures for the LLM.

Create `data/structural_examples.jsonl`:

```jsonl
{"cnl":"Every student is eligible.","ast":{"type":"forall","variable":"x","children":[{"type":"implies","children":[{"type":"atomic","name":"student","arguments":["x"]},{"type":"atomic","name":"eligible","arguments":["x"]}]}]}}
{"cnl":"If a student studies, then the student passes.","ast":{"type":"forall","variable":"x","children":[{"type":"implies","children":[{"type":"atomic","name":"student_studies","arguments":["x"]},{"type":"atomic","name":"student_passes","arguments":["x"]}]}]}}
{"cnl":"A student passes if and only if the student submits the thesis.","ast":{"type":"forall","variable":"x","children":[{"type":"iff","children":[{"type":"atomic","name":"student_passes","arguments":["x"]},{"type":"atomic","name":"student_submits_thesis","arguments":["x"]}]}]}}
{"cnl":"A student can enter the lab only if the student has approval.","ast":{"type":"forall","variable":"x","children":[{"type":"implies","children":[{"type":"atomic","name":"student_can_enter_lab","arguments":["x"]},{"type":"atomic","name":"student_has_approval","arguments":["x"]}]}]}}
{"cnl":"Some student has housing.","ast":{"type":"exists","variable":"x","children":[{"type":"and","children":[{"type":"atomic","name":"student","arguments":["x"]},{"type":"atomic","name":"has_housing","arguments":["x"]}]}]}}
{"cnl":"John is a student.","ast":{"type":"atomic","name":"student","arguments":["john"]}}
```

Create `src/stage2_rag.py`:

```python
import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


class StructuralRAG:
    def __init__(self, examples_path: str = "data/structural_examples.jsonl"):
        self.examples_path = Path(examples_path)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.items: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.load()

    def load(self):
        rows = []
        with self.examples_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        self.items = rows
        texts = [r["cnl"] for r in rows]
        self.embeddings = self.embedder.encode(texts, convert_to_numpy=True)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if self.embeddings is None or len(self.items) == 0:
            return []

        query_emb = self.embedder.encode([query], convert_to_numpy=True)[0]

        scores = []
        for emb in self.embeddings:
            denom = np.linalg.norm(query_emb) * np.linalg.norm(emb)
            score = float(np.dot(query_emb, emb) / denom) if denom else 0.0
            scores.append(score)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.items[i] | {"score": scores[i]} for i in top_indices]

    def format_examples(self, query: str, top_k: int = 3) -> str:
        matches = self.retrieve(query, top_k=top_k)

        blocks = []
        for m in matches:
            blocks.append(
                "CNL:\n"
                f"{m['cnl']}\n"
                "AST:\n"
                f"{json.dumps(m['ast'], ensure_ascii=False)}"
            )

        return "\n\n".join(blocks)
```

---

## 10. Stage 3: Second Parser / AST Compiler

Stage 3 is where the model turns CNL into typed AST.

Important rule:

> The LLM can propose AST. Python decides whether it is valid.

Create `src/stage3_ast.py`:

```python
import json

from ollama import Client

from .config import PipelineConfig
from .json_utils import extract_json_object
from .schemas import Stage1Output, Stage3Output
from .stage2_rag import StructuralRAG


STAGE3_SYSTEM_PROMPT = """
You are Stage 3 of a neurosymbolic logic parser.

Your job:
Compile CNL statements into typed recursive LogicNode AST JSON.

You must NOT solve the question.
You must NOT add facts that are not in the premise.
You must preserve:
- quantifiers
- IF direction
- only-if direction
- iff direction
- classical negation
- modal uncertainty

Allowed node types:
atomic, and, or, not, implies, iff, forall, exists, equation

Atomic predicate rules:
1. Predicate names must be lowercase snake_case.
2. Use variables like x, y for quantified rules.
3. Use constants like john, mary, quantum_lab for named entities.
4. Do not put "not" inside predicate names.
   Correct: {"type":"not","children":[{"type":"atomic","name":"has_housing","arguments":["x"]}]}
   Wrong: {"type":"atomic","name":"not_has_housing","arguments":["x"]}

Quantifier rules:
- Every/All/Any -> forall
- Some/At least one/A -> exists only when the sentence asserts existence
- Generic rules usually become forall x: antecedent -> consequent

Only-if rule:
- "A only if B" means A -> B.

IFF:
- "A if and only if B" means A <-> B.

Return JSON only in this exact shape:
{
  "compiled": [
    {
      "premise_id": "P1",
      "kind": "RULE",
      "cnl": "...",
      "ast": {...},
      "solver_ready": false,
      "needs_review": false,
      "unsupported": false,
      "notes": []
    }
  ]
}
"""


class ASTCompiler:
    def __init__(self, config: PipelineConfig, rag: StructuralRAG):
        self.config = config
        self.rag = rag
        self.client = Client(host=config.ollama_host)

    def compile(self, stage1: Stage1Output) -> Stage3Output:
        cnl_text = "\n".join(
            f"{s.premise_id} [{s.kind_hint}]: {s.cnl}"
            for s in stage1.statements
        )

        rag_context = self.rag.format_examples(cnl_text, top_k=self.config.rag_top_k)

        user_prompt = f"""
Reference examples:
{rag_context}

Now compile these CNL statements:
{cnl_text}
"""

        response = self.client.chat(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": STAGE3_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": self.config.temperature,
                "seed": self.config.seed,
            },
            format="json",
        )

        raw_text = response["message"]["content"]
        data = extract_json_object(raw_text)
        return Stage3Output.model_validate(data)
```

---

## 11. Stage 4: Python Validator

This validator is what prevents the LLM from damaging the solver.

Create `src/stage4_validate.py`:

```python
from dataclasses import dataclass, field

from .schemas import LogicNode, Stage3Output


@dataclass
class ValidationIssue:
    premise_id: str
    severity: str  # "error", "warning"
    message: str


@dataclass
class ValidationReport:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)


class LogicValidator:
    def validate_stage3(self, output: Stage3Output) -> ValidationReport:
        issues: list[ValidationIssue] = []

        for item in output.compiled:
            issues.extend(self.validate_node(item.premise_id, item.ast))

            if item.kind == "ONLY_IF_RULE":
                if not self.contains_node_type(item.ast, "implies"):
                    issues.append(
                        ValidationIssue(
                            item.premise_id,
                            "error",
                            "ONLY_IF_RULE must compile to an implies node.",
                        )
                    )

            if item.kind == "IFF":
                if not self.contains_node_type(item.ast, "iff"):
                    issues.append(
                        ValidationIssue(
                            item.premise_id,
                            "error",
                            "IFF premise must compile to an iff node.",
                        )
                    )

        return ValidationReport(ok=not any(i.severity == "error" for i in issues), issues=issues)

    def validate_node(self, premise_id: str, node: LogicNode) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        bound_vars: set[str] = set()

        self._walk(node, premise_id, bound_vars, issues)
        return issues

    def _walk(
        self,
        node: LogicNode,
        premise_id: str,
        bound_vars: set[str],
        issues: list[ValidationIssue],
    ):
        if node.type in {"forall", "exists"}:
            if not node.variable:
                issues.append(
                    ValidationIssue(premise_id, "error", f"{node.type} missing variable")
                )
                return

            new_bound = set(bound_vars)
            new_bound.add(node.variable)

            for child in node.children:
                self._walk(child, premise_id, new_bound, issues)
            return

        if node.type == "atomic":
            for arg in node.arguments:
                # constants are allowed. Variables should be single-letter x/y/z.
                if arg in {"x", "y", "z"} and arg not in bound_vars:
                    issues.append(
                        ValidationIssue(
                            premise_id,
                            "error",
                            f"Unbound variable '{arg}' in atomic predicate {node.name}",
                        )
                    )

            if node.name and node.name.startswith("not_"):
                issues.append(
                    ValidationIssue(
                        premise_id,
                        "error",
                        f"Predicate name '{node.name}' contains negation. Use a NOT node instead.",
                    )
                )

        if node.type == "not":
            if len(node.children) != 1:
                issues.append(
                    ValidationIssue(premise_id, "error", "NOT node must have exactly one child.")
                )

            child = node.children[0] if node.children else None
            if child and child.type == "not":
                issues.append(
                    ValidationIssue(
                        premise_id,
                        "warning",
                        "Double negation found. Consider simplifying.",
                    )
                )

        if node.type == "implies":
            if len(node.children) != 2:
                issues.append(
                    ValidationIssue(premise_id, "error", "IMPLIES node must have exactly 2 children.")
                )

        for child in node.children:
            self._walk(child, premise_id, bound_vars, issues)

    def contains_node_type(self, node: LogicNode, target: str) -> bool:
        if node.type == target:
            return True
        return any(self.contains_node_type(child, target) for child in node.children)
```

---

## 12. Stage 5: Repair Loop

If validation fails, do not manually guess the fix. Feed the precise validation error back into the LLM.

Create `src/stage5_repair.py`:

```python
from ollama import Client

from .config import PipelineConfig
from .json_utils import extract_json_object
from .schemas import Stage3Output
from .stage4_validate import ValidationReport


REPAIR_SYSTEM_PROMPT = """
You are a JSON AST repair component.

You receive:
1. A previous invalid Stage 3 JSON output.
2. Python validator errors.

Your job:
Repair the AST JSON only.

Rules:
- Do not solve the question.
- Do not add new premises.
- Fix only the structural errors.
- Preserve premise_id, kind, and cnl.
- Return JSON only.
"""


class RepairLoop:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = Client(host=config.ollama_host)

    def repair(self, bad_output: Stage3Output, report: ValidationReport) -> Stage3Output:
        issue_text = "\n".join(
            f"{i.premise_id} [{i.severity}]: {i.message}"
            for i in report.issues
        )

        prompt = f"""
Previous invalid JSON:
{bad_output.model_dump_json(indent=2)}

Validator errors:
{issue_text}

Return repaired JSON with the same schema.
"""

        response = self.client.chat(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": 0.0,
                "seed": self.config.seed,
            },
            format="json",
        )

        raw_text = response["message"]["content"]
        data = extract_json_object(raw_text)
        return Stage3Output.model_validate(data)
```

---

## 13. Stage 6: Solver Export

Start with a simple Horn-style exporter.

This exporter supports:

```text
forall x: A(x) -> B(x)
FACT: A(john)
EXISTS: exists x: A(x)
```

Create `src/stage6_solver_export.py`:

```python
from dataclasses import dataclass

from .schemas import LogicNode, Stage3Output


@dataclass
class HornFact:
    predicate: str
    args: tuple[str, ...]
    negated: bool = False


@dataclass
class HornRule:
    body: list[HornFact]
    head: HornFact


@dataclass
class HornProgram:
    facts: list[HornFact]
    rules: list[HornRule]
    unsupported: list[str]


class HornExporter:
    def export(self, parsed: Stage3Output) -> HornProgram:
        facts: list[HornFact] = []
        rules: list[HornRule] = []
        unsupported: list[str] = []

        for item in parsed.compiled:
            try:
                self._export_node(item.ast, facts, rules)
            except Exception as e:
                unsupported.append(f"{item.premise_id}: {e}")

        return HornProgram(facts=facts, rules=rules, unsupported=unsupported)

    def _export_node(self, node: LogicNode, facts: list[HornFact], rules: list[HornRule]):
        if node.type == "atomic":
            facts.append(HornFact(node.name, tuple(node.arguments)))
            return

        if node.type == "not":
            child = node.children[0]
            if child.type != "atomic":
                raise ValueError("Only NOT atomic is supported in Horn exporter")
            facts.append(HornFact(child.name, tuple(child.arguments), negated=True))
            return

        if node.type == "forall":
            self._export_node(node.children[0], facts, rules)
            return

        if node.type == "implies":
            body_node, head_node = node.children

            body_facts = self._node_to_fact_list(body_node)
            head_fact = self._node_to_single_fact(head_node)

            rules.append(HornRule(body=body_facts, head=head_fact))
            return

        if node.type == "exists":
            # Simple version:
            # exists x: student(x) AND has_housing(x)
            # becomes a skolem constant.
            skolem = "skolem_entity"
            for fact in self._node_to_fact_list(node.children[0]):
                facts.append(
                    HornFact(
                        predicate=fact.predicate,
                        args=tuple(skolem if a == node.variable else a for a in fact.args),
                        negated=fact.negated,
                    )
                )
            return

        raise ValueError(f"Unsupported node type for Horn export: {node.type}")

    def _node_to_fact_list(self, node: LogicNode) -> list[HornFact]:
        if node.type == "and":
            out = []
            for child in node.children:
                out.extend(self._node_to_fact_list(child))
            return out

        return [self._node_to_single_fact(node)]

    def _node_to_single_fact(self, node: LogicNode) -> HornFact:
        if node.type == "atomic":
            return HornFact(node.name, tuple(node.arguments))

        if node.type == "not":
            child = node.children[0]
            if child.type != "atomic":
                raise ValueError("Only NOT atomic can become a HornFact")
            return HornFact(child.name, tuple(child.arguments), negated=True)

        raise ValueError(f"Cannot convert {node.type} to HornFact")
```

This is intentionally simple. You can improve it later for:

```text
OR
IFF lowering
nested rules
deontic rules
modal uncertainty
numeric comparisons
```

---

## 14. Full Pipeline Orchestrator

Create `src/pipeline.py`:

```python
from .config import PipelineConfig
from .stage1_cnl import CNLRewriter
from .stage2_rag import StructuralRAG
from .stage3_ast import ASTCompiler
from .stage4_validate import LogicValidator
from .stage5_repair import RepairLoop
from .stage6_solver_export import HornExporter, HornProgram
from .schemas import Stage1Output, Stage3Output


class LogicPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

        self.stage1 = CNLRewriter(config)
        self.rag = StructuralRAG()
        self.stage3 = ASTCompiler(config, self.rag)
        self.validator = LogicValidator()
        self.repair_loop = RepairLoop(config)
        self.exporter = HornExporter()

    def parse(self, premises: list[str]) -> tuple[Stage1Output, Stage3Output]:
        stage1_output = self.stage1.rewrite(premises)
        stage3_output = self.stage3.compile(stage1_output)

        for _ in range(self.config.max_repair_attempts + 1):
            report = self.validator.validate_stage3(stage3_output)
            if report.ok:
                return stage1_output, stage3_output

            stage3_output = self.repair_loop.repair(stage3_output, report)

        final_report = self.validator.validate_stage3(stage3_output)
        if not final_report.ok:
            messages = "\n".join(
                f"{i.premise_id} [{i.severity}]: {i.message}"
                for i in final_report.issues
            )
            raise ValueError(f"Pipeline failed validation after repair attempts:\n{messages}")

        return stage1_output, stage3_output

    def parse_and_export_horn(self, premises: list[str]) -> HornProgram:
        _, parsed = self.parse(premises)
        return self.exporter.export(parsed)
```

---

## 15. Run One Example

Create `scripts/run_one.py`:

```python
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import PipelineConfig
from src.pipeline import LogicPipeline


def main():
    premises = [
        "Every scholarship recipient receives priority housing.",
        "Students without housing cannot participate in the Quantum Physics lab.",
        "Mina is a scholarship recipient.",
    ]

    config = PipelineConfig(
        model_name="qwen2.5:7b-instruct",
        max_repair_attempts=2,
    )

    pipeline = LogicPipeline(config)
    stage1, stage3 = pipeline.parse(premises)

    print("\n===== STAGE 1 CNL =====")
    print(stage1.model_dump_json(indent=2))

    print("\n===== STAGE 3 AST =====")
    print(stage3.model_dump_json(indent=2))

    horn = pipeline.exporter.export(stage3)

    print("\n===== HORN EXPORT =====")
    print(json.dumps(
        {
            "facts": [f.__dict__ for f in horn.facts],
            "rules": [
                {
                    "body": [b.__dict__ for b in r.body],
                    "head": r.head.__dict__,
                }
                for r in horn.rules
            ],
            "unsupported": horn.unsupported,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
```

Run:

```bash
python scripts/run_one.py
```

---

## 16. Batch Runner for JSONL

Create `scripts/run_jsonl.py`:

```python
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import PipelineConfig
from src.pipeline import LogicPipeline


def main():
    input_path = Path("data/sample_inputs.jsonl")
    output_path = Path("artifacts/predictions.jsonl")
    output_path.parent.mkdir(exist_ok=True)

    config = PipelineConfig(model_name="qwen2.5:7b-instruct")
    pipeline = LogicPipeline(config)

    with input_path.open("r", encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
        for line_no, line in enumerate(f_in, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            premises = row["premises"]

            try:
                stage1, stage3 = pipeline.parse(premises)
                record = {
                    "id": row.get("id", f"row-{line_no}"),
                    "ok": True,
                    "stage1": stage1.model_dump(),
                    "stage3": stage3.model_dump(),
                }
            except Exception as e:
                record = {
                    "id": row.get("id", f"row-{line_no}"),
                    "ok": False,
                    "error": str(e),
                }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{line_no}] {record['id']} ok={record['ok']}")

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
```

Create `data/sample_inputs.jsonl`:

```jsonl
{"id":"case-1","premises":["Every scholarship recipient receives priority housing.","Students without housing cannot participate in the Quantum Physics lab.","Mina is a scholarship recipient."],"question":"Can Mina participate in the Quantum Physics lab?"}
{"id":"case-2","premises":["A student can enter the lab only if the student has approval.","John can enter the lab."],"question":"Does John have approval?"}
```

Run:

```bash
python scripts/run_jsonl.py
```

---

## 17. Testing

### 17.1 Schema Test

Create `tests/test_schemas.py`:

```python
from src.schemas import LogicNode


def test_atomic_node_valid():
    node = LogicNode(type="atomic", name="student", arguments=["john"])
    assert node.name == "student"


def test_implies_node_valid():
    node = LogicNode(
        type="implies",
        children=[
            LogicNode(type="atomic", name="student", arguments=["x"]),
            LogicNode(type="atomic", name="eligible", arguments=["x"]),
        ],
    )
    assert node.type == "implies"
```

### 17.2 Validator Test

Create `tests/test_validator.py`:

```python
from src.schemas import CompiledPremise, LogicNode, Stage3Output
from src.stage4_validate import LogicValidator


def test_unbound_variable_detected():
    output = Stage3Output(
        compiled=[
            CompiledPremise(
                premise_id="P1",
                kind="FACT",
                cnl="x is a student",
                ast=LogicNode(type="atomic", name="student", arguments=["x"]),
            )
        ]
    )

    report = LogicValidator().validate_stage3(output)
    assert not report.ok
    assert any("Unbound variable" in i.message for i in report.issues)
```

### 17.3 Run Tests

```bash
pytest -q
```

---

## 18. How to Handle Each Premise Type

### FACT

Example:

```text
John is a student.
```

AST:

```json
{
  "type": "atomic",
  "name": "student",
  "arguments": ["john"]
}
```

---

### EXISTS

Example:

```text
Some student has housing.
```

AST:

```json
{
  "type": "exists",
  "variable": "x",
  "children": [
    {
      "type": "and",
      "children": [
        {"type": "atomic", "name": "student", "arguments": ["x"]},
        {"type": "atomic", "name": "has_housing", "arguments": ["x"]}
      ]
    }
  ]
}
```

---

### FORALL

Example:

```text
Every student is eligible.
```

AST:

```json
{
  "type": "forall",
  "variable": "x",
  "children": [
    {
      "type": "implies",
      "children": [
        {"type": "atomic", "name": "student", "arguments": ["x"]},
        {"type": "atomic", "name": "eligible", "arguments": ["x"]}
      ]
    }
  ]
}
```

---

### RULE

Example:

```text
If a student studies, then the student passes.
```

AST:

```json
{
  "type": "forall",
  "variable": "x",
  "children": [
    {
      "type": "implies",
      "children": [
        {"type": "atomic", "name": "studies", "arguments": ["x"]},
        {"type": "atomic", "name": "passes", "arguments": ["x"]}
      ]
    }
  ]
}
```

---

### ONLY_IF_RULE

Example:

```text
A student enters the lab only if the student has approval.
```

Meaning:

```text
enters_lab(x) -> has_approval(x)
```

AST:

```json
{
  "type": "forall",
  "variable": "x",
  "children": [
    {
      "type": "implies",
      "children": [
        {"type": "atomic", "name": "enters_lab", "arguments": ["x"]},
        {"type": "atomic", "name": "has_approval", "arguments": ["x"]}
      ]
    }
  ]
}
```

---

### IFF

Example:

```text
A student passes if and only if the student submits the thesis.
```

AST:

```json
{
  "type": "forall",
  "variable": "x",
  "children": [
    {
      "type": "iff",
      "children": [
        {"type": "atomic", "name": "passes", "arguments": ["x"]},
        {"type": "atomic", "name": "submits_thesis", "arguments": ["x"]}
      ]
    }
  ]
}
```

For Horn export, lower IFF into two rules:

```text
passes(x) -> submits_thesis(x)
submits_thesis(x) -> passes(x)
```

---

### NON_IF_RULE

Example:

```text
Passing Philosophy grants eligibility for the Quantum Physics lab.
```

Stage 1 should rewrite:

```text
If a student passes Philosophy, then the student is eligible for the Quantum Physics lab.
```

Then Stage 3 compiles it as a normal rule.

---

### OBLIGATION_RULE

Example:

```text
Wearing protective equipment is mandatory in science laboratories.
```

Recommended AST:

```json
{
  "type": "forall",
  "variable": "x",
  "children": [
    {
      "type": "implies",
      "children": [
        {"type": "atomic", "name": "in_science_laboratory", "arguments": ["x"]},
        {"type": "atomic", "name": "obligated_wear_protective_equipment", "arguments": ["x"]}
      ]
    }
  ]
}
```

Do not mix obligation with normal fact unless your solver supports deontic logic.

---

### META

Use META for nested logic.

Example:

```text
If the rule "students who pass thesis receive grants" is active, then John receives a grant.
```

Stage 1 should mark this as `META`.

You can either:

1. Send it to a special meta-logic solver, or
2. Mark it `needs_review`, or
3. Lower it only if your solver supports rules as objects.

---

## 19. Negation Policy

Use this consistently.

### Correct

```json
{
  "type": "not",
  "children": [
    {"type": "atomic", "name": "has_housing", "arguments": ["x"]}
  ]
}
```

### Wrong

```json
{"type": "atomic", "name": "not_has_housing", "arguments": ["x"]}
```

### Normalize these forms

| Raw text | Normalized meaning |
|---|---|
| without housing | NOT has_housing(x) |
| fail to maintain GPA | NOT maintain_gpa(x) |
| cannot participate | NOT participate(x) |
| non-participation | NOT participate(x) |
| loss of housing | NOT has_housing(x) |
| revocation of scholarship | NOT scholarship_recipient(x) |

### Important modal case

```text
not necessarily energy efficient
```

This is **not** the same as:

```text
NOT energy_efficient(x)
```

It means uncertainty:

```text
not guaranteed energy efficient
```

Recommended handling:

```json
{
  "kind": "FORALL",
  "risk_flags": ["modal_not_necessarily"],
  "notes": ["Do not lower to classical NOT without policy decision."]
}
```

---

## 20. Validation Rules You Should Enforce

Add these checks before sending anything to the solver:

```text
1. Every AST must pass Pydantic validation.
2. Every variable used in an atomic predicate must be bound by forall/exists.
3. "not_" predicate names are forbidden.
4. ONLY_IF_RULE must become implication in the correct direction.
5. IFF must become iff or two implications.
6. "not necessarily" must not become classical NOT.
7. Every rule must have exactly one antecedent side and one consequent side.
8. EXISTS must use a deterministic Skolem constant if exported to Horn.
9. Unsupported META/nested rules must be marked needs_review unless you implement meta-logic.
10. The solver must never receive unknown node types.
```

---

## 21. Solver-Readiness Classifier

Add this after validation:

```python
def classify_solver_readiness(kind: str, ast_type: str, risk_flags: list[str]) -> str:
    if "modal_not_necessarily" in risk_flags:
        return "needs_review"

    if "modal_scope_ambiguous" in risk_flags:
        return "needs_review"

    if kind in {"META", "OBLIGATION_RULE"}:
        return "needs_review"

    if ast_type in {"atomic", "forall", "exists", "implies", "and", "not"}:
        return "solver_ready"

    if ast_type in {"or", "iff", "equation"}:
        return "needs_lowering"

    return "unsupported"
```

Use three output buckets:

```text
solver_ready
needs_review
unsupported
```

Your target should be:

```text
unsupported = 0
needs_review = small and explainable
solver_ready = as high as possible
```

---

## 22. Repair Strategy

Do not use vague repair prompts.

Bad:

```text
Fix this JSON.
```

Good:

```text
The validator found:
P3 error: Unbound variable x in predicate has_housing.
Repair by adding a forall or exists quantifier if the premise is generic.
Return JSON only.
```

Repair prompt should include:

```text
1. Original raw premise
2. Stage 1 CNL
3. Bad AST
4. Validator error
5. Exact allowed schema
```

---

## 23. Performance Plan for Under 1 Minute

For local models under 8B:

```text
1. Run Stage 1 once per problem, not once per premise if the problem is small.
2. Run Stage 3 on all premises together if there are fewer than 20 premises.
3. For large input, batch premises in groups of 8–12.
4. Keep max repair attempts at 1 or 2.
5. Use temperature 0.
6. Keep Structural RAG examples short.
7. Use small embedding model: all-MiniLM-L6-v2.
8. Cache embeddings.
9. Cache Stage 1 output by hash of raw premise.
10. Cache Stage 3 output by hash of CNL.
```

Recommended runtime layout:

```text
Input size: 5–15 premises
Local 7B/8B model
Target total time: < 60 seconds

Budget:
Stage 1: 10–25 seconds
RAG: < 1 second
Stage 3: 10–25 seconds
Validation: < 1 second
Repair: only if needed
```

---

## 24. What to Improve Later

After the basic pipeline works, improve in this order:

```text
1. Add more structural RAG examples.
2. Add deterministic handling for easy FACT/FORALL/RULE cases.
3. Add predicate canonicalization.
4. Add IFF lowering.
5. Add EXISTS Skolemization.
6. Add question parser.
7. Add solver.
8. Add explanation generator.
9. Add evaluation dashboard.
```

---

## 25. Question Parser

After premise parsing is stable, parse questions separately.

Example:

```text
Can Mina participate in the Quantum Physics lab?
```

Question AST:

```json
{
  "query": {
    "type": "atomic",
    "name": "participate_quantum_physics_lab",
    "arguments": ["mina"]
  }
}
```

For multiple choice:

```json
{
  "choices": {
    "A": {"type": "atomic", "name": "eligible", "arguments": ["mina"]},
    "B": {"type": "not", "children": [{"type": "atomic", "name": "eligible", "arguments": ["mina"]}]}
  }
}
```

Important:

```text
The question parser should not use premise facts to rewrite choices.
It only parses the choices.
The solver decides which choice is entailed.
```

---

## 26. Evaluation Metrics

Track these numbers:

```text
stage1_ok
stage1_needs_attention
stage3_schema_valid
stage3_validation_valid
solver_ready
needs_review
unsupported
repair_success_rate
answer_accuracy
average_runtime_seconds
```

A good early target:

```text
stage3_validation_valid >= 95%
unsupported = 0
needs_review <= 10%
average_runtime < 60s
```

---

## 27. Minimum Working Command Flow

From a clean repo:

```bash
mkdir logic_pipeline
cd logic_pipeline

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

ollama serve
ollama pull qwen2.5:7b-instruct

python scripts/run_one.py
pytest -q
python scripts/run_jsonl.py
```

---

## 28. Final Implementation Checklist

Before connecting to a real solver, make sure you have:

```text
[ ] Pydantic schemas working
[ ] Stage 1 CNL output working
[ ] Stage 2 Structural RAG examples loaded
[ ] Stage 3 AST compiler working
[ ] Validator catches unbound variables
[ ] Validator catches bad negation
[ ] Repair loop can fix simple structural errors
[ ] Horn exporter works for FACT/RULE/FORALL/EXISTS
[ ] IFF lowering implemented or marked needs_lowering
[ ] Modal cases marked needs_review
[ ] META cases marked needs_review
[ ] Batch runner saves JSONL outputs
[ ] Evaluation report counts solver_ready / needs_review / unsupported
```

---

## 29. Best Practical Version of Your Pipeline

The best practical version is:

```text
First Parser:
    Raw English -> CNL + kind hints + risk flags

Structural RAG:
    CNL -> retrieve similar CNL/AST examples

Second Parser:
    CNL + examples -> typed AST

Python Validator:
    reject invalid AST

Repair Loop:
    repair only using exact validator errors

Solver Export:
    only export solver_ready records

Solver:
    answer question

Explanation:
    translate solver trace back into human language
```

This design keeps the local LLM useful without trusting it too much.

The most important rule:

> **Never let the LLM be the source of truth.  
> Let the LLM propose structure.  
> Let Python validate.  
> Let the solver reason.**
