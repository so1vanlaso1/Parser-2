from .config import PipelineConfig
from .json_utils import extract_json_object
from .llm_client import ChatModel
from .schemas import Stage1Output, Stage3Output
from .stage2_rag import StructuralRAG


STAGE3_SYSTEM_PROMPT = """\
/no_think

You are Stage 3 of a neurosymbolic logic parser.
You are a JSON transducer, not a solver.

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

Output rules:
- Output ONLY valid JSON.
- First character must be {.
- Do not write analysis.
- Do not write explanations.
- Do not write markdown.
- Do not include comments.
- Do not output any node type outside the allowed node types.
- End after the JSON object with <END_JSON>.

Atomic predicate rules:
1. Predicate names must be lowercase snake_case.
2. Use variables like x, y for quantified rules.
3. Use constants like john, mary, quantum_lab for named entities.
4. Do not put "not" inside predicate names.
   Correct: {"type":"not","children":[{"type":"atomic","name":"has_housing","arguments":["x"]}]}
   Wrong: {"type":"atomic","name":"not_has_housing","arguments":["x"]}
5. Reuse the same predicate name for the same concept across all premises.
   Example: "requires extensive hyperparameter tuning", "has extensive hyperparameter tuning",
   and "has been extensively tuned" should use one canonical predicate.
6. Every atomic node MUST have at least 1 argument. Never produce {"type":"atomic","name":"foo","arguments":[]}.

STRICT SHAPE RULES (CRITICAL — violating these causes validation failure):
- forall/exists: MUST have exactly 1 child and a "variable" field.
- implies: MUST have exactly 2 children (antecedent, consequent).
- not: MUST have exactly 1 child.
- and/or/iff: MUST have at least 2 children.
- atomic: MUST have "name" (non-empty) and "arguments" (at least 1 element).

Quantifier rules:
- Every/All/Any -> forall
- Some/At least one/A -> exists only when the sentence asserts existence
- Generic rules usually become forall x: antecedent -> consequent

Only-if rule:
- "A only if B" means A -> B.

IFF:
- "A if and only if B" means A <-> B.

Nested quantifier / Mixed scope rules:
When a sentence mixes existential and universal quantifiers, nest them properly.
Example: "If there exists a student who passes, then every teacher celebrates."
AST:
{
  "type": "implies",
  "children": [
    {
      "type": "exists",
      "variable": "x",
      "children": [
        {"type": "and", "children": [
          {"type": "atomic", "name": "student", "arguments": ["x"]},
          {"type": "atomic", "name": "passes", "arguments": ["x"]}
        ]}
      ]
    },
    {
      "type": "forall",
      "variable": "y",
      "children": [
        {"type": "implies", "children": [
          {"type": "atomic", "name": "teacher", "arguments": ["y"]},
          {"type": "atomic", "name": "celebrates", "arguments": ["y"]}
        ]}
      ]
    }
  ]
}

Obligation / Deontic rules:
For OBLIGATION_RULE premises, represent the obligation using an "obligated_" prefix
on the predicate name within a forall wrapper.
Example: "It is mandatory to wear goggles in science laboratories."
AST:
{
  "type": "forall",
  "variable": "x",
  "children": [
    {"type": "implies", "children": [
      {"type": "atomic", "name": "in_science_laboratory", "arguments": ["x"]},
      {"type": "atomic", "name": "obligated_wear_goggles", "arguments": ["x"]}
    ]}
  ]
}

META / Nested implication rules:
For META premises with nested implications, preserve the nesting.
Example: "If passing the exam implies graduation, then students who pass are eligible."
AST:
{
  "type": "implies",
  "children": [
    {
      "type": "forall",
      "variable": "x",
      "children": [
        {"type": "implies", "children": [
          {"type": "atomic", "name": "passes_exam", "arguments": ["x"]},
          {"type": "atomic", "name": "graduates", "arguments": ["x"]}
        ]}
      ]
    },
    {
      "type": "forall",
      "variable": "y",
      "children": [
        {"type": "implies", "children": [
          {"type": "atomic", "name": "passes_exam", "arguments": ["y"]},
          {"type": "atomic", "name": "eligible", "arguments": ["y"]}
        ]}
      ]
    }
  ]
}

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

Do not output anything after <END_JSON>.
"""

class ASTCompiler:
    """Stage 3 — compiles CNL statements into typed LogicNode AST."""

    def __init__(self, config: PipelineConfig, rag: StructuralRAG, llm: ChatModel):
        self.config = config
        self.rag = rag
        self.llm = llm

    def compile(self, stage1: Stage1Output) -> Stage3Output:
        cnl_text = "\n".join(
            f"{s.premise_id} [{s.kind_hint}]: {s.cnl}"
            for s in stage1.statements
        )

        rag_context = self.rag.format_examples(cnl_text, top_k=self.config.rag_top_k)

        user_prompt = f"""\
Reference examples (use these as structural guidance):
{rag_context}

Now compile these CNL statements into AST:
{cnl_text}
"""

        raw_text = self.llm.generate(
            STAGE3_SYSTEM_PROMPT,
            user_prompt,
            max_new_tokens=self._token_budget(len(stage1.statements)),
        )
        data = extract_json_object(raw_text)
        return Stage3Output.model_validate(data)

    def _token_budget(self, statement_count: int) -> int:
        return min(
            self.config.max_new_tokens,
            max(self.config.stage3_max_new_tokens, 450 + 350 * statement_count),
        )
