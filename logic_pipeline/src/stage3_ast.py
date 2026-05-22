from .config import PipelineConfig
from .json_utils import extract_json_object
from .llm_client import ChatModel
from .schemas import Stage1Output, Stage3Output
from .stage2_rag import StructuralRAG


STAGE3_SYSTEM_PROMPT = """\
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
        )
        data = extract_json_object(raw_text)
        return Stage3Output.model_validate(data)
