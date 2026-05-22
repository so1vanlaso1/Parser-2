from .config import PipelineConfig
from .json_utils import extract_json_object
from .llm_client import ChatModel
from .schemas import Stage1Output


STAGE1_SYSTEM_PROMPT = """\
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
8. Return JSON only. No explanation text.

Return this exact JSON shape:
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
    """Stage 1 — rewrites raw English premises into Controlled Natural Language."""

    def __init__(self, config: PipelineConfig, llm: ChatModel):
        self.config = config
        self.llm = llm

    def rewrite(self, premises: list[str]) -> Stage1Output:
        numbered = "\n".join([f"P{i+1}: {p}" for i, p in enumerate(premises)])

        raw_text = self.llm.generate(
            STAGE1_SYSTEM_PROMPT,
            f"Rewrite these premises into CNL:\n{numbered}",
        )
        data = extract_json_object(raw_text)
        return Stage1Output.model_validate(data)
