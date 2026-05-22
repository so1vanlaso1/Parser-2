from .config import PipelineConfig
from .json_utils import extract_json_object
from .llm_client import ChatModel
from .schemas import QuestionParse


QUESTION_PARSER_SYSTEM_PROMPT = """\
/no_think

You are a question parser for a neurosymbolic logic pipeline.
You are a JSON transducer, not a solver.

Your job:
Parse a question (and its choices, if any) into structured LogicNode AST.

You must NOT solve the question.
You must NOT use premise facts to rewrite choices.
You only parse the question and choices into AST structure.

Allowed LogicNode type values:
atomic, and, or, not, implies, iff, forall, exists, equation

Forbidden type values:
inference, claim, choice, statement, because, explanation, predicate

Predicate naming rules:
1. Predicate names must be lowercase snake_case.
2. Use constants for named entities (e.g., john, mina, alphanet).
3. Do not put "not" inside predicate names - use a NOT node.
4. If known predicate names are provided, reuse a known predicate when it fits the same meaning.
5. Do not invent synonyms for a known predicate.

Reason clauses:
- For multiple-choice entailment questions, parse only the claim being inferred.
- Ignore "because ..." reason text unless the question asks whether the explanation itself is valid.
- Do not create a "because" or "inference" node.

For yes/no questions, return a "query" node representing what is being asked.
For multiple-choice questions, return a "choices" map with each choice as a LogicNode.

Output rules:
- Output ONLY valid JSON.
- First character must be {.
- Do not write analysis.
- Do not write explanations.
- Do not write markdown.
- Do not include comments.
- Every query or choice value must be a valid LogicNode using only the allowed node types.
- End after the JSON object with <END_JSON>.

Return this exact JSON shape:
{
  "question": "the original question text",
  "query": {...} or null,
  "choices": {
    "A": {...},
    "B": {...}
  }
}

If there are no choices, set "choices" to {}.
If it's a multiple-choice question, set "query" to null.

Examples:
Choice text: "The AlphaNet model does not require hyperparameter tuning because it achieves high accuracy."
Choice AST:
{
  "type": "not",
  "children": [
    {"type": "atomic", "name": "requires_hyperparameter_tuning", "arguments": ["alphanet"]}
  ]
}

Choice text: "The AlphaNet model has been extensively tuned because it achieves high accuracy and processes data quickly."
Choice AST:
{"type": "atomic", "name": "has_extensive_hyperparameter_tuning", "arguments": ["alphanet"]}

Do not output anything after <END_JSON>.
"""


class QuestionParser:
    """Parses question text + choices into QuestionParse AST."""

    def __init__(self, config: PipelineConfig, llm: ChatModel):
        self.config = config
        self.llm = llm

    def parse(
        self,
        question_text: str,
        choices: dict[str, str] | None = None,
        known_predicates: list[str] | None = None,
    ) -> QuestionParse:
        if not question_text:
            return QuestionParse(question="")

        user_content = f"Question: {question_text}"
        if known_predicates:
            predicates = "\n".join(f"- {name}" for name in known_predicates)
            user_content += f"\n\nKnown predicate names from premises:\n{predicates}"
        if choices:
            choices_text = "\n".join(f"{k}: {v}" for k, v in choices.items())
            user_content += f"\n\nChoices:\n{choices_text}"

        raw_text = self.llm.generate(
            QUESTION_PARSER_SYSTEM_PROMPT,
            user_content,
            max_new_tokens=self.config.question_max_new_tokens,
        )
        data = extract_json_object(raw_text)
        return QuestionParse.model_validate(data)
