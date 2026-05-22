from .config import PipelineConfig
from .json_utils import extract_json_object
from .llm_client import ChatModel
from .schemas import QuestionParse


QUESTION_PARSER_SYSTEM_PROMPT = """\
You are a question parser for a neurosymbolic logic pipeline.

Your job:
Parse a question (and its choices, if any) into structured LogicNode AST.

You must NOT solve the question.
You must NOT use premise facts to rewrite choices.
You only parse the question and choices into AST structure.

Predicate naming rules:
1. Predicate names must be lowercase snake_case.
2. Use constants for named entities (e.g., john, mina, alphanet).
3. Do not put "not" inside predicate names — use a NOT node.

For yes/no questions, return a "query" node representing what is being asked.
For multiple-choice questions, return a "choices" map with each choice as a LogicNode.

Return JSON only in this exact shape:
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
    ) -> QuestionParse:
        if not question_text:
            return QuestionParse(question="")

        user_content = f"Question: {question_text}"
        if choices:
            choices_text = "\n".join(f"{k}: {v}" for k, v in choices.items())
            user_content += f"\n\nChoices:\n{choices_text}"

        raw_text = self.llm.generate(
            QUESTION_PARSER_SYSTEM_PROMPT,
            user_content,
        )
        data = extract_json_object(raw_text)
        return QuestionParse.model_validate(data)
