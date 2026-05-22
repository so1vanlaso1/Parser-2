from .config import PipelineConfig
from .json_utils import extract_json_object
from .llm_client import ChatModel
from .schemas import Stage3Output
from .stage4_validate import ValidationReport


REPAIR_SYSTEM_PROMPT = """\
/no_think

You are a JSON AST repair component.
You are a JSON transducer, not a solver.

You receive:
1. A previous invalid Stage 3 JSON output.
2. Python validator errors.

Your job:
Repair the AST JSON only.

Rules:
- Do not solve the question.
- Do not add new premises.
- Fix only the structural errors reported by the validator.
- Preserve premise_id, kind, and cnl.
- Output ONLY valid JSON in the same schema.
- First character must be {.
- Do not write analysis, explanations, markdown, or comments.
- End after the JSON object with <END_JSON>.
- Do not output anything after <END_JSON>.
"""


class RepairLoop:
    """Stage 5 — feeds validation errors back into the LLM to repair the AST."""

    def __init__(self, config: PipelineConfig, llm: ChatModel):
        self.config = config
        self.llm = llm

    def repair(self, bad_output: Stage3Output, report: ValidationReport) -> Stage3Output:
        issue_text = "\n".join(
            f"{i.premise_id} [{i.severity}]: {i.message}"
            for i in report.issues
        )

        prompt = f"""\
Previous invalid JSON:
{bad_output.model_dump_json(indent=2)}

Validator errors:
{issue_text}

Return repaired JSON with the same schema.
"""

        raw_text = self.llm.generate(
            REPAIR_SYSTEM_PROMPT,
            prompt,
            temperature=0.0,
            max_new_tokens=self._token_budget(len(bad_output.compiled)),
        )
        data = extract_json_object(raw_text)
        return Stage3Output.model_validate(data)

    def _token_budget(self, premise_count: int) -> int:
        return min(
            self.config.max_new_tokens,
            max(self.config.repair_max_new_tokens, 450 + 300 * premise_count),
        )
