import re

from pydantic import BaseModel, Field


class RawLogicProblem(BaseModel):
    """
    Represents one problem from the JSONL dataset.
    Adapted for the fixed_smoke_logic_406.jsonl format which uses
    'premises-NL' instead of 'premises'.
    """
    id: str
    premises: list[str]
    question: str | None = None
    choices: dict[str, str] = Field(default_factory=dict)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces."""
    return " ".join(text.strip().split())


def extract_choices_from_question(question_text: str) -> tuple[str, dict[str, str]]:
    """
    Extract multiple-choice options from question text.

    Handles formats like:
        A) Option text
        A. Option text
        A: Option text

    Returns (clean_question, choices_dict).
    """
    if not question_text:
        return question_text or "", {}

    # Pattern: letter followed by ) or . or : then the option text.
    # Split question into the question stem and the choices.
    choice_pattern = re.compile(
        r"(?:^|\n)\s*([A-Z])\s*[.):\-]\s*(.+?)(?=(?:\n\s*[A-Z]\s*[.):\-])|$)",
        re.DOTALL,
    )

    matches = choice_pattern.findall(question_text)
    if not matches:
        return normalize_whitespace(question_text), {}

    choices = {}
    for letter, text in matches:
        choices[letter.strip()] = normalize_whitespace(text)

    # Extract the question stem (everything before the first choice).
    first_choice_match = re.search(r"\n?\s*[A-Z]\s*[.):\-]", question_text)
    if first_choice_match:
        stem = question_text[: first_choice_match.start()]
    else:
        stem = question_text

    return normalize_whitespace(stem), choices


def load_problem(raw: dict) -> RawLogicProblem:
    """
    Load a problem from a raw JSONL dict.
    Supports both 'premises-NL' (dataset format) and 'premises' (guide format).
    """
    # Get premises from either field name.
    premises_raw = raw.get("premises-NL") or raw.get("premises", [])
    premises = [normalize_whitespace(p) for p in premises_raw]

    # Get question and extract embedded choices.
    question_raw = raw.get("question", "")
    question_stem, extracted_choices = extract_choices_from_question(question_raw)

    # Use explicitly provided choices if available, otherwise use extracted.
    explicit_choices = raw.get("choices", {})
    choices = (
        {k: normalize_whitespace(v) for k, v in explicit_choices.items()}
        if explicit_choices
        else extracted_choices
    )

    return RawLogicProblem(
        id=raw.get("id", "unknown"),
        premises=premises,
        question=question_stem if question_stem else None,
        choices=choices,
    )
