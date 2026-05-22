import json
import re
from typing import Any


def _trim_end_marker(text: str) -> str:
    marker = "<END_JSON>"
    if marker in text:
        return text.split(marker, 1)[0].strip()
    return text


def _extract_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in response:\n{text[:500]}")

    in_string = False
    escaped = False
    depth = 0

    for index in range(start, len(text)):
        char = text[index]

        if escaped:
            escaped = False
            continue

        if char == "\\" and in_string:
            escaped = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError(
        "JSON object appears truncated before the closing brace. "
        "Increase max_new_tokens for this LLM call."
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Extract the first valid JSON object from a model response.
    Local models often wrap JSON in markdown fences or extra commentary.
    """
    text = _trim_end_marker(text.strip())

    # Try direct parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences if present.
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, flags=re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    candidate = _extract_balanced_json_object(text)
    return json.loads(candidate)


def extract_json_array_or_object(text: str) -> Any:
    """
    Extract a JSON object or array from potentially noisy model output.
    """
    text = _trim_end_marker(text.strip())

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences.
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, flags=re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
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
