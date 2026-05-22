import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Extract the first valid JSON object from a model response.
    Local models often wrap JSON in markdown fences or extra commentary.
    """
    text = text.strip()

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

    # Greedy brace match.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response:\n{text[:500]}")

    candidate = match.group(0)
    return json.loads(candidate)


def extract_json_array_or_object(text: str) -> Any:
    """
    Extract a JSON object or array from potentially noisy model output.
    """
    text = text.strip()

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
