from __future__ import annotations

"""Small adapters for Stage 6 validators.

Stage 1/2 code uses Pydantic models internally, while run artifacts are plain
JSON dictionaries. These helpers keep validator modules focused on checks.
"""

from typing import Any, Iterable


def get_value(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def iter_children(node: Any) -> Iterable[Any]:
    return as_list(get_value(node, "children", []))


def node_type(node: Any, *, uppercase: bool = False) -> str | None:
    value = get_value(node, "type")
    if value is None:
        return None
    text = str(value.value) if hasattr(value, "value") else str(value)
    return text.upper() if uppercase else text.lower()


def skeleton_kind(skeleton: Any) -> str | None:
    value = get_value(skeleton, "kind")
    if value is None:
        return None
    return str(value.value) if hasattr(value, "value") else str(value)


def atom_dict(atom: Any) -> dict[str, Any]:
    if isinstance(atom, dict):
        return atom
    if hasattr(atom, "model_dump"):
        return atom.model_dump(mode="json")
    return {
        "name": get_value(atom, "name"),
        "arguments": list(get_value(atom, "arguments", []) or []),
        "negated": get_value(atom, "negated", False),
        "source_phrase": get_value(atom, "source_phrase"),
        "confidence": get_value(atom, "confidence", 1.0),
    }
