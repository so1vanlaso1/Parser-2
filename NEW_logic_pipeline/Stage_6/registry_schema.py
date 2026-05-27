from __future__ import annotations

"""Schema helpers for externally supplied predicate registries."""

import json
from pathlib import Path
from typing import Any


REQUIRED_PREDICATE_FIELDS = {
    "arity": int,
    "roles": list,
    "solver_safe": bool,
}

RESERVED_REGISTRY_KEYS = {
    "__argument_values__",
    "__incompatible_role_sets__",
}


def load_registry_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML registry config and return normalized registry data.

    JSON is parsed directly. YAML is supported when PyYAML is installed. JSON
    syntax is valid YAML, so config files can keep a ``.yaml`` extension without
    forcing PyYAML as a runtime dependency.
    """

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _load_yaml(text, config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Registry config must be an object: {config_path}")
    return normalize_registry_config(payload)


def normalize_registry_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Accept either a flat registry or a config with a top-level predicates map."""

    if not config:
        return {}

    source = dict(config)
    predicates = source.pop("predicates", None)
    if predicates is None:
        registry = dict(source)
    else:
        if not isinstance(predicates, dict):
            raise ValueError("Registry field 'predicates' must be an object.")
        registry = dict(predicates)
        for key in RESERVED_REGISTRY_KEYS:
            if key in source:
                registry[key] = source[key]

    validate_registry_shape(registry)
    return registry


def validate_registry_shape(registry: dict[str, Any]) -> None:
    for name, metadata in registry.items():
        if name in RESERVED_REGISTRY_KEYS:
            continue
        if not isinstance(metadata, dict):
            raise ValueError(f"Registry entry {name!r} must be an object.")
        for field_name, field_type in REQUIRED_PREDICATE_FIELDS.items():
            if field_name not in metadata:
                raise ValueError(f"Registry entry {name!r} missing {field_name!r}.")
            if not isinstance(metadata[field_name], field_type):
                expected = field_type.__name__
                raise ValueError(f"Registry entry {name!r}.{field_name} must be {expected}.")
        arity = metadata["arity"]
        roles = metadata["roles"]
        if len(roles) != arity:
            raise ValueError(f"Registry entry {name!r} has arity {arity}, but {len(roles)} role(s).")


def incompatible_role_sets(registry: dict[str, Any]) -> list[set[str]]:
    raw_sets = registry.get("__incompatible_role_sets__", []) or []
    output: list[set[str]] = []
    for raw in raw_sets:
        if isinstance(raw, (list, tuple, set)):
            values = {str(item) for item in raw if str(item)}
            if len(values) >= 2:
                output.append(values)
    return output


def _load_yaml(text: str, path: Path) -> Any:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise ValueError(
            f"Registry config is not JSON and PyYAML is not installed: {path}"
        ) from exc
    return yaml.safe_load(text)


__all__ = [
    "REQUIRED_PREDICATE_FIELDS",
    "RESERVED_REGISTRY_KEYS",
    "incompatible_role_sets",
    "load_registry_config",
    "normalize_registry_config",
    "validate_registry_shape",
]
