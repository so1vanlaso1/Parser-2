from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


Severity = Literal["info", "warning", "error", "fatal"]


@dataclass
class SemanticPolicy:
    logical_cue_words: dict[str, list[str]] = field(
        default_factory=lambda: {
            "and": ["and"],
            "or": ["or"],
            "negation": ["not", "no", "never", "cannot", "without"],
        }
    )
    allowed_dynamic_predicates: bool = False
    require_evidence_links: bool = True
    require_domain_restrictions: bool = True
    disjunction_requires_or_or_group: bool = True
    numeric_arguments_must_be_strings: bool = True
    important_source_mention_roles: set[str] = field(
        default_factory=lambda: {
            "domain_type",
            "object",
            "object_type",
            "quantity",
            "grade",
            "entity",
        }
    )

    @classmethod
    def from_mapping(cls, value: "SemanticPolicy | dict[str, Any] | None") -> "SemanticPolicy":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        data = dict(value)
        if "important_source_mention_roles" in data:
            data["important_source_mention_roles"] = set(data["important_source_mention_roles"] or [])
        return cls(**data)

    @classmethod
    def from_file(cls, path: str | Path) -> "SemanticPolicy":
        config_path = Path(path)
        text = config_path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = _load_yaml(text, config_path)
        if not isinstance(payload, dict):
            raise ValueError(f"Semantic policy config must be an object: {config_path}")
        return cls.from_mapping(payload)


@dataclass
class ValidationIssue:
    code: str
    severity: Severity
    message: str
    premise_id: str | None = None
    request_id: str | None = None
    path: list[int] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    suggested_stage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    parse_valid: bool
    direct_solver_ready: bool
    needs_lowering: bool
    needs_meta_resolution: bool
    needs_review: bool
    unsupported: bool
    issues: list[ValidationIssue]
    readiness_reasons: list[str]
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["issues"] = [issue.to_dict() for issue in self.issues]
        return data


def _load_yaml(text: str, path: Path) -> Any:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise ValueError(
            f"Semantic policy config is not JSON and PyYAML is not installed: {path}"
        ) from exc
    return yaml.safe_load(text)
