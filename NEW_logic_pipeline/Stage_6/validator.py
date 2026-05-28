from __future__ import annotations

from .argument_validator import validate_argument_roles
from .ast_validator import validate_asts
from .predicate_validator import build_effective_registry
from .predicate_validator import validate_predicates
from .registry_schema import normalize_registry_config
from .semantic_validator import validate_semantics
from .solver_readiness import classify_solver_readiness
from .structural_validator import validate_skeletons
from .validation_models import SemanticPolicy, ValidationReport


DEFAULT_SOLVER_CAPABILITIES = {
    "supports_or": False,
    "supports_exists": False,
    "supports_iff": False,
    "supports_meta": False,
    "supports_cardinality": False,
    "supports_classical_negation": True,
}


class Stage6Validator:
    def __init__(
        self,
        predicate_registry: dict | None = None,
        solver_capabilities: dict | None = None,
        semantic_policy: SemanticPolicy | dict | None = None,
    ):
        self.predicate_registry = normalize_registry_config(predicate_registry or {})
        self.solver_capabilities = solver_capabilities or DEFAULT_SOLVER_CAPABILITIES
        self.semantic_policy = SemanticPolicy.from_mapping(semantic_policy)

    def validate(self, parsed_record: dict) -> ValidationReport:
        issues = []
        effective_registry = build_effective_registry(parsed_record, self.predicate_registry)

        issues.extend(validate_skeletons(parsed_record))
        issues.extend(
            validate_predicates(
                parsed_record,
                effective_registry,
                allow_dynamic_predicates=self.semantic_policy.allowed_dynamic_predicates,
            )
        )
        issues.extend(validate_argument_roles(parsed_record, effective_registry))
        issues.extend(validate_semantics(parsed_record, effective_registry, self.semantic_policy))
        issues.extend(
            validate_asts(
                parsed_record,
                effective_registry,
                allow_dynamic_predicates=self.semantic_policy.allowed_dynamic_predicates,
            )
        )

        readiness = classify_solver_readiness(
            parsed_record=parsed_record,
            issues=issues,
            solver_capabilities=self.solver_capabilities,
            registry=effective_registry,
        )
        issues.extend(readiness["issues"])

        summary = {
            "issue_count": len(issues),
            "fatal_count": sum(1 for issue in issues if issue.severity == "fatal"),
            "error_count": sum(1 for issue in issues if issue.severity == "error"),
            "warning_count": sum(1 for issue in issues if issue.severity == "warning"),
            "info_count": sum(1 for issue in issues if issue.severity == "info"),
            "issue_codes": sorted(set(issue.code for issue in issues)),
        }

        return ValidationReport(
            parse_valid=readiness["parse_valid"],
            direct_solver_ready=readiness["direct_solver_ready"],
            needs_lowering=readiness["needs_lowering"],
            needs_meta_resolution=readiness["needs_meta_resolution"],
            needs_review=readiness["needs_review"],
            unsupported=readiness["unsupported"],
            issues=issues,
            readiness_reasons=readiness["reasons"],
            summary=summary,
        )
