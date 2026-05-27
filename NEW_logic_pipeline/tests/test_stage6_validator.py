from pathlib import Path

from NEW_logic_pipeline.Stage_6.registry_schema import load_registry_config
from NEW_logic_pipeline.Stage_6.validator import Stage6Validator


ROOT = Path(__file__).resolve().parents[2]
EDUCATION_REGISTRY = load_registry_config(ROOT / "configs" / "registries" / "education_registry.yaml")


def test_formula_like_leaf_rejected():
    parsed = {
        "skeletons": [
            {
                "premise_id": "P1",
                "kind": "META",
                "formula_tree": {
                    "type": "implies",
                    "children": [
                        {"type": "leaf", "text": "there exists at least one student", "children": []},
                        {"type": "leaf", "text": "if a student asks questions then they attend tutorials", "children": []},
                    ],
                },
            }
        ],
        "atomization_results": [],
    }

    report = Stage6Validator(predicate_registry=EDUCATION_REGISTRY).validate(parsed)

    assert not report.parse_valid
    assert "STRUCT_FORMULA_LIKE_LEAF" in report.summary["issue_codes"]


def test_unknown_predicate_rejected():
    parsed = {
        "skeletons": [],
        "atomization_results": [
            {
                "premise_id": "P1",
                "request_id": "P1_body",
                "phrase": "is happy",
                "atoms": [{"name": "happy", "arguments": ["x"], "negated": False}],
            }
        ],
    }

    report = Stage6Validator(predicate_registry=EDUCATION_REGISTRY).validate(parsed)

    assert not report.parse_valid
    assert "PRED_UNKNOWN" in report.summary["issue_codes"]


def test_or_became_and_rejected_from_logical_cue_metadata():
    parsed = {
        "skeletons": [],
        "atomization_requests": [
            {
                "premise_id": "P1",
                "request_id": "P1_consequent",
                "phrase": "they can earn an A or A+",
                "role": "consequent",
                "logical_cues": ["or"],
            }
        ],
        "atomization_results": [
            {
                "premise_id": "P1",
                "request_id": "P1_consequent",
                "phrase": "they can earn an A or A+",
                "atoms": [
                    {"name": "earned_grade", "arguments": ["x", "a"], "negated": False},
                    {"name": "earned_grade", "arguments": ["x", "a_plus"], "negated": False},
                ],
            }
        ],
    }

    report = Stage6Validator(predicate_registry=EDUCATION_REGISTRY).validate(parsed)

    assert not report.parse_valid
    assert "SEMANTIC_OR_BECAME_AND" in report.summary["issue_codes"]


def test_disjunction_group_comes_from_registry_argument_metadata():
    registry = {
        "status": {"arity": 2, "roles": ["actor", "status_group"], "solver_safe": True},
        "notify": {"arity": 1, "roles": ["actor"], "solver_safe": True},
        "__argument_values__": {
            "active_or_pending": {"role": "status_group", "encodes_disjunction": True}
        },
    }
    parsed = {
        "skeletons": [],
        "atomization_requests": [
            {
                "premise_id": "P1",
                "request_id": "P1_body",
                "phrase": "entity is active or pending and receives notice",
                "logical_cues": ["or"],
            }
        ],
        "atomization_results": [
            {
                "premise_id": "P1",
                "request_id": "P1_body",
                "phrase": "entity is active or pending and receives notice",
                "atoms": [
                    {"name": "status", "arguments": ["x", "active_or_pending"], "negated": False},
                    {"name": "notify", "arguments": ["x"], "negated": False},
                ],
            }
        ],
    }

    report = Stage6Validator(predicate_registry=registry).validate(parsed)

    assert "SEMANTIC_OR_BECAME_AND" not in report.summary["issue_codes"]


def test_same_value_with_incompatible_roles_is_rejected_by_type_unification():
    registry = {
        "uses_resource": {"arity": 2, "roles": ["actor", "resource"], "solver_safe": True},
        "__incompatible_role_sets__": [["actor", "resource"]],
    }
    parsed = {
        "skeletons": [],
        "atomization_results": [
            {
                "premise_id": "P1",
                "request_id": "P1_antecedent",
                "phrase": "operator uses a resource",
                "atoms": [{"name": "uses_resource", "arguments": ["x", "x"], "negated": False}],
            }
        ],
    }

    report = Stage6Validator(predicate_registry=registry).validate(parsed)

    assert not report.parse_valid
    assert "ARG_ROLE_MISMATCH" in report.summary["issue_codes"]


def test_typed_argument_role_mismatch_rejected():
    registry = {
        "uses_resource": {"arity": 2, "roles": ["actor", "resource"], "solver_safe": True},
    }
    parsed = {
        "skeletons": [],
        "atomization_results": [
            {
                "premise_id": "P1",
                "request_id": "P1_antecedent",
                "phrase": "operator uses a resource",
                "atoms": [
                    {
                        "name": "uses_resource",
                        "arguments": [
                            {"value": "x", "kind": "variable", "semantic_type": "actor"},
                            {"value": "x", "kind": "variable", "semantic_type": "actor"},
                        ],
                        "negated": False,
                    }
                ],
            }
        ],
    }

    report = Stage6Validator(predicate_registry=registry).validate(parsed)

    assert not report.parse_valid
    assert "ARG_ROLE_MISMATCH" in report.summary["issue_codes"]


def test_required_domain_atom_metadata_drives_domain_restriction_check():
    registry = {
        "applicant": {"arity": 1, "roles": ["person"], "solver_safe": True},
        "eligible": {"arity": 1, "roles": ["person"], "solver_safe": True},
    }
    parsed = {
        "skeletons": [],
        "atomization_requests": [
            {
                "premise_id": "P1",
                "request_id": "P1_antecedent",
                "phrase": "an applicant is eligible",
                "required_domain_atoms": [
                    {"predicate": "applicant", "arguments": ["x"], "negated": False}
                ],
            }
        ],
        "atomization_results": [
            {
                "premise_id": "P1",
                "request_id": "P1_antecedent",
                "phrase": "an applicant is eligible",
                "atoms": [{"name": "eligible", "arguments": ["x"], "negated": False}],
            }
        ],
    }

    report = Stage6Validator(predicate_registry=registry).validate(parsed)

    assert "SEMANTIC_DOMAIN_RESTRICTION_LOST" in report.summary["issue_codes"]


def test_source_mention_preservation_uses_evidence_links():
    registry = {
        "read_policy": {"arity": 1, "roles": ["actor"], "solver_safe": True},
    }
    parsed = {
        "skeletons": [],
        "atomization_requests": [
            {
                "premise_id": "P1",
                "request_id": "P1_body",
                "phrase": "read the policy",
                "source_mentions": [
                    {
                        "id": "m_obj_1",
                        "surface": "policy",
                        "semantic_role": "object",
                        "canonical": "policy",
                    }
                ],
            }
        ],
        "atomization_results": [
            {
                "premise_id": "P1",
                "request_id": "P1_body",
                "phrase": "read the policy",
                "atoms": [{"name": "read_policy", "arguments": ["x"], "negated": False}],
            }
        ],
    }

    report = Stage6Validator(predicate_registry=registry).validate(parsed)

    assert "SEMANTIC_SOURCE_MENTION_DROPPED" in report.summary["issue_codes"]


def test_registry_metadata_controls_lowering():
    registry = {
        "age_at_least": {
            "arity": 2,
            "roles": ["person", "count"],
            "solver_safe": False,
            "requires_lowering": True,
        }
    }
    parsed = {
        "skeletons": [],
        "atomization_results": [
            {
                "premise_id": "P1",
                "request_id": "P1_body",
                "phrase": "person is at least 18",
                "atoms": [{"name": "age_at_least", "arguments": ["x", "18"], "negated": False}],
            }
        ],
    }

    report = Stage6Validator(predicate_registry=registry).validate(parsed)

    assert report.parse_valid
    assert not report.direct_solver_ready
    assert report.needs_lowering
    assert "READY_CARDINALITY_REQUIRES_LOWERING" in report.summary["issue_codes"]


def test_missing_ast_blocks_direct_solver_readiness_without_invalidating_parse():
    parsed = {
        "skeletons": [],
        "atomization_results": [
            {
                "premise_id": "P1",
                "request_id": "P1_body",
                "phrase": "student",
                "atoms": [{"name": "student", "arguments": ["x"], "negated": False}],
            }
        ],
    }

    report = Stage6Validator(predicate_registry=EDUCATION_REGISTRY).validate(parsed)

    assert report.parse_valid
    assert not report.direct_solver_ready
    assert "AST_MISSING" in report.summary["issue_codes"]
    assert "ast_missing" in report.readiness_reasons


def test_stage6_code_does_not_embed_education_domain_words():
    forbidden = {
        "a_or_a_plus",
        "grade_count",
        "grade_count_at_least",
        "material",
        "tutorials",
        "exam",
        "student",
        "tuan",
        "laura",
        "mastered_subject",
        "scholarship",
    }
    stage6_dir = ROOT / "NEW_logic_pipeline" / "Stage_6"
    source = "\n".join(path.read_text(encoding="utf-8").lower() for path in stage6_dir.glob("*.py"))

    assert not {word for word in forbidden if word in source}
