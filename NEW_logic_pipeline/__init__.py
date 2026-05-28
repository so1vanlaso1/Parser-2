"""Public package exports for NEW_logic_pipeline."""
from __future__ import annotations

import sys

from .Stage_1 import connector_registry as _connector_registry
from .Stage_1 import logic_skeleton as _logic_skeleton
from .Stage_1 import operator_router as _operator_router
from .Stage_1 import skeleton_builder as _skeleton_builder
from .Stage_1.logic_skeleton import FormulaSkeleton, LogicSkeleton, MatchedEvidence, TextSpan
from .Stage_1.skeleton_builder import build_skeleton, build_skeletons
from .Stage_2 import atomization_requests as _atomization_requests
from .Stage_2 import leaf_atomizer as _leaf_atomizer
from .Stage_2 import local_model_config as _local_model_config
from .Stage_2 import model_backends as _model_backends
from .Stage_2 import predicate_canonicalizer as _predicate_canonicalizer
from .Stage_2.atomization_requests import (
    AtomizationRequest,
    AtomizationResult,
    PredicateAtom,
    collect_atomization_requests,
    collect_batch_atomization_requests,
    collect_formula_leaf_requests,
)
from .Stage_2.leaf_atomizer import (
    PhraseConsistencyCache,
    atomize_request,
    atomize_requests,
    build_atomizer_prompt,
    parse_atomizer_response,
    sanitize_raw_atom_payload,
    validate_atomization_result,
)
from .Stage_2.local_model_config import get_local_transformers_config
from .Stage_2.model_backends import LocalTransformersConfig, LocalTransformersLLM, ModelMode, create_local_llm
from .Stage_2.predicate_canonicalizer import (
    CANONICAL_PREDICATES,
    DEFAULT_KNOWN_PREDICATES,
    canonicalize_atomization_results,
)
from .Stage_5 import ast_builder as _ast_builder
from .Stage_5.ast_builder import LogicNode, build_ast, build_asts
from .Stage_6 import SemanticPolicy, Stage6Validator, ValidationIssue, ValidationReport

# Compatibility aliases for tests and callers that imported Stage 1 modules
# before they were moved under NEW_logic_pipeline.Stage_1.
sys.modules.setdefault(__name__ + ".connector_registry", _connector_registry)
sys.modules.setdefault(__name__ + ".logic_skeleton", _logic_skeleton)
sys.modules.setdefault(__name__ + ".operator_router", _operator_router)
sys.modules.setdefault(__name__ + ".skeleton_builder", _skeleton_builder)
sys.modules.setdefault(__name__ + ".atomization_requests", _atomization_requests)
sys.modules.setdefault(__name__ + ".leaf_atomizer", _leaf_atomizer)
sys.modules.setdefault(__name__ + ".local_model_config", _local_model_config)
sys.modules.setdefault(__name__ + ".model_backends", _model_backends)
sys.modules.setdefault(__name__ + ".predicate_canonicalizer", _predicate_canonicalizer)
sys.modules.setdefault(__name__ + ".ast_builder", _ast_builder)

__all__ = [
    "build_skeleton",
    "build_skeletons",
    "LogicSkeleton",
    "FormulaSkeleton",
    "TextSpan",
    "MatchedEvidence",
    "PredicateAtom",
    "AtomizationRequest",
    "AtomizationResult",
    "collect_atomization_requests",
    "collect_batch_atomization_requests",
    "collect_formula_leaf_requests",
    "build_atomizer_prompt",
    "parse_atomizer_response",
    "sanitize_raw_atom_payload",
    "validate_atomization_result",
    "atomize_request",
    "atomize_requests",
    "PhraseConsistencyCache",
    "ModelMode",
    "LocalTransformersConfig",
    "LocalTransformersLLM",
    "create_local_llm",
    "get_local_transformers_config",
    "CANONICAL_PREDICATES",
    "DEFAULT_KNOWN_PREDICATES",
    "canonicalize_atomization_results",
    "LogicNode",
    "build_ast",
    "build_asts",
    "Stage6Validator",
    "SemanticPolicy",
    "ValidationIssue",
    "ValidationReport",
]
