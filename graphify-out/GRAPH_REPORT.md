# Graph Report - .  (2026-05-28)

## Corpus Check
- Corpus is ~40,379 words - fits in a single context window. You may not need a graph.

## Summary
- 676 nodes · 1508 edges · 27 communities (25 shown, 2 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 66 edges (avg confidence: 0.69)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Stage 6 Validators|Stage 6 Validators]]
- [[_COMMUNITY_Logic Skeleton AST|Logic Skeleton AST]]
- [[_COMMUNITY_Leaf Atomization Backend|Leaf Atomization Backend]]
- [[_COMMUNITY_Skeleton Builder Rules|Skeleton Builder Rules]]
- [[_COMMUNITY_Runner And Registry|Runner And Registry]]
- [[_COMMUNITY_AST Validation Flow|AST Validation Flow]]
- [[_COMMUNITY_Connector Routing|Connector Routing]]
- [[_COMMUNITY_Predicate Canonicalizer|Predicate Canonicalizer]]
- [[_COMMUNITY_CIR Pipeline Docs|CIR Pipeline Docs]]
- [[_COMMUNITY_Atomization Request Helpers|Atomization Request Helpers]]
- [[_COMMUNITY_Run Summary A|Run Summary A]]
- [[_COMMUNITY_Run Summary B|Run Summary B]]
- [[_COMMUNITY_Run Summary C|Run Summary C]]
- [[_COMMUNITY_Run Summary D|Run Summary D]]
- [[_COMMUNITY_Run Summary E|Run Summary E]]
- [[_COMMUNITY_Readiness Issue Taxonomy|Readiness Issue Taxonomy]]
- [[_COMMUNITY_Parser Repair Plan|Parser Repair Plan]]
- [[_COMMUNITY_Registry Validation|Registry Validation]]
- [[_COMMUNITY_Semantic Checks|Semantic Checks]]
- [[_COMMUNITY_Stage 6 Public API|Stage 6 Public API]]
- [[_COMMUNITY_Disjunction Checks|Disjunction Checks]]
- [[_COMMUNITY_LLM Backend Runbook|LLM Backend Runbook]]
- [[_COMMUNITY_Skeleton Test Cases|Skeleton Test Cases]]
- [[_COMMUNITY_AST Compiler Tests|AST Compiler Tests]]
- [[_COMMUNITY_Phrase Key Utility|Phrase Key Utility]]
- [[_COMMUNITY_Validation Issue Model|Validation Issue Model]]

## God Nodes (most connected - your core abstractions)
1. `get_value()` - 33 edges
2. `LogicSkeleton` - 29 edges
3. `PredicateAtom` - 29 edges
4. `build_skeleton()` - 28 edges
5. `ValidationIssue` - 27 edges
6. `Stage6Validator` - 27 edges
7. `AtomizationResult` - 24 edges
8. `build_ast()` - 22 edges
9. `FormulaSkeleton` - 21 edges
10. `classify_operator()` - 20 edges

## Surprising Connections (you probably didn't know these)
- `Stage 4 Validator Shape Checks` --semantically_similar_to--> `Validate Formula Tree`  [INFERRED] [semantically similar]
  improvement.md → NEW_logic_pipeline/Stage_6/structural_validator.py
- `Verification and Testing Strategy` --rationale_for--> `Stage 6 Validator Test Suite`  [INFERRED]
  improvement.md → NEW_logic_pipeline/tests/test_stage6_validator.py
- `RULE Kind Too Broad Diagnosis` --conceptually_related_to--> `Validate Skeletons`  [INFERRED]
  improvement.md → NEW_logic_pipeline/Stage_6/structural_validator.py
- `Stage 3 AST Compiler Prompts` --conceptually_related_to--> `AST Builder Test Suite`  [INFERRED]
  improvement.md → NEW_logic_pipeline/tests/test_ast_builder.py
- `Local Model Backends Test Suite` --conceptually_related_to--> `Qwen Hugging Face Backend`  [INFERRED]
  NEW_logic_pipeline/tests/test_local_model_backends.py → haha.md

## Hyperedges (group relationships)
- **Stage12 Parse Pipeline Flow** — skeleton_builder_build_skeletons, atomization_requests_collect_batch_atomization_requests, leaf_atomizer_atomize_requests, predicate_canonicalizer_canonicalize_atomization_results, ast_builder_build_asts, run_stage6_validation [EXTRACTED 1.00]
- **Stage 1 Skeleton Routing** — connector_registry_ConnectorRegistry, operator_router_classify_operator, skeleton_builder_build_skeleton, logic_skeleton_LogicSkeleton [EXTRACTED 1.00]
- **Stage 2 Atomization Flow** — atomization_requests_AtomizationRequest, leaf_atomizer_build_atomizer_prompt, model_backends_LocalTransformersLLM, atomization_requests_AtomizationResult, predicate_canonicalizer_canonicalize_atomization_results [INFERRED 0.95]
- **Stage 6 Validation Pipeline** — validator_Stage6Validator, validator_validate, structural_validator_validate_skeletons, predicate_validator_validate_predicates, semantic_validator_validate_semantics, external_validate_argument_roles, external_validate_asts, solver_readiness_classify_solver_readiness, validation_models_ValidationReport [EXTRACTED 1.00]
- **Semantic Preservation Checks** — semantic_validator_or_preservation, semantic_validator_source_mentions, semantic_validator_domain_restrictions, semantic_validator_numeric_typing, semantic_validator_modal_not_necessarily, semantic_validator_deontic_unresolved, validation_models_SemanticPolicy [EXTRACTED 1.00]
- **Parser Repair Strategy** — improvement_rule_kind_too_broad, improvement_relax_pydantic_schema_validation, improvement_stage4_validator_shape_checks, improvement_stage1_cnl_heuristics, improvement_stage3_ast_compiler_prompts, improvement_verification_testing_strategy [EXTRACTED 1.00]
- **CIR-Only Compiler Flow** — neurosymbolic_logic_parser_full_implementation_guide_cir_boundary, next_cir_only_parser_migration, next_stage3_qwen_to_python_cir, next_targeted_cir_repair [EXTRACTED 1.00]
- **Validation Safety System** — stage6_deterministic_quality_gate, stage6_predicate_registry_schema, stage6_solver_readiness_classifier, semantic_policy_constraints, education_registry_predicate_registry [INFERRED 0.85]
- **Parser Pipeline Documentation Cluster** — neurosymbolic_logic_parser_full_implementation_guide_stage_pipeline, summary_compiler_style_pipeline, walkthrough_parser_only_pipeline, next_cir_only_parser_migration [INFERRED 0.85]

## Communities (27 total, 2 thin omitted)

### Community 0 - "Stage 6 Validators"
Cohesion: 0.10
Nodes (57): collect_type_constraints(), _infer_kind(), normalize_argument(), validate_argument_roles(), validate_type_unification(), _bad_child_count(), _check_phrase_consistency(), _main_property_predicates() (+49 more)

### Community 1 - "Logic Skeleton AST"
Cohesion: 0.07
Nodes (61): BaseModel, LogicSkeleton, MatchedEvidence, A raw English phrase span that later stages atomize.      `negation_hint` is onl, Debug evidence explaining why the router selected a skeleton kind., The main Stage 1 output.      The object is deliberately not solver-ready. It on, SkeletonBuildResult, TextSpan (+53 more)

### Community 2 - "Leaf Atomization Backend"
Cohesion: 0.07
Nodes (51): Public package exports for NEW_logic_pipeline., AtomizationRequest, is_formula_like_leaf(), Stage 2 leaf extraction and atomization helpers., _apply_request_level_checks(), atomize_request(), atomize_requests(), build_atomizer_prompt() (+43 more)

### Community 3 - "Skeleton Builder Rules"
Cohesion: 0.11
Nodes (52): FormulaSkeleton, Recursive text-only formula skeleton for META/nested premises., _append_unique(), _base_skeleton(), build_meta_formula_tree(), build_skeleton(), _clean_part(), detect_modality_hint() (+44 more)

### Community 4 - "Runner And Registry"
Cohesion: 0.07
Nodes (41): build_local_config(), _configure_stdout(), get_premises(), load_jsonl_rows(), main(), make_run_dir(), parse_args(), parse_row() (+33 more)

### Community 5 - "AST Validation Flow"
Cohesion: 0.06
Nodes (50): Type Constraint Unification, validate_argument_roles, LogicNode, build_ast, build_asts, Formula Tree to AST Conversion, validate_ast_node, validate_asts (+42 more)

### Community 6 - "Connector Routing"
Cohesion: 0.10
Nodes (36): _clean_part(), ConnectorEntry, ConnectorMatch, ConnectorRegistry, _cue_pattern(), _cue_span_from_match(), default_connector_registry(), _default_split_pattern() (+28 more)

### Community 7 - "Predicate Canonicalizer"
Cohesion: 0.17
Nodes (31): PredicateAtom, _argument_key(), _argument_text(), _ascii_fold(), _canonical_argument(), _canonical_count(), _canonical_grade(), _canonicalize_atom() (+23 more)

### Community 8 - "CIR Pipeline Docs"
Cohesion: 0.08
Nodes (30): Grade Group And Grade Count Lowering, Education Predicate Registry, Empty General Logic Registry, CIR-Only Semantic Boundary, Multi-Stage Neurosymbolic Logic Parser Implementation Guide, LLM Python Solver Responsibility Split, LogicNode AST, Stage 0 To Solver Pipeline (+22 more)

### Community 9 - "Atomization Request Helpers"
Cohesion: 0.16
Nodes (28): _article_mentions(), _canonical_surface(), _clean_phrase(), _clean_text(), _detect_modality_hint(), _detect_negation_hint(), _first_flag_with_prefix(), _formula_request_id() (+20 more)

### Community 10 - "Run Summary A"
Cohesion: 0.10
Nodes (20): artifacts_dir, finished_at, input, llm_call_count, model_config, device_map, max_new_tokens, minicpm_model_id (+12 more)

### Community 11 - "Run Summary B"
Cohesion: 0.10
Nodes (20): artifacts_dir, finished_at, input, llm_call_count, model_config, device_map, max_new_tokens, minicpm_model_id (+12 more)

### Community 12 - "Run Summary C"
Cohesion: 0.10
Nodes (20): artifacts_dir, finished_at, input, llm_call_count, model_config, device_map, max_new_tokens, minicpm_model_id (+12 more)

### Community 13 - "Run Summary D"
Cohesion: 0.10
Nodes (20): artifacts_dir, finished_at, input, llm_call_count, model_config, device_map, max_new_tokens, minicpm_model_id (+12 more)

### Community 14 - "Run Summary E"
Cohesion: 0.10
Nodes (20): artifacts_dir, finished_at, input, llm_call_count, model_config, device_map, max_new_tokens, minicpm_model_id (+12 more)

### Community 15 - "Readiness Issue Taxonomy"
Cohesion: 0.20
Nodes (11): Stage 6 Issue Code Taxonomy, Iter Atoms, Validate Predicates, Any Atom Requires Lowering, AST Contains, Atomization Unsupported Results, Classify Solver Readiness, Direct Solver Readiness (+3 more)

### Community 16 - "Parser Repair Plan"
Cohesion: 0.22
Nodes (10): Parser Improvement Plan, Relax Pydantic Schema Validation, RULE Kind Too Broad Diagnosis, Stage 1 CNL Heuristics, Stage 4 Validator Shape Checks, Validate Formula Tree, Validate Skeletons, Relative Clause Probe (+2 more)

### Community 17 - "Registry Validation"
Cohesion: 0.25
Nodes (9): Verification and Testing Strategy, Incompatible Role Sets, Load Registry Config, Normalize Registry Config, Validate Registry Shape, Stage 6 Validator Test Suite, SemanticPolicy, Stage6Validator (+1 more)

### Community 18 - "Semantic Checks"
Cohesion: 0.29
Nodes (7): Atom Equivalent Present, Deontic Unresolved Check, Domain Restriction Preservation Check, Modal Not Necessarily Check, Numeric Argument Typing Check, Source Mention Preservation Check, Validate Semantics

### Community 19 - "Stage 6 Public API"
Cohesion: 0.29
Nodes (7): Validate Argument Roles, Stage 6 Public API, Build Effective Registry, Dynamic Predicate Vocabulary, Predicate Canonicalizer Test Suite, ValidationReport, Stage6Validator.validate

### Community 20 - "Disjunction Checks"
Cohesion: 0.33
Nodes (6): Atom Encodes Disjunction, OR Preservation Check, Request Has Logical Cue, Formula Like Leaf Detection, Atomization Requests Test Suite, Leaf Atomizer Test Suite

### Community 21 - "LLM Backend Runbook"
Cohesion: 0.50
Nodes (4): Ollama Fallback Backend, Qwen Hugging Face Backend, Parser Setup and Runbook, Local Model Backends Test Suite

### Community 22 - "Skeleton Test Cases"
Cohesion: 0.83
Nodes (3): _assert_node_contains(), _contains_type(), test_100_more_skeleton_cases()

### Community 23 - "AST Compiler Tests"
Cohesion: 0.67
Nodes (3): Validate ASTs, Stage 3 AST Compiler Prompts, AST Builder Test Suite

## Ambiguous Edges - Review These
- `CIR-Only Parser Migration` → `Parser-2 README`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to

## Knowledge Gaps
- **126 isolated node(s):** `started_at`, `input`, `artifacts_dir`, `mode`, `minicpm_model_id` (+121 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `CIR-Only Parser Migration` and `Parser-2 README`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `ValidationIssue` connect `Stage 6 Validators` to `Leaf Atomization Backend`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `Stage6Validator` connect `Runner And Registry` to `Stage 6 Validators`, `Logic Skeleton AST`, `Leaf Atomization Backend`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Why does `build_skeleton()` connect `Skeleton Builder Rules` to `Logic Skeleton AST`, `Leaf Atomization Backend`, `Connector Routing`, `Skeleton Test Cases`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `LogicSkeleton` (e.g. with `PredicateAtom` and `AtomizationRequest`) actually correct?**
  _`LogicSkeleton` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `PredicateAtom` (e.g. with `FormulaSkeleton` and `LogicSkeleton`) actually correct?**
  _`PredicateAtom` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Log every LLM prompt/response while preserving the generate(prompt) API.`, `started_at`, `input` to the rest of the system?**
  _170 weakly-connected nodes found - possible documentation gaps or missing edges._