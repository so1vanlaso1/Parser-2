# Graph Report - .  (2026-05-22)

## Corpus Check
- 27 files · ~81,758 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 197 nodes · 484 edges · 12 communities (10 shown, 2 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 36 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]

## God Nodes (most connected - your core abstractions)
1. `LogicNode` - 46 edges
2. `LogicValidator` - 29 edges
3. `PipelineConfig` - 22 edges
4. `CompiledPremise` - 21 edges
5. `LogicPipeline` - 19 edges
6. `_make_stage3()` - 18 edges
7. `Stage3Output` - 17 edges
8. `extract_json_object()` - 15 edges
9. `ASTCompiler` - 13 edges
10. `ChatModel` - 12 edges

## Surprising Connections (you probably didn't know these)
- `test_atomic_node_valid()` --calls--> `LogicNode`  [EXTRACTED]
  tests/test_schemas.py → src/schemas.py
- `test_implies_node_valid()` --calls--> `LogicNode`  [EXTRACTED]
  tests/test_schemas.py → src/schemas.py
- `test_not_node_valid()` --calls--> `LogicNode`  [EXTRACTED]
  tests/test_schemas.py → src/schemas.py
- `test_forall_node_valid()` --calls--> `LogicNode`  [EXTRACTED]
  tests/test_schemas.py → src/schemas.py
- `test_exists_node_valid()` --calls--> `LogicNode`  [EXTRACTED]
  tests/test_schemas.py → src/schemas.py

## Communities (12 total, 2 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.12
Nodes (43): CompiledPremise, LogicNode, classify_solver_readiness(), LogicValidator, Python-side validator for Stage 3 AST output.      Catches structural errors tha, Classify a compiled premise into solver-readiness buckets.      Returns: "solver, _make_stage3(), Tests for the Stage 4 LogicValidator. (+35 more)

### Community 1 - "Community 1"
Cohesion: 0.14
Nodes (21): BaseModel, Protocol, main(), Run the full parse pipeline on a single hardcoded example.  Usage:     python sc, PipelineConfig, ChatModel, Generate assistant text from a system/user prompt pair., LogicPipeline (+13 more)

### Community 2 - "Community 2"
Cohesion: 0.07
Nodes (28): Tests for Pydantic schemas — LogicNode shape validation.  After the schema relax, atomic without name — previously crashed, now accepted with warning., atomic without arguments — previously crashed, now accepted with warning., implies with 1 child — previously crashed, now accepted with warning., not with 2 children — previously crashed, now accepted with warning., forall without variable — previously crashed, now accepted with warning., forall with 0 children — previously crashed, now accepted with warning., exists with 2 children — previously crashed, now accepted with warning. (+20 more)

### Community 3 - "Community 3"
Cohesion: 0.16
Nodes (17): Run the full parse pipeline on premises + optional question.          Returns Fu, canonicalize_node(), canonicalize_predicate_name(), canonicalize_question_parse(), canonicalize_stage3(), collect_predicate_names(), QuestionParser, Parses question text + choices into QuestionParse AST. (+9 more)

### Community 4 - "Community 4"
Cohesion: 0.16
Nodes (10): create_chat_model(), HuggingFaceChatModel, OllamaChatModel, Compatibility adapter for the previous Ollama backend., Writes LLM prompts and streamed outputs to terminal and a text file., Shared Hugging Face Transformers chat adapter.      Qwen/Qwen3.5-4B is publish, _torch_dtype(), TraceWriter (+2 more)

### Community 5 - "Community 5"
Cohesion: 0.19
Nodes (8): _extract_balanced_json_object(), extract_json_array_or_object(), extract_json_object(), Extract the first valid JSON object from a model response.     Local models ofte, Extract a JSON object or array from potentially noisy model output., _trim_end_marker(), test_json_extractor_ignores_end_marker(), test_json_extractor_reports_truncated_object()

### Community 6 - "Community 6"
Cohesion: 0.24
Nodes (10): main(), Batch runner: parse every problem in a JSONL file and auto-advance.  Reads premi, extract_choices_from_question(), load_problem(), normalize_whitespace(), Collapse runs of whitespace into single spaces., Extract multiple-choice options from question text.      Handles formats like:, Load a problem from a raw JSONL dict.     Supports both 'premises-NL' (dataset f (+2 more)

### Community 7 - "Community 7"
Cohesion: 0.38
Nodes (3): Structural memory retriever.      Stores CNL -> AST example pairs and retrieves, Format top-k retrieved examples for injection into Stage 3 prompt., StructuralRAG

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Extraction Notice

Semantic extraction for document files was skipped because no LLM API key was configured in this shell. This graph contains deterministic code/AST relationships only.
