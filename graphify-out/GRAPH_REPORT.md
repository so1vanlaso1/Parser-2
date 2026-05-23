# Graph Report - C:/Users/hi/Desktop/Praser  (2026-05-22)

## Corpus Check
- 28 files · ~88,689 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 269 nodes · 627 edges · 16 communities (14 shown, 2 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 49 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Should Node|Should Node]]
- [[_COMMUNITY_Parse Init|Parse Init  ]]
- [[_COMMUNITY_Previously Crashed|Previously Crashed]]
- [[_COMMUNITY_Astcompiler Logicpipeline|Astcompiler Logicpipeline]]
- [[_COMMUNITY_Basemodel Check Shape|Basemodel Check Shape]]
- [[_COMMUNITY_Astcompiler Atom To Node|Astcompiler Atom To Node]]
- [[_COMMUNITY_Parse Full|Parse Full]]
- [[_COMMUNITY_Init   Validator|Init   Validator]]
- [[_COMMUNITY_Init   Generate|Init   Generate]]
- [[_COMMUNITY_Problem Jsonl|Problem Jsonl]]
- [[_COMMUNITY_Structural Memory|Structural Memory]]
- [[_COMMUNITY_Chatmodel Huggingfacechatmodel|Chatmodel Huggingfacechatmodel]]
- [[_COMMUNITY_Hugging Face|Hugging Face]]
- [[_COMMUNITY_Lenient Shape|Lenient Shape]]

## God Nodes (most connected - your core abstractions)
1. `LogicNode` - 53 edges
2. `ASTCompiler` - 36 edges
3. `LogicValidator` - 29 edges
4. `PipelineConfig` - 26 edges
5. `CompiledPremise` - 24 edges
6. `LogicPipeline` - 19 edges
7. `Stage3Output` - 18 edges
8. `_make_stage3()` - 18 edges
9. `extract_json_object()` - 16 edges
10. `Stage1Output` - 16 edges

## Surprising Connections (you probably didn't know these)
- `FakeRAG` --uses--> `PipelineConfig`  [INFERRED]
  tests/test_stage3_frame_compiler.py → src/config.py
- `FakeLLM` --uses--> `PipelineConfig`  [INFERRED]
  tests/test_stage3_frame_compiler.py → src/config.py
- `test_json_extractor_reports_truncated_object()` --calls--> `extract_json_object()`  [EXTRACTED]
  tests/test_pipeline_fixes.py → src/json_utils.py
- `test_json_extractor_ignores_end_marker()` --calls--> `extract_json_object()`  [EXTRACTED]
  tests/test_pipeline_fixes.py → src/json_utils.py
- `FakeRAG` --uses--> `ASTCompiler`  [INFERRED]
  tests/test_stage3_frame_compiler.py → src/stage3_ast.py

## Hyperedges (group relationships)
- **h1_pipeline_orchestration** — pipeline.LogicPipeline.parse, stage1_cnl.CNLRewriter.rewrite, stage3_ast.ASTCompiler.compile, stage4_validate.LogicValidator.validate_stage3, question_parser.QuestionParser.parse [INFERRED 1.00]
- **h2_json_transduction_stack** — json_utils.extract_json_object, stage1_cnl.CNLRewriter.rewrite, stage3_ast.ASTCompiler._extract_frames, stage3_ast.ASTCompiler._compile_full_ast, question_parser.QuestionParser.parse [INFERRED 1.00]
- **h3_model_backend_selection** — config.PipelineConfig, llm_client.create_chat_model, llm_client.HuggingFaceChatModel, llm_client.OllamaChatModel, pipeline.LogicPipeline.__init__ [INFERRED 1.00]
- **h.validation_repair_loop** —  [INFERRED 1.00]
- **h.pipeline_documentation_cluster** —  [INFERRED 1.00]
- **h.stage3_to_stage5_feedback** —  [INFERRED 0.75]

## Communities (16 total, 2 thin omitted)

### Community 0 - "Should Node"
Cohesion: 0.08
Nodes (47): CompiledPremise, Stage3Output, classify_solver_readiness(), LogicValidator, Python-side validator for Stage 3 AST output.      Catches structural errors tha, Check if an implies node contains another implies node in its subtree., Classify a compiled premise into solver-readiness buckets.      Returns: "solver, ValidationIssue (+39 more)

### Community 1 - "Parse Init  "
Cohesion: 0.10
Nodes (28): Protocol, main(), Batch runner: parse every problem in a JSONL file and auto-advance.  Reads premi, main(), Run the full parse pipeline on a single hardcoded example.  Usage:     python sc, PipelineConfig, _extract_balanced_json_object(), extract_json_array_or_object() (+20 more)

### Community 2 - "Previously Crashed"
Cohesion: 0.11
Nodes (29): LogicNode, Tests for Pydantic schemas — LogicNode shape validation.  After the schema relax, atomic without name — previously crashed, now accepted with warning., atomic without arguments — previously crashed, now accepted with warning., implies with 1 child — previously crashed, now accepted with warning., not with 2 children — previously crashed, now accepted with warning., forall without variable — previously crashed, now accepted with warning., forall with 0 children — previously crashed, now accepted with warning. (+21 more)

### Community 3 - "Astcompiler Logicpipeline"
Cohesion: 0.13
Nodes (21): PipelineConfig, extract_json_object, create_chat_model, LogicPipeline.__init__, LogicPipeline.parse, canonicalize_question_parse, canonicalize_stage3, collect_predicate_names (+13 more)

### Community 4 - "Basemodel Check Shape"
Cohesion: 0.21
Nodes (14): BaseModel, CNLStatement, FullParseResult, PredicateAtom, PredicateFrame, PredicateFrameOutput, PredicateGroup, RiskFlag (+6 more)

### Community 6 - "Parse Full"
Cohesion: 0.18
Nodes (14): Run the full parse pipeline on premises + optional question.          Returns Fu, canonicalize_node(), canonicalize_predicate_name(), canonicalize_question_parse(), canonicalize_stage3(), collect_predicate_names(), has_explicit_numeric_condition(), remove_false_numeric_flags() (+6 more)

### Community 7 - "Init   Validator"
Cohesion: 0.16
Nodes (17): __init__.py, stage5_repair.py, test_pipeline_fixes.py, test_schemas.py, test_stage3_frame_compiler.py, test_validator.py, tests/__init__.py, Frame-based AST compilation (+9 more)

### Community 8 - "Init   Generate"
Cohesion: 0.24
Nodes (6): HuggingFaceChatModel, OllamaChatModel, Compatibility adapter for the previous Ollama backend., Writes LLM prompts and streamed outputs to terminal and a text file., Shared Hugging Face Transformers chat adapter.      Qwen/Qwen3.5-4B is publish, TraceWriter

### Community 9 - "Problem Jsonl"
Cohesion: 0.33
Nodes (8): extract_choices_from_question(), load_problem(), normalize_whitespace(), Collapse runs of whitespace into single spaces., Extract multiple-choice options from question text.      Handles formats like:, Load a problem from a raw JSONL dict.     Supports both 'premises-NL' (dataset f, Represents one problem from the JSONL dataset.     Adapted for the fixed_smoke_l, RawLogicProblem

### Community 10 - "Structural Memory"
Cohesion: 0.38
Nodes (3): Structural memory retriever.      Stores CNL -> AST example pairs and retrieves, Format top-k retrieved examples for injection into Stage 3 prompt., StructuralRAG

### Community 11 - "Chatmodel Huggingfacechatmodel"
Cohesion: 0.67
Nodes (4): ChatModel, HuggingFaceChatModel, OllamaChatModel, TraceWriter

### Community 12 - "Hugging Face"
Cohesion: 1.00
Nodes (3): Hugging Face Qwen/Qwen3.5-4B backend, haha.md, requirements.txt

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LogicNode` connect `Previously Crashed` to `Should Node`, `Parse Init  `, `Basemodel Check Shape`, `Astcompiler Atom To Node`, `Parse Full`?**
  _High betweenness centrality (0.235) - this node is a cross-community bridge._
- **Why does `ASTCompiler` connect `Astcompiler Atom To Node` to `Should Node`, `Parse Init  `, `Previously Crashed`, `Basemodel Check Shape`, `Structural Memory`?**
  _High betweenness centrality (0.126) - this node is a cross-community bridge._
- **Why does `PipelineConfig` connect `Parse Init  ` to `Init   Generate`, `Basemodel Check Shape`, `Astcompiler Atom To Node`, `Parse Full`?**
  _High betweenness centrality (0.109) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `LogicNode` (e.g. with `ASTCompiler` and `ValidationIssue`) actually correct?**
  _`LogicNode` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `ASTCompiler` (e.g. with `LogicPipeline` and `PipelineConfig`) actually correct?**
  _`ASTCompiler` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `LogicValidator` (e.g. with `LogicPipeline` and `LogicNode`) actually correct?**
  _`LogicValidator` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `PipelineConfig` (e.g. with `ChatModel` and `TraceWriter`) actually correct?**
  _`PipelineConfig` has 11 INFERRED edges - model-reasoned connections that need verification._