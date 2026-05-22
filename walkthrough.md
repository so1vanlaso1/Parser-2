# Walkthrough: Neurosymbolic Logic Parser Pipeline

## What Was Built

A complete **parser-only** neurosymbolic pipeline that takes `Premise-NL + Question` and produces validated `LogicNode AST` — ready for a future solver. The solver (Stage 6-7) is excluded per request.

---

## Architecture

```text
[Raw JSONL: premises-NL + question]
        │
        ▼
[Stage 0: Input Loader]  ──  load_problem() + choice extraction
        │
        ▼
[Stage 1: CNL Rewriter]  ──  HF Transformers → controlled natural language
        │
        ▼
[Stage 2: Structural RAG] ── all-MiniLM-L6-v2 embedding similarity
        │
        ▼
[Stage 3: AST Compiler]  ──  HF Transformers → typed LogicNode AST
        │
        ▼
[Stage 4: Validator]      ──  Python checks: vars, negation, arity
        │
        ▼
[Stage 5: Repair Loop]   ──  feed errors back to the LLM (max 2 tries)
        │
        ▼
[Question Parser]         ──  HF Transformers → query/choices AST
        │
        ▼
[FullParseResult]         ──  premises + question, classified for solver
```

---

## Files Created (18 total)

### Core Infrastructure
| File | Purpose |
|------|---------|
| [requirements.txt](file:///c:/Users/hi/Desktop/Praser/logic_pipeline/requirements.txt) | Dependencies |
| [src/__init__.py](file:///c:/Users/hi/Desktop/Praser/logic_pipeline/src/__init__.py) | Package init |
| [src/config.py](file:///c:/Users/hi/Desktop/Praser/logic_pipeline/src/config.py) | `PipelineConfig` dataclass |
| [src/json_utils.py](file:///c:/Users/hi/Desktop/Praser/logic_pipeline/src/json_utils.py) | Safe JSON extraction from noisy LLM output |

### Pipeline Stages
| File | Stage | What It Does |
|------|-------|-------------|
| [src/schemas.py](file:///c:/Users/hi/Desktop/Praser/logic_pipeline/src/schemas.py) | — | All Pydantic models (LogicNode, CNLStatement, etc.) |
| [src/stage0_input.py](file:///c:/Users/hi/Desktop/Praser/logic_pipeline/src/stage0_input.py) | 0 | Loads JSONL, extracts choices from question text |
| [src/stage1_cnl.py](file:///c:/Users/hi/Desktop/Praser/logic_pipeline/src/stage1_cnl.py) | 1 | CNL Rewriter via shared LLM backend |
| [src/stage2_rag.py](file:///c:/Users/hi/Desktop/Praser/logic_pipeline/src/stage2_rag.py) | 2 | Structural RAG with sentence embeddings |
| [src/stage3_ast.py](file:///c:/Users/hi/Desktop/Praser/logic_pipeline/src/stage3_ast.py) | 3 | AST Compiler via shared LLM backend + RAG context |
| [src/stage4_validate.py](file:///c:/Users/hi/Desktop/Praser/logic_pipeline/src/stage4_validate.py) | 4 | Python validator + solver readiness classifier |
| [src/stage5_repair.py](file:///c:/Users/hi/Desktop/Praser/logic_pipeline/src/stage5_repair.py) | 5 | Repair loop — feeds errors back to the LLM |
| [src/question_parser.py](file:///c:/Users/hi/Desktop/Praser/logic_pipeline/src/question_parser.py) | Q | Question + choices → AST |
| [src/pipeline.py](file:///c:/Users/hi/Desktop/Praser/logic_pipeline/src/pipeline.py) | — | Orchestrates all stages |

### Data
| File | Purpose |
|------|---------|
| [data/structural_examples.jsonl](file:///c:/Users/hi/Desktop/Praser/logic_pipeline/data/structural_examples.jsonl) | 20 seed CNL→AST examples for RAG |

### Scripts
| File | Purpose |
|------|---------|
| [scripts/run_one.py](file:///c:/Users/hi/Desktop/Praser/logic_pipeline/scripts/run_one.py) | Quick single-problem test |
| [scripts/run_jsonl.py](file:///c:/Users/hi/Desktop/Praser/logic_pipeline/scripts/run_jsonl.py) | Batch auto-advance runner with `--start`/`--limit` |

### Tests
| File | Tests |
|------|-------|
| [tests/test_schemas.py](file:///c:/Users/hi/Desktop/Praser/logic_pipeline/tests/test_schemas.py) | 12 tests — LogicNode shape validation |
| [tests/test_validator.py](file:///c:/Users/hi/Desktop/Praser/logic_pipeline/tests/test_validator.py) | 12 tests — validator + readiness classifier |

---

## Test Results

```
24 passed in 0.35s
```

All schema and validator tests pass.

---

## How to Run

### Single problem (smoke test):
```bash
cd logic_pipeline
python scripts/run_one.py
```

### Batch (auto-advance through the dataset):
```bash
# First 5 problems
python scripts/run_jsonl.py --limit 5

# All 100 problems
python scripts/run_jsonl.py

# Start from problem 10, process 20
python scripts/run_jsonl.py --start 10 --limit 20
```

> [!IMPORTANT]
> The default backend is Hugging Face Transformers with `Qwen/Qwen3.5-4B`.
> The Hugging Face loader uses 4-bit bitsandbytes quantization by default to reduce VRAM use.
> Install dependencies with `pip install -r requirements.txt`, then run the scripts normally.
> The first run downloads the model weights from Hugging Face, so it can take time and requires enough disk/RAM plus a CUDA GPU for the default 4-bit path.
> LLM prompts and streamed model output are printed to the terminal and appended to `logic_pipeline/artifacts/llm_io.txt`.
> To change the transcript file, run `python scripts/run_jsonl.py --llm-trace artifacts/my_trace.txt --limit 5`.
> To disable tracing, run `python scripts/run_jsonl.py --no-llm-trace --limit 5`.
> To disable quantization, run `python scripts/run_jsonl.py --no-4bit --limit 5`.
>
> The old Ollama backend remains available for compatibility:
> `python scripts/run_jsonl.py --provider ollama --model qwen2.5:7b-instruct --limit 5`.

---

## Key Design Decisions

1. **LLM proposes, Python validates** — the LLM never solves; it only reformulates (Stage 1) and compiles (Stage 3). Python enforces structural correctness.
2. **Auto-advance** — the batch runner auto-continues to the next problem after each parse, with progressive save to `artifacts/predictions.jsonl`.
3. **Adapted for dataset format** — Stage 0 reads `premises-NL` (not `premises`) and auto-extracts `A) ...` style choices from question text.
4. **20 RAG seed examples** — expanded from the guide's 6 to cover negation rules, multi-variable premises, meta-level implications, and named entities.
5. **Solver readiness classifier** — each parsed premise gets tagged `solver_ready`, `needs_review`, or `unsupported` for when you add the solver later.
