"""
Batch runner: parse every problem in a JSONL file and auto-advance.

Reads premises-NL + question from each line, runs the full parse pipeline,
saves results to artifacts/predictions.jsonl, and automatically continues
to the next problem.

Usage:
    python scripts/run_jsonl.py                          # all problems
    python scripts/run_jsonl.py --limit 5                # first 5
    python scripts/run_jsonl.py --start 10 --limit 5     # problems 10-14
    python scripts/run_jsonl.py --input path/to/file.jsonl
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PipelineConfig
from src.pipeline import LogicPipeline
from src.stage0_input import load_problem


def main():
    parser = argparse.ArgumentParser(description="Batch parse JSONL logic problems")
    parser.add_argument(
        "--input", "-i",
        default=str(ROOT.parent / "fixed_smoke_logic_406.jsonl"),
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(ROOT / "artifacts" / "predictions.jsonl"),
        help="Path to output predictions JSONL",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--limit", type=int, default=None, help="Max problems to process")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Hugging Face model id")
    parser.add_argument(
        "--provider",
        choices=["huggingface", "ollama"],
        default="huggingface",
        help="Model backend",
    )
    parser.add_argument("--repair", type=int, default=2, help="Max repair attempts")
    parser.add_argument(
        "--no-frame-ast",
        action="store_true",
        help="Disable Stage 3 predicate-frame extraction and use the old full-AST LLM compiler",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable Hugging Face 4-bit quantized loading",
    )
    parser.add_argument(
        "--no-llm-trace",
        action="store_true",
        help="Disable live LLM prompt/response tracing",
    )
    parser.add_argument(
        "--llm-trace",
        default=str(ROOT / "artifacts" / "llm_io.txt"),
        help="Path to the LLM prompt/response transcript text file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("run_jsonl")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    with input_path.open("r", encoding="utf-8") as f:
        all_lines = [line for line in f if line.strip()]

    total = len(all_lines)
    start = args.start
    end = min(start + args.limit, total) if args.limit else total
    lines_to_process = all_lines[start:end]

    log.info("Input: %s (%d total problems)", input_path.name, total)
    log.info("Processing problems %d-%d (%d problems)", start, end - 1, len(lines_to_process))

    config = PipelineConfig(
        model_name=args.model,
        model_provider=args.provider,
        hf_load_in_4bit=not args.no_4bit,
        llm_live_trace=not args.no_llm_trace,
        llm_trace_path=args.llm_trace,
        max_repair_attempts=args.repair,
        enable_frame_ast_compiler=not args.no_frame_ast,
    )
    rag_path = str(ROOT / "data" / "structural_examples.jsonl")
    pipeline = LogicPipeline(config, rag_path=rag_path)

    stats = {"ok": 0, "fail": 0, "total_time": 0.0}

    with output_path.open("w", encoding="utf-8") as f_out:
        for idx, line in enumerate(lines_to_process):
            problem_num = start + idx
            raw = json.loads(line)
            problem = load_problem(raw)

            log.info(
                "--- [%d/%d] %s (%d premises) ---",
                problem_num + 1, total, problem.id, len(problem.premises),
            )

            t0 = time.time()
            try:
                result = pipeline.parse(
                    premises=problem.premises,
                    question=problem.question,
                    choices=problem.choices if problem.choices else None,
                )

                elapsed = time.time() - t0
                stats["total_time"] += elapsed

                ready = sum(1 for p in result.premises if p.solver_ready)
                review = sum(1 for p in result.premises if p.needs_review)
                unsupported = sum(1 for p in result.premises if p.unsupported)
                ok = result.status == "success"
                stats["ok" if ok else "fail"] += 1

                record = {
                    "id": problem.id,
                    "ok": ok,
                    "status": result.status,
                    "elapsed_seconds": round(elapsed, 2),
                    "premise_count": len(problem.premises),
                    "solver_ready": ready,
                    "needs_review": review,
                    "unsupported": unsupported,
                    "parse_result": result.model_dump(),
                }
                if result.error:
                    record["error"] = result.error

                if ok:
                    log.info(
                        "  parsed %s (%.1fs) - %d ready, %d review, %d unsupported",
                        problem.id, elapsed, ready, review, unsupported,
                    )
                else:
                    log.error(
                        "  failed %s (%.1fs) - %s",
                        problem.id, elapsed, result.error or result.status,
                    )

            except Exception as e:
                elapsed = time.time() - t0
                stats["fail"] += 1
                stats["total_time"] += elapsed

                record = {
                    "id": problem.id,
                    "ok": False,
                    "status": "failed",
                    "elapsed_seconds": round(elapsed, 2),
                    "error": str(e),
                }

                log.error("  failed %s (%.1fs) - %s", problem.id, elapsed, e)

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

    print("\n" + "=" * 60)
    print("BATCH RUN COMPLETE")
    print("=" * 60)
    print(f"  Processed: {stats['ok'] + stats['fail']}")
    print(f"  Succeeded: {stats['ok']}")
    print(f"  Failed:    {stats['fail']}")
    print(f"  Total time: {stats['total_time']:.1f}s")
    if stats["ok"] + stats["fail"] > 0:
        avg = stats["total_time"] / (stats["ok"] + stats["fail"])
        print(f"  Avg time:  {avg:.1f}s per problem")
    print(f"  Output:    {output_path}")


if __name__ == "__main__":
    main()
