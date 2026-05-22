"""
Run the full parse pipeline on a single hardcoded example.

Usage:
    python scripts/run_one.py
"""
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PipelineConfig
from src.pipeline import LogicPipeline


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    premises = [
        "Every scholarship recipient receives priority housing.",
        "Students without housing cannot participate in the Quantum Physics lab.",
        "Mina is a scholarship recipient.",
    ]
    question = "Can Mina participate in the Quantum Physics lab?"

    config = PipelineConfig(
        model_name="Qwen/Qwen3.5-4B",
        model_provider="huggingface",
        hf_load_in_4bit=True,
        llm_trace_path=str(ROOT / "artifacts" / "llm_io.txt"),
        max_repair_attempts=2,
    )

    rag_path = str(ROOT / "data" / "structural_examples.jsonl")
    pipeline = LogicPipeline(config, rag_path=rag_path)

    result = pipeline.parse(premises, question=question)

    print("\n" + "=" * 60)
    print("FULL PARSE RESULT")
    print("=" * 60)
    print(result.model_dump_json(indent=2))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for p in result.premises:
        status = "✓ ready" if p.solver_ready else ("⚠ review" if p.needs_review else "✗ unsupported")
        print(f"  {p.premise_id} [{p.kind}] {status}: {p.cnl[:80]}")

    if result.question:
        print(f"\n  Question: {result.question.question}")
        if result.question.query:
            print(f"  Query AST type: {result.question.query.type}")
        if result.question.choices:
            print(f"  Choices: {list(result.question.choices.keys())}")


if __name__ == "__main__":
    main()
