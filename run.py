from __future__ import annotations

"""Run Stage 1 and Stage 2 over rows from the smoke-test JSONL file."""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from NEW_logic_pipeline.Stage_1.skeleton_builder import build_skeletons
from NEW_logic_pipeline.Stage_2.atomization_requests import collect_batch_atomization_requests
from NEW_logic_pipeline.Stage_2.leaf_atomizer import atomize_requests
from NEW_logic_pipeline.Stage_2.model_backends import LocalTransformersConfig, create_local_llm
from NEW_logic_pipeline.Stage_2.predicate_canonicalizer import (
    DEFAULT_KNOWN_PREDICATES,
    canonicalize_atomization_results,
)
from NEW_logic_pipeline.Stage_6.registry_schema import load_registry_config
from NEW_logic_pipeline.Stage_6.validation_models import SemanticPolicy
from NEW_logic_pipeline.Stage_6.validator import Stage6Validator


DEFAULT_INPUT = Path("fixed_smoke_logic_406.jsonl")
DEFAULT_ARTIFACTS_DIR = Path("artifacts")
DEFAULT_STAGE6_REGISTRY = Path("configs") / "registries" / "education_registry.yaml"
DEFAULT_STAGE6_POLICY = Path("configs") / "semantic_policies" / "education_policy.yaml"


class TracingLLM:
    """Log every LLM prompt/response while preserving the generate(prompt) API."""

    def __init__(self, llm: Any, trace_path: Path, print_io: bool = False):
        self.llm = llm
        self.trace_path = trace_path
        self.print_io = print_io
        self.call_count = 0
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        call_id = self.call_count
        started = time.perf_counter()
        record: dict[str, Any] = {
            "call_id": call_id,
            "prompt": prompt,
        }

        if self.print_io:
            print(f"\n--- LLM INPUT #{call_id} ---")
            print(prompt)
            print(f"--- END LLM INPUT #{call_id} ---", flush=True)

        try:
            response = self.llm.generate(prompt)
        except Exception as exc:
            record["error"] = repr(exc)
            record["duration_seconds"] = round(time.perf_counter() - started, 3)
            self._write_record(record)
            raise

        record["response"] = response
        record["duration_seconds"] = round(time.perf_counter() - started, 3)
        self._write_record(record)

        if self.print_io:
            print(f"\n--- LLM OUTPUT #{call_id} ---")
            print(response)
            print(f"--- END LLM OUTPUT #{call_id} ---", flush=True)

        return response

    def _write_record(self, record: dict[str, Any]) -> None:
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    args = parse_args(argv)

    rows = load_jsonl_rows(
        args.input,
        start=args.start,
        limit=args.limit,
        row_ids=args.row_id,
    )
    if not rows:
        print("No rows selected.")
        return 1

    config = build_local_config(args)
    run_dir = make_run_dir(args.artifacts_dir, args.run_name)
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    llm_io_path = run_dir / "llm_io.jsonl"
    llm_io_path.write_text("", encoding="utf-8")

    print(f"Selected {len(rows)} row(s).")
    print(f"Model mode: {config.mode}")
    print(f"Artifacts: {run_dir}")
    print("Loading local Transformers model...")

    llm = TracingLLM(create_local_llm(config), llm_io_path, print_io=args.print_llm_io)

    summary: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(args.input),
        "artifacts_dir": str(run_dir),
        "model_config": asdict(config),
        "row_count": len(rows),
        "rows": [],
    }

    with results_path.open("w", encoding="utf-8") as results_file:
        for selected_index, row_info in enumerate(rows, start=1):
            row_number = row_info["row_number"]
            row = row_info["row"]
            row_id = str(row.get("id") or row.get("source_record_id") or f"row-{row_number}")
            premises = get_premises(row)

            print(f"[{selected_index}/{len(rows)}] Row {row_number} ({row_id}): {len(premises)} premise(s)")

            row_started = time.perf_counter()
            row_output = parse_row(
                row=row,
                row_number=row_number,
                row_id=row_id,
                premises=premises,
                llm=llm,
                known_predicates=args.known_predicate,
                stage6_registry=args.stage6_registry,
                stage6_policy=args.stage6_policy,
            )
            row_output["duration_seconds"] = round(time.perf_counter() - row_started, 3)

            results_file.write(json.dumps(row_output, ensure_ascii=False) + "\n")
            results_file.flush()

            row_summary = {
                "row_number": row_number,
                "id": row_id,
                "premise_count": len(premises),
                "skeleton_count": row_output["counts"]["skeletons"],
                "request_count": row_output["counts"]["atomization_requests"],
                "result_count": row_output["counts"]["atomization_results"],
                "needs_review_count": row_output["counts"]["needs_review_results"],
                "duration_seconds": row_output["duration_seconds"],
            }
            summary["rows"].append(row_summary)
            print(
                "  "
                f"requests={row_summary['request_count']} "
                f"results={row_summary['result_count']} "
                f"needs_review={row_summary['needs_review_count']}"
            )

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary["llm_call_count"] = llm.call_count
    summary["outputs"] = {
        "results_jsonl": str(results_path),
        "summary_json": str(summary_path),
        "llm_io_jsonl": str(llm_io_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Done.")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"LLM I/O: {llm_io_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NEW_logic_pipeline Stage 1 and Stage 2 over smoke-test JSONL rows.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input smoke-test JSONL path.")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR, help="Directory for run artifacts.")
    parser.add_argument("--run-name", default=None, help="Optional artifact subdirectory name.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based row offset to start from.")
    parser.add_argument("--limit", type=int, default=1, help="Number of rows to parse. Use 0 for all selected rows.")
    parser.add_argument("--row-id", action="append", default=None, help="Specific row id to parse. May be repeated.")
    parser.add_argument("--known-predicate", action="append", default=[], help="Known predicate name to pass into atomization.")
    parser.add_argument(
        "--stage6-registry",
        type=Path,
        default=DEFAULT_STAGE6_REGISTRY,
        help="External Stage 6 predicate registry config.",
    )
    parser.add_argument(
        "--stage6-policy",
        type=Path,
        default=DEFAULT_STAGE6_POLICY,
        help="External Stage 6 semantic policy config.",
    )
    parser.add_argument(
        "--model",
        choices=["minicpm_hf", "qwen_hf_4bit"],
        default=os.getenv("ATOMIZER_MODEL", "minicpm_hf"),
        help="Local Transformers model mode.",
    )
    parser.add_argument(
        "--minicpm-model-id",
        default=os.getenv("MINICPM_MODEL_ID", "openbmb/MiniCPM5-1B"),
        help="Hugging Face model id for minicpm_hf.",
    )
    parser.add_argument(
        "--qwen-model-path",
        default=os.getenv("QWEN_HF_MODEL_PATH"),
        help="Local Hugging Face model folder for qwen_hf_4bit.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.getenv("ATOMIZER_MAX_NEW_TOKENS", "512")),
        help="Max new tokens per atomizer request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("ATOMIZER_TEMPERATURE", "0.0")),
        help="Generation temperature.",
    )
    parser.add_argument("--top-p", type=float, default=float(os.getenv("ATOMIZER_TOP_P", "0.95")), help="Top-p sampling value.")
    parser.add_argument("--device-map", default=os.getenv("ATOMIZER_DEVICE_MAP", "auto"), help="Transformers device_map.")
    parser.add_argument("--torch-dtype", default=os.getenv("ATOMIZER_TORCH_DTYPE", "auto"), help="Transformers torch_dtype for non-4-bit mode.")
    parser.add_argument("--print-llm-io", action="store_true", help="Print every LLM prompt and response in the terminal.")
    return parser.parse_args(argv)


def build_local_config(args: argparse.Namespace) -> LocalTransformersConfig:
    if args.model == "qwen_hf_4bit" and not args.qwen_model_path:
        raise ValueError("--qwen-model-path or QWEN_HF_MODEL_PATH is required for qwen_hf_4bit.")

    return LocalTransformersConfig(
        mode=args.model,
        minicpm_model_id=args.minicpm_model_id,
        qwen_model_path=args.qwen_model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )


def load_jsonl_rows(
    path: Path,
    *,
    start: int,
    limit: int,
    row_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if start < 0:
        raise ValueError("--start must be >= 0.")
    if limit < 0:
        raise ValueError("--limit must be >= 0.")
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

    wanted_ids = set(row_ids or [])
    selected: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            row_id = str(row.get("id") or row.get("source_record_id") or "")
            if wanted_ids and row_id not in wanted_ids:
                continue
            selected.append({"row_number": row_number, "row": row})

    selected = selected[start:]
    if limit:
        selected = selected[:limit]
    return selected


def make_run_dir(artifacts_dir: Path, run_name: str | None) -> Path:
    name = run_name or f"stage12_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = artifacts_dir / name
    if run_dir.exists():
        suffix = datetime.now().strftime("%f")
        run_dir = artifacts_dir / f"{name}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_row(
    *,
    row: dict[str, Any],
    row_number: int,
    row_id: str,
    premises: list[str],
    llm: Any,
    known_predicates: list[str],
    stage6_registry: Path | dict[str, Any] | None = DEFAULT_STAGE6_REGISTRY,
    stage6_policy: Path | dict[str, Any] | SemanticPolicy | None = DEFAULT_STAGE6_POLICY,
) -> dict[str, Any]:
    skeletons = build_skeletons(premises)
    effective_known_predicates = sorted({*DEFAULT_KNOWN_PREDICATES, *known_predicates})
    requests = collect_batch_atomization_requests(skeletons, known_predicates=effective_known_predicates)
    raw_results = atomize_requests(requests, llm)
    results, canonicalization_summary = canonicalize_atomization_results(raw_results)

    needs_review_atomization_count = sum(1 for result in results if result.needs_review)
    needs_review_skeleton_count = sum(1 for skeleton in skeletons if skeleton.needs_review)
    formula_like_leaf_count = sum(
        1
        for request in requests
        if "formula_like_leaf_detected" in request.notes
    ) + sum(
        1
        for result in results
        if result.unsupported_reason == "formula_like_leaf_requires_recursive_parse"
    )
    canonicalization_conflict_count = int(canonicalization_summary.get("conflict_count", 0))
    row_needs_review = any(
        (
            needs_review_skeleton_count,
            needs_review_atomization_count,
            formula_like_leaf_count,
            canonicalization_conflict_count,
        )
    )
    needs_review_total = (
        needs_review_skeleton_count
        + needs_review_atomization_count
        + formula_like_leaf_count
        + canonicalization_conflict_count
    )

    output = {
        "row_number": row_number,
        "id": row_id,
        "source_record_id": row.get("source_record_id"),
        "question_index": row.get("question_index"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "premises_nl": premises,
        "skeletons": [skeleton.model_dump(mode="json") for skeleton in skeletons],
        "atomization_requests": [request.model_dump(mode="json") for request in requests],
        "atomization_results": [result.model_dump(mode="json") for result in results],
        "canonicalization": canonicalization_summary,
        "counts": {
            "premises": len(premises),
            "skeletons": len(skeletons),
            "atomization_requests": len(requests),
            "atomization_results": len(results),
            "needs_review_results": needs_review_total,
            "needs_review_atomization_results": needs_review_atomization_count,
            "needs_review_skeletons": needs_review_skeleton_count,
            "formula_like_leaf_requests": formula_like_leaf_count,
            "predicate_canonicalization_conflicts": canonicalization_conflict_count,
            "row_needs_review": row_needs_review,
        },
    }
    registry = (
        load_registry_config(stage6_registry)
        if isinstance(stage6_registry, (str, Path))
        else stage6_registry
    )
    semantic_policy = (
        SemanticPolicy.from_file(stage6_policy)
        if isinstance(stage6_policy, (str, Path))
        else SemanticPolicy.from_mapping(stage6_policy)
    )
    validation_report = Stage6Validator(
        predicate_registry=registry,
        semantic_policy=semantic_policy,
    ).validate(output)
    output["counts"].update(
        {
            "validation_issues": validation_report.summary["issue_count"],
            "validation_errors": validation_report.summary["error_count"],
            "validation_warnings": validation_report.summary["warning_count"],
            "validation_infos": validation_report.summary["info_count"],
        }
    )
    legacy_reasons = _readiness_reasons(
        needs_review_skeleton_count=needs_review_skeleton_count,
        needs_review_atomization_count=needs_review_atomization_count,
        formula_like_leaf_count=formula_like_leaf_count,
        canonicalization_conflict_count=canonicalization_conflict_count,
    )
    combined_reasons = sorted(set([*validation_report.readiness_reasons, *legacy_reasons]))

    output["validation"] = validation_report.to_dict()
    output["readiness"] = {
        "parse_valid": validation_report.parse_valid,
        "direct_solver_ready": validation_report.direct_solver_ready and not row_needs_review,
        "needs_lowering": validation_report.needs_lowering,
        "needs_meta_resolution": validation_report.needs_meta_resolution,
        "needs_review": validation_report.needs_review or row_needs_review,
        "unsupported": validation_report.unsupported or any(result.unsupported_reason for result in results),
        "reasons": combined_reasons,
    }
    return output


def _readiness_reasons(
    *,
    needs_review_skeleton_count: int,
    needs_review_atomization_count: int,
    formula_like_leaf_count: int,
    canonicalization_conflict_count: int,
) -> list[str]:
    reasons: list[str] = []
    if needs_review_skeleton_count:
        reasons.append("skeleton_needs_review")
    if needs_review_atomization_count:
        reasons.append("atomization_needs_review")
    if formula_like_leaf_count:
        reasons.append("formula_like_leaf_detected")
    if canonicalization_conflict_count:
        reasons.append("predicate_canonicalization_conflict")
    return reasons


def get_premises(row: dict[str, Any]) -> list[str]:
    for key in ("premises-NL", "premises_nl", "premises"):
        value = row.get(key)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return value
    raise ValueError("Row does not contain a string list under premises-NL, premises_nl, or premises.")


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
