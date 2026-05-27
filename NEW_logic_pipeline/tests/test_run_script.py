import json
import shutil
import uuid
from pathlib import Path

from run import TracingLLM, get_premises, load_jsonl_rows, parse_row


class FakeLLM:
    def generate(self, prompt: str) -> str:
        return f"response to {prompt}"


class CleanAtomLLM:
    def generate(self, prompt: str) -> str:
        return '{"atoms":[],"needs_review":false,"unsupported_reason":null,"notes":[]}'


def test_load_jsonl_rows_honors_start_and_limit():
    workdir = _workspace_tmp_dir()
    try:
        path = workdir / "rows.jsonl"
        rows = [
            {"id": "a", "premises-NL": ["A"]},
            {"id": "b", "premises-NL": ["B"]},
            {"id": "c", "premises-NL": ["C"]},
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

        selected = load_jsonl_rows(path, start=1, limit=1)

        assert len(selected) == 1
        assert selected[0]["row_number"] == 2
        assert selected[0]["row"]["id"] == "b"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_load_jsonl_rows_filters_by_row_id():
    workdir = _workspace_tmp_dir()
    try:
        path = workdir / "rows.jsonl"
        rows = [
            {"id": "a", "premises-NL": ["A"]},
            {"id": "b", "premises-NL": ["B"]},
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

        selected = load_jsonl_rows(path, start=0, limit=0, row_ids=["b"])

        assert [item["row"]["id"] for item in selected] == ["b"]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_get_premises_prefers_smoke_key():
    assert get_premises({"premises-NL": ["A", "B"]}) == ["A", "B"]


def test_tracing_llm_writes_prompt_and_response():
    workdir = _workspace_tmp_dir()
    try:
        trace_path = workdir / "llm_io.jsonl"
        llm = TracingLLM(FakeLLM(), trace_path)

        response = llm.generate("prompt")

        assert response == "response to prompt"
        records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        assert records == [
            {
                "call_id": 1,
                "prompt": "prompt",
                "response": "response to prompt",
                "duration_seconds": records[0]["duration_seconds"],
            }
        ]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_parse_row_counts_structural_review_in_readiness():
    premise = (
        "If there exists at least one student who is attending tutorials, "
        "then (if a student is not asking questions, they are not attending tutorials)."
    )

    output = parse_row(
        row={"id": "row-a", "premises-NL": [premise]},
        row_number=1,
        row_id="row-a",
        premises=[premise],
        llm=CleanAtomLLM(),
        known_predicates=[],
    )

    assert output["counts"]["needs_review_skeletons"] == 1
    assert output["counts"]["needs_review_atomization_results"] == 0
    assert output["counts"]["needs_review_results"] >= 1
    assert output["readiness"]["needs_review"] is True
    assert "skeleton_needs_review" in output["readiness"]["reasons"]


def _workspace_tmp_dir() -> Path:
    path = Path(".pytest_cache") / "run_script_tests" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path
