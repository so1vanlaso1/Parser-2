from NEW_logic_pipeline.Stage_2.atomization_requests import AtomizationRequest
from NEW_logic_pipeline.Stage_2.leaf_atomizer import (
    atomize_request,
    atomize_requests,
    build_atomizer_prompt,
    parse_atomizer_response,
)


class FakeLLM:
    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str):
        return self.response


class SequenceLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def generate(self, prompt: str):
        return self.responses.pop(0)


class RaisingLLM:
    def generate(self, prompt: str):
        raise AssertionError("formula-like META leaf should not be sent to the LLM")


def _request(**overrides):
    values = {
        "request_id": "P1_antecedent",
        "premise_id": "P1",
        "role": "antecedent",
        "phrase": "a student does not maintain GPA",
        "variable": "x",
    }
    values.update(overrides)
    return AtomizationRequest(**values)


def test_parse_valid_atomizer_json():
    raw = """
    {
      "atoms": [
        {
          "name": "student",
          "arguments": ["x"],
          "negated": false,
          "confidence": 0.95
        },
        {
          "name": "maintain_gpa",
          "arguments": ["x"],
          "negated": true,
          "confidence": 0.95
        }
      ],
      "needs_review": false,
      "unsupported_reason": null,
      "notes": []
    }
    """

    result = parse_atomizer_response(raw, _request())

    assert len(result.atoms) == 2
    assert result.atoms[1].name == "maintain_gpa"
    assert result.atoms[1].negated is True
    assert result.needs_review is False


def test_invalid_json_returns_needs_review():
    result = parse_atomizer_response("not json", _request())

    assert result.needs_review is True
    assert result.unsupported_reason == "invalid_json_from_atomizer"
    assert result.atoms == []


def test_code_fenced_json_is_parsed():
    raw = """
    ```json
    {
      "atoms": [
        {
          "name": "certified",
          "arguments": ["john"],
          "negated": false
        }
      ],
      "needs_review": false,
      "unsupported_reason": null,
      "notes": []
    }
    ```
    """

    result = parse_atomizer_response(raw, _request(phrase="John is certified", role="body"))

    assert len(result.atoms) == 1
    assert result.atoms[0].name == "certified"
    assert result.atoms[0].arguments == ["john"]


def test_bad_predicate_name_gets_needs_review():
    raw = """
    {
      "atoms": [
        {
          "name": "not maintain GPA",
          "arguments": ["x"],
          "negated": false
        }
      ],
      "needs_review": false,
      "unsupported_reason": null,
      "notes": []
    }
    """

    result = parse_atomizer_response(raw, _request())

    assert result.needs_review is True
    assert any("invalid predicate name" in note for note in result.notes)


def test_modal_unsupported_result():
    raw = """
    {
      "atoms": [],
      "needs_review": true,
      "unsupported_reason": "modal_not_necessarily_is_not_classical_negation",
      "notes": []
    }
    """

    result = parse_atomizer_response(
        raw,
        _request(
            request_id="P1_body",
            role="body",
            phrase="Every smart home device is not necessarily energy efficient",
            modality_hint="modal_not_necessarily",
        ),
    )

    assert result.needs_review is True
    assert result.unsupported_reason == "modal_not_necessarily_is_not_classical_negation"
    assert result.atoms == []


def test_batch_does_not_crash_on_one_bad_output():
    requests = [
        _request(request_id="P1_body", role="body", phrase="John is certified"),
        _request(request_id="P2_body", role="body", phrase="not json phrase"),
    ]
    llm = SequenceLLM(
        [
            '{"atoms":[{"name":"certified","arguments":["john"],"negated":false}],"needs_review":false,"unsupported_reason":null,"notes":[]}',
            "not json",
        ]
    )

    results = atomize_requests(requests, llm)

    assert len(results) == 2
    assert results[0].atoms[0].name == "certified"
    assert results[1].needs_review is True
    assert results[1].unsupported_reason == "invalid_json_from_atomizer"


def test_atomizer_rejects_formula_like_meta_leaf_without_llm_call():
    result = atomize_request(
        _request(
            request_id="P1_meta_leaf_1",
            role="meta_leaf",
            phrase="(if a student is not asking questions, they are not attending tutorials)",
            skeleton_kind="META",
        ),
        RaisingLLM(),
    )

    assert result.needs_review is True
    assert result.unsupported_reason == "formula_like_leaf_requires_recursive_parse"
    assert result.atoms == []


def test_phrase_cache_key_keeps_negation_separate():
    requests = [
        _request(request_id="P1_body", role="body", phrase="has completed the registration form", negation_hint=False),
        _request(request_id="P2_body", role="body", phrase="has not completed the registration form", negation_hint=True),
    ]
    llm = SequenceLLM(
        [
            '{"atoms":[{"name":"completed_registration_form","arguments":["x"],"negated":false}],"needs_review":false,"unsupported_reason":null,"notes":[]}',
            '{"atoms":[{"name":"completed_registration_form","arguments":["x"],"negated":true}],"needs_review":false,"unsupported_reason":null,"notes":[]}',
        ]
    )

    results = atomize_requests(requests, llm)

    assert len(results) == 2
    assert results[0].atoms[0].negated is False
    assert results[1].atoms[0].negated is True


def test_prompt_contains_named_numeric_object_rules():
    prompt = build_atomizer_prompt(_request(phrase="Nam has a GPA of 3.2", role="body"))

    assert "For named FACT phrases, use the named constant, not x" in prompt
    assert "Credits are not grades" in prompt
    assert "GPA values must be preserved" in prompt
    assert "Do not create predicate names containing it" in prompt
