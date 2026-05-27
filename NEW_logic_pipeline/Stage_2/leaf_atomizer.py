from __future__ import annotations

"""Stage 5: LLM leaf atomizer.

Input: AtomizationRequest.
Output: AtomizationResult with flat PredicateAtom objects only.

No AST, solver export, META resolution, predicate registry, or final answer logic
belongs here.
"""

from typing import Any
import json
import re

from pydantic import ValidationError

try:
    from .atomization_requests import AtomizationRequest, AtomizationResult, PredicateAtom, is_formula_like_leaf
except Exception:  # pragma: no cover
    from atomization_requests import AtomizationRequest, AtomizationResult, PredicateAtom, is_formula_like_leaf


VALID_PREDICATE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
VALID_ARGUMENT_RE = re.compile(r"^(?:[a-z][a-z0-9_]*|[0-9]+)$")
VARIABLE_NAMES = {"u", "v", "w", "x", "y", "z"}
ENCODED_NEGATION_PREFIXES = ("not_", "non_", "no_")
MODAL_OR_DEONTIC_HINTS = {
    "modal_not_necessarily",
    "not_necessarily",
    "modal_uncertainty",
    "modal",
    "deontic_obligation",
    "deontic_permission",
    "deontic_prohibition",
    "deontic",
    "unknown_structure",
}


def build_atomizer_prompt(request: AtomizationRequest) -> str:
    input_payload = {
        "phrase": request.phrase,
        "variable": request.variable,
        "role": request.role,
        "known_predicates": request.known_predicates,
        "negation_hint": request.negation_hint,
        "modality_hint": request.modality_hint,
        "logical_cues": request.logical_cues,
        "required_domain_atoms": request.required_domain_atoms,
        "source_mentions": request.source_mentions,
        "skeleton_kind": request.skeleton_kind,
        "formula_path": request.formula_path,
    }

    return f"""/no_think

You are an atomizer for a neurosymbolic logic parser.

Your only task:
Convert ONE small English phrase into flat predicate atoms.

You are not a solver.
You must not infer unstated facts.
You must not create implication.
You must not create quantifiers.
You must not create iff.
You must not create a full AST.
You must not decide solver readiness.
You must not decide the final answer.
You must not flatten META formulas.

Return valid JSON only.

Rules:
1. Predicate names must be lowercase snake_case.
2. Do not put negation inside predicate names. Use "negated": true.
3. Use the provided variable exactly for generic entities.
4. Use constants only for named individuals, normalized to lowercase snake_case.
5. Reuse known predicates when the meaning clearly matches.
6. Prefer binary predicates for subject-scoped relations: mastered_subject(person, subject), has_knowledge_of_subject(person, subject), explain_subject(person, subject).
7. Represent grade counts explicitly when present: grade_count(tuan, a, 3) or grade_count_at_least(x, a_or_a_plus, 5).
8. If a phrase has a subject class plus a property, output both atoms.
9. If the phrase is modal/deontic/uncertain/unknown, set needs_review true unless it is a simple lexical atom that is clearly safe.
10. Treat "must have", "must be", and "must satisfy" as logical necessity in rule consequents, not deontic obligation.
11. If the phrase cannot be safely atomized, return empty atoms and unsupported_reason.
12. Do not add information not present in the phrase.
13. Preserve every required_domain_atom unless it is explicitly added elsewhere.
14. Add evidence_links using source_mentions ids for atoms that represent those mentions.

Negation hint rule:
- negation_hint means classical negation supplied by the skeleton.
- If negation_hint is true and the phrase itself is not already negative, negate the main property atom.
- If the phrase contains a subject class plus a property, do NOT negate the subject class; negate only the property.
- If modality_hint is modal_not_necessarily or not_necessarily, do NOT treat "not" as classical negation.
- If you cannot identify the main property safely, return needs_review true.

Input:
{json.dumps(input_payload, indent=2)}

Output schema:
{{
  "atoms": [
    {{
      "name": "predicate_name",
      "arguments": ["{request.variable}"],
      "negated": false,
      "evidence_links": [],
      "confidence": 0.9
    }}
  ],
  "needs_review": false,
  "unsupported_reason": null,
  "notes": []
}}

Examples:

Input phrase: a student does not maintain GPA
Output:
{{
  "atoms": [
    {{"name": "student", "arguments": ["x"], "negated": false, "confidence": 0.95}},
    {{"name": "maintain_gpa", "arguments": ["x"], "negated": true, "confidence": 0.95}}
  ],
  "needs_review": false,
  "unsupported_reason": null,
  "notes": []
}}

Input phrase: John is certified
Output:
{{
  "atoms": [
    {{"name": "certified", "arguments": ["john"], "negated": false, "confidence": 0.95}}
  ],
  "needs_review": false,
  "unsupported_reason": null,
  "notes": []
}}

Input phrase: Every smart home device is not necessarily energy efficient
Output:
{{
  "atoms": [],
  "needs_review": true,
  "unsupported_reason": "modal_not_necessarily_is_not_classical_negation",
  "notes": ["Do not convert not necessarily into classical NOT."]
}}

Input phrase: wearing goggles is mandatory in science laboratories
Output:
{{
  "atoms": [],
  "needs_review": true,
  "unsupported_reason": "deontic_obligation_requires_special_handling",
  "notes": ["Obligation should not be treated as ordinary fact."]
}}

Input phrase: laboratory access
negation_hint: true
Output:
{{
  "atoms": [
    {{"name": "laboratory_access", "arguments": ["x"], "negated": true, "confidence": 0.9}}
  ],
  "needs_review": false,
  "unsupported_reason": null,
  "notes": ["Applied negation_hint to the main property atom."]
}}
"""


def parse_atomizer_response(raw_text: Any, request: AtomizationRequest) -> AtomizationResult:
    raw = _response_to_text(raw_text)
    json_text = _extract_json_object(raw)
    if json_text is None:
        return _invalid_result(request, "invalid_json_from_atomizer", ["Atomizer response did not contain a JSON object."])
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return _invalid_result(request, "invalid_json_from_atomizer", [f"Atomizer JSON parse failed: {exc.msg}."])
    if not isinstance(payload, dict):
        return _invalid_result(request, "invalid_atomizer_schema", ["Atomizer JSON root must be an object."])

    try:
        atoms_payload = payload.get("atoms", [])
        if atoms_payload is None:
            atoms_payload = []
        if not isinstance(atoms_payload, list):
            raise TypeError("atoms must be a list")
        atoms = [PredicateAtom.model_validate(item) for item in atoms_payload]
    except (TypeError, ValidationError) as exc:
        return _invalid_result(request, "invalid_atomizer_schema", [f"Atomizer atom schema validation failed: {exc}"])

    for atom in atoms:
        if atom.source_phrase is None:
            atom.source_phrase = request.phrase

    result = AtomizationResult(
        request_id=request.request_id,
        premise_id=request.premise_id,
        role=request.role,
        phrase=request.phrase,
        variable=request.variable,
        atoms=atoms,
        source_mentions=list(request.source_mentions),
        domain_restriction_added_elsewhere=bool(payload.get("domain_restriction_added_elsewhere", False)),
        needs_review=bool(payload.get("needs_review", False)),
        unsupported_reason=_none_if_empty(payload.get("unsupported_reason")),
        notes=_coerce_notes(payload.get("notes", [])),
        formula_path=list(request.formula_path),
    )
    return validate_atomization_result(result)


def validate_atomization_result(result: AtomizationResult) -> AtomizationResult:
    notes = list(result.notes)
    needs_review = bool(result.needs_review)

    if result.unsupported_reason and result.atoms:
        needs_review = True
        notes.append("unsupported_reason was set but atoms were returned; keeping atoms but marking needs_review.")

    for atom in result.atoms:
        original_name = atom.name
        if not VALID_PREDICATE_RE.fullmatch(str(original_name or "")):
            needs_review = True
            notes.append(f"invalid predicate name: {original_name!r}.")

        atom.name = _to_snake_case(atom.name)

        if atom.name != original_name:
            needs_review = True
            notes.append(f"normalized predicate name from {original_name!r} to {atom.name!r}.")

        if not VALID_PREDICATE_RE.fullmatch(atom.name):
            needs_review = True
            notes.append(f"invalid predicate name: {atom.name!r}.")

        if _has_encoded_negation(atom.name):
            needs_review = True
            notes.append(f"predicate appears to encode negation: {atom.name!r}; use atom.negated instead.")

        token_count = len(atom.name.split("_"))
        if token_count > 6:
            needs_review = True
            notes.append(f"predicate name is too long ({token_count} tokens): {atom.name!r}.")

        if not atom.arguments:
            needs_review = True
            notes.append(f"atom {atom.name!r} has no arguments.")

        normalized_args: list[Any] = []
        for arg in atom.arguments:
            if isinstance(arg, dict):
                normalized = dict(arg)
                normalized["value"] = _to_snake_case(str(normalized.get("value", "")))
                if not VALID_ARGUMENT_RE.fullmatch(normalized["value"]):
                    needs_review = True
                    notes.append(f"invalid argument {arg!r} for atom {atom.name!r}.")
                normalized_args.append(normalized)
            else:
                norm_arg = _to_snake_case(str(arg))
                if not VALID_ARGUMENT_RE.fullmatch(norm_arg):
                    needs_review = True
                    notes.append(f"invalid argument {arg!r} for atom {atom.name!r}.")
                normalized_args.append(norm_arg)
        atom.arguments = normalized_args

        if atom.confidence < 0 or atom.confidence > 1:
            needs_review = True
            notes.append(f"confidence outside [0,1] for atom {atom.name!r}.")
            atom.confidence = min(1.0, max(0.0, atom.confidence))

    result.needs_review = needs_review
    result.notes = _dedupe(notes)
    return result


def atomize_request(request: AtomizationRequest, llm: Any) -> AtomizationResult:
    if _request_is_formula_like_leaf(request):
        return _invalid_result(
            request,
            "formula_like_leaf_requires_recursive_parse",
            ["Formula-like META leaves must be recursively parsed before atomization."],
        )

    prompt = build_atomizer_prompt(request)
    try:
        if hasattr(llm, "generate"):
            raw = llm.generate(prompt)
        elif callable(llm):
            raw = llm(prompt)
        else:
            return _invalid_result(request, "invalid_llm_interface", ["LLM object must be callable or expose generate(prompt)."])
    except Exception as exc:  # pragma: no cover
        return _invalid_result(request, "llm_call_failed", [f"LLM call failed: {exc}"])

    parsed = parse_atomizer_response(raw, request)
    parsed = _apply_request_level_checks(parsed, request)
    return validate_atomization_result(parsed)


def atomize_requests(requests: list[AtomizationRequest], llm: Any) -> list[AtomizationResult]:
    return [atomize_request(request, llm) for request in requests]


def _apply_request_level_checks(result: AtomizationResult, request: AtomizationRequest) -> AtomizationResult:
    notes = list(result.notes)

    if request.negation_hint and result.atoms and not any(atom.negated for atom in result.atoms):
        result.needs_review = True
        notes.append("request.negation_hint=True but no returned atom is negated.")

    if request.negation_hint and len(result.atoms) > 1:
        # Warn if all atoms were negated, because subject class atoms should
        # usually remain positive.
        if all(atom.negated for atom in result.atoms):
            result.needs_review = True
            notes.append("negation_hint appears applied to every atom; subject/restrictor atoms should usually stay positive.")

    result = _validate_request_modality(result, request)
    result.notes = _dedupe([*notes, *result.notes])
    return result


def _validate_request_modality(result: AtomizationResult, request: AtomizationRequest) -> AtomizationResult:
    hint = _normalize_hint(request.modality_hint)
    notes = list(result.notes)

    if not hint:
        return result

    if re.search(r"\bmust\s+(have|be|satisfy|meet)\b", request.phrase, flags=re.I):
        return result

    if hint in {"unknown_structure"}:
        result.needs_review = True
        result.unsupported_reason = result.unsupported_reason or "unknown_structure_requires_review"
        notes.append("Unknown skeleton structure should not be treated as solver-ready atomization.")

    if hint in MODAL_OR_DEONTIC_HINTS and result.atoms and not result.needs_review:
        result.needs_review = True
        notes.append(f"request has modality/deontic hint {hint!r}; atomization requires review.")

    if hint in {"modal_not_necessarily", "not_necessarily"}:
        if any(atom.negated for atom in result.atoms):
            result.needs_review = True
            result.unsupported_reason = result.unsupported_reason or "modal_not_necessarily_is_not_classical_negation"
            notes.append("not necessarily must not become classical negation.")
        elif not result.unsupported_reason:
            result.unsupported_reason = "modal_not_necessarily_is_not_classical_negation"
            result.needs_review = True

    if hint.startswith("deontic") and result.atoms and not result.unsupported_reason:
        result.needs_review = True
        result.unsupported_reason = "deontic_statement_requires_special_handling"

    result.notes = _dedupe(notes)
    return result


def _extract_json_object(raw_text: str) -> str | None:
    text = raw_text.strip()
    if not text:
        return None

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    return _first_balanced_json_object(text)


def _first_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _invalid_result(request: AtomizationRequest, reason: str, notes: list[str] | None = None) -> AtomizationResult:
    return AtomizationResult(
        request_id=request.request_id,
        premise_id=request.premise_id,
        role=request.role,
        phrase=request.phrase,
        variable=request.variable,
        atoms=[],
        source_mentions=list(request.source_mentions),
        needs_review=True,
        unsupported_reason=reason,
        notes=notes or [],
        formula_path=list(request.formula_path),
    )


def _coerce_notes(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def _response_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # Common wrapper shapes from LLM adapters.
        for key in ("text", "content", "response", "output"):
            if key in value:
                return str(value[key])
        return json.dumps(value)
    return str(value)


def _to_snake_case(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    text = re.sub(r"_+", "_", text)
    return text.lower()


def _has_encoded_negation(value: str) -> bool:
    lowered = value.lower()
    return any(lowered.startswith(prefix) for prefix in ENCODED_NEGATION_PREFIXES)


def _normalize_hint(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace(" ", "_")


def _none_if_empty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _request_is_formula_like_leaf(request: AtomizationRequest) -> bool:
    if "formula_like_leaf_detected" in request.notes:
        return True
    return str(request.skeleton_kind or "").upper() == "META" and is_formula_like_leaf(request.phrase)


__all__ = [
    "build_atomizer_prompt",
    "parse_atomizer_response",
    "validate_atomization_result",
    "atomize_request",
    "atomize_requests",
]
