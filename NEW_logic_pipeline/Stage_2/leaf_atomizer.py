from __future__ import annotations

"""Stage 3: LLM leaf atomizer.

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
VALID_ARGUMENT_RE = re.compile(r"^(?:[a-z][a-z0-9_]*|[0-9]+(?:\.[0-9]+)?)$")
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
        "resolved_references": request.resolved_references,
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
15. For named FACT phrases, use the named constant, not x.
16. Preserve numeric values as arguments, including decimals.
17. Credits are not grades; use credits_per_semester(person, credit_count) for "credits per semester".
18. GPA values must be preserved: has_gpa(person, gpa_value).
19. Preserve stated course, subject, test, form, waiver, project, and requirement objects as arguments when the predicate signature needs an object.
20. Do not infer student(nam) or other domain/class atoms unless explicitly stated in the phrase or required_domain_atoms.
21. Do not create predicate names containing it, this, that, or them.
22. If a pronoun reference is unresolved, return needs_review true with empty atoms and unsupported_reason like "unresolved_pronoun_it".

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

Input phrase: Nam earns 15 credits per semester
Output:
{{
  "atoms": [
    {{"name": "credits_per_semester", "arguments": ["nam", "15"], "negated": false, "evidence_links": ["m1", "m2"], "confidence": 0.95}}
  ],
  "needs_review": false,
  "unsupported_reason": null,
  "notes": []
}}

Input phrase: Nam failed Operating Systems, a core course
Output:
{{
  "atoms": [
    {{"name": "failed_subject", "arguments": ["nam", "operating_systems"], "negated": false, "confidence": 0.95}},
    {{"name": "core_course", "arguments": ["operating_systems"], "negated": false, "confidence": 0.95}}
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


def sanitize_raw_atom_payload(item: dict) -> dict:
    """Normalize an LLM-returned atom dict before Pydantic validation.

    The LLM sometimes returns ``evidence_links`` as a list of dicts
    (e.g. ``[{"id": "m1", ...}]``) instead of the expected ``list[str]``.
    It may also return arguments as dicts.  This helper coerces the payload
    so that ``PredicateAtom.model_validate`` succeeds.
    """
    item = dict(item)

    # --- evidence_links: list[dict] → list[str] ---
    links = item.get("evidence_links", [])
    fixed_links: list[str] = []
    if isinstance(links, list):
        for link in links:
            if isinstance(link, str):
                fixed_links.append(link)
            elif isinstance(link, dict):
                # Accept any reasonable identifier field
                for key in ("id", "ID", "mention_id", "link_id"):
                    if key in link:
                        fixed_links.append(str(link[key]))
                        break
                else:
                    # Last resort: stringify the whole dict
                    fixed_links.append(str(link))
    item["evidence_links"] = fixed_links

    # --- arguments: ensure each element is a plain value ---
    args = item.get("arguments", [])
    if isinstance(args, list):
        fixed_args = []
        for arg in args:
            if isinstance(arg, dict) and "value" in arg:
                fixed_args.append(arg["value"])
            else:
                fixed_args.append(arg)
        item["arguments"] = fixed_args

    return item


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
        atoms = [
            PredicateAtom.model_validate(sanitize_raw_atom_payload(item))
            for item in atoms_payload
        ]
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
        resolved_references=dict(request.resolved_references),
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
                normalized["value"] = _normalize_argument_text(str(normalized.get("value", "")))
                if not VALID_ARGUMENT_RE.fullmatch(normalized["value"]):
                    needs_review = True
                    notes.append(f"invalid argument {arg!r} for atom {atom.name!r}.")
                normalized_args.append(normalized)
            else:
                norm_arg = _normalize_argument_text(str(arg))
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
    """Atomize a batch of requests, using a phrase consistency cache.

    If the same normalized phrase has already been successfully atomized in this
    batch, the cached atoms are reused (with variable substitution) instead of
    calling the LLM again.  This ensures semantic consistency across premises.
    """
    cache = PhraseConsistencyCache()
    results: list[AtomizationResult] = []
    for request in requests:
        cached = cache.lookup(request)
        if cached is not None:
            results.append(cached)
        else:
            result = atomize_request(request, llm)
            cache.store(request, result)
            results.append(result)
    return results


class PhraseConsistencyCache:
    """General phrase normalization cache for atomization consistency.

    Normalizes phrases using purely linguistic rules (stripping articles,
    copulae, pronouns, whitespace) — no domain-specific logic.  When two
    requests share the same normalized key, the atoms from the first successful
    result are reused with variable substitution.
    """

    def __init__(self) -> None:
        self._cache: dict[str, AtomizationResult] = {}

    @staticmethod
    def _normalize_phrase(phrase: str) -> str:
        """Produce a canonical key from a phrase using general linguistic rules."""
        text = re.sub(r"\s+", " ", phrase.strip()).lower()
        # Strip leading articles and pronouns (general English, not domain words)
        text = re.sub(
            r"^(a|an|the|this|that|these|those|some|any|each|every|all)\s+",
            "",
            text,
        )
        # Strip leading copulae / auxiliary verbs + subject pronouns
        text = re.sub(
            r"^(he|she|it|they|we|you|one)\s+",
            "",
            text,
        )
        text = re.sub(
            r"^(is|are|was|were|has|have|had|does|do|did)\s+",
            "",
            text,
        )
        # Strip "not" marker for keying — negation is tracked separately
        text = re.sub(r"^not\s+", "", text)
        # Collapse remaining whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _cache_key(cls, request: AtomizationRequest) -> str:
        phrase_key = cls._normalize_phrase(request.phrase)
        explicit_negation = _explicit_classical_negation(request.phrase)
        role = str(request.role or "")
        skeleton_kind = str(request.skeleton_kind or "")
        return "|".join(
            [
                phrase_key,
                f"negation_hint={bool(request.negation_hint)}",
                f"explicit_negation={explicit_negation}",
                f"role={role}",
                f"skeleton_kind={skeleton_kind}",
            ]
        )

    def lookup(self, request: AtomizationRequest) -> AtomizationResult | None:
        """Return a cached result with variables substituted, or None."""
        key = self._cache_key(request)
        if not key or key not in self._cache:
            return None
        cached = self._cache[key]
        if cached.needs_review or cached.unsupported_reason:
            return None
        if not cached.atoms:
            return None
        # Clone and substitute variable
        result = cached.model_copy(deep=True)
        result.request_id = request.request_id
        result.premise_id = request.premise_id
        result.role = request.role
        result.phrase = request.phrase
        result.variable = request.variable
        result.formula_path = list(request.formula_path)
        result.source_mentions = list(request.source_mentions)
        result.notes = [*result.notes, f"reused cached atoms for normalized phrase: {key!r}"]
        # Substitute cached variable → request variable
        old_var = cached.variable
        new_var = request.variable
        if old_var != new_var:
            for atom in result.atoms:
                atom.arguments = [
                    new_var if (isinstance(arg, str) and arg == old_var) else arg
                    for arg in atom.arguments
                ]
                atom.source_phrase = request.phrase
        return result

    def store(self, request: AtomizationRequest, result: AtomizationResult) -> None:
        """Cache a successful atomization result."""
        key = self._cache_key(request)
        if not key:
            return
        if result.needs_review or result.unsupported_reason:
            return
        if not result.atoms:
            return
        if key not in self._cache:
            self._cache[key] = result


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
        resolved_references=dict(request.resolved_references),
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


def _normalize_argument_text(value: str) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text):
        return text
    return _to_snake_case(text)


def _has_encoded_negation(value: str) -> bool:
    lowered = value.lower()
    return any(lowered.startswith(prefix) for prefix in ENCODED_NEGATION_PREFIXES)


def _explicit_classical_negation(text: str) -> bool:
    lower = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    if "not necessarily" in lower:
        return False
    return bool(
        re.search(
            r"\b(no|not|never|without|cannot|can't|does\s+not|do\s+not|did\s+not|has\s+not|have\s+not|had\s+not|is\s+not|are\s+not|was\s+not|were\s+not)\b",
            lower,
        )
    )


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
