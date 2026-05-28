from __future__ import annotations

"""Deterministic predicate cleanup after LLM leaf atomization.

The atomizer is allowed to understand English, but it is not trusted to keep a
stable predicate vocabulary or bind named constants consistently. This pass is
small and conservative: it merges known synonyms, fixes obvious subject-scoped
arities, and emits review notes when a predicate cannot be made solver-safe.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .atomization_requests import AtomizationResult, PredicateAtom

VALID_PREDICATE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


@dataclass(frozen=True)
class PredicateSignature:
    name: str
    arity: int
    arg_roles: tuple[str, ...]


CANONICAL_PREDICATES: dict[str, str] = {
    "attend_tutorials": "attending_tutorials",
    "attends_tutorials": "attending_tutorials",
    "tutorial_attendance": "attending_tutorials",
    "ask_question": "ask_questions",
    "asking_questions": "ask_questions",
    "master": "mastered_subject",
    "masters": "mastered_subject",
    "master_subject": "mastered_subject",
    "mastered_it": "mastered_subject",
    "master_it": "mastered_subject",
    "explain": "explain_subject",
    "explains": "explain_subject",
    "explain_to": "explain_subject",
    "explain_it": "explain_subject",
    "cannot_explain": "explain_subject",
    "understand": "understand_subject",
    "understands": "understand_subject",
    "understand_it": "understand_subject",
    "understand_subject": "understand_subject",
    "has_knowledge_of": "has_knowledge_of_subject",
    "have_knowledge_of": "has_knowledge_of_subject",
    "have_knowledge_of_it": "has_knowledge_of_subject",
    "has_knowledge_of_it": "has_knowledge_of_subject",
    "knowledge_of": "has_knowledge_of_subject",
    "earned": "earned_grade",
    "earn": "earned_grade",
    "earn_grade": "earned_grade",
    "has_grade": "earned_grade",
    "have_completed_registration_form": "completed_registration_form",
    "has_completed_registration_form": "completed_registration_form",
    "complete_registration_form": "completed_registration_form",
    "completed_form": "completed_registration_form",
    "signed_waiver": "signed_liability_waiver",
    "has_signed_liability_waiver": "signed_liability_waiver",
    "eligibility": "eligible",
    "is_eligible": "eligible",
    "taken_test": "taken_required_test",
    "has_taken_test": "taken_required_test",
    "has_taken_required_test": "taken_required_test",
    "take_required_test": "taken_required_test",
    "meet_gpa_requirement": "meets_gpa_requirement",
    "requirement_met": "meets_gpa_requirement",
    "retaken_it": "retaken_subject",
    "retake_it": "retaken_subject",
    "retaken": "retaken_subject",
    "failed": "failed_subject",
    "fail": "failed_subject",
}

SUBJECT_SCOPED_PREDICATES = {
    "mastered_subject",
    "explain_subject",
    "understand_subject",
    "friends_understand_subject",
    "has_knowledge_of_subject",
    "failed_subject",
    "retaken_subject",
}

PREDICATE_SIGNATURES: dict[str, PredicateSignature] = {
    "student": PredicateSignature("student", 1, ("person",)),
    "subject": PredicateSignature("subject", 1, ("subject",)),
    "attending_tutorials": PredicateSignature("attending_tutorials", 1, ("person",)),
    "ask_questions": PredicateSignature("ask_questions", 1, ("person",)),
    "preparing_for_exam": PredicateSignature("preparing_for_exam", 1, ("person",)),
    "understand_material": PredicateSignature("understand_material", 1, ("person",)),
    "contains_knowledge": PredicateSignature("contains_knowledge", 1, ("subject",)),
    "mastered_subject": PredicateSignature("mastered_subject", 2, ("person", "subject")),
    "explain_subject": PredicateSignature("explain_subject", 2, ("person", "subject")),
    "understand_subject": PredicateSignature("understand_subject", 2, ("person", "subject")),
    "friends_understand_subject": PredicateSignature("friends_understand_subject", 2, ("person", "subject")),
    "has_knowledge_of_subject": PredicateSignature("has_knowledge_of_subject", 2, ("person", "subject")),
    "grade_count": PredicateSignature("grade_count", 3, ("person", "grade", "count")),
    "grade_count_at_least": PredicateSignature("grade_count_at_least", 3, ("person", "grade_group", "count")),
    "earned_grade": PredicateSignature("earned_grade", 2, ("person", "grade")),
    "earned_grade_in_subject": PredicateSignature("earned_grade_in_subject", 3, ("person", "grade", "subject")),
    "has_gpa": PredicateSignature("has_gpa", 2, ("person", "gpa_value")),
    "credits_per_semester": PredicateSignature("credits_per_semester", 2, ("person", "credit_count")),
    "failed_subject": PredicateSignature("failed_subject", 2, ("person", "subject")),
    "retaken_subject": PredicateSignature("retaken_subject", 2, ("person", "subject")),
    "completed_registration_form": PredicateSignature("completed_registration_form", 1, ("person",)),
    "signed_liability_waiver": PredicateSignature("signed_liability_waiver", 1, ("person",)),
    "eligible": PredicateSignature("eligible", 1, ("person",)),
    "core_course": PredicateSignature("core_course", 1, ("subject",)),
    "taken_required_test": PredicateSignature("taken_required_test", 1, ("person",)),
    "meets_gpa_requirement": PredicateSignature("meets_gpa_requirement", 1, ("person",)),
}
EXPECTED_ARITY = {name: signature.arity for name, signature in PREDICATE_SIGNATURES.items()}

DEFAULT_KNOWN_PREDICATES = sorted(
    {
        *CANONICAL_PREDICATES.values(),
        "student",
        "subject",
        "contains_knowledge",
        "grade_count",
        "grade_count_at_least",
        "earned_grade_in_subject",
        "friends_understand_subject",
        "scholarship",
        *PREDICATE_SIGNATURES.keys(),
    }
)

VARIABLE_NAMES = {"u", "v", "w", "x", "y", "z"}
BLOCKED_NAME_STARTS = {
    "a",
    "an",
    "all",
    "any",
    "each",
    "every",
    "for",
    "if",
    "no",
    "some",
    "the",
    "there",
    "they",
    "when",
    "whenever",
}
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def canonicalize_atomization_results(results: list[AtomizationResult]) -> tuple[list[AtomizationResult], dict[str, Any]]:
    canonicalized: list[AtomizationResult] = []
    changed_atoms = 0
    conflict_count = 0

    for result in results:
        rewritten = result.model_copy(deep=True)
        deterministic = _deterministic_grade_rewrite(rewritten)
        if deterministic is not None:
            rewritten = deterministic
            changed_atoms += 1

        notes = list(rewritten.notes)
        atoms: list[PredicateAtom] = []
        for atom in rewritten.atoms:
            before = (atom.name, tuple(atom.arguments), atom.negated)
            _canonicalize_atom(atom, rewritten)
            after = (atom.name, tuple(atom.arguments), atom.negated)
            if after != before:
                changed_atoms += 1
                notes.append(
                    f"canonicalized atom {before[0]!r}{list(before[1])!r} -> {after[0]!r}{list(after[1])!r}."
                )
            atoms.append(atom)

        rewritten.atoms = _drop_implicit_pronoun_consequent_domains(
            _dedupe_atoms(atoms),
            rewritten,
        )
        rewritten.atoms = _drop_unsafe_inferred_domain_atoms(rewritten.atoms, rewritten)
        notes = _dedupe_strings([*notes, *rewritten.notes])

        for atom in rewritten.atoms:
            expected = EXPECTED_ARITY.get(atom.name)
            if expected is not None and len(atom.arguments) != expected:
                conflict_count += 1
                rewritten.needs_review = True
                notes.append(
                    f"predicate_canonicalization_conflict: {atom.name!r} expects arity {expected}, got {len(atom.arguments)}."
                )

        rewritten.notes = _dedupe_strings(notes)
        _clear_logical_must_have_review(rewritten)
        canonicalized.append(rewritten)

    # --- Open registry: collect all clean predicates seen in this batch ---
    seen_predicates: set[str] = set()
    observed_arities: dict[str, set[int]] = {}
    for result in canonicalized:
        for atom in result.atoms:
            # Only register predicates from results that are not flagged for review
            if not result.needs_review and VALID_PREDICATE_RE.fullmatch(atom.name):
                seen_predicates.add(atom.name)
                observed_arities.setdefault(atom.name, set()).add(len(atom.arguments))

    summary = {
        "changed_atoms": changed_atoms,
        "conflict_count": conflict_count,
        "canonical_predicates": sorted(set(DEFAULT_KNOWN_PREDICATES) | seen_predicates),
        "predicate_signatures": _signature_summary(seen_predicates, observed_arities),
    }
    return canonicalized, summary


def _canonicalize_atom(atom: PredicateAtom, result: AtomizationResult) -> None:
    phrase = atom.source_phrase or result.phrase
    original_name = atom.name
    atom.name = CANONICAL_PREDICATES.get(atom.name, atom.name)
    atom.arguments = [_canonical_argument(arg) for arg in atom.arguments]

    if _is_friends_understand_phrase(phrase, original_name):
        owner = _extract_possessive_owner(phrase)
        actor = owner or _first_argument_or_variable(atom, result)
        atom.name = "friends_understand_subject"
        atom.arguments = [actor, "subject"]
        return

    named_subject = _extract_named_subject(phrase)
    if named_subject and _should_bind_named_subject(atom, phrase):
        if atom.arguments:
            atom.arguments[0] = named_subject
        else:
            atom.arguments = [named_subject]

    if _predicate_contains_unresolved_pronoun(original_name) and not _has_resolved_references(result):
        result.needs_review = True
        result.unsupported_reason = result.unsupported_reason or _unresolved_pronoun_reason(original_name)

    if atom.name in SUBJECT_SCOPED_PREDICATES:
        actor = _first_argument_or_variable(atom, result)
        subject = _subject_argument(atom, phrase, result)
        atom.arguments = [actor, subject]
        return

    signature = PREDICATE_SIGNATURES.get(atom.name)
    if signature and signature.arity == 1 and len(atom.arguments) > 1:
        atom.arguments = [_first_argument_or_variable(atom, result)]


def _deterministic_grade_rewrite(result: AtomizationResult) -> AtomizationResult | None:
    phrase = result.phrase
    folded = _ascii_fold(phrase).lower()
    named = _extract_named_subject(phrase)
    variable = result.variable or "x"

    credits = re.match(
        r"^(?P<person>[a-z][a-z0-9_ .'-]*)\s+earns?\s+(?P<count>\d+(?:\.\d+)?)\s+credits?\s+per\s+semester$",
        folded,
        flags=re.I,
    )
    if credits and named:
        rewritten = result.model_copy(deep=True)
        rewritten.atoms = [
            PredicateAtom(
                name="credits_per_semester",
                arguments=[named, credits.group("count")],
                negated=False,
                source_phrase=phrase,
                confidence=0.99,
            )
        ]
        rewritten.needs_review = False
        rewritten.unsupported_reason = None
        rewritten.notes = _dedupe_strings([*rewritten.notes, "deterministic credits-per-semester rewrite applied."])
        return rewritten

    gpa = re.match(
        r"^(?P<person>[a-z][a-z0-9_ .'-]*)\s+has\s+(?:a\s+)?gpa\s+of\s+(?P<value>\d+(?:\.\d+)?)$",
        folded,
        flags=re.I,
    )
    if gpa and named:
        rewritten = result.model_copy(deep=True)
        rewritten.atoms = [
            PredicateAtom(
                name="has_gpa",
                arguments=[named, gpa.group("value")],
                negated=False,
                source_phrase=phrase,
                confidence=0.99,
            )
        ]
        rewritten.needs_review = False
        rewritten.unsupported_reason = None
        rewritten.notes = _dedupe_strings([*rewritten.notes, "deterministic GPA rewrite applied."])
        return rewritten

    failed_subject = re.match(
        r"^(?P<person>[a-z][a-z0-9_ .'-]*)\s+failed\s+(?P<subject>[a-z][a-z0-9_ .'-]*?)(?:,\s+a\s+(?P<attribute>core\s+course))?$",
        folded,
        flags=re.I,
    )
    if failed_subject and named:
        subject = _normalize_constant(failed_subject.group("subject"))
        atoms = [
            PredicateAtom(
                name="failed_subject",
                arguments=[named, subject],
                negated=False,
                source_phrase=phrase,
                confidence=0.99,
            )
        ]
        if failed_subject.group("attribute"):
            atoms.append(
                PredicateAtom(
                    name="core_course",
                    arguments=[subject],
                    negated=False,
                    source_phrase=phrase,
                    confidence=0.99,
                )
            )
        rewritten = result.model_copy(deep=True)
        rewritten.atoms = atoms
        rewritten.needs_review = False
        rewritten.unsupported_reason = None
        rewritten.notes = _dedupe_strings([*rewritten.notes, "deterministic failed-subject rewrite applied."])
        return rewritten

    retaken_pronoun = re.match(
        r"^(?P<person>[a-z][a-z0-9_ .'-]*)\s+has\s+not\s+retaken\s+(?P<object>it|this|that)\s+yet$",
        folded,
        flags=re.I,
    )
    if retaken_pronoun and named:
        resolved = _resolved_reference(result, retaken_pronoun.group("object"))
        rewritten = result.model_copy(deep=True)
        rewritten.atoms = [
            PredicateAtom(
                name="retaken_subject",
                arguments=[named, resolved or "subject"],
                negated=True,
                source_phrase=phrase,
                confidence=0.99 if resolved else 0.7,
            )
        ]
        rewritten.needs_review = not bool(resolved)
        rewritten.unsupported_reason = None if resolved else f"unresolved_pronoun_{retaken_pronoun.group('object')}"
        rewritten.notes = _dedupe_strings([*rewritten.notes, "deterministic retaken-subject rewrite applied."])
        return rewritten

    additional = re.match(
        r"^(?P<person>[a-z][a-z0-9_ .'-]*)\s+has\s+not\s+earned\s+any\s+additional\s+(?P<grade>a\+|a\s+plus|a)\s+grades?$",
        folded,
        flags=re.I,
    )
    if additional and named:
        grade = _canonical_grade(additional.group("grade"))
        rewritten = result.model_copy(deep=True)
        rewritten.atoms = [
            PredicateAtom(name="grade_count", arguments=[named, grade, "0"], negated=False, source_phrase=phrase, confidence=0.99)
        ]
        rewritten.needs_review = False
        rewritten.unsupported_reason = None
        rewritten.notes = _dedupe_strings([*rewritten.notes, "deterministic grade-count rewrite applied."])
        return rewritten

    count_fact = re.match(
        r"^(?P<person>[a-z][a-z0-9_ .'-]*)\s+has\s+earned\s+(?P<count>[a-z0-9]+)\s+(?P<grade>a\+|a\s+plus|a)\s+grades?$",
        folded,
        flags=re.I,
    )
    if count_fact and named:
        count = _canonical_count(count_fact.group("count"))
        grade = _canonical_grade(count_fact.group("grade"))
        rewritten = result.model_copy(deep=True)
        rewritten.atoms = [
            PredicateAtom(name="grade_count", arguments=[named, grade, count], negated=False, source_phrase=phrase, confidence=0.99)
        ]
        rewritten.needs_review = False
        rewritten.unsupported_reason = None
        rewritten.notes = _dedupe_strings([*rewritten.notes, "deterministic grade-count rewrite applied."])
        return rewritten

    threshold = re.match(
        r"^a\s+student\s+earns?\s+at\s+least\s+(?P<count>[a-z0-9]+)\s+a\s+or\s+a\+\s+grades?$",
        folded,
        flags=re.I,
    )
    if threshold:
        count = _canonical_count(threshold.group("count"))
        rewritten = result.model_copy(deep=True)
        rewritten.atoms = [
            PredicateAtom(name="student", arguments=[variable], negated=False, source_phrase=phrase, confidence=0.99),
            PredicateAtom(name="grade_count_at_least", arguments=[variable, "a_or_a_plus", count], negated=False, source_phrase=phrase, confidence=0.99),
        ]
        rewritten.needs_review = False
        rewritten.unsupported_reason = None
        rewritten.notes = _dedupe_strings([*rewritten.notes, "deterministic grade-threshold rewrite applied."])
        return rewritten

    grade_in_subject = re.match(
        r"^a\s+student\s+earns?\s+an?\s+(?P<grade>a\+|a\s+plus|a)\s+in\s+a\s+subject$",
        folded,
        flags=re.I,
    )
    if grade_in_subject:
        grade = _canonical_grade(grade_in_subject.group("grade"))
        rewritten = result.model_copy(deep=True)
        rewritten.atoms = [
            PredicateAtom(name="student", arguments=[variable], negated=False, source_phrase=phrase, confidence=0.99),
            PredicateAtom(name="earned_grade_in_subject", arguments=[variable, grade, "subject"], negated=False, source_phrase=phrase, confidence=0.99),
        ]
        rewritten.needs_review = False
        rewritten.unsupported_reason = None
        rewritten.notes = _dedupe_strings([*rewritten.notes, "deterministic grade-subject rewrite applied."])
        return rewritten

    return None


def _clear_logical_must_have_review(result: AtomizationResult) -> None:
    if not re.search(r"\bmust\s+(have|be|satisfy|meet)\b", result.phrase, flags=re.I):
        return
    if result.unsupported_reason == "deontic_statement_requires_special_handling":
        result.unsupported_reason = None
    result.notes = [
        note
        for note in result.notes
        if "deontic" not in note.lower() and "modality/deontic" not in note.lower()
    ]
    if result.unsupported_reason is None and not any("conflict" in note.lower() or "invalid" in note.lower() for note in result.notes):
        result.needs_review = False


def _is_friends_understand_phrase(phrase: str, predicate_name: str) -> bool:
    folded = _ascii_fold(phrase).lower()
    if "friend" not in folded:
        return False
    if "understand" not in folded:
        return False
    canonical_name = CANONICAL_PREDICATES.get(predicate_name, predicate_name)
    return canonical_name in {"understand_subject", "friends_understand_subject"}


def _extract_possessive_owner(phrase: str) -> str | None:
    match = re.search(r"\b(?P<name>[A-ZÀ-Ỹ][\wÀ-ỹ.'-]*)['’]s\s+friends\b", phrase)
    if not match:
        return None
    return _normalize_constant(match.group("name"))


def _extract_named_subject(phrase: str) -> str | None:
    clean = phrase.strip()
    if not clean:
        return None
    first = clean.split()[0].strip(" ,.;:()")
    lowered = first.lower().removesuffix("'s").removesuffix("’s")
    if lowered in BLOCKED_NAME_STARTS:
        return None
    if first.lower() in {"dr.", "prof.", "professor"}:
        words = clean.split()
        if len(words) >= 2:
            return _normalize_constant(f"{words[0]} {words[1].strip(' ,.;:()')}")
    bare = first.removesuffix("'s").removesuffix("’s")
    if not bare:
        return None
    if bare[0].isupper() or any(ord(char) > 127 for char in bare) or re.match(r"^[A-Z]{2,}[0-9]*$", bare):
        return _normalize_constant(bare)
    return None


def _should_bind_named_subject(atom: PredicateAtom, phrase: str) -> bool:
    signature = PREDICATE_SIGNATURES.get(atom.name)
    if signature is not None and signature.arg_roles and signature.arg_roles[0] != "person":
        return False
    folded = _ascii_fold(phrase).lower()
    if "'s friends" in folded or "’s friends" in folded:
        return atom.name not in {"friends_understand_subject"}
    return True


def _first_argument_or_variable(atom: PredicateAtom, result: AtomizationResult) -> str:
    if atom.arguments:
        return _argument_text(atom.arguments[0])
    return result.variable or "x"


def _subject_argument(atom: PredicateAtom, phrase: str, result: AtomizationResult) -> str:
    if len(atom.arguments) >= 2 and _argument_text(atom.arguments[1]) not in {"it", "this", "that", "friend", "friends"}:
        return _argument_text(atom.arguments[1])
    for pronoun in ("it", "this", "that", "them"):
        resolved = _resolved_reference(result, pronoun)
        if resolved:
            return resolved
    folded = _ascii_fold(phrase).lower()
    if "subject" in folded or re.search(r"\bit\b", folded):
        return "subject"
    return _argument_text(atom.arguments[1]) if len(atom.arguments) >= 2 else "subject"


def _canonical_argument(argument: Any) -> Any:
    if isinstance(argument, dict):
        normalized = dict(argument)
        normalized["value"] = _canonical_argument(normalized.get("value", ""))
        return normalized
    raw = str(argument).strip()
    if re.fullmatch(r"[0-9]+", raw):
        return raw
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", raw):
        return raw
    value = _normalize_constant(str(argument))
    mapping = {
        "a_plus": "a_plus",
        "a": "a",
        "a_or_a": "a_or_a_plus",
        "a_or_a_plus": "a_or_a_plus",
        "it": "subject",
        "the_subject": "subject",
        "friend": "friends",
    }
    return mapping.get(value, value)


def _resolved_reference(result: AtomizationResult, pronoun: str) -> str | None:
    refs = getattr(result, "resolved_references", {}) or {}
    value = refs.get(pronoun.lower())
    return _normalize_constant(value) if value else None


def _has_resolved_references(result: AtomizationResult) -> bool:
    refs = getattr(result, "resolved_references", {}) or {}
    return any(str(value).strip() for value in refs.values())


def _predicate_contains_unresolved_pronoun(name: str) -> bool:
    return bool(re.search(r"(?:^|_)(it|this|that|them)(?:_|$)", str(name or "")))


def _unresolved_pronoun_reason(name: str) -> str:
    match = re.search(r"(?:^|_)(it|this|that|them)(?:_|$)", str(name or ""))
    return f"unresolved_pronoun_{match.group(1)}" if match else "unresolved_pronoun"


def _canonical_grade(value: str) -> str:
    folded = _ascii_fold(value).lower().replace(" ", "_")
    if folded in {"a+", "a_plus"}:
        return "a_plus"
    return "a"


def _canonical_count(value: str) -> str:
    lowered = value.lower()
    return NUMBER_WORDS.get(lowered, lowered)


def _normalize_constant(value: str) -> str:
    folded = _ascii_fold(value)
    folded = re.sub(r"['’]s$", "", folded)
    folded = re.sub(r"[^A-Za-z0-9]+", "_", folded).strip("_").lower()
    folded = re.sub(r"_+", "_", folded)
    if not folded:
        return "unknown"
    if folded[0].isdigit():
        return f"id_{folded}"
    return folded


def _ascii_fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _dedupe_atoms(atoms: list[PredicateAtom]) -> list[PredicateAtom]:
    seen: set[tuple[str, tuple[str, ...], bool]] = set()
    output: list[PredicateAtom] = []
    for atom in atoms:
        key = (atom.name, tuple(_argument_key(arg) for arg in atom.arguments), atom.negated)
        if key in seen:
            continue
        seen.add(key)
        output.append(atom)
    return output


def _drop_implicit_pronoun_consequent_domains(
    atoms: list[PredicateAtom],
    result: AtomizationResult,
) -> list[PredicateAtom]:
    if result.role != "consequent" or len(atoms) <= 1:
        return atoms
    if not re.match(r"^(he|she|it|they|we|you|one)\b", _ascii_fold(result.phrase).lower()):
        return atoms
    variable = result.variable or "x"
    filtered = [
        atom
        for atom in atoms
        if not (
            len(atom.arguments) == 1
            and _argument_text(atom.arguments[0]) == variable
            and "_" not in atom.name
            and not atom.negated
        )
    ]
    return filtered or atoms


def _drop_unsafe_inferred_domain_atoms(
    atoms: list[PredicateAtom],
    result: AtomizationResult,
) -> list[PredicateAtom]:
    if len(atoms) <= 1 or not _extract_named_subject(result.phrase):
        return atoms
    mentioned = {
        str(mention.get("canonical"))
        for mention in getattr(result, "source_mentions", []) or []
        if str(mention.get("semantic_role")) in {"domain_type", "object_type", "subject", "attribute"}
    }
    required = {
        str(item.get("predicate"))
        for item in getattr(result, "required_domain_atoms", []) or []
    }
    filtered: list[PredicateAtom] = []
    for atom in atoms:
        signature = PREDICATE_SIGNATURES.get(atom.name)
        looks_like_class_atom = (
            signature is not None
            and signature.arity == 1
            and signature.arg_roles[0] in {"person", "subject"}
            and atom.name in {"student", "subject"}
            and atom.name not in mentioned
            and atom.name not in required
            and not atom.negated
        )
        if looks_like_class_atom:
            result.notes = _dedupe_strings([*result.notes, f"dropped unsafe inferred domain atom: {atom.name}."])
            continue
        filtered.append(atom)
    return filtered or atoms


def _argument_key(argument: Any) -> str:
    return _argument_text(argument)


def _argument_text(argument: Any) -> str:
    if isinstance(argument, dict):
        return str(argument.get("value", ""))
    return str(argument)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _signature_summary(seen_predicates: set[str], observed_arities: dict[str, set[int]]) -> dict[str, dict[str, Any]]:
    signatures: dict[str, dict[str, Any]] = {}
    for name in sorted(seen_predicates):
        known = PREDICATE_SIGNATURES.get(name)
        if known is not None:
            signatures[name] = {"arity": known.arity, "roles": list(known.arg_roles)}
            continue
        arities = observed_arities.get(name, set())
        if len(arities) != 1:
            continue
        arity = next(iter(arities))
        if arity > 0:
            signatures[name] = {"arity": arity, "roles": ["entity"] * arity}
    return signatures


__all__ = [
    "CANONICAL_PREDICATES",
    "DEFAULT_KNOWN_PREDICATES",
    "PREDICATE_SIGNATURES",
    "PredicateSignature",
    "canonicalize_atomization_results",
]
