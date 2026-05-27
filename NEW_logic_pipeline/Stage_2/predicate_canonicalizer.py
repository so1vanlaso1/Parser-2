from __future__ import annotations

"""Deterministic predicate cleanup after LLM leaf atomization.

The atomizer is allowed to understand English, but it is not trusted to keep a
stable predicate vocabulary or bind named constants consistently. This pass is
small and conservative: it merges known synonyms, fixes obvious subject-scoped
arities, and emits review notes when a predicate cannot be made solver-safe.
"""

import re
import unicodedata
from typing import Any

from .atomization_requests import AtomizationResult, PredicateAtom


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
}

SUBJECT_SCOPED_PREDICATES = {
    "mastered_subject",
    "explain_subject",
    "understand_subject",
    "friends_understand_subject",
    "has_knowledge_of_subject",
}

EXPECTED_ARITY = {
    "attending_tutorials": 1,
    "ask_questions": 1,
    "mastered_subject": 2,
    "explain_subject": 2,
    "understand_subject": 2,
    "friends_understand_subject": 2,
    "has_knowledge_of_subject": 2,
    "grade_count": 3,
    "grade_count_at_least": 3,
    "earned_grade_in_subject": 3,
}

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

        rewritten.atoms = _dedupe_atoms(atoms)

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

    summary = {
        "changed_atoms": changed_atoms,
        "conflict_count": conflict_count,
        "canonical_predicates": DEFAULT_KNOWN_PREDICATES,
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

    if atom.name in SUBJECT_SCOPED_PREDICATES:
        actor = _first_argument_or_variable(atom, result)
        subject = _subject_argument(atom, phrase)
        atom.arguments = [actor, subject]


def _deterministic_grade_rewrite(result: AtomizationResult) -> AtomizationResult | None:
    phrase = result.phrase
    folded = _ascii_fold(phrase).lower()
    named = _extract_named_subject(phrase)
    variable = result.variable or "x"

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
    folded = _ascii_fold(phrase).lower()
    if "'s friends" in folded or "’s friends" in folded:
        return atom.name not in {"friends_understand_subject"}
    return True


def _first_argument_or_variable(atom: PredicateAtom, result: AtomizationResult) -> str:
    if atom.arguments:
        return _argument_text(atom.arguments[0])
    return result.variable or "x"


def _subject_argument(atom: PredicateAtom, phrase: str) -> str:
    if len(atom.arguments) >= 2 and _argument_text(atom.arguments[1]) not in {"it", "this", "that", "friend", "friends"}:
        return _argument_text(atom.arguments[1])
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


__all__ = [
    "CANONICAL_PREDICATES",
    "DEFAULT_KNOWN_PREDICATES",
    "canonicalize_atomization_results",
]
