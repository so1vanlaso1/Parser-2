from NEW_logic_pipeline.Stage_2.atomization_requests import AtomizationResult, PredicateAtom
from NEW_logic_pipeline.Stage_2.predicate_canonicalizer import canonicalize_atomization_results


def _result(phrase, atoms, **overrides):
    values = {
        "request_id": "P1_body",
        "premise_id": "P1",
        "role": "body",
        "phrase": phrase,
        "variable": "x",
        "atoms": atoms,
    }
    values.update(overrides)
    return AtomizationResult(**values)


def test_canonicalizes_tutorial_predicate_synonyms():
    results, summary = canonicalize_atomization_results(
        [
            _result(
                "they are not attending tutorials",
                [PredicateAtom(name="attend_tutorials", arguments=["x"], negated=True)],
            )
        ]
    )

    assert results[0].atoms[0].name == "attending_tutorials"
    assert summary["changed_atoms"] == 1


def test_named_grade_count_fact_uses_constant_and_numeric_count():
    results, _ = canonicalize_atomization_results(
        [
            _result(
                "Tuấn has earned three A grades",
                [
                    PredicateAtom(name="earned", arguments=["x"]),
                    PredicateAtom(name="has_grade", arguments=["x", "three_a"]),
                ],
            )
        ]
    )

    atom = results[0].atoms[0]
    assert atom.name == "grade_count"
    assert atom.arguments == ["tuan", "a", "3"]
    assert results[0].needs_review is False


def test_subject_relations_are_binary_and_named_entities_are_bound():
    results, _ = canonicalize_atomization_results(
        [
            _result(
                "Tuấn has not mastered it",
                [PredicateAtom(name="master_it", arguments=["x"], negated=True)],
                role="consequent",
            ),
            _result(
                "Tuấn's friends do not understand a subject",
                [PredicateAtom(name="understand", arguments=["x"], negated=True)],
                role="antecedent",
            ),
        ]
    )

    assert results[0].atoms[0].name == "mastered_subject"
    assert results[0].atoms[0].arguments == ["tuan", "subject"]
    assert results[0].atoms[0].negated is True
    assert results[1].atoms[0].name == "friends_understand_subject"
    assert results[1].atoms[0].arguments == ["tuan", "subject"]


def test_grade_threshold_rewrite():
    results, _ = canonicalize_atomization_results(
        [
            _result(
                "a student earns at least five A or A+ grades",
                [PredicateAtom(name="earn", arguments=["x"])],
                role="antecedent",
            )
        ]
    )

    assert [(atom.name, atom.arguments) for atom in results[0].atoms] == [
        ("student", ["x"]),
        ("grade_count_at_least", ["x", "a_or_a_plus", "5"]),
    ]
    assert results[0].needs_review is False


def test_must_have_is_not_kept_as_deontic_after_canonicalization():
    results, _ = canonicalize_atomization_results(
        [
            _result(
                "they must have mastered the subject",
                [PredicateAtom(name="master_it", arguments=["x"])],
                role="consequent",
                needs_review=True,
                unsupported_reason="deontic_statement_requires_special_handling",
                notes=["request has modality/deontic hint 'deontic_obligation'; atomization requires review."],
            )
        ]
    )

    assert results[0].atoms[0].name == "mastered_subject"
    assert results[0].atoms[0].arguments == ["x", "subject"]
    assert results[0].needs_review is False
    assert results[0].unsupported_reason is None
