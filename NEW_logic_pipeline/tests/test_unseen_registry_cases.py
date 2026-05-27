from NEW_logic_pipeline import build_skeleton


def test_named_event_fact_generic_verb():
    skeleton = build_skeleton("P1", "Laura emailed the supervisor yesterday.")

    assert skeleton.kind == "FACT"
    assert skeleton.body is not None
    assert "Laura emailed" in skeleton.body.text
    assert skeleton.matched_rule == "fact.named_event_or_attribute_assertion"
    assert skeleton.matched_evidence is not None


def test_unregistered_motivates_is_not_guessed_as_non_if_rule():
    skeleton = build_skeleton("P1", "Positive feedback motivates students.")

    assert skeleton.kind == "UNKNOWN"
    assert skeleton.needs_review is True


def test_registered_negative_connector_blocks():
    skeleton = build_skeleton("P1", "Lack of approval blocks laboratory access.")

    assert skeleton.kind == "NON_IF_RULE"
    assert skeleton.antecedent is not None
    assert skeleton.consequent is not None
    assert "Lack of approval" in skeleton.antecedent.text
    assert "laboratory access" in skeleton.consequent.text
    assert skeleton.consequent.negation_hint is True
    assert skeleton.matched_rule == "non_if_rule.blocks"


def test_unregistered_supports_is_unknown():
    skeleton = build_skeleton("P1", "Regular revision supports long-term memory.")

    assert skeleton.kind == "UNKNOWN"
    assert skeleton.needs_review is True


def test_mid_sentence_provided_that_rule():
    skeleton = build_skeleton("P1", "A student succeeds provided that they practice daily.")

    assert skeleton.kind == "RULE"
    assert skeleton.antecedent is not None
    assert skeleton.consequent is not None
    assert "practice daily" in skeleton.antecedent.text
    assert "student succeeds" in skeleton.consequent.text
    assert skeleton.matched_rule == "rule.conditional_cue.provided_that"
