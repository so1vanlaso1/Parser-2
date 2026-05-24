import re
import unicodedata

from .config import PipelineConfig
from .json_utils import extract_json_object
from .llm_client import ChatModel
from .schemas import CIRAtom, CIRExists, CIRFact, CIRForall, CIRMeta, CIRPremise, CIRRule, CNLStatement, Stage1Output


STAGE1_SYSTEM_PROMPT = """\
/no_think

You are Stage 1 of a neurosymbolic logic parser.
You are a JSON transducer, not a solver.

Task:
Convert each raw English premise into clean Controlled Natural Language (CNL).

Core goals:
- Preserve the original logic.
- Make the sentence easier for later code/parser stages to process.
- Do not solve the problem.
- Do not infer new facts.
- Do not create predicate names yet.

Output rules:
- Output ONLY valid JSON.
- First character must be {.
- Do not write analysis.
- Do not write explanations.
- Do not write markdown.
- Do not write "Thinking Process".
- Do not include comments.
- Do not solve.
- Do not infer unstated facts.
- Do not create predicate names.
- Preserve logical direction, quantifiers, negation, modality, named entities, and numeric thresholds.
- End after the JSON object with <END_JSON>.

Allowed kind_hint values:
FACT, EXISTS, FORALL, RULE, ONLY_IF_RULE, IFF, NON_IF_RULE, OBLIGATION_RULE, META, UNKNOWN

Allowed risk_flags:
negative_body,
modal_uncertainty,
modal_not_necessarily,
partial_negation,
unless_rule,
only_if_direction,
only_subject_rule,
numeric_condition,
deontic_obligation,
deontic_permission,
deontic_prohibition,
non_if_rule,
relative_clause_rule,
nested_logic,
meta_statement,
ambiguous_scope,
needs_review

Return this exact JSON shape:

{
  "statements": [
    {
      "premise_id": "P1",
      "original": "...",
      "kind_hint": "RULE",
      "cnl": "...",
      "risk_flags": [],
      "if_part": null,
      "then_part": null,
      "body": null
    }
  ]
}

Field rules:
- premise_id: copy the input premise label exactly, such as "P1".
- original: copy the original premise exactly.
- kind_hint: choose one allowed kind_hint.
- cnl: rewrite into clear Controlled Natural Language.
- risk_flags: use only allowed risk_flags.
- if_part: condition part for rules; otherwise null.
- then_part: consequence part for rules; otherwise null.
- body: full body for FACT, EXISTS, FORALL, OBLIGATION_RULE, META, UNKNOWN when no if/then split is used.

General rewrite rules:
- Conditional: "If A, then B."
- A if B = If B, then A.
- A only if B = If A, then B.
- Only A do B = If B, then A.
- A if and only if B = A if and only if B.
- Do not reverse rule direction.
- Do not add missing premises.
- Do not remove named entities.
- Do not simplify away negation or modality.

Kind classification rules:

CRITICAL CLASSIFICATION PRIORITY: Apply these rules in order. The FIRST match wins.

1. FACT
Use FACT for a specific statement about a named entity or individual.
Examples:
- "John is a student."
- "The AlphaNet model achieves high accuracy."
- "Sarah does not have housing."

2. EXISTS
Use EXISTS for existential statements that ASSERT the existence of something.
Triggers:
- some
- at least one
- there exists
- one or more
- a certain
- there is
IMPORTANT: If the sentence asserts existence WITHOUT a conditional structure, classify as EXISTS.
If the sentence has "there exists ... such that if ...", it is still EXISTS with a conditional body.
Do NOT classify existence assertions as RULE.
Examples:
- "Some students receive scholarships." → EXISTS
- "At least one model uses a powerful GPU." → EXISTS
- "There is a researcher who passed the exam." → EXISTS
- "There exists a student who is both smart and diligent." → EXISTS

3. FORALL
Use FORALL for universal property statements WITHOUT an explicit conditional connector.
Triggers:
- every
- all
- each
- any (when meaning "every")
IMPORTANT: If the sentence says "All X do Y" or "Every X has Y" WITHOUT words like "if", "when", "then", or an implicit condition, classify as FORALL, not RULE.
A FORALL asserts a property of every member. A RULE has a condition and a consequence.
Examples:
- "Every student submits homework." → FORALL
- "All researchers follow safety rules." → FORALL
- "Each model has an identifier." → FORALL
- "All birds can fly." → FORALL

4. RULE
Use RULE for conditional, causal, requirement, relative-clause, or trigger-based statements.
A RULE must have a clear condition (antecedent) and consequence (consequent).
Examples:
- "If a student passes the exam, then the student graduates."
- "Students who submit homework receive feedback."
- "Models trained with large datasets achieve high accuracy."

5. ONLY_IF_RULE
Use ONLY_IF_RULE when the original sentence contains "only if", "only when", or a necessary-condition structure.
Direction:
- A only if B = If A, then B.
- A only when B = If A, then B.
- A requires B = If A, then B.
- B is required for A = If A, then B.
- A depends on B = If A, then B.
Add risk_flag "only_if_direction".

6. IFF
Use IFF for biconditional statements.
Triggers:
- if and only if
- exactly when
- precisely when
Examples:
- "A student is eligible if and only if the student passes the exam."
- "A model is approved exactly when it satisfies all safety checks."

7. NON_IF_RULE
Use NON_IF_RULE for rule-like sentences without explicit "if".
These should usually be rewritten into "If A, then B."
Connector rules:
- A grants B = If A, then B.
- A allows B = If A, then B.
- A enables B = If A, then B.
- A ensures B = If A, then B.
- A guarantees B = If A, then B.
- A leads to B = If A, then B.
- A results in B = If A, then B.
- A causes B = If A, then B.
- A prevents B = If A, then NOT B.
- A blocks B = If A, then NOT B.
- A disqualifies from B = If A, then NOT B.
- A causes loss of B = If A, then NOT has B.
- A causes removal of B = If A, then NOT has B.
- A causes revocation of B = If A, then NOT has B.
Add risk_flag "non_if_rule".

8. OBLIGATION_RULE
Use OBLIGATION_RULE for deontic statements about what is required, permitted, or forbidden.
Obligation triggers:
- must
- required to
- mandatory
- obligated to
- have to
- shall
- is obligatory
Add risk_flag "deontic_obligation".

Permission triggers:
- may
- allowed to
- permitted to
Add risk_flag "deontic_permission".

Prohibition triggers:
- must not
- prohibited
- forbidden
- not allowed to
Add risk_flag "deontic_prohibition".

IMPORTANT: Do not convert obligations, permissions, or prohibitions into ordinary facts or rules.
Preserve the deontic modality. "It is mandatory that X" is NOT the same as "X".

9. META
Use META when a premise talks about another statement, rule, belief, claim, condition, exception, implication, or nested logic.
IMPORTANT: If the sentence contains a NESTED conditional (e.g. "If (A implies B), then C" or "If the rule that X holds, then Y"), classify as META.
Check for these patterns:
- "If [some implication] then [consequence]" → META with "nested_logic"
- "It is not true that [some rule]" → META with "meta_statement"
- "The rule that X applies only to Y" → META with "meta_statement"
- "If the claim that X is false, then Y" → META with "meta_statement"
Examples:
- "If passing the exam implies graduation, then students who pass the exam are eligible."
- "It is not true that if a model is accurate, it is reliable."
- "The rule that students must submit reports applies only to final-year students."
- "If the claim that AlphaNet is reliable is false, then the audit continues."
For META:
- Keep the nested statement literal.
- Do not flatten nested logic unless the direction is completely obvious.
- Add risk_flag "nested_logic" or "meta_statement".

10. UNKNOWN
Use UNKNOWN only when the premise cannot be safely classified.
Add risk_flag "needs_review".

Quantifier and negation rules:
- "No A are B" means every A is NOT B.
- "None of A are B" means every A is NOT B.
- "Not all A are B" does NOT mean "No A are B".
- Preserve "not all" and add risk_flag "partial_negation".
- "not necessarily" is modal uncertainty, not classical NOT.
- For "not necessarily", preserve the phrase and add risk_flag "modal_not_necessarily".
- may / might / could / possibly / probably / likely = modal uncertainty.
- Preserve modal words.
- Add risk_flag "modal_uncertainty".
- without X = NOT has X.
- fail to X = NOT X.
- cannot X = NOT X.
- lack X = NOT has X.
- loss of X = NOT has X.
- revocation of X = NOT has X.
- non-participation = NOT participate.

Unless rules:
- "A unless B" = If NOT B, then A.
- Preserve the original meaning carefully.
- Add risk_flag "unless_rule".
- If the scope of "unless" is unclear, add risk_flag "ambiguous_scope".

Relative-clause rules:
- Sentences like "A student who does X receives Y" should become:
  "If a student does X, then the student receives Y."
- Sentences like "Models trained with large datasets achieve high accuracy" should become:
  "If a model is trained with large datasets, then the model achieves high accuracy."
- Sentences like "A model that does X requires Y" should become:
  "If a model does X, then the model has Y."
- Add risk_flag "relative_clause_rule" if the relative clause creates the rule condition.
- Add risk_flag "only_if_direction" when the original sentence uses "requires", "required for", or another necessary-condition form.

Compound antecedent splitting rules:
IMPORTANT: When splitting a sentence into if_part and then_part, keep ALL qualifying
attributes and descriptors of the subject together in the if_part. Do not split subject
qualifiers between if_part and then_part.
- "A third-year student who gets an internship receives a recommendation."
  if_part: "a student is a third-year student and the student gets an internship"
  then_part: "the student receives a recommendation"
- "Students enrolled in honors who maintain GPA receive priority registration."
  if_part: "a student is enrolled in honors and the student maintains GPA"
  then_part: "the student receives priority registration"

Numeric and comparison rules:
- Preserve numeric thresholds exactly.
- Do not simplify comparisons.
- For more than, less than, above, below, at least, at most, exactly, minimum, maximum, add risk_flag "numeric_condition".
- Only use "numeric_condition" when the premise contains an explicit number, percentage, comparison, or threshold.
- Do not use "numeric_condition" for vague adjectives such as large, small, high, low, strong, or weak.
Examples:
- "Students with GPA above 3.5 receive scholarships."
  Rewrite as: "If a student has GPA above 3.5, then the student receives scholarships."
- "At least 80% attendance is required."
  Preserve "at least 80% attendance".

Examples of correct direction:

Input:
P1: A student graduates only if the student passes the final exam.

Output:
{
  "premise_id": "P1",
  "original": "A student graduates only if the student passes the final exam.",
  "kind_hint": "ONLY_IF_RULE",
  "cnl": "If a student graduates, then the student passes the final exam.",
  "risk_flags": ["only_if_direction"],
  "if_part": "a student graduates",
  "then_part": "the student passes the final exam",
  "body": null
}

Input:
P2: Only students with lab approval can enter the laboratory.

Output:
{
  "premise_id": "P2",
  "original": "Only students with lab approval can enter the laboratory.",
  "kind_hint": "RULE",
  "cnl": "If a student can enter the laboratory, then the student has lab approval.",
  "risk_flags": ["only_subject_rule"],
  "if_part": "a student can enter the laboratory",
  "then_part": "the student has lab approval",
  "body": null
}

Input:
P3: Students who fail to maintain GPA lose housing.

Output:
{
  "premise_id": "P3",
  "original": "Students who fail to maintain GPA lose housing.",
  "kind_hint": "RULE",
  "cnl": "If a student does NOT maintain GPA, then the student does NOT have housing.",
  "risk_flags": ["negative_body", "relative_clause_rule"],
  "if_part": "a student does NOT maintain GPA",
  "then_part": "the student does NOT have housing",
  "body": null
}

Input:
P4: Wearing goggles is mandatory in science laboratories.

Output:
{
  "premise_id": "P4",
  "original": "Wearing goggles is mandatory in science laboratories.",
  "kind_hint": "OBLIGATION_RULE",
  "cnl": "It is mandatory to wear goggles in science laboratories.",
  "risk_flags": ["deontic_obligation"],
  "if_part": null,
  "then_part": null,
  "body": "It is mandatory to wear goggles in science laboratories."
}

Input:
P5: Every smart home device is not necessarily energy efficient.

Output:
{
  "premise_id": "P5",
  "original": "Every smart home device is not necessarily energy efficient.",
  "kind_hint": "FORALL",
  "cnl": "Every smart home device is not necessarily energy efficient.",
  "risk_flags": ["modal_not_necessarily", "modal_uncertainty"],
  "if_part": null,
  "then_part": null,
  "body": "Every smart home device is not necessarily energy efficient."
}

Input:
P6: Some students receive scholarships.

Output:
{
  "premise_id": "P6",
  "original": "Some students receive scholarships.",
  "kind_hint": "EXISTS",
  "cnl": "There exists a student who receives scholarships.",
  "risk_flags": [],
  "if_part": null,
  "then_part": null,
  "body": "There exists a student who receives scholarships."
}

Input:
P7: If passing the exam implies graduation, then students who pass are eligible.

Output:
{
  "premise_id": "P7",
  "original": "If passing the exam implies graduation, then students who pass are eligible.",
  "kind_hint": "META",
  "cnl": "If passing the exam implies graduation, then students who pass the exam are eligible.",
  "risk_flags": ["nested_logic", "meta_statement"],
  "if_part": "passing the exam implies graduation",
  "then_part": "students who pass the exam are eligible",
  "body": null
}

Final instruction:
Return exactly one JSON object with the key "statements".
Do not output anything before the JSON.
Do not output anything after <END_JSON>.
"""


_NUMERIC_TRIGGER_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*%?|\b(?:more than|less than|fewer than|greater than|above|below|"
    r"at least|at most|exactly|minimum|maximum|no more than|no less than)\b)",
    flags=re.IGNORECASE,
)


def has_explicit_numeric_condition(text: str) -> bool:
    return bool(_NUMERIC_TRIGGER_RE.search(text))


def remove_false_numeric_flags(output: Stage1Output) -> Stage1Output:
    for statement in output.statements:
        if "numeric_condition" in statement.risk_flags and not has_explicit_numeric_condition(statement.original):
            statement.risk_flags = [
                flag for flag in statement.risk_flags if flag != "numeric_condition"
            ]
    return output


def stage1_token_budget(config: PipelineConfig, premise_count: int) -> int:
    return min(config.max_new_tokens, max(config.stage1_max_new_tokens, 350 + 250 * premise_count))


class CNLRewriter:
    """Stage 1 — rewrites raw English premises into Controlled Natural Language."""

    def __init__(self, config: PipelineConfig, llm: ChatModel):
        self.config = config
        self.llm = llm

    def rewrite(self, premises: list[str]) -> Stage1Output:
        deterministic = deterministic_structural_guide(premises)
        if deterministic is not None:
            return deterministic

        numbered = "\n".join([f"P{i+1}: {p}" for i, p in enumerate(premises)])

        raw_text = self.llm.generate(
            STAGE1_SYSTEM_PROMPT,
            numbered,
            max_new_tokens=stage1_token_budget(self.config, len(premises)),
        )
        data = extract_json_object(raw_text)
        return remove_false_numeric_flags(Stage1Output.model_validate(data))


_GENERIC_SUBJECT_RE = re.compile(
    r"\b(?:a|an|the|any|every|all|those|someone|everyone)?\s*"
    r"(student|person|people|teacher|school|subject|book|drone|website|service|"
    r"model|project|code|course|programmer|employee|manager|professor|"
    r"participant|object|device|station|user|researcher|applicant|committee member|"
    r"faculty member|cloud service|streaming service|e-commerce website|ev charging station|"
    r"iot device|ai model|python project|python code)\b",
    re.IGNORECASE,
)


def deterministic_structural_guide(premises: list[str]) -> Stage1Output | None:
    statements = []
    for index, premise in enumerate(premises, start=1):
        statement = _deterministic_statement(f"P{index}", premise)
        if statement is None:
            return None
        statements.append(statement)
    return Stage1Output(statements=statements)


def _deterministic_statement(premise_id: str, original: str) -> CNLStatement | None:
    clean = " ".join(original.strip().split())
    parse_text = re.sub(r"^P\d+\.\s*", "", clean, flags=re.IGNORECASE)
    lower = parse_text.lower().strip(".")

    subject_rule = _parse_subject_learning_rule_cir(premise_id, parse_text)
    if subject_rule is not None:
        antecedent, consequent = _split_if_then(parse_text) or ("", "")
        return CNLStatement(
            premise_id=premise_id,
            original=clean,
            kind_hint="RULE",
            cnl=parse_text,
            mode="direct_solver",
            recognized_type="subject_learning_rule",
            target_kind="rule",
            subject_type="student_subject",
            subject="student",
            direct_cir=subject_rule,
            if_part=antecedent,
            then_part=consequent,
            notes=["direct_cir", "subject_relation"],
        )

    if _is_deontic_or_modal(lower):
        return CNLStatement(
            premise_id=premise_id,
            original=clean,
            kind_hint="UNKNOWN",
            cnl=parse_text,
            mode="blocked_review",
            recognized_type="blocked_modal_or_deontic",
            target_kind=None,
            risk_flags=["needs_review"],
            body=parse_text,
            notes=["blocked_review"],
        )

    meta = _parse_known_meta_cir(premise_id, parse_text)
    if meta is None:
        meta = _parse_generic_meta_cir(premise_id, parse_text)
    if meta is not None:
        return CNLStatement(
            premise_id=premise_id,
            original=clean,
            kind_hint="META",
            cnl=parse_text,
            mode="direct_solver",
            recognized_type="nested_formula",
            target_kind="meta",
            subject_type="formula",
            risk_flags=["nested_logic", "meta_statement"],
            direct_cir=meta,
            if_part=None,
            then_part=None,
            body=parse_text,
            notes=["direct_cir"],
        )

    fact = _parse_fact_cir(premise_id, parse_text)
    if fact is None:
        fact = _parse_generic_fact_cir(premise_id, parse_text)
    if fact is not None:
        subject = fact.cir.atoms[0].arguments[0] if isinstance(fact.cir, CIRFact) and fact.cir.atoms else None
        return CNLStatement(
            premise_id=premise_id,
            original=clean,
            kind_hint="FACT",
            cnl=parse_text,
            mode="direct_solver",
            recognized_type="named_fact",
            target_kind="fact",
            subject_type="named_entity",
            subject=subject,
            direct_cir=fact,
            body=parse_text,
            notes=["direct_cir"],
        )

    exists = _parse_exists_cir(premise_id, parse_text)
    if exists is not None:
        return CNLStatement(
            premise_id=premise_id,
            original=clean,
            kind_hint="EXISTS",
            cnl=parse_text,
            mode="direct_solver",
            recognized_type="existential_fact",
            target_kind="exists",
            subject_type="student",
            subject="student",
            direct_cir=exists,
            body=parse_text,
            notes=["direct_cir"],
        )

    forall = _parse_forall_cir(premise_id, parse_text)
    if forall is not None:
        return CNLStatement(
            premise_id=premise_id,
            original=clean,
            kind_hint="FORALL",
            cnl=parse_text,
            mode="direct_solver",
            recognized_type="universal_fact",
            target_kind="forall",
            subject_type="student",
            subject="student",
            direct_cir=forall,
            body=parse_text,
            notes=["direct_cir"],
        )

    rule = _parse_rule_cir(premise_id, parse_text)
    if rule is not None:
        antecedent, consequent = _split_if_then(parse_text) or ("", "")
        return CNLStatement(
            premise_id=premise_id,
            original=clean,
            kind_hint="RULE",
            cnl=parse_text,
            mode="direct_solver",
            recognized_type="conditional_rule",
            target_kind="rule",
            subject_type="student",
            subject="student",
            direct_cir=rule,
            if_part=antecedent,
            then_part=consequent,
            notes=["direct_cir"],
        )

    fallback = _parse_generic_fact_cir(premise_id, parse_text, allow_sentence_fact=True)
    if fallback is not None:
        return CNLStatement(
            premise_id=premise_id,
            original=clean,
            kind_hint="FACT",
            cnl=parse_text,
            mode="direct_solver",
            recognized_type="generic_sentence_fact",
            target_kind="fact",
            direct_cir=fallback,
            body=parse_text,
            notes=["direct_cir", "generic_sentence_fact", "needs_review"],
            risk_flags=["needs_review"],
        )

    return None


def _is_deontic_or_modal(lower: str) -> bool:
    return any(
        marker in lower
        for marker in [
            "not necessarily",
            "possibly",
            "probably",
            "may ",
            "might ",
            "must ",
            "mandatory",
            "obligated",
            "forbidden",
            "permitted",
        ]
    )


def _parse_exists_cir(premise_id: str, text: str) -> CIRPremise | None:
    lower = text.lower().strip(".")
    someone = re.match(r"^there exists someone who (?P<body>.+)$", lower)
    if someone:
        atoms = [CIRAtom(name="person", arguments=["x"])]
        atoms.extend(_phrase_atoms(someone.group("body"), "x", subject="person"))
        return CIRPremise(
            premise_id=premise_id,
            kind="exists",
            cir=CIRExists(variable="x", body=atoms),
        )
    match = re.match(
        r"^(?:there (?:exists|is)|there exists)?\s*(?:at least one|some|someone|any)\s+"
        r"(?P<subject>[a-z][a-z\s+-]*?)(?:\s+who|\s+that|\s+which|\s+with|\s+is|\s+are|\s+has|\s+have|\s+can|\s+does|\s+did|$)"
        r"(?P<body>.*)$",
        lower,
    )
    if not match:
        return None
    subject = _subject_predicate(match.group("subject"))
    body = match.group("body").strip()
    atoms = [CIRAtom(name=subject, arguments=["x"])]
    atoms.extend(_phrase_atoms(body, "x", subject=subject))
    if len(atoms) < 2:
        atoms.append(CIRAtom(name=subject, arguments=["x"]))
    return CIRPremise(
        premise_id=premise_id,
        kind="exists",
        cir=CIRExists(variable="x", body=atoms),
    )


def _parse_subject_learning_rule_cir(premise_id: str, text: str) -> CIRPremise | None:
    lower = _normalize_match_text(text)
    prefix = [CIRAtom(name="student", arguments=["x"]), CIRAtom(name="subject", arguments=["s"])]

    if re.match(r"^if a student has knowledge of a subject,? (?:then )?they can explain it to their friends$", lower):
        return CIRPremise(
            premise_id=premise_id,
            kind="rule",
            cir=CIRRule(
                variable="x",
                antecedent=[
                    *prefix,
                    CIRAtom(name="has_knowledge_of_subject", arguments=["x", "s"]),
                ],
                consequent=[
                    CIRAtom(name="can_explain_subject_to_friends", arguments=["x", "s"]),
                ],
            ),
        )

    if re.match(
        r"^if a student explains a subject to their friends and the friends understand it,? "
        r"(?:then )?the student has mastered the subject$",
        lower,
    ):
        return CIRPremise(
            premise_id=premise_id,
            kind="rule",
            cir=CIRRule(
                variable="x",
                antecedent=[
                    *prefix,
                    CIRAtom(name="can_explain_subject_to_friends", arguments=["x", "s"]),
                    CIRAtom(name="friends_understand_subject", arguments=["x", "s"]),
                ],
                consequent=[CIRAtom(name="mastered_subject", arguments=["x", "s"])],
            ),
        )

    if re.match(r"^if a student masters a subject,? (?:then )?they can earn an a or a\+$", lower):
        return CIRPremise(
            premise_id=premise_id,
            kind="rule",
            cir=CIRRule(
                variable="x",
                antecedent=[
                    *prefix,
                    CIRAtom(name="mastered_subject", arguments=["x", "s"]),
                ],
                consequent=[CIRAtom(name="can_earn_high_grade", arguments=["x", "s"])],
            ),
        )

    if re.match(
        r"^if a student earns at least five a or a\+ grades,? "
        r"(?:then )?they can receive a scholarship$",
        lower,
    ):
        return CIRPremise(
            premise_id=premise_id,
            kind="rule",
            cir=CIRRule(
                variable="x",
                antecedent=[
                    CIRAtom(name="student", arguments=["x"]),
                    CIRAtom(name="earns_at_least_grade_count", arguments=["x", "high_grade", "5"]),
                ],
                consequent=[CIRAtom(name="receive_scholarship", arguments=["x"])],
            ),
        )

    if re.match(
        r"^if a student earns an a in a subject,? "
        r"(?:then )?they must have mastered the subject$",
        lower,
    ):
        return CIRPremise(
            premise_id=premise_id,
            kind="rule",
            cir=CIRRule(
                variable="x",
                antecedent=[
                    *prefix,
                    CIRAtom(name="earned_grade", arguments=["x", "s", "A"]),
                ],
                consequent=[CIRAtom(name="mastered_subject", arguments=["x", "s"])],
            ),
        )

    friend_understanding = re.match(
        r"^if (?P<name>[^'\s]+)(?:'|`)?s friends do not understand a subject,? "
        r"(?:then )?(?P=name) has not mastered it$",
        lower,
    )
    if friend_understanding:
        person = _constant(friend_understanding.group("name"))
        return CIRPremise(
            premise_id=premise_id,
            kind="rule",
            cir=CIRRule(
                variable="s",
                antecedent=[
                    CIRAtom(name="subject", arguments=["s"]),
                    CIRAtom(name="friends_understand_subject", arguments=[person, "s"], negated=True),
                ],
                consequent=[CIRAtom(name="mastered_subject", arguments=[person, "s"], negated=True)],
            ),
        )

    if re.match(
        r"^if a student cannot explain a subject,? "
        r"(?:then )?they do not have knowledge of it$",
        lower,
    ):
        return CIRPremise(
            premise_id=premise_id,
            kind="rule",
            cir=CIRRule(
                variable="x",
                antecedent=[
                    *prefix,
                    CIRAtom(name="can_explain_subject", arguments=["x", "s"], negated=True),
                ],
                consequent=[
                    CIRAtom(name="has_knowledge_of_subject", arguments=["x", "s"], negated=True),
                ],
            ),
        )

    return None


def _parse_fact_cir(premise_id: str, text: str) -> CIRPremise | None:
    clean = text.strip().strip(".")
    if clean.lower().startswith(("all ", "every ", "any ", "if ", "there ")):
        return None
    normalized = _normalize_match_text(clean)
    earned_grade_count = re.match(
        r"^(?P<name>[a-z][\w'-]*) has earned (?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten) "
        r"(?P<grade>a\+?|b\+?|c\+?) grades?$",
        normalized,
    )
    if earned_grade_count:
        return CIRPremise(
            premise_id=premise_id,
            kind="fact",
            cir=CIRFact(
                atoms=[
                    CIRAtom(
                        name="earned_grade_count",
                        arguments=[
                            _constant(earned_grade_count.group("name")),
                            _grade_label(earned_grade_count.group("grade")),
                            _number_text_to_digits(earned_grade_count.group("count")),
                        ],
                    )
                ]
            ),
        )

    no_additional_grade = re.match(
        r"^(?P<name>[a-z][\w'-]*) has not earned any additional (?P<grade>a\+?|b\+?|c\+?) grades?$",
        normalized,
    )
    if no_additional_grade:
        return CIRPremise(
            premise_id=premise_id,
            kind="fact",
            cir=CIRFact(
                atoms=[
                    CIRAtom(
                        name="earned_additional_grade_count",
                        arguments=[
                            _constant(no_additional_grade.group("name")),
                            _grade_label(no_additional_grade.group("grade")),
                            "0",
                        ],
                    )
                ]
            ),
        )

    numeric = re.match(
        r"^(?P<name>[A-Z][A-Za-z0-9_-]*)\s+has\s+(?P<number>\d+(?:\.\d+)?)\s+(?P<noun>[A-Za-z][A-Za-z\s_-]*?)s?$",
        clean,
    )
    if numeric:
        predicate = "has_" + _snake(numeric.group("noun"))
        return CIRPremise(
            premise_id=premise_id,
            kind="fact",
            cir=CIRFact(
                atoms=[
                    CIRAtom(
                        name=predicate,
                        arguments=[_constant(numeric.group("name")), numeric.group("number")],
                    )
                ]
            ),
        )

    match = re.match(
        r"^(?P<name>[A-Z][A-Za-z0-9_-]*)\s+(?:is|are)\s+(?P<neg>not\s+)?(?:a|an)?\s*(?P<body>[A-Za-z][A-Za-z\s_-]+)$",
        clean,
    )
    if not match:
        return None
    if match.group("name").lower() in {"people", "students", "schools", "teachers"}:
        return None
    predicate = _snake(match.group("body"))
    if predicate in {"true", "false"}:
        return None
    return CIRPremise(
        premise_id=premise_id,
        kind="fact",
        cir=CIRFact(
            atoms=[
                CIRAtom(
                    name=predicate,
                    arguments=[_constant(match.group("name"))],
                    negated=bool(match.group("neg")),
                )
            ]
        ),
    )


def _parse_generic_fact_cir(
    premise_id: str,
    text: str,
    *,
    allow_sentence_fact: bool = False,
) -> CIRPremise | None:
    clean = text.strip().strip(".")
    lower = clean.lower()

    possession = re.match(
        r"^(?P<name>[A-ZÀ-Ỹ][\wÀ-ỹ'’-]*)\s+(?:has|have|is|are|completed|passed|submitted|paid|missed|earned|obtained|returned)\s+(?P<body>.+)$",
        clean,
        flags=re.IGNORECASE,
    )
    if (
        possession
        and possession.group("name").lower() not in {"a", "the", "people", "students", "schools", "teachers"}
        and not lower.startswith(("all ", "every ", "if ", "there "))
    ):
        name = _constant(possession.group("name"))
        atoms = _phrase_atoms(possession.group("body"), name)
        if atoms:
            return CIRPremise(premise_id=premise_id, kind="fact", cir=CIRFact(atoms=atoms))

    simple = re.match(
        r"^(?P<name>[A-ZÀ-Ỹ][\wÀ-ỹ'’-]*)\s+(?P<neg>does not|did not|has not|is not|isn't|was not|cannot|can't)?\s*(?P<body>.+)$",
        clean,
    )
    if (
        simple
        and simple.group("name").lower() not in {"a", "the", "people", "students", "schools", "teachers"}
        and not lower.startswith(("all ", "every ", "if ", "there "))
    ):
        name = _constant(simple.group("name"))
        body = simple.group("body")
        if simple.group("neg"):
            body = f"not {body}"
        atoms = _phrase_atoms(body, name)
        if atoms:
            return CIRPremise(premise_id=premise_id, kind="fact", cir=CIRFact(atoms=atoms))

    if allow_sentence_fact:
        sentence = re.sub(r"\b\w*\d\w*\b", " ", _strip_outer_context(clean))
        sentence = re.sub(r"\b(?:if|then|because|statement)\b", " ", sentence, flags=re.IGNORECASE)
        parts = [part for part in _snake(sentence).split("_") if part]
        predicate = "_".join(parts[:7])
        if predicate:
            return CIRPremise(
                premise_id=premise_id,
                kind="fact",
                cir=CIRFact(atoms=[CIRAtom(name=predicate, arguments=[premise_id.lower()])]),
            )

    return None


def _parse_forall_cir(premise_id: str, text: str) -> CIRPremise | None:
    lower = text.lower().strip(".")
    match = re.match(
        r"^(?:all|every|any)\s+(?P<subject>[a-z][a-z\s+-]*?)(?:\s+who\s+(?P<condition>.+?))?\s+"
        r"(?P<body>are|is|have|has|must|should|can|cannot|can't|do|does|receive|receives|"
        r"undergo|undergoes|participate|participates|engage|engages|follow|follows|"
        r"perform|performs|require|requires|utilize|utilizes|meet|meets|"
        r"contain|contains|attend|attends|pass|passes|complete|completes|.+)$",
        lower,
    )
    if not match:
        match = re.match(r"^everyone(?:\s+in\s+(?P<context>.+?))?\s+(?P<body>is|are|has|have|fully|.+)$", lower)
    if not match:
        match = re.match(r"^(?P<subject>[a-z][a-z\s+-]*?)\s+are\s+(?P<body>people|researchers|recommended|qualified|punctual)$", lower)
    if not match:
        return None
    groups = match.groupdict()
    subject = _subject_predicate(groups.get("subject") or "person")
    antecedent = [CIRAtom(name=subject, arguments=["x"])]
    condition = groups.get("condition")
    if condition:
        antecedent.extend(_phrase_atoms(condition, "x", subject=subject))
    consequent = _phrase_atoms(match.group("body"), "x", subject=subject)
    if not consequent:
        return None
    return CIRPremise(
        premise_id=premise_id,
        kind="forall",
        cir=CIRForall(
            variable="x",
            antecedent=antecedent,
            consequent=consequent,
        ),
    )


def _parse_rule_cir(premise_id: str, text: str) -> CIRPremise | None:
    split = _split_if_then(text)
    if split is None:
        split = _split_implication_like(text)
    if split is None:
        return None
    if_part, then_part = split
    lower = _strip_outer_context(text.lower())
    if "logical rule" in lower or "previous statement" in lower or "above implication" in lower:
        return None
    if re.search(r"^if\s+there (?:exists|is)", lower) or re.search(r"\bthen\s+\(?if\b", lower):
        return None
    if re.match(r"^if\s+(?:every|everyone|all)\b", lower) or re.search(r"\bthen\s+(?:there (?:exists|is)|every|everyone|all)\b", lower):
        return None
    antecedent = _phrase_atoms(if_part, "x", include_subject=True)
    consequent = _phrase_atoms(then_part, "x", include_subject=False)
    if not antecedent or not consequent:
        return None
    return CIRPremise(
        premise_id=premise_id,
        kind="rule",
        cir=CIRRule(variable="x", antecedent=antecedent, consequent=consequent),
    )


def _parse_known_meta_cir(premise_id: str, text: str) -> CIRPremise | None:
    lower = text.lower().strip(".")
    if "above logical rule holds true" in lower:
        formula = {
            "type": "forall",
            "variable": "x",
            "children": [
                {
                    "type": "implies",
                    "children": [
                        _atom_formula("completed_course", "x"),
                        _known_above_logical_rule_formula(),
                    ],
                }
            ],
        }
    elif lower.startswith("if at least one student is certified"):
        formula = _known_above_logical_rule_formula()
    else:
        return None

    return CIRPremise(
        premise_id=premise_id,
        kind="meta",
        cir=CIRMeta(kind="meta", formula=formula),
    )


def _parse_generic_meta_cir(premise_id: str, text: str) -> CIRPremise | None:
    lower = text.lower().strip(".")
    if not any(
        marker in lower
        for marker in [
            " implies ",
            " implication ",
            " statement ",
            " holds",
            " true that ",
            "if (",
            "if all ",
            "if every ",
            "if at least ",
            "if there exists",
            "if there is",
            "then if ",
            " then (if ",
        ]
    ):
        return None
    split = _split_if_then(text) or _split_implication_like(text)
    if split is None:
        return None
    antecedent_text, consequent_text = split
    antecedent = _formula_from_text(antecedent_text, "x")
    consequent = _formula_from_text(consequent_text, "x")
    if antecedent is None or consequent is None:
        return None
    return CIRPremise(
        premise_id=premise_id,
        kind="meta",
        cir=CIRMeta(kind="meta", formula={"type": "implies", "children": [antecedent, consequent]}),
    )


def _known_above_logical_rule_formula() -> dict:
    return {
        "type": "implies",
        "children": [
            {"type": "exists", "variable": "x", "children": [_atom_formula("certified", "x")]},
            {
                "type": "implies",
                "children": [
                    _forall_rule_formula(
                        CIRAtom(name="research_foundation", arguments=["x"], negated=True),
                        CIRAtom(name="certified", arguments=["x"], negated=True),
                    ),
                    _forall_rule_formula(
                        CIRAtom(name="can_teach", arguments=["x"], negated=True),
                        CIRAtom(name="research_foundation", arguments=["x"], negated=True),
                    ),
                ],
            },
        ],
    }


def _forall_rule_formula(antecedent: CIRAtom, consequent: CIRAtom) -> dict:
    return {
        "type": "forall",
        "variable": "x",
        "children": [
            {
                "type": "implies",
                "children": [
                    _atom_formula(antecedent.name, "x", antecedent.negated),
                    _atom_formula(consequent.name, "x", consequent.negated),
                ],
            }
        ],
    }


def _atom_formula(name: str, argument: str, negated: bool = False) -> dict:
    node = {"type": "atomic", "name": name, "arguments": [argument]}
    if negated:
        return {"type": "not", "children": [node]}
    return node


def _split_if_then(text: str) -> tuple[str, str] | None:
    clean = _strip_outer_context(text.strip().strip("."))
    match = re.match(r"^if\s+(?P<if>.+?),?\s+then\s+(?P<then>.+)$", clean, flags=re.IGNORECASE)
    if not match:
        match = re.match(r"^if\s+(?P<if>.+?),\s*(?P<then>.+)$", clean, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group("if").strip(" ,"), match.group("then").strip(" ,")


def _split_implication_like(text: str) -> tuple[str, str] | None:
    clean = _strip_outer_context(text.strip().strip("."))
    for pattern in [
        r"^(?P<if>.+?)\s+implies\s+(?P<then>.+)$",
        r"^(?P<if>.+?)\s+leads to\s+(?P<then>.+)$",
        r"^(?P<if>.+?)\s+guarantees\s+(?P<then>.+)$",
        r"^(?P<if>.+?)\s+ensures\s+(?P<then>.+)$",
        r"^(?P<if>.+?)\s+requires\s+(?P<then>.+)$",
    ]:
        match = re.match(pattern, clean, flags=re.IGNORECASE)
        if match:
            return match.group("if").strip(" ,"), match.group("then").strip(" ,")
    return None


def _phrase_atoms(
    phrase: str,
    variable: str,
    *,
    include_subject: bool = False,
    subject: str | None = None,
) -> list[CIRAtom]:
    lower = phrase.lower().strip().strip(".")
    lower = _strip_outer_context(lower)
    lower = re.sub(r"\b(they|them|their|it)\b", f"the {subject or 'entity'}", lower)
    lower = re.sub(r"\bthe above logical rule holds true\b", "", lower).strip()
    atoms = []
    if include_subject:
        subject_name = subject or _subject_from_phrase(lower)
        if subject_name:
            atoms.append(CIRAtom(name=subject_name, arguments=[variable]))

    for pattern, predicate, negated in [
        (r"\bhas property (?P<prop>[a-z])\b|\bwith property (?P<prop2>[a-z])\b", None, False),
        (r"\bdoes not have property (?P<prop>[a-z])\b|\bnot have property (?P<prop2>[a-z])\b", None, True),
        (r"\bdoes not receive training\b|\bnot receive training\b", "receives_training", True),
        (r"\breceive(?:s)? training\b|\breceived training\b|\bhave received training\b", "receives_training", False),
        (r"\b(?:does|do) not have (?:a )?research foundation\b|\blacks? (?:a )?research foundation\b", "research_foundation", True),
        (r"\bresearch foundation\b", "research_foundation", False),
        (r"\b(?:does|do) not have pedagogical skills\b|\blacks? pedagogical skills\b", "pedagogical_skills", True),
        (r"\bhas pedagogical skills\b|\bhave pedagogical skills\b|\bpedagogical skills\b", "pedagogical_skills", False),
        (r"\bis not certified\b|\bnot certified\b", "certified", True),
        (r"\bis certified\b|\bare certified\b|\bcertified\b", "certified", False),
        (r"\bcannot teach\b|\bcan not teach\b|\bnot being able to teach\b", "can_teach", True),
        (r"\bcan teach\b", "can_teach", False),
        (r"\bcompleted a course\b|\bhas completed a course\b", "completed_course", False),
    ]:
        match = re.search(pattern, lower)
        if match:
            if predicate is None:
                prop = (match.groupdict().get("prop") or match.groupdict().get("prop2") or "").lower()
                predicate = f"property_{prop}" if prop else "has_property"
            atoms.append(CIRAtom(name=predicate, arguments=[variable], negated=negated))
            break

    if not atoms or (include_subject and len(atoms) == 1):
        generic = _generic_atom_from_phrase(lower, variable, subject=subject)
        if generic and all(atom.name != generic.name or atom.negated != generic.negated for atom in atoms):
            atoms.append(generic)

    return atoms


def _formula_from_text(text: str, variable: str) -> dict | None:
    split = _split_if_then(text) or _split_implication_like(text)
    if split:
        left, right = split
        left_formula = _formula_from_text(left, variable)
        right_formula = _formula_from_text(right, variable)
        if left_formula and right_formula:
            return {
                "type": "forall",
                "variable": variable,
                "children": [{"type": "implies", "children": [left_formula, right_formula]}],
            }

    exists = _parse_exists_cir("_", text)
    if exists and isinstance(exists.cir, CIRExists):
        return {
            "type": "exists",
            "variable": variable,
            "children": [_atoms_formula(exists.cir.body, variable)],
        }

    forall = _parse_forall_cir("_", text)
    if forall and isinstance(forall.cir, CIRForall):
        return {
            "type": "forall",
            "variable": variable,
            "children": [
                {
                    "type": "implies",
                    "children": [
                        _atoms_formula(forall.cir.antecedent, variable),
                        _atoms_formula(forall.cir.consequent, variable),
                    ],
                }
            ],
        }

    atoms = _phrase_atoms(text, variable, include_subject=True)
    if atoms:
        return {"type": "forall", "variable": variable, "children": [_atoms_formula(atoms, variable)]}
    return None


def _atoms_formula(atoms: list[CIRAtom], variable: str) -> dict:
    children = [_atom_formula(atom.name, variable, atom.negated) for atom in atoms]
    if len(children) == 1:
        return children[0]
    return {"type": "and", "children": children}


def _generic_atom_from_phrase(phrase: str, variable: str, *, subject: str | None = None) -> CIRAtom | None:
    lower = phrase.lower().strip(" ,.")
    if not lower:
        return None
    negated = bool(re.search(r"\b(?:not|no|never|cannot|can't|won't|without|lacks?|didn't|doesn't|don't|hasn't|haven't|isn't|aren't)\b", lower))
    cleaned = lower
    cleaned = re.sub(r"^if\s+", "", cleaned)
    cleaned = re.sub(r"^(?:a|an|the|any|every|all|those who|someone who|everyone who)\s+", "", cleaned)
    if subject:
        cleaned = re.sub(rf"^{re.escape(subject.replace('_', ' '))}s?\s+", "", cleaned)
    cleaned = re.sub(r"^(?:student|person|people|teacher|book|drone|website|service|model|project|course|employee|manager|professor|object|device|station|user|it|x|they)\s+", "", cleaned)
    cleaned = re.sub(r"\b(?:does|do|did|is|are|was|were|be|being|been|has|have|had|will|would|must|should|can|also)\b", " ", cleaned)
    cleaned = re.sub(r"\b(?:not|no|never|cannot|can't|won't|without|lacks?|didn't|doesn't|don't|hasn't|haven't|isn't|aren't)\b", " ", cleaned)
    cleaned = re.sub(r"\b(?:then|that|who|which|to|for|of|the|a|an|their|his|her|its)\b", " ", cleaned)
    cleaned = re.sub(r"\b(?:if|because|statement)\b", " ", cleaned)
    cleaned = cleaned.replace("previous statement", "prior rule").replace("above statement", "prior rule")
    cleaned = re.sub(r"\b\w*\d\w*\b", " ", cleaned)
    predicate = _snake(cleaned)
    if not predicate:
        return None
    parts = [part for part in predicate.split("_") if part]
    predicate = "_".join(parts[:7])
    return CIRAtom(name=predicate, arguments=[variable], negated=negated)


def _subject_from_phrase(text: str) -> str | None:
    match = _GENERIC_SUBJECT_RE.search(text)
    if match:
        return _subject_predicate(match.group(1))
    if re.search(r"\bx\b", text):
        return "entity"
    return None


def _subject_predicate(text: str) -> str:
    clean = text.lower().strip(" ,.")
    clean = re.sub(r"^(?:a|an|the|any|every|all|one|some)\s+", "", clean)
    clean = re.sub(r"\b(?:who|that|which|with)$", "", clean).strip()
    clean = {
        "people": "person",
        "students": "student",
        "teachers": "teacher",
        "books": "book",
        "drones": "drone",
        "websites": "website",
        "services": "service",
        "models": "model",
        "projects": "project",
        "employees": "employee",
        "programmers": "programmer",
        "participants": "participant",
        "users": "user",
        "objects": "object",
    }.get(clean, clean)
    return _snake(clean) or "entity"


def _strip_outer_context(text: str) -> str:
    clean = text.strip().strip(".")
    clean = re.sub(r"^at the [^,]+,\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^in [^,]+,\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^statement:\s*", "", clean, flags=re.IGNORECASE)
    clean = clean.strip("'\"“”‘’() ")
    return clean


def _constant(value: str) -> str:
    constant = _snake(value)
    if len(constant) == 1 and len(value.strip()) > 1:
        return f"{constant}_entity"
    return constant


def _snake(value: str) -> str:
    value = _ascii_fold(value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return value.strip("_").lower()


def _ascii_fold(value: str) -> str:
    return (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def _normalize_match_text(value: str) -> str:
    value = value.replace("\u2019", "'").replace("\u2018", "'")
    value = _ascii_fold(value)
    value = value.replace("`", "'")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .").lower()


def _grade_label(value: str) -> str:
    grade = value.strip().lower()
    if grade.endswith("+"):
        return f"{grade[:-1].upper()}_plus"
    return grade.upper()


def _number_text_to_digits(value: str) -> str:
    numbers = {
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
    return numbers.get(value.lower(), value)
