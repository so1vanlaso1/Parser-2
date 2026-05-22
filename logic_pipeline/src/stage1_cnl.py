import re

from .config import PipelineConfig
from .json_utils import extract_json_object
from .llm_client import ChatModel
from .schemas import Stage1Output


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
        numbered = "\n".join([f"P{i+1}: {p}" for i, p in enumerate(premises)])

        raw_text = self.llm.generate(
            STAGE1_SYSTEM_PROMPT,
            numbered,
            max_new_tokens=stage1_token_budget(self.config, len(premises)),
        )
        data = extract_json_object(raw_text)
        return remove_false_numeric_flags(Stage1Output.model_validate(data))
