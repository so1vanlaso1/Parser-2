# CIR-Only Parser Migration

## Summary
Migrate the repo from “LLM may compile AST” to a strict staged compiler:

`raw input -> Stage 1 structural guide/direct CIR -> Stage 2 CIR examples -> Stage 3A Qwen CIR extraction -> Stage 3B Python CIR compiler -> validator -> targeted CIR repair -> question parser -> solver bridge`

This includes code and guide updates. Qwen may extract meaning as CIR JSON, but must never emit `LogicNode`, solver syntax, or final predicate logic.

## Key Changes

- Add CIR schemas in [schemas.py](/C:/Users/hi/Desktop/Praser/logic_pipeline/src/schemas.py):
  `CIRAtom`, `CIRFact`, `CIRExists`, `CIRForall`, `CIRRule`, `CIRMeta`, `CIRLink`, `CIRPremise`, `Stage3CIROutput`.
  Allowed CIR kinds are exactly `fact`, `exists`, `forall`, `rule`, `meta`.
- Replace Stage 1’s current CNL-only contract with a structural guide model:
  `mode`, `recognized_type`, `target_kind`, `subject_type`, `subject`, `slots`, `risk_flags`, `notes`, optional `direct_cir`.
- Keep backward-compatible fields (`cnl`, `kind_hint`) during migration only where tests or existing pipeline code need them, but make CIR the source of truth for compilation.
- Add Stage 0 choice detection in [stage0_input.py](/C:/Users/hi/Desktop/Praser/logic_pipeline/src/stage0_input.py):
  tag choices as `symbolic_formula`, `natural_language`, or `open_ended`; symbolic choices must bypass normal atomization.

## Implementation Changes

- Stage 1 in [stage1_cnl.py](/C:/Users/hi/Desktop/Praser/logic_pipeline/src/stage1_cnl.py):
  - Add a deterministic pre-pass for simple named facts and safe direct rules.
  - Emit `direct_solver` with `direct_cir` for simple safe premises such as named constants and numeric facts.
  - Emit `llm_guided` for structurally recognizable hard cases such as nested quantifier implications.
  - Emit `blocked_review` for modal-heavy, deontic-unsafe, open-ended, or ambiguous premises.
  - Do not call Stage 3/Qwen for `direct_solver` premises.

- Stage 2 in [stage2_rag.py](/C:/Users/hi/Desktop/Praser/logic_pipeline/src/stage2_rag.py):
  - Change examples from `CNL -> AST` to `structural guide/CNL -> CIR`.
  - Update [structural_examples.jsonl](/C:/Users/hi/Desktop/Praser/logic_pipeline/data/structural_examples.jsonl) to store `cir` instead of `ast`.
  - Format retrieved examples as `Input` and `CIR JSON`, never `AST`.

- Stage 3 in [stage3_ast.py](/C:/Users/hi/Desktop/Praser/logic_pipeline/src/stage3_ast.py):
  - Rename behavior conceptually to CIR extraction plus deterministic compilation.
  - Replace `PredicateFrameOutput` prompt with `Stage3CIROutput` prompt.
  - Delete or disable `_compile_full_ast` fallback.
  - Failed CIR extraction goes to targeted CIR repair or `needs_review`, never full AST generation.
  - Add deterministic `CIR -> LogicNode` compiler:
    facts compile to atom conjunctions, rules compile to quantified implications, quantifiers preserve scope, and meta CIR compiles to formula-level implication trees.

- META handling in [meta_formula.py](/C:/Users/hi/Desktop/Praser/logic_pipeline/src/meta_formula.py):
  - Preserve current formula-level resolution behavior.
  - Change input source from text split/leaf atomizer where possible to CIR meta objects.
  - Represent nested links like `p1 -> (p2 -> p3)` without flattening into Horn rules.

- Validator and repair:
  - Extend [stage4_validate.py](/C:/Users/hi/Desktop/Praser/logic_pipeline/src/stage4_validate.py) issues with stable `code` and `repair_hint`.
  - Add checks for named fact using variable, unbound variable, number in predicate name, sentence-like predicate, modal/deontic hard truth, flattened meta, symbolic choice atomized as text, and open-ended query as predicate.
  - Replace AST repair in [stage5_repair.py](/C:/Users/hi/Desktop/Praser/logic_pipeline/src/stage5_repair.py) with CIR repair first.
  - Use deterministic repairs for simple error codes and LLM CIR repair only for meta/nested semantic fixes.

- Question parser in [question_parser.py](/C:/Users/hi/Desktop/Praser/logic_pipeline/src/question_parser.py):
  - Split into three paths:
    natural-language choices -> CIR extraction then deterministic compile,
    symbolic choices -> symbolic formula parser,
    open-ended questions -> `blocked_review`.
  - Add a small symbolic parser for formulas containing `¬`, `->`, `→`, parentheses, and known predicate names.
  - Prevent symbolic formulas from becoming long text predicates.

- Canonicalizer in [predicate_canonicalizer.py](/C:/Users/hi/Desktop/Praser/logic_pipeline/src/predicate_canonicalizer.py):
  - Keep predicate normalization but add guards against merging semantic roles.
  - Reject numeric values embedded in predicate names.
  - Do not merge obligation with actual truth, eligibility with access, or modal possibility with fact.

- Documentation:
  - Update [neurosymbolic_logic_parser_full_implementation_guide.md](/C:/Users/hi/Desktop/Praser/neurosymbolic_logic_parser_full_implementation_guide.md) and [walkthrough.md](/C:/Users/hi/Desktop/Praser/walkthrough.md) to describe the CIR-only contract and remove “CNL -> AST” examples.

## Test Plan

- Add schema tests for valid and invalid CIR objects.
- Add Stage 1 tests for:
  direct named fact, numeric fact, nested quantifier implication, deontic blocked/review behavior.
- Add Stage 3 tests proving:
  direct CIR skips Qwen, Qwen returns CIR only, no full AST fallback is called, CIR compiles deterministically to `LogicNode`.
- Add validator tests for:
  `NAMED_FACT_USED_VARIABLE`, `NUMBER_IN_PREDICATE_NAME`, `SENTENCE_LIKE_PREDICATE`, `META_FLATTENED_TO_HORN`, `SYMBOLIC_CHOICE_ATOMIZED_AS_TEXT`.
- Add repair tests for deterministic subject replacement, numeric predicate split, modal demotion, deontic conversion, and meta-only CIR re-extraction.
- Add question parser tests for natural-language choices, symbolic choices, and open-ended questions.
- Run existing unit tests plus focused smoke coverage from [fixed_smoke_logic_406.jsonl](/C:/Users/hi/Desktop/Praser/fixed_smoke_logic_406.jsonl).

## Assumptions

- Public output can still expose `LogicNode` through `CompiledPremise.ast`; the new invariant is that only Python creates it.
- CIR is the internal semantic boundary, not the final solver API.
- META premises remain `needs_review` unless current formula-resolution logic can safely materialize a consequent.
- Deontic and modal statements are not solver-ready unless represented explicitly as deontic/modal CIR, not plain facts.
