from .config import PipelineConfig
from .json_utils import extract_json_object
from .llm_client import ChatModel
from .schemas import (
    CompiledPremise,
    CNLStatement,
    LogicNode,
    PredicateFrame,
    PredicateFrameOutput,
    PredicateGroup,
    Stage1Output,
    Stage3Output,
)
from .stage2_rag import StructuralRAG
from .meta_formula import (
    atomize_leaf_with_llm,
    formula_tree_to_logic_node,
    is_direct_solver_ready_formula,
    is_higher_order_or_meta,
    process_formula_node,
    split_meta_formula,
)


STAGE3_FRAME_SYSTEM_PROMPT = """\
/no_think

You are Stage 3-lite of a neurosymbolic logic parser.
You are a JSON predicate-frame extractor, not a solver.

Your job:
For each CNL statement, identify only:
- predicate names
- predicate arguments
- negation
- rule relationship: antecedent vs consequent vs body
- one main quantified variable when present

Do NOT build recursive AST JSON. Python will build the AST deterministically.
Do NOT solve the question.
Do NOT add facts that are not in the premise.

Output rules:
- Output ONLY valid JSON.
- First character must be {.
- Do not write analysis.
- Do not write explanations.
- Do not write markdown.
- Do not include comments.
- End after the JSON object with <END_JSON>.

Predicate rules:
1. Predicate names must be lowercase snake_case.
2. Do not put negation in predicate names. Use "negated": true.
3. Use variables like x, y for generic quantified entities.
4. Use constants like john, mary, tuan, quantum_lab for named entities.
5. Every atom should have at least one argument.
6. Reuse the same predicate name for the same concept across all premises.

Frame shape:
{
  "frames": [
    {
      "premise_id": "P1",
      "kind": "RULE",
      "cnl": "...",
      "variable": "x",
      "antecedent": {
        "connective": "and",
        "atoms": [
          {"name": "student", "arguments": ["x"], "negated": false}
        ]
      },
      "consequent": {
        "connective": "and",
        "atoms": [
          {"name": "eligible", "arguments": ["x"], "negated": false}
        ]
      },
      "body": null,
      "unsupported": false,
      "notes": []
    }
  ]
}

Use these fields by kind:
- FACT: put all atoms in body. Use constants for named individuals.
- EXISTS: put all asserted atoms in body and set variable.
- FORALL: usually set antecedent to the class restriction and consequent to the asserted property.
- RULE, ONLY_IF_RULE, NON_IF_RULE: set antecedent and consequent.
- OBLIGATION_RULE: set antecedent for the situation and consequent for the obligated/permitted/forbidden predicate.
  Use prefixes obligated_, permitted_, or forbidden_ on the consequent predicate name.
- IFF: set antecedent and consequent.
- META or UNKNOWN: set unsupported true unless the statement is clearly representable as flat atoms.

Examples:

Input:
P1 [FORALL]: Every student is preparing for an exam.
Output frame:
{
  "premise_id": "P1",
  "kind": "FORALL",
  "cnl": "Every student is preparing for an exam.",
  "variable": "x",
  "antecedent": {"connective": "and", "atoms": [{"name": "student", "arguments": ["x"], "negated": false}]},
  "consequent": {"connective": "and", "atoms": [{"name": "preparing_for_exam", "arguments": ["x"], "negated": false}]},
  "body": null,
  "unsupported": false,
  "notes": []
}

Input:
P2 [RULE]: If a student does NOT ask questions, then the student does NOT attend tutorials.
Output frame:
{
  "premise_id": "P2",
  "kind": "RULE",
  "cnl": "If a student does NOT ask questions, then the student does NOT attend tutorials.",
  "variable": "x",
  "antecedent": {"connective": "and", "atoms": [
    {"name": "student", "arguments": ["x"], "negated": false},
    {"name": "asks_questions", "arguments": ["x"], "negated": true}
  ]},
  "consequent": {"connective": "and", "atoms": [
    {"name": "attends_tutorials", "arguments": ["x"], "negated": true}
  ]},
  "body": null,
  "unsupported": false,
  "notes": []
}

Input:
P3 [EXISTS]: There exists a student who attends tutorials.
Output frame:
{
  "premise_id": "P3",
  "kind": "EXISTS",
  "cnl": "There exists a student who attends tutorials.",
  "variable": "x",
  "antecedent": null,
  "consequent": null,
  "body": {"connective": "and", "atoms": [
    {"name": "student", "arguments": ["x"], "negated": false},
    {"name": "attends_tutorials", "arguments": ["x"], "negated": false}
  ]},
  "unsupported": false,
  "notes": []
}

Return exactly one JSON object with the key "frames".
Do not output anything before the JSON.
Do not output anything after <END_JSON>.
"""


STAGE3_SYSTEM_PROMPT = """\
/no_think

You are Stage 3 of a neurosymbolic logic parser.
You are a JSON transducer, not a solver.

Your job:
Compile CNL statements into typed recursive LogicNode AST JSON.

You must NOT solve the question.
You must NOT add facts that are not in the premise.
You must preserve:
- quantifiers
- IF direction
- only-if direction
- iff direction
- classical negation
- modal uncertainty

Allowed node types:
atomic, and, or, not, implies, iff, forall, exists, equation

Output rules:
- Output ONLY valid JSON.
- First character must be {.
- Do not write analysis.
- Do not write explanations.
- Do not write markdown.
- Do not include comments.
- Do not output any node type outside the allowed node types.
- End after the JSON object with <END_JSON>.

Atomic predicate rules:
1. Predicate names must be lowercase snake_case.
2. Use variables like x, y for quantified rules.
3. Use constants like john, mary, quantum_lab for named entities.
4. Do not put "not" inside predicate names.
   Correct: {"type":"not","children":[{"type":"atomic","name":"has_housing","arguments":["x"]}]}
   Wrong: {"type":"atomic","name":"not_has_housing","arguments":["x"]}
5. Reuse the same predicate name for the same concept across all premises.
   Example: "requires extensive hyperparameter tuning", "has extensive hyperparameter tuning",
   and "has been extensively tuned" should use one canonical predicate.
6. Every atomic node MUST have at least 1 argument. Never produce {"type":"atomic","name":"foo","arguments":[]}.

STRICT SHAPE RULES (CRITICAL — violating these causes validation failure):
- forall/exists: MUST have exactly 1 child and a "variable" field.
- implies: MUST have exactly 2 children (antecedent, consequent).
- not: MUST have exactly 1 child.
- and/or/iff: MUST have at least 2 children.
- atomic: MUST have "name" (non-empty) and "arguments" (at least 1 element).

Quantifier rules:
- Every/All/Any -> forall
- Some/At least one/A -> exists only when the sentence asserts existence
- Generic rules usually become forall x: antecedent -> consequent

Only-if rule:
- "A only if B" means A -> B.

IFF:
- "A if and only if B" means A <-> B.

Nested quantifier / Mixed scope rules:
When a sentence mixes existential and universal quantifiers, nest them properly.
Example: "If there exists a student who passes, then every teacher celebrates."
AST:
{
  "type": "implies",
  "children": [
    {
      "type": "exists",
      "variable": "x",
      "children": [
        {"type": "and", "children": [
          {"type": "atomic", "name": "student", "arguments": ["x"]},
          {"type": "atomic", "name": "passes", "arguments": ["x"]}
        ]}
      ]
    },
    {
      "type": "forall",
      "variable": "y",
      "children": [
        {"type": "implies", "children": [
          {"type": "atomic", "name": "teacher", "arguments": ["y"]},
          {"type": "atomic", "name": "celebrates", "arguments": ["y"]}
        ]}
      ]
    }
  ]
}

Obligation / Deontic rules:
For OBLIGATION_RULE premises, represent the obligation using an "obligated_" prefix
on the predicate name within a forall wrapper.
Example: "It is mandatory to wear goggles in science laboratories."
AST:
{
  "type": "forall",
  "variable": "x",
  "children": [
    {"type": "implies", "children": [
      {"type": "atomic", "name": "in_science_laboratory", "arguments": ["x"]},
      {"type": "atomic", "name": "obligated_wear_goggles", "arguments": ["x"]}
    ]}
  ]
}

META / Nested implication rules:
For META premises with nested implications, preserve the nesting.
Example: "If passing the exam implies graduation, then students who pass are eligible."
AST:
{
  "type": "implies",
  "children": [
    {
      "type": "forall",
      "variable": "x",
      "children": [
        {"type": "implies", "children": [
          {"type": "atomic", "name": "passes_exam", "arguments": ["x"]},
          {"type": "atomic", "name": "graduates", "arguments": ["x"]}
        ]}
      ]
    },
    {
      "type": "forall",
      "variable": "y",
      "children": [
        {"type": "implies", "children": [
          {"type": "atomic", "name": "passes_exam", "arguments": ["y"]},
          {"type": "atomic", "name": "eligible", "arguments": ["y"]}
        ]}
      ]
    }
  ]
}

Return JSON only in this exact shape:
{
  "compiled": [
    {
      "premise_id": "P1",
      "kind": "RULE",
      "cnl": "...",
      "ast": {...},
      "solver_ready": false,
      "needs_review": false,
      "unsupported": false,
      "notes": []
    }
  ]
}

Do not output anything after <END_JSON>.
"""

class ASTCompiler:
    """Stage 3 - compiles CNL statements into typed LogicNode AST."""

    def __init__(self, config: PipelineConfig, rag: StructuralRAG, llm: ChatModel):
        self.config = config
        self.rag = rag
        self.llm = llm

    def compile(self, stage1: Stage1Output) -> Stage3Output:
        if getattr(self.config, "enable_frame_ast_compiler", True):
            try:
                frames = self._extract_frames(stage1)
                return self._compile_frames_with_fallback(stage1, frames)
            except Exception:
                # Preserve the previous behavior if the shallow frame response
                # is malformed or cannot be validated.
                return self._compile_full_ast(stage1)

        return self._compile_full_ast(stage1)

    def _extract_frames(self, stage1: Stage1Output) -> PredicateFrameOutput:
        cnl_text = "\n".join(
            f"{s.premise_id} [{s.kind_hint}]: {s.cnl}"
            for s in stage1.statements
        )

        raw_text = self.llm.generate(
            STAGE3_FRAME_SYSTEM_PROMPT,
            cnl_text,
            max_new_tokens=self._frame_token_budget(len(stage1.statements)),
        )
        data = extract_json_object(raw_text)
        return PredicateFrameOutput.model_validate(data)

    def _compile_frames_with_fallback(
        self,
        stage1: Stage1Output,
        frames: PredicateFrameOutput,
    ) -> Stage3Output:
        frame_by_id = {frame.premise_id: frame for frame in frames.frames}
        compiled_by_id: dict[str, CompiledPremise] = {}
        fallback_statements = []

        for statement in stage1.statements:
            frame = frame_by_id.get(statement.premise_id)
            if frame is not None:
                frame = frame.model_copy(update={"kind": statement.kind_hint, "cnl": statement.cnl})

            if self._statement_is_meta(statement, frame):
                try:
                    compiled_by_id[statement.premise_id] = self._compile_meta_premise(statement)
                except Exception as exc:
                    compiled_by_id[statement.premise_id] = self._build_unsupported_meta_premise(
                        statement,
                        str(exc),
                    )
                continue

            if frame is None or self._frame_needs_fallback(frame):
                fallback_statements.append(statement)
                continue

            try:
                compiled_by_id[statement.premise_id] = self._build_compiled_premise(frame)
            except ValueError:
                fallback_statements.append(statement)

        if fallback_statements:
            fallback_output = self._compile_full_ast(Stage1Output(statements=fallback_statements))
            for item in fallback_output.compiled:
                compiled_by_id[item.premise_id] = item

        return Stage3Output(
            compiled=[compiled_by_id[statement.premise_id] for statement in stage1.statements]
        )

    def _frame_needs_fallback(self, frame: PredicateFrame) -> bool:
        return frame.unsupported or frame.kind == "UNKNOWN"

    def _statement_is_meta(
        self,
        statement: CNLStatement,
        frame: PredicateFrame | None,
    ) -> bool:
        return (
            statement.kind_hint == "META"
            or (frame is not None and frame.kind == "META")
            or is_higher_order_or_meta(statement.cnl)
            or is_higher_order_or_meta(statement.original)
        )

    def _compile_meta_premise(self, statement: CNLStatement) -> CompiledPremise:
        text_tree = split_meta_formula(statement.cnl or statement.original)

        def llm_atomizer(phrase: str, variable: str):
            return atomize_leaf_with_llm(
                phrase,
                variable,
                self.llm,
                max_new_tokens=getattr(self.config, "stage3_meta_atomizer_max_new_tokens", 300),
            )

        formula_tree, flat_atoms = process_formula_node(
            text_tree,
            llm_atomizer,
            premise_id=statement.premise_id,
        )
        ast = formula_tree_to_logic_node(formula_tree, flat_atoms)
        notes = list(statement.risk_flags)
        if "nested_logic" not in notes:
            notes.append("nested_logic")

        return CompiledPremise(
            premise_id=statement.premise_id,
            kind="META",
            cnl=statement.cnl,
            ast=ast,
            solver_ready=False,
            needs_review=True,
            unsupported=False,
            direct_solver_ready=is_direct_solver_ready_formula(formula_tree),
            meta_resolvable=False,
            flat_atoms=flat_atoms,
            formula_tree=formula_tree,
            solver_export=[],
            meta_links=[],
            notes=notes,
        )

    def _build_unsupported_meta_premise(
        self,
        statement: CNLStatement,
        reason: str,
    ) -> CompiledPremise:
        notes = list(statement.risk_flags)
        for note in ["nested_logic", f"meta_split_failed: {reason}"]:
            if note not in notes:
                notes.append(note)

        return CompiledPremise(
            premise_id=statement.premise_id,
            kind="META",
            cnl=statement.cnl,
            ast=LogicNode(
                type="atomic",
                name="unsupported_meta_formula",
                arguments=[statement.premise_id.lower()],
                source_premise_id=statement.premise_id,
            ),
            solver_ready=False,
            needs_review=True,
            unsupported=True,
            direct_solver_ready=False,
            meta_resolvable=False,
            flat_atoms=[],
            formula_tree=None,
            solver_export=[],
            meta_links=[
                {
                    "type": "resolution",
                    "status": "unresolved",
                    "reason": "meta formula could not be split safely",
                }
            ],
            notes=notes,
        )

    def _build_compiled_premise(self, frame: PredicateFrame) -> CompiledPremise:
        return CompiledPremise(
            premise_id=frame.premise_id,
            kind=frame.kind,
            cnl=frame.cnl,
            ast=self._build_ast(frame),
            notes=frame.notes,
        )

    def _build_ast(self, frame: PredicateFrame) -> LogicNode:
        if frame.kind == "FACT":
            return self._group_to_node(self._require_group(frame.body, "body"), frame.premise_id)

        if frame.kind == "EXISTS":
            body = self._group_to_node(self._require_group(frame.body, "body"), frame.premise_id)
            variable = self._resolve_variable(frame, body)
            if not variable:
                raise ValueError("EXISTS frame missing variable")
            return LogicNode(type="exists", variable=variable, children=[body])

        if frame.kind in {"FORALL", "RULE", "ONLY_IF_RULE", "NON_IF_RULE", "OBLIGATION_RULE"}:
            body = self._build_conditional_or_body(frame, relation="implies")
            return self._wrap_forall_if_needed(body, self._resolve_variable(frame, body))

        if frame.kind == "IFF":
            body = self._build_conditional_or_body(frame, relation="iff")
            return self._wrap_forall_if_needed(body, self._resolve_variable(frame, body))

        raise ValueError(f"Unsupported frame kind: {frame.kind}")

    def _build_conditional_or_body(self, frame: PredicateFrame, *, relation: str) -> LogicNode:
        if frame.antecedent and frame.consequent:
            return LogicNode(
                type=relation,
                children=[
                    self._group_to_node(frame.antecedent, frame.premise_id),
                    self._group_to_node(frame.consequent, frame.premise_id),
                ],
            )

        if frame.body:
            return self._group_to_node(frame.body, frame.premise_id)

        raise ValueError(f"{frame.kind} frame missing antecedent/consequent or body")

    def _require_group(self, group: PredicateGroup | None, field_name: str) -> PredicateGroup:
        if group is None:
            raise ValueError(f"Missing predicate group: {field_name}")
        return group

    def _group_to_node(self, group: PredicateGroup, premise_id: str) -> LogicNode:
        if not group.atoms:
            raise ValueError("Predicate group has no atoms")

        children = [self._atom_to_node(atom, premise_id) for atom in group.atoms]
        if len(children) == 1:
            return children[0]

        return LogicNode(type=group.connective, children=children)

    def _atom_to_node(self, atom, premise_id: str) -> LogicNode:
        if not atom.name or not atom.arguments:
            raise ValueError("Predicate atom missing name or arguments")

        node = LogicNode(
            type="atomic",
            name=atom.name,
            arguments=atom.arguments,
            source_premise_id=premise_id,
        )
        if atom.negated:
            return LogicNode(type="not", children=[node], source_premise_id=premise_id)
        return node

    def _wrap_forall_if_needed(self, node: LogicNode, variable: str | None) -> LogicNode:
        if not variable:
            return node
        return LogicNode(type="forall", variable=variable, children=[node])

    def _resolve_variable(self, frame: PredicateFrame, node: LogicNode) -> str | None:
        if frame.variable:
            return frame.variable
        return self._infer_variable(node)

    def _infer_variable(self, node: LogicNode) -> str | None:
        if node.type == "atomic":
            for arg in node.arguments:
                if len(arg) == 1 and arg.isalpha() and arg.islower():
                    return arg

        for child in node.children:
            variable = self._infer_variable(child)
            if variable:
                return variable

        if isinstance(node.left, LogicNode):
            variable = self._infer_variable(node.left)
            if variable:
                return variable

        if isinstance(node.right, LogicNode):
            variable = self._infer_variable(node.right)
            if variable:
                return variable

        return None

    def _compile_full_ast(self, stage1: Stage1Output) -> Stage3Output:
        cnl_text = "\n".join(
            f"{s.premise_id} [{s.kind_hint}]: {s.cnl}"
            for s in stage1.statements
        )

        rag_context = self.rag.format_examples(cnl_text, top_k=self.config.rag_top_k)

        user_prompt = f"""\
Reference examples (use these as structural guidance):
{rag_context}

Now compile these CNL statements into AST:
{cnl_text}
"""

        raw_text = self.llm.generate(
            STAGE3_SYSTEM_PROMPT,
            user_prompt,
            max_new_tokens=self._token_budget(len(stage1.statements)),
        )
        data = extract_json_object(raw_text)
        return Stage3Output.model_validate(data)

    def _token_budget(self, statement_count: int) -> int:
        return min(
            self.config.max_new_tokens,
            max(self.config.stage3_max_new_tokens, 450 + 350 * statement_count),
        )

    def _frame_token_budget(self, statement_count: int) -> int:
        return min(
            self.config.max_new_tokens,
            max(
                getattr(self.config, "stage3_frame_max_new_tokens", 500),
                250 + 180 * statement_count,
            ),
        )
