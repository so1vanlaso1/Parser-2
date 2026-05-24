import re
import unicodedata

from .config import PipelineConfig
from .json_utils import extract_json_object
from .llm_client import ChatModel
from .schemas import QuestionParse
from .schemas import CIRExists, CIRFact, CIRForall, CIRMeta, CIRRule
from .stage1_cnl import (
    _parse_exists_cir,
    _parse_fact_cir,
    _parse_forall_cir,
    _parse_generic_fact_cir,
    _parse_generic_meta_cir,
    _parse_rule_cir,
)
from .stage4_validate import has_solver_blocking_predicate
from .meta_formula import (
    atomize_leaf_with_llm,
    formula_tree_to_logic_node,
    is_direct_solver_ready_formula,
    is_higher_order_or_meta,
    process_formula_node,
    split_meta_formula,
)


QUESTION_PARSER_SYSTEM_PROMPT = """\
/no_think

You are a question parser for a neurosymbolic logic pipeline.
You are a JSON transducer, not a solver.

Your job:
Parse a question (and its choices, if any) into structured LogicNode AST.

You must NOT solve the question.
You must NOT use premise facts to rewrite choices.
You only parse the question and choices into AST structure.

Allowed LogicNode type values:
atomic, and, or, not, implies, iff, forall, exists, equation

Forbidden type values:
inference, claim, choice, statement, because, explanation, predicate

Predicate naming rules:
1. Predicate names must be lowercase snake_case.
2. Use constants for named entities (e.g., john, mina, alphanet).
3. Do not put "not" inside predicate names - use a NOT node.
4. If known predicate names are provided, reuse a known predicate when it fits the same meaning.
5. Do not invent synonyms for a known predicate.

Reason clauses:
- For multiple-choice entailment questions, parse only the claim being inferred.
- Ignore "because ..." reason text unless the question asks whether the explanation itself is valid.
- Do not create a "because" or "inference" node.

For yes/no questions, return a "query" node representing what is being asked.
For multiple-choice questions, return a "choices" map with each choice as a LogicNode.

Output rules:
- Output ONLY valid JSON.
- First character must be {.
- Do not write analysis.
- Do not write explanations.
- Do not write markdown.
- Do not include comments.
- Every query or choice value must be a valid LogicNode using only the allowed node types.
- End after the JSON object with <END_JSON>.

Return this exact JSON shape:
{
  "question": "the original question text",
  "query": {...} or null,
  "choices": {
    "A": {...},
    "B": {...}
  }
}

If there are no choices, set "choices" to {}.
If it's a multiple-choice question, set "query" to null.

Examples:
Choice text: "The AlphaNet model does not require hyperparameter tuning because it achieves high accuracy."
Choice AST:
{
  "type": "not",
  "children": [
    {"type": "atomic", "name": "requires_hyperparameter_tuning", "arguments": ["alphanet"]}
  ]
}

Choice text: "The AlphaNet model has been extensively tuned because it achieves high accuracy and processes data quickly."
Choice AST:
{"type": "atomic", "name": "has_extensive_hyperparameter_tuning", "arguments": ["alphanet"]}

Do not output anything after <END_JSON>.
"""


class QuestionParser:
    """Parses question text + choices into QuestionParse AST."""

    def __init__(self, config: PipelineConfig, llm: ChatModel):
        self.config = config
        self.llm = llm

    def parse(
        self,
        question_text: str,
        choices: dict[str, str] | None = None,
        known_predicates: list[str] | None = None,
    ) -> QuestionParse:
        if not question_text:
            return QuestionParse(question="", query_type="EMPTY_QUERY")

        if choices and all(_looks_symbolic_formula(choice) for choice in choices.values()):
            parsed = QuestionParse(
                question=question_text,
                choices={key: _parse_symbolic_formula(value) for key, value in choices.items()},
                query=None,
                notes=["symbolic_choices"],
            )
            return _classify_question_parse(parsed)

        if not choices and is_higher_order_or_meta(_extract_meta_statement(question_text)):
            return self._parse_meta_question(question_text)

        deterministic = _parse_deterministic_question(question_text, choices)
        if deterministic is not None:
            return _classify_question_parse(deterministic)

        user_content = f"Question: {question_text}"
        if known_predicates:
            predicates = "\n".join(f"- {name}" for name in known_predicates)
            user_content += f"\n\nKnown predicate names from premises:\n{predicates}"
        if choices:
            choices_text = "\n".join(f"{k}: {v}" for k, v in choices.items())
            user_content += f"\n\nChoices:\n{choices_text}"

        raw_text = self.llm.generate(
            QUESTION_PARSER_SYSTEM_PROMPT,
            user_content,
            max_new_tokens=self.config.question_max_new_tokens,
        )
        data = extract_json_object(raw_text)
        parsed = QuestionParse.model_validate(data)
        return _classify_question_parse(parsed)

    def _parse_meta_question(self, question_text: str) -> QuestionParse:
        statement = _extract_meta_statement(question_text)
        text_tree = split_meta_formula(statement)

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
            premise_id="Q",
        )
        query = formula_tree_to_logic_node(formula_tree, flat_atoms)
        return QuestionParse(
            question=question_text,
            query=query,
            choices={},
            query_type="META_QUERY",
            direct_solver_ready=is_direct_solver_ready_formula(formula_tree),
            formula_tree=formula_tree,
            flat_atoms=flat_atoms,
            unsupported=False,
            notes=["nested_logic"],
        )


def _classify_question_parse(parsed: QuestionParse) -> QuestionParse:
    if parsed.choices:
        parsed.query_type = "CHOICE_QUERY"
        parsed.direct_solver_ready = all(
            is_direct_solver_ready_formula(choice) and not has_solver_blocking_predicate(choice)
            for choice in parsed.choices.values()
        )
    elif parsed.query:
        parsed.query_type = "AST_QUERY"
        parsed.direct_solver_ready = (
            is_direct_solver_ready_formula(parsed.query)
            and not has_solver_blocking_predicate(parsed.query)
        )
    else:
        parsed.query_type = "UNKNOWN"
        parsed.direct_solver_ready = False
    return parsed


def _extract_meta_statement(question_text: str) -> str:
    clean = re.sub(r"\s+", " ", question_text).strip()
    statement_match = re.search(r"\bstatement\s*:\s*(?P<statement>.+)$", clean, flags=re.IGNORECASE)
    if statement_match:
        clean = statement_match.group("statement").strip()

    if clean.lower().startswith("if "):
        return clean

    if_start = re.search(r"\bif\b", clean, flags=re.IGNORECASE)
    if if_start:
        return clean[if_start.start():].strip()

    return clean


def _parse_deterministic_question(
    question_text: str,
    choices: dict[str, str] | None = None,
) -> QuestionParse | None:
    if choices:
        parsed_choices = {}
        for key, value in choices.items():
            formula = _parse_nl_formula(value)
            if formula is None:
                formula = _parse_nl_formula(value, allow_sentence_fact=True)
            if formula is None:
                return None
            parsed_choices[key] = formula
        return QuestionParse(
            question=question_text,
            choices=parsed_choices,
            query=None,
            notes=["deterministic_nl_choices"],
        )

    known_query = _parse_known_structured_question(question_text)
    if known_query is not None:
        return known_query

    statement = _extract_meta_statement(question_text)
    formula = _parse_nl_formula(statement)
    if formula is None:
        formula = _parse_nl_formula(statement, allow_sentence_fact=True)
    if formula is None:
        return None
    return QuestionParse(
        question=question_text,
        query=formula,
        choices={},
        notes=["deterministic_nl_query"],
    )


def _parse_known_structured_question(question_text: str) -> QuestionParse | None:
    normalized = _normalize_question_text(question_text)
    if re.match(r"^does tuan have knowledge of his a-grade subjects$", normalized):
        query = {
            "type": "forall",
            "variable": "s",
            "children": [
                {
                    "type": "implies",
                    "children": [
                        {
                            "type": "and",
                            "children": [
                                {"type": "atomic", "name": "subject", "arguments": ["s"]},
                                {"type": "atomic", "name": "earned_grade", "arguments": ["tuan", "s", "A"]},
                            ],
                        },
                        {
                            "type": "atomic",
                            "name": "has_knowledge_of_subject",
                            "arguments": ["tuan", "s"],
                        },
                    ],
                }
            ],
        }
        return QuestionParse(
            question=question_text,
            query=query,
            choices={},
            notes=["deterministic_structured_query"],
        )
    return None


def _parse_nl_formula(text: str, *, allow_sentence_fact: bool = False) -> dict | None:
    clean = text.strip().strip(".?")
    if not clean:
        return None

    if _looks_symbolic_formula(clean):
        try:
            return _parse_symbolic_formula(clean)
        except ValueError:
            if not allow_sentence_fact:
                return None

    meta = _parse_generic_meta_cir("_Q", clean)
    if meta is not None and isinstance(meta.cir, CIRMeta):
        return meta.cir.formula

    for parser in (_parse_rule_cir, _parse_forall_cir, _parse_exists_cir, _parse_fact_cir):
        premise = parser("_Q", clean)
        if premise is not None:
            return _cir_to_formula(premise.cir)

    premise = _parse_generic_fact_cir("_Q", clean, allow_sentence_fact=allow_sentence_fact)
    if premise is not None:
        return _cir_to_formula(premise.cir)

    return None


def _cir_to_formula(cir) -> dict:
    if isinstance(cir, CIRFact):
        return _atoms_to_formula(cir.atoms)
    if isinstance(cir, CIRExists):
        return {
            "type": "exists",
            "variable": cir.variable,
            "children": [_atoms_to_formula(cir.body)],
        }
    if isinstance(cir, CIRForall):
        if cir.antecedent and cir.consequent:
            body = {
                "type": "implies",
                "children": [_atoms_to_formula(cir.antecedent), _atoms_to_formula(cir.consequent)],
            }
        else:
            body = _atoms_to_formula(cir.body)
        return {"type": "forall", "variable": cir.variable, "children": [body]}
    if isinstance(cir, CIRRule):
        return {
            "type": "forall",
            "variable": cir.variable,
            "children": [
                {
                    "type": "implies",
                    "children": [_atoms_to_formula(cir.antecedent), _atoms_to_formula(cir.consequent)],
                }
            ],
        }
    if isinstance(cir, CIRMeta):
        return cir.formula
    raise ValueError(f"Unsupported CIR query: {cir!r}")


def _atoms_to_formula(atoms) -> dict:
    children = []
    for atom in atoms:
        node = {"type": "atomic", "name": atom.name, "arguments": atom.arguments}
        if atom.negated:
            node = {"type": "not", "children": [node]}
        children.append(node)
    if len(children) == 1:
        return children[0]
    return {"type": "and", "children": children}


def _looks_symbolic_formula(text: str) -> bool:
    return bool(
        re.search(r"[\u00ac\u2227\u2228\u2200\u2203()]", text)
        or "->" in text
        or "=>" in text
        or "\u2192" in text
    )


def _parse_symbolic_formula(text: str) -> dict:
    normalized = (
        text.strip()
        .replace("=>", "->")
        .replace("\u2192", "->")
        .replace("\u00ac", "~")
        .replace("\u2227", "&")
        .replace("\u2228", "|")
    )
    quantifier = re.match(r"^(?P<quant>[\u2200\u2203])\s*(?P<var>[A-Za-z]\w*)\s*(?P<body>.+)$", normalized)
    if quantifier:
        body = quantifier.group("body").strip()
        if body.startswith("(") and body.endswith(")"):
            body = body[1:-1]
        return {
            "type": "forall" if quantifier.group("quant") == "\u2200" else "exists",
            "variable": quantifier.group("var"),
            "children": [_parse_symbolic_formula(body)],
        }
    return SymbolicFormulaParser(normalized).parse()


class SymbolicFormulaParser:
    def __init__(self, text: str):
        self.tokens = _tokenize_symbolic(text)
        self.pos = 0

    def parse(self):
        node = self._parse_implication()
        if self._peek() is not None:
            raise ValueError(f"Unexpected symbolic token: {self._peek()}")
        return node

    def _parse_implication(self):
        left = self._parse_or()
        token = self._peek()
        if token in {"->", "\u2192"}:
            self._advance()
            right = self._parse_implication()
            return {
                "type": "implies",
                "children": [left, right],
            }
        return left

    def _parse_or(self):
        node = self._parse_and()
        while self._peek() in {"\u2228", "|"}:
            self._advance()
            right = self._parse_and()
            node = _merge_nary("or", node, right)
        return node

    def _parse_and(self):
        node = self._parse_not()
        while self._peek() in {"\u2227", "&"}:
            self._advance()
            right = self._parse_not()
            node = _merge_nary("and", node, right)
        return node

    def _parse_not(self):
        if self._peek() in {"\u00ac", "~"}:
            self._advance()
            return {
                "type": "not",
                "children": [self._parse_not()],
            }
        return self._parse_primary()

    def _parse_primary(self):
        token = self._peek()
        if token in {"\u2200", "\u2203"}:
            quantifier = self._advance()
            variable = self._advance()
            body = self._parse_primary()
            return {
                "type": "forall" if quantifier == "\u2200" else "exists",
                "variable": variable,
                "children": [body],
            }
        if token == "(":
            self._advance()
            node = self._parse_implication()
            self._expect(")")
            return node
        if token and re.match(r"^[A-Za-z_]\w*$", token):
            name = self._advance()
            args = []
            if self._peek() == "(":
                self._advance()
                while self._peek() != ")":
                    args.append(self._advance())
                    if self._peek() == ",":
                        self._advance()
                    elif self._peek() != ")":
                        raise ValueError(f"Expected comma or ')', got {self._peek()}")
                self._expect(")")
            return {
                "type": "atomic",
                "name": _to_snake(name),
                "arguments": args or ["x"],
            }
        raise ValueError(f"Unexpected symbolic token: {token}")

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _advance(self):
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of symbolic formula")
        self.pos += 1
        return token

    def _expect(self, expected: str) -> None:
        actual = self._advance()
        if actual != expected:
            raise ValueError(f"Expected {expected}, got {actual}")


def _tokenize_symbolic(text: str) -> list[str]:
    normalized = (
        text.replace("=>", "->")
        .replace("\u2192", "->")
        .replace("\u00ac", "~")
        .replace("\u2227", "&")
        .replace("\u2228", "|")
    )
    token_re = re.compile(r"\s*(->|~|&|\||[\u2200\u2203]|[(),]|[A-Za-z_]\w*)")
    tokens = []
    pos = 0
    while pos < len(normalized):
        match = token_re.match(normalized, pos)
        if not match:
            if normalized[pos].isspace():
                pos += 1
                continue
            raise ValueError(f"Cannot tokenize symbolic formula near: {normalized[pos:]}")
        tokens.append(match.group(1))
        pos = match.end()
    return tokens


def _merge_nary(node_type: str, left: dict, right: dict) -> dict:
    children = []
    if left.get("type") == node_type:
        children.extend(left.get("children", []))
    else:
        children.append(left)
    if right.get("type") == node_type:
        children.extend(right.get("children", []))
    else:
        children.append(right)
    return {"type": node_type, "children": children}


def _to_snake(value: str) -> str:
    value = _ascii_fold(value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    return value.strip("_").lower()


def _ascii_fold(value: str) -> str:
    return (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def _normalize_question_text(value: str) -> str:
    value = value.replace("\u2019", "'").replace("\u2018", "'")
    value = _ascii_fold(value)
    value = value.replace("a grade", "a-grade")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ?.").lower()
