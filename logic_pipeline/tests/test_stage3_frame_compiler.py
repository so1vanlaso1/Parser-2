import json

from src.config import PipelineConfig
from src.meta_formula import resolve_meta_premises
from src.predicate_canonicalizer import canonicalize_stage3
from src.schemas import CNLStatement, Stage1Output
from src.stage3_ast import ASTCompiler


class FakeRAG:
    def format_examples(self, _text: str, top_k: int = 3) -> str:
        return ""


class FakeLLM:
    def __init__(self, responses: list[dict]):
        self.responses = [json.dumps(response) for response in responses]
        self.calls: list[tuple[str, str, int | None]] = []

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        self.calls.append((system_prompt, user_prompt, max_new_tokens))
        return self.responses.pop(0)


def _compiler(responses: list[dict]) -> ASTCompiler:
    config = PipelineConfig(max_new_tokens=4096, llm_live_trace=False)
    return ASTCompiler(config, FakeRAG(), FakeLLM(responses))


def test_frame_compiler_builds_rule_ast_without_full_ast_prompt():
    stage1 = Stage1Output(
        statements=[
            CNLStatement(
                premise_id="P1",
                original="If a student does not ask questions, then they do not attend tutorials.",
                kind_hint="RULE",
                cnl=(
                    "If a student does NOT ask questions, then the student does "
                    "NOT attend tutorials."
                ),
                if_part="a student does NOT ask questions",
                then_part="the student does NOT attend tutorials",
            )
        ]
    )
    compiler = _compiler(
        [
            {
                "frames": [
                    {
                        "premise_id": "P1",
                        "kind": "RULE",
                        "cnl": stage1.statements[0].cnl,
                        "variable": "x",
                        "antecedent": {
                            "connective": "and",
                            "atoms": [
                                {"name": "student", "arguments": ["x"], "negated": False},
                                {
                                    "name": "asks_questions",
                                    "arguments": ["x"],
                                    "negated": True,
                                },
                            ],
                        },
                        "consequent": {
                            "connective": "and",
                            "atoms": [
                                {
                                    "name": "attends_tutorials",
                                    "arguments": ["x"],
                                    "negated": True,
                                }
                            ],
                        },
                        "body": None,
                        "unsupported": False,
                        "notes": [],
                    }
                ]
            }
        ]
    )

    output = compiler.compile(stage1)

    assert len(compiler.llm.calls) == 1
    ast = output.compiled[0].ast
    assert ast.type == "forall"
    assert ast.variable == "x"
    implies = ast.children[0]
    assert implies.type == "implies"
    antecedent, consequent = implies.children
    assert antecedent.type == "and"
    assert antecedent.children[1].type == "not"
    assert consequent.type == "not"
    assert consequent.children[0].name == "attends_tutorials"


def test_meta_frame_builds_formula_graph_without_full_ast_prompt():
    stage1 = Stage1Output(
        statements=[
            CNLStatement(
                premise_id="P1",
                original="Mina is a student.",
                kind_hint="FACT",
                cnl="Mina is a student.",
                body="Mina is a student.",
            ),
            CNLStatement(
                premise_id="P2",
                original="If passing implies graduation, then students are eligible.",
                kind_hint="META",
                cnl="If passing implies graduation, then students are eligible.",
                if_part="passing implies graduation",
                then_part="students are eligible",
            ),
        ]
    )
    compiler = _compiler(
        [
            {
                "frames": [
                    {
                        "premise_id": "P1",
                        "kind": "FACT",
                        "cnl": "Mina is a student.",
                        "variable": None,
                        "antecedent": None,
                        "consequent": None,
                        "body": {
                            "connective": "and",
                            "atoms": [
                                {"name": "student", "arguments": ["mina"], "negated": False}
                            ],
                        },
                        "unsupported": False,
                        "notes": [],
                    },
                    {
                        "premise_id": "P2",
                        "kind": "META",
                        "cnl": "If passing implies graduation, then students are eligible.",
                        "variable": None,
                        "antecedent": None,
                        "consequent": None,
                        "body": None,
                        "unsupported": True,
                        "notes": ["nested implication"],
                    },
                ]
            },
            {
                "atoms": [{"predicate": "passes", "arguments": ["y"], "negated": False}]
            },
            {
                "atoms": [{"predicate": "graduates", "arguments": ["y"], "negated": False}]
            },
            {
                "atoms": [{"predicate": "student", "arguments": ["x"], "negated": False}]
            },
            {
                "atoms": [{"predicate": "eligible", "arguments": ["x"], "negated": False}]
            },
        ]
    )

    output = compiler.compile(stage1)

    assert len(compiler.llm.calls) == 5
    assert [item.premise_id for item in output.compiled] == ["P1", "P2"]
    assert output.compiled[0].ast.name == "student"
    assert output.compiled[0].ast.arguments == ["mina"]
    meta = output.compiled[1]
    assert meta.kind == "META"
    assert meta.direct_solver_ready is False
    assert len(meta.flat_atoms) == 4
    assert meta.formula_tree["type"] == "implies"
    assert meta.formula_tree["children"][0]["type"] == "forall"
    assert meta.formula_tree["children"][1]["type"] == "forall"
    assert all("Stage 3 of a neurosymbolic logic parser" not in call[0] for call in compiler.llm.calls[1:])


def test_meta_resolver_marks_p4_p5_style_formulas_redundant():
    stage1 = Stage1Output(
        statements=[
            CNLStatement(
                premise_id="P2",
                original="If a student is studying, then the student is asking questions.",
                kind_hint="RULE",
                cnl="If a student is studying, then the student is asking questions.",
            ),
            CNLStatement(
                premise_id="P3",
                original="Every student is attending tutorials.",
                kind_hint="FORALL",
                cnl="Every student is attending tutorials.",
            ),
            CNLStatement(
                premise_id="P4",
                original=(
                    "If a student studying implies they are asking questions, "
                    "then every student is attending tutorials."
                ),
                kind_hint="META",
                cnl=(
                    "If a student studying implies they are asking questions, "
                    "then every student is attending tutorials."
                ),
            ),
            CNLStatement(
                premise_id="P5",
                original=(
                    "If every student is attending tutorials, then if a student is studying, "
                    "the student is asking questions."
                ),
                kind_hint="META",
                cnl=(
                    "If every student is attending tutorials, then if a student is studying, "
                    "the student is asking questions."
                ),
            ),
        ]
    )
    compiler = _compiler(
        [
            {
                "frames": [
                    {
                        "premise_id": "P2",
                        "kind": "RULE",
                        "cnl": stage1.statements[0].cnl,
                        "variable": "x",
                        "antecedent": {
                            "connective": "and",
                            "atoms": [
                                {"name": "student", "arguments": ["x"], "negated": False},
                                {"name": "studying", "arguments": ["x"], "negated": False},
                            ],
                        },
                        "consequent": {
                            "connective": "and",
                            "atoms": [
                                {"name": "asks_questions", "arguments": ["x"], "negated": False}
                            ],
                        },
                        "body": None,
                        "unsupported": False,
                        "notes": [],
                    },
                    {
                        "premise_id": "P3",
                        "kind": "FORALL",
                        "cnl": stage1.statements[1].cnl,
                        "variable": "x",
                        "antecedent": {
                            "connective": "and",
                            "atoms": [
                                {"name": "student", "arguments": ["x"], "negated": False}
                            ],
                        },
                        "consequent": {
                            "connective": "and",
                            "atoms": [
                                {
                                    "name": "attends_tutorials",
                                    "arguments": ["x"],
                                    "negated": False,
                                }
                            ],
                        },
                        "body": None,
                        "unsupported": False,
                        "notes": [],
                    },
                    {
                        "premise_id": "P4",
                        "kind": "META",
                        "cnl": stage1.statements[2].cnl,
                        "variable": None,
                        "antecedent": None,
                        "consequent": None,
                        "body": None,
                        "unsupported": True,
                        "notes": ["nested implication"],
                    },
                    {
                        "premise_id": "P5",
                        "kind": "META",
                        "cnl": stage1.statements[3].cnl,
                        "variable": None,
                        "antecedent": None,
                        "consequent": None,
                        "body": None,
                        "unsupported": True,
                        "notes": ["nested implication"],
                    },
                ]
            },
            {
                "atoms": [
                    {"predicate": "student", "arguments": ["y"], "negated": False},
                    {"predicate": "studying", "arguments": ["y"], "negated": False},
                ]
            },
            {
                "atoms": [{"predicate": "asks_questions", "arguments": ["y"], "negated": False}]
            },
            {
                "atoms": [{"predicate": "student", "arguments": ["x"], "negated": False}]
            },
            {
                "atoms": [
                    {"predicate": "attends_tutorials", "arguments": ["x"], "negated": False}
                ]
            },
            {
                "atoms": [{"predicate": "student", "arguments": ["y"], "negated": False}]
            },
            {
                "atoms": [
                    {"predicate": "attends_tutorials", "arguments": ["y"], "negated": False}
                ]
            },
            {
                "atoms": [
                    {"predicate": "student", "arguments": ["x"], "negated": False},
                    {"predicate": "studying", "arguments": ["x"], "negated": False},
                ]
            },
            {
                "atoms": [{"predicate": "asks_questions", "arguments": ["x"], "negated": False}]
            },
        ]
    )

    output = resolve_meta_premises(canonicalize_stage3(compiler.compile(stage1)))
    by_id = {item.premise_id: item for item in output.compiled}

    assert by_id["P4"].meta_resolvable is True
    assert by_id["P5"].meta_resolvable is True
    assert by_id["P4"].solver_export == []
    assert by_id["P5"].solver_export == []
    assert any(link.get("status") == "redundant" for link in by_id["P4"].meta_links)
    assert any(link.get("status") == "redundant" for link in by_id["P5"].meta_links)
