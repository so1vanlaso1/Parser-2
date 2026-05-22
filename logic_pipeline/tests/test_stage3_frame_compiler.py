import json

from src.config import PipelineConfig
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


def test_unsupported_frame_falls_back_for_that_premise_only():
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
                "compiled": [
                    {
                        "premise_id": "P2",
                        "kind": "META",
                        "cnl": "If passing implies graduation, then students are eligible.",
                        "ast": {
                            "type": "forall",
                            "variable": "x",
                            "children": [
                                {
                                    "type": "implies",
                                    "children": [
                                        {
                                            "type": "implies",
                                            "children": [
                                                {
                                                    "type": "atomic",
                                                    "name": "passes",
                                                    "arguments": ["x"],
                                                },
                                                {
                                                    "type": "atomic",
                                                    "name": "graduates",
                                                    "arguments": ["x"],
                                                },
                                            ],
                                        },
                                        {
                                            "type": "atomic",
                                            "name": "eligible",
                                            "arguments": ["x"],
                                        },
                                    ],
                                }
                            ],
                        },
                        "solver_ready": False,
                        "needs_review": False,
                        "unsupported": False,
                        "notes": [],
                    }
                ]
            },
        ]
    )

    output = compiler.compile(stage1)

    assert len(compiler.llm.calls) == 2
    assert [item.premise_id for item in output.compiled] == ["P1", "P2"]
    assert output.compiled[0].ast.name == "student"
    assert output.compiled[0].ast.arguments == ["mina"]
    assert output.compiled[1].ast.type == "forall"
