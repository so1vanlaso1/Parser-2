import logging
import time

from .config import PipelineConfig
from .llm_client import create_chat_model
from .schemas import FullParseResult, Stage1Output, Stage3Output, QuestionParse
from .stage1_cnl import CNLRewriter
from .stage2_rag import StructuralRAG
from .stage3_ast import ASTCompiler
from .stage4_validate import LogicValidator, classify_solver_readiness
from .stage5_repair import RepairLoop
from .question_parser import QuestionParser
from .predicate_canonicalizer import (
    canonicalize_question_parse,
    canonicalize_stage3,
    collect_predicate_names,
)
from .meta_formula import is_direct_solver_ready_formula, resolve_meta_premises

logger = logging.getLogger(__name__)


class LogicPipeline:
    """
    Full parse pipeline orchestrator.

    Flow:
        Stage 1 (CNL Rewriter)
        → Stage 2 (Structural RAG — automatic)
        → Stage 3 (AST Compiler)
        → Stage 4 (Validator)
        → Stage 5 (Repair Loop, if needed)
        → Question Parser (if question provided)
        → FullParseResult
    """

    def __init__(self, config: PipelineConfig, rag_path: str = "data/structural_examples.jsonl"):
        self.config = config

        logger.info("Initializing pipeline components...")
        self.llm = create_chat_model(config)
        self.stage1 = CNLRewriter(config, self.llm)
        self.rag = StructuralRAG(examples_path=rag_path)
        self.stage3 = ASTCompiler(config, self.rag, self.llm)
        self.validator = LogicValidator()
        self.repair_loop = RepairLoop(config, self.llm)
        self.question_parser = QuestionParser(config, self.llm)
        logger.info("Pipeline ready.")

    def parse(
        self,
        premises: list[str],
        question: str | None = None,
        choices: dict[str, str] | None = None,
    ) -> FullParseResult:
        """
        Run the full parse pipeline on premises + optional question.

        Returns FullParseResult with compiled premises and optional question parse.
        """
        t0 = time.time()

        # ── Stage 1: CNL Rewriting ──────────────────────────────────────
        logger.info("Stage 1: Rewriting %d premises to CNL...", len(premises))
        t1 = time.time()
        stage1_output = self.stage1.rewrite(premises)
        logger.info("Stage 1 complete (%.1fs) — %d statements", time.time() - t1, len(stage1_output.statements))

        # ── Stage 3: AST Compilation (Stage 2 RAG is embedded) ──────────
        logger.info("Stage 3: Compiling CNL to AST...")
        t3 = time.time()
        stage3_output = resolve_meta_premises(
            canonicalize_stage3(self.stage3.compile(stage1_output))
        )
        logger.info("Stage 3 complete (%.1fs) — %d compiled", time.time() - t3, len(stage3_output.compiled))

        # ── Stage 4 + 5: Validate & Repair Loop ────────────────────────
        final_report = None
        for attempt in range(self.config.max_repair_attempts + 1):
            report = self.validator.validate_stage3(stage3_output)
            if report.ok:
                logger.info("Validation passed (attempt %d)", attempt + 1)
                final_report = report
                break

            error_count = sum(1 for i in report.issues if i.severity == "error")
            warn_count = sum(1 for i in report.issues if i.severity == "warning")
            logger.warning(
                "Validation failed (attempt %d): %d errors, %d warnings",
                attempt + 1, error_count, warn_count,
            )
            for issue in report.issues:
                logger.warning("  %s [%s]: %s", issue.premise_id, issue.severity, issue.message)

            if attempt < self.config.max_repair_attempts:
                logger.info("Stage 5: Repairing...")
                t5 = time.time()
                stage3_output = resolve_meta_premises(
                    canonicalize_stage3(self.repair_loop.repair(stage3_output, report))
                )
                logger.info("Repair complete (%.1fs)", time.time() - t5)
            else:
                final_report = report

        # Use the last report from the loop if available; avoid redundant re-validation.
        if final_report is None:
            final_report = self.validator.validate_stage3(stage3_output)

        if not final_report.ok:
            error_messages = "\n".join(
                f"  {i.premise_id} [{i.severity}]: {i.message}"
                for i in final_report.issues
                if i.severity == "error"
            )
            logger.error("Pipeline failed validation after all repair attempts:\n%s", error_messages)

        result_status = "success"
        result_error: str | None = None
        if not final_report.ok:
            result_status = "failed"
            result_error = "validation_failed"

        # ── Classify solver readiness ───────────────────────────────────
        for item in stage3_output.compiled:
            if item.kind == "META":
                item.direct_solver_ready = (
                    is_direct_solver_ready_formula(item.formula_tree)
                    if item.formula_tree
                    else False
                )
                item.solver_ready = False
                item.needs_review = not item.meta_resolvable or item.unsupported
                continue

            item.direct_solver_ready = is_direct_solver_ready_formula(item.formula_tree or item.ast)

            status = classify_solver_readiness(
                item.kind,
                item.ast.type,
                [f.strip() for f in (item.notes or [])],
            )
            item.solver_ready = (status == "solver_ready" and item.direct_solver_ready)
            item.needs_review = (status == "needs_review") or (
                status == "solver_ready" and not item.direct_solver_ready
            )
            item.unsupported = (status == "unsupported")
            item.solver_export = [
                item.ast.model_dump(exclude_none=True)
            ] if item.solver_ready else []

        # ── Question Parsing ────────────────────────────────────────────
        question_parse: QuestionParse | None = None
        question_parse_valid = True
        if question:
            logger.info("Parsing question...")
            tq = time.time()
            try:
                question_parse = canonicalize_question_parse(
                    self.question_parser.parse(
                        question,
                        choices,
                        known_predicates=collect_predicate_names(stage3_output),
                    )
                )
                logger.info("Question parse complete (%.1fs)", time.time() - tq)
            except Exception as e:
                logger.error("Question parse failed: %s", e)
                question_parse = QuestionParse(question=question)
                question_parse_valid = False
                result_status = "failed"
                result_error = "question_parse_failed"

        total_time = time.time() - t0
        logger.info("Pipeline complete (%.1fs total)", total_time)

        return FullParseResult(
            premises=stage3_output.compiled,
            question=question_parse,
            status=result_status,
            error=result_error,
            question_parse_valid=question_parse_valid,
        )
