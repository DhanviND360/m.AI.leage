"""Evaluator Agent: verifies implementation against ActionPlan acceptance criteria.

Uses tests, file inspection, build results, and targeted LLM evaluation to
produce PASS / PARTIAL / FAIL verdicts with concrete evidence for every
requirement. Detects local-model overcapacity and triggers Copilot escalation
when the local model cannot complete the task.
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from mileage.agents.evaluator_schemas import (
    AcceptanceCriterionResult,
    EscalationReason,
    EvaluationReport,
    RequirementResult,
    RequirementVerdict,
)
from mileage.agents.planner_schemas import ActionPlan
from mileage.agents.project_tester import ProjectTester, TestRunResult
from mileage.core.logger import logger
from mileage.core.workspace import WorkspaceManager
from mileage.models.ollama_client import OllamaClient
from mileage.models.schemas import ChatMessage, MessageRole
from mileage.tools.permissions import ToolPermission


EVALUATOR_SYSTEM_PROMPT = """\
You are the m.AI.leage Evaluator — a strict, evidence-based quality gate.
Your job is to verify whether a codebase satisfies a set of requirements and
acceptance criteria from an ActionPlan.

RULES:
1. Output ONLY valid JSON matching the schema below.
2. For EACH requirement, output a verdict: "pass", "partial", or "fail".
3. Always cite concrete evidence: a test name, a file path + line, or a build log excerpt.
4. Set confidence to 0.0–1.0 based on how certain you are of the verdict.
5. If you cannot verify a requirement, set verdict to "fail" and confidence < 0.3.
6. Never claim pass without evidence.

OUTPUT JSON SCHEMA:
{
  "requirements": [
    {
      "requirement_id": "R1",
      "verdict": "pass" | "partial" | "fail",
      "confidence": 0.0–1.0,
      "evidence": "concrete proof string",
      "verification_method": "test_result" | "file_check" | "llm_eval"
    }
  ],
  "acceptance_criteria": [
    {
      "criterion_id": "AC1",
      "verdict": "pass" | "partial" | "fail",
      "confidence": 0.0–1.0,
      "evidence": "concrete proof string",
      "verification_method": "test_result" | "file_check" | "llm_eval"
    }
  ],
  "summary": "One-paragraph assessment of overall implementation quality"
}
"""


class EvaluatorAgent:
    """Compares implementation state against an ActionPlan and produces an EvaluationReport.

    Verification strategy (layered):
    1. Run project-native tests and parse pass/fail counts.
    2. Inspect referenced files to confirm they exist and contain expected structures.
    3. Use targeted LLM evaluation for requirements that cannot be checked mechanically.
    4. Aggregate verdicts and compute overcapacity signals.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        model_name: str,
        workspace: WorkspaceManager,
        permission: Optional[ToolPermission] = None,
        low_confidence_threshold: float = 0.4,
        overcapacity_fail_ratio: float = 0.5,
    ):
        self.client = ollama_client
        self.model_name = model_name
        self.workspace = workspace
        self.permission = permission or ToolPermission(
            allowed_directories=[str(workspace.root_dir)]
        )
        self.low_confidence_threshold = low_confidence_threshold
        self.overcapacity_fail_ratio = overcapacity_fail_ratio

    def evaluate(
        self,
        plan: ActionPlan,
        session_record=None,
        custom_test_cmd: Optional[str] = None,
    ) -> EvaluationReport:
        """Run full evaluation pipeline and return a structured report."""
        start_t = time.perf_counter()
        report = EvaluationReport(
            plan_id=plan.plan_id,
            goal=plan.goal,
            model_name=self.model_name,
        )

        # ── 1. Run project-native tests ──────────────────────────────
        test_result = self._run_tests(custom_test_cmd)
        report.tests_passed = test_result.passed
        report.test_summary = (
            f"{test_result.framework}: {test_result.tests_passed} passed, "
            f"{test_result.tests_failed} failed, {test_result.tests_errors} errors"
        )
        if not test_result.passed:
            report.test_failure_details = test_result.failure_summary

        # ── 2. Verify referenced files exist ─────────────────────────
        for fref in plan.files:
            fpath = self.workspace.root_dir / fref.path
            if fpath.is_file():
                report.files_verified.append(fref.path)

        # ── 3. Mechanical requirement checks ─────────────────────────
        for req in plan.requirements:
            result = self._check_requirement_mechanically(req, test_result, plan)
            report.requirement_results.append(result)

        for ac in plan.acceptance_criteria:
            result = self._check_acceptance_mechanically(ac, test_result, plan)
            report.acceptance_results.append(result)

        # ── 4. Targeted LLM evaluation for uncertain verdicts ────────
        uncertain_reqs = [
            r for r in report.requirement_results
            if r.confidence < self.low_confidence_threshold
        ]
        uncertain_acs = [
            r for r in report.acceptance_results
            if r.confidence < self.low_confidence_threshold
        ]

        if uncertain_reqs or uncertain_acs:
            self._llm_evaluate(plan, report, uncertain_reqs, uncertain_acs)
            report.llm_evaluation_used = True

        # ── 5. Compute aggregates and overcapacity ───────────────────
        report.compute_aggregates()
        self._detect_overcapacity(report, session_record)

        report.evaluation_duration_ms = round(
            (time.perf_counter() - start_t) * 1000, 2
        )

        return report

    def _run_tests(self, custom_test_cmd: Optional[str] = None) -> TestRunResult:
        """Execute project-native tests."""
        try:
            tester = ProjectTester(
                workspace_root=self.workspace.root_dir,
                permission=self.permission,
                custom_test_cmd=custom_test_cmd,
            )
            return tester.run()
        except Exception as e:
            logger.warning("Evaluator test run failed: %s", e)
            return TestRunResult(
                passed=False,
                exit_code=1,
                command="",
                failure_summary=f"Test execution error: {str(e)}",
            )

    def _check_requirement_mechanically(
        self, req, test_result: TestRunResult, plan: ActionPlan,
    ) -> RequirementResult:
        """Check a requirement using test results and file presence."""
        # If tests pass, requirements about "tests passing" get PASS
        evidence_parts = []
        confidence = 0.2  # base

        # Check if target files exist
        target_files = [f for f in plan.files if f.role == "target"]
        for tf in target_files:
            fpath = self.workspace.root_dir / tf.path
            if fpath.is_file():
                evidence_parts.append(f"File '{tf.path}' exists ({fpath.stat().st_size} bytes)")
                confidence += 0.1

        # Test-based evidence
        if test_result.passed:
            evidence_parts.append(
                f"All tests pass ({test_result.tests_passed} passed, "
                f"{test_result.tests_failed} failed)"
            )
            confidence += 0.4
        elif test_result.total_tests > 0:
            evidence_parts.append(
                f"Tests partial: {test_result.tests_passed}/{test_result.total_tests} passed"
            )
            confidence += 0.15

        evidence = "; ".join(evidence_parts) if evidence_parts else "No mechanical evidence available"
        confidence = min(confidence, 1.0)

        # Determine verdict
        if test_result.passed and confidence >= 0.5:
            verdict = RequirementVerdict.PASS
        elif confidence >= 0.3 and test_result.tests_passed > 0:
            verdict = RequirementVerdict.PARTIAL
        else:
            verdict = RequirementVerdict.FAIL

        return RequirementResult(
            requirement_id=req.id,
            description=req.description,
            priority=req.priority,
            verdict=verdict,
            confidence=round(confidence, 2),
            evidence=evidence,
            verification_method="test_result" if test_result.total_tests > 0 else "file_check",
        )

    def _check_acceptance_mechanically(
        self, ac, test_result: TestRunResult, plan: ActionPlan,
    ) -> AcceptanceCriterionResult:
        """Check an acceptance criterion using test results."""
        evidence_parts = []
        confidence = 0.2

        if test_result.passed:
            evidence_parts.append(f"All {test_result.tests_passed} tests pass")
            confidence += 0.4
        elif test_result.tests_passed > 0:
            evidence_parts.append(
                f"Partial: {test_result.tests_passed}/{test_result.total_tests} pass"
            )
            confidence += 0.15

        evidence = "; ".join(evidence_parts) if evidence_parts else "No mechanical evidence available"
        confidence = min(confidence, 1.0)

        if test_result.passed and confidence >= 0.5:
            verdict = RequirementVerdict.PASS
        elif confidence >= 0.3 and test_result.tests_passed > 0:
            verdict = RequirementVerdict.PARTIAL
        else:
            verdict = RequirementVerdict.FAIL

        return AcceptanceCriterionResult(
            criterion_id=ac.id,
            description=ac.description,
            verdict=verdict,
            confidence=round(confidence, 2),
            evidence=evidence,
            verification_method="test_result" if test_result.total_tests > 0 else "file_check",
        )

    def _llm_evaluate(
        self,
        plan: ActionPlan,
        report: EvaluationReport,
        uncertain_reqs: List[RequirementResult],
        uncertain_acs: List[AcceptanceCriterionResult],
    ) -> None:
        """Use targeted LLM evaluation to resolve uncertain requirements."""
        # Gather file context for uncertain items
        file_context_parts = []
        for fref in plan.files[:5]:  # limit context window
            fpath = self.workspace.root_dir / fref.path
            if fpath.is_file():
                try:
                    content = self.workspace.read_file_safely(fref.path, max_lines=100)
                    file_context_parts.append(f"=== {fref.path} ===\n{content}")
                except Exception:
                    pass

        file_context = "\n\n".join(file_context_parts) if file_context_parts else "[No files available]"

        # Build evaluation prompt
        items_to_check = []
        for r in uncertain_reqs:
            items_to_check.append(f"- Requirement {r.requirement_id}: {r.description}")
        for a in uncertain_acs:
            items_to_check.append(f"- Criterion {a.criterion_id}: {a.description}")

        items_str = "\n".join(items_to_check)

        user_prompt = (
            f"GOAL: {plan.goal}\n\n"
            f"ITEMS TO VERIFY:\n{items_str}\n\n"
            f"PROJECT FILES:\n{file_context}\n\n"
            f"TEST RESULTS: {report.test_summary}\n"
            f"{'TEST FAILURES: ' + report.test_failure_details if report.test_failure_details else ''}\n\n"
            f"Evaluate each item above. Output JSON per the schema."
        )

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=EVALUATOR_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ]

        try:
            self.client.start_daemon(timeout_seconds=5.0)
            payload = {
                "model": self.model_name,
                "messages": [{"role": m.role.value, "content": m.content} for m in messages],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 2048},
            }
            with self.client._get_http_client(timeout=60.0) as http_client:
                resp = http_client.post("/api/chat", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = data.get("message", {}).get("content", "")
                    report.evaluation_tokens += data.get("eval_count", 0)
                    report.evaluation_tokens += data.get("prompt_eval_count", 0)
                    self._apply_llm_verdicts(report, raw_text, uncertain_reqs, uncertain_acs)
                    report.llm_evaluation_summary = raw_text[:500]
        except Exception as e:
            logger.warning("LLM evaluation call failed: %s", e)
            report.llm_evaluation_summary = f"LLM evaluation failed: {str(e)}"

    def _apply_llm_verdicts(
        self,
        report: EvaluationReport,
        raw_text: str,
        uncertain_reqs: List[RequirementResult],
        uncertain_acs: List[AcceptanceCriterionResult],
    ) -> None:
        """Parse LLM JSON and apply verdicts to uncertain items."""
        parsed = self._extract_json(raw_text)

        llm_reqs = {
            r.get("requirement_id", ""): r
            for r in parsed.get("requirements", [])
        }
        llm_acs = {
            a.get("criterion_id", ""): a
            for a in parsed.get("acceptance_criteria", [])
        }

        for r in uncertain_reqs:
            llm_r = llm_reqs.get(r.requirement_id, {})
            if llm_r:
                v = llm_r.get("verdict", "").lower()
                if v in ("pass", "partial", "fail"):
                    r.verdict = RequirementVerdict(v)
                conf = llm_r.get("confidence")
                if isinstance(conf, (int, float)):
                    r.confidence = max(r.confidence, min(conf, 1.0))
                ev = llm_r.get("evidence", "")
                if ev:
                    r.evidence = f"{r.evidence}; LLM: {ev}" if r.evidence else f"LLM: {ev}"
                r.verification_method = "llm_eval"

        for a in uncertain_acs:
            llm_a = llm_acs.get(a.criterion_id, {})
            if llm_a:
                v = llm_a.get("verdict", "").lower()
                if v in ("pass", "partial", "fail"):
                    a.verdict = RequirementVerdict(v)
                conf = llm_a.get("confidence")
                if isinstance(conf, (int, float)):
                    a.confidence = max(a.confidence, min(conf, 1.0))
                ev = llm_a.get("evidence", "")
                if ev:
                    a.evidence = f"{a.evidence}; LLM: {ev}" if a.evidence else f"LLM: {ev}"
                a.verification_method = "llm_eval"

    def _detect_overcapacity(self, report: EvaluationReport, session_record=None) -> None:
        """Detect if the local model is overcapacity based on evaluation signals."""
        signals: List[str] = []

        # Signal 1: High fail ratio
        if report.total_requirements > 0:
            fail_ratio = report.fail_count / report.total_requirements
            if fail_ratio >= self.overcapacity_fail_ratio:
                signals.append(
                    f"High fail ratio: {report.fail_count}/{report.total_requirements} "
                    f"requirements failed ({fail_ratio:.0%})"
                )

        # Signal 2: Low overall confidence
        if report.overall_confidence < self.low_confidence_threshold:
            signals.append(
                f"Low evaluation confidence: {report.overall_confidence:.2f} "
                f"(threshold: {self.low_confidence_threshold})"
            )

        # Signal 3: Stagnation from coding session
        if session_record:
            stag = getattr(session_record, "stagnation_report", None)
            if stag and getattr(stag, "is_stagnant", False):
                signals.append(
                    f"Coding agent stagnated: {getattr(stag, 'reason', 'unknown')}"
                )

            # Repeated loop detection
            total_iters = getattr(session_record, "total_iterations", 0)
            if total_iters >= 5 and not report.tests_passed:
                signals.append(
                    f"Excessive iterations ({total_iters}) without test pass"
                )

        # Signal 4: Tests still failing after agent work
        if not report.tests_passed and report.test_failure_details:
            signals.append("Tests still failing after local agent execution")

        report.overcapacity_signals = signals
        report.overcapacity_detected = len(signals) >= 2

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from model output."""
        if not text:
            return {}
        clean = text.strip()

        if "```" in clean:
            matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
            if matches:
                clean = matches[0].strip()

        try:
            return json.loads(clean)
        except Exception:
            pass

        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(clean[start:end + 1])
            except Exception:
                pass

        return {}
