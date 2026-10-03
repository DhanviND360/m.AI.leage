"""Gemma-powered planner agent for m.AI.leage.

Accepts voice transcripts, text, images, and project files as inputs.
Converts every input into a strict Pydantic ActionPlan schema.
Never directly modifies project files — output is structured JSON only.

Uses Gemma via local Ollama for all inference (multimodal when images provided).
"""

import base64
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from mileage.agents.file_summarizer import build_file_context, estimate_tokens
from mileage.agents.planner_schemas import (
    ActionPlan,
    Complexity,
    InputType,
    PlannerInput,
)
from mileage.core.logger import logger
from mileage.core.workspace import WorkspaceManager
from mileage.models.ollama_client import OllamaClient


# ── Compact, reusable system prompt ──────────────────────────────────────────
PLANNER_SYSTEM_PROMPT = """\
You are the m.AI.leage Planner — a structured planning engine.
Your ONLY job is to analyze user input and produce a JSON action plan.

RULES:
1. Output ONLY valid JSON matching the schema below. No markdown, no explanation.
2. Extract a clear goal, requirements, file references, constraints, complexity, and acceptance criteria.
3. If an image is provided, describe what you see and incorporate it into the plan.
4. Never suggest modifying files directly. Your output is a plan, not code.
5. Keep all text concise. Goal ≤ 1 sentence. Requirements ≤ 3 sentences each.
6. Estimate complexity: trivial, low, medium, high, critical.
7. Every requirement gets an id (R1, R2, ...) and a priority (must, should, could).
8. Every acceptance criterion gets an id (AC1, AC2, ...) and a testable description.

OUTPUT JSON SCHEMA:
{
  "goal": "string",
  "context": "string or null",
  "requirements": [{"id": "R1", "description": "string", "priority": "must|should|could"}],
  "files": [{"path": "string", "role": "target|context|output", "summary": "string or null"}],
  "constraints": [{"description": "string", "reason": "string or null"}],
  "complexity": "trivial|low|medium|high|critical",
  "acceptance_criteria": [{"id": "AC1", "description": "string", "verification": "string or null"}],
  "image_analysis": "string or null"
}
"""


class PlannerAgent:
    """Converts arbitrary inputs into strict ActionPlan schemas via local Gemma.

    Architecture:
        PlannerInput → prompt assembly → Gemma (Ollama) → JSON parse → ActionPlan

    The planner is read-only w.r.t. project files. It reads them for context
    but never writes or modifies anything.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        model_name: str,
        workspace: Optional[WorkspaceManager] = None,
        max_context_tokens: int = 4096,
    ):
        self.client = ollama_client
        self.model_name = model_name
        self.workspace = workspace
        self.max_context_tokens = max_context_tokens

    # ── Public API ───────────────────────────────────────────────────────

    def plan(self, planner_input: PlannerInput) -> ActionPlan:
        """Generate a structured ActionPlan from the given input.

        This is the primary entry point. Accepts text, images, and file paths.
        Always returns a validated ActionPlan — never raw text.
        """
        start_time = time.perf_counter()

        # 1. Build the prompt
        messages = self._build_messages(planner_input)

        # 2. Call Gemma via Ollama
        raw_response = self._call_model(messages, has_images=bool(planner_input.image_paths))

        # 3. Parse JSON from response (with robust repair & context fallback)
        plan_dict = self._extract_json(raw_response, user_input_text=planner_input.text)

        # 4. Build validated ActionPlan
        latency_ms = (time.perf_counter() - start_time) * 1000
        plan = self._build_action_plan(
            plan_dict=plan_dict,
            planner_input=planner_input,
            raw_response=raw_response,
            latency_ms=latency_ms,
        )

        logger.info(
            "Planner generated plan '%s' (complexity=%s, %d requirements, %.0fms)",
            plan.plan_id,
            plan.complexity.value,
            len(plan.requirements),
            plan.latency_ms,
        )

        return plan

    def plan_from_text(self, text: str) -> ActionPlan:
        """Convenience: plan from a plain text prompt."""
        return self.plan(PlannerInput(text=text))

    def plan_from_voice(self, transcript: str) -> ActionPlan:
        """Convenience: plan from a voice transcript."""
        inp = PlannerInput(text=f"[Voice Transcript] {transcript}")
        plan = self.plan(inp)
        plan.input_type = InputType.VOICE_TRANSCRIPT
        return plan

    def plan_from_image(self, image_path: str, text: Optional[str] = None) -> ActionPlan:
        """Convenience: plan from an image (uses Gemma multimodal)."""
        return self.plan(PlannerInput(text=text, image_paths=[image_path]))

    def plan_from_files(self, file_paths: List[str], text: Optional[str] = None) -> ActionPlan:
        """Convenience: plan from project files with optional text."""
        return self.plan(PlannerInput(text=text, file_paths=file_paths))

    # ── Prompt Assembly ──────────────────────────────────────────────────

    def _build_messages(self, inp: PlannerInput) -> List[Dict[str, Any]]:
        """Assemble the message list for the Ollama API call."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT}
        ]

        # Build user content
        user_parts: List[str] = []

        # Workspace context
        if inp.workspace_context:
            user_parts.append(f"[Workspace Context]\n{inp.workspace_context}")
        elif self.workspace:
            try:
                stats = self.workspace.get_stats()
                ws_ctx = (
                    f"[Workspace Context]\n"
                    f"Project: {stats.project_name}\n"
                    f"Files: {stats.total_files}\n"
                    f"Extensions: {dict(stats.extension_counts)}\n"
                )
                user_parts.append(ws_ctx)
            except Exception:
                pass

        # File context (with intelligent summarization)
        if inp.file_paths and self.workspace:
            max_tokens = inp.max_context_tokens or self.max_context_tokens
            file_context, tokens_used = build_file_context(
                file_paths=inp.file_paths,
                workspace_root=self.workspace.root_dir,
                max_total_tokens=max_tokens,
            )
            if file_context.strip():
                user_parts.append(f"[File Context ({tokens_used} tokens)]\n{file_context}")

        # Main text / transcript
        if inp.text:
            user_parts.append(f"[User Request]\n{inp.text}")

        # Explicit instruction to prevent local LLM JSON failures
        user_parts.append(
            "Respond ONLY with a valid JSON object matching the schema. "
            "Do NOT include markdown backticks (```json), commentary, or thinking tags. "
            "Ensure the plan includes at least 1 requirement and at least 1 acceptance criterion."
        )

        user_content = "\n\n".join(user_parts) or "[No input provided — generate a sample plan]"

        user_message: Dict[str, Any] = {
            "role": "user",
            "content": user_content,
        }

        # Multimodal: attach images as base64
        if inp.image_paths:
            images_b64 = self._encode_images(inp.image_paths)
            if images_b64:
                user_message["images"] = images_b64

        messages.append(user_message)
        return messages

    def _encode_images(self, image_paths: List[str]) -> List[str]:
        """Encode images to base64 for Ollama's multimodal API."""
        encoded: List[str] = []
        for img_path in image_paths:
            p = Path(img_path)
            if not p.is_file():
                logger.warning("Image not found: %s", img_path)
                continue
            try:
                raw = p.read_bytes()
                b64 = base64.b64encode(raw).decode("utf-8")
                encoded.append(b64)
                logger.debug("Encoded image %s (%d bytes)", p.name, len(raw))
            except Exception as e:
                logger.warning("Failed to encode image %s: %s", img_path, e)
        return encoded

    # ── Model Interaction ────────────────────────────────────────────────

    def _call_model(
        self,
        messages: List[Dict[str, Any]],
        has_images: bool = False,
    ) -> str:
        """Send messages to Gemma via Ollama and return the raw response text."""
        # Ensure daemon is running and model is available
        self.client.start_daemon(timeout_seconds=5.0)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2,       # Low temp for deterministic structured output
                "num_predict": 4096,      # Sufficient for rich JSON plan without truncation
            },
        }

        try:
            with self.client._get_http_client(timeout=60.0) as client:
                resp = client.post("/api/chat", json=payload)

                if resp.status_code != 200:
                    logger.error("Ollama planner call failed: HTTP %d: %s", resp.status_code, resp.text)
                    raise RuntimeError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")

                data = resp.json()
                return data.get("message", {}).get("content", "")

        except Exception as e:
            logger.error("Planner model call failed: %s", e)
            raise

    # ── JSON Extraction & Validation ─────────────────────────────────────

    def _extract_json(self, raw: str, user_input_text: Optional[str] = None) -> Dict[str, Any]:
        """Extract and parse JSON from the model's response.

        Handles edge cases: think tags, markdown fencing, trailing text, nested JSON,
        trailing commas, unclosed braces from token truncation.
        """
        # Strip reasoning tokens / think tags (<think>...</think>) from reasoning models
        cleaned_raw = re.sub(r'(?is)<think>.*?</think>', '', raw).strip()
        text = cleaned_raw if cleaned_raw else raw.strip()

        # Helper to safely clean common LLM JSON defects
        def _try_parse(candidate_str: str) -> Optional[Dict[str, Any]]:
            # 1. Direct parse
            try:
                parsed = json.loads(candidate_str, strict=False)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

            # 2. Strip trailing commas before } or ]
            cleaned = re.sub(r',\s*([\]}])', r'\1', candidate_str)
            try:
                parsed = json.loads(cleaned, strict=False)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

            # 3. Balance unclosed brackets and braces (handles token truncation)
            opens_curly = cleaned.count("{") - cleaned.count("}")
            opens_sq = cleaned.count("[") - cleaned.count("]")
            if opens_curly > 0 or opens_sq > 0:
                balanced = cleaned + ("]" * max(0, opens_sq)) + ("}" * max(0, opens_curly))
                balanced = re.sub(r',\s*([\]}])', r'\1', balanced)
                try:
                    parsed = json.loads(balanced, strict=False)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass

            return None

        # 1. Direct parse
        res = _try_parse(text)
        if res:
            return res

        # 2. Extract from markdown code fences
        m_fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if m_fence:
            res = _try_parse(m_fence.group(1))
            if res:
                return res

        for fence in ("```json", "```"):
            if fence in text:
                parts = text.split(fence)
                if len(parts) > 1:
                    candidate = parts[1].split("```")[0].strip()
                    res = _try_parse(candidate)
                    if res:
                        return res

        # 3. Find outermost { ... } block
        brace_start = text.find("{")
        if brace_start != -1:
            brace_end = text.rfind("}")
            candidate = text[brace_start : brace_end + 1] if brace_end > brace_start else text[brace_start:]
            res = _try_parse(candidate)
            if res:
                return res

        # 4. Regex extraction of key fields if valid JSON structure was partially broken
        extracted_goal = None
        m_goal = re.search(r'"goal"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text)
        if m_goal:
            extracted_goal = m_goal.group(1).replace(r'\"', '"')

        extracted_reqs = []
        for m_req in re.finditer(r'"description"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text):
            desc = m_req.group(1).replace(r'\"', '"')
            if desc and desc != extracted_goal:
                extracted_reqs.append({"id": f"R{len(extracted_reqs) + 1}", "description": desc, "priority": "must"})

        if extracted_goal or extracted_reqs:
            clean_user_txt = re.sub(r'^\[.*?\]\s*', '', user_input_text or "").strip()
            effective_goal = extracted_goal or clean_user_txt[:120] or "Execute task"
            reqs = extracted_reqs or [{"id": "R1", "description": f"Implement {effective_goal}", "priority": "must"}]
            criteria = [{"id": f"AC{i+1}", "description": f"Verify {r['description']}"} for i, r in enumerate(reqs)]
            return {
                "goal": effective_goal,
                "requirements": reqs,
                "files": [],
                "constraints": [],
                "complexity": "medium",
                "acceptance_criteria": criteria,
            }

        # 5. Intelligent Fallback — wrap user intent as an actionable plan
        logger.warning("Could not parse JSON from planner response; using structured fallback plan.")
        clean_user_txt = re.sub(r'^\[.*?\]\s*', '', user_input_text or text[:120]).strip()
        effective_goal = clean_user_txt or "Complete requested task"

        return {
            "goal": effective_goal,
            "requirements": [
                {
                    "id": "R1",
                    "description": f"Implement core functionality: {effective_goal}",
                    "priority": "must",
                }
            ],
            "files": [],
            "constraints": [],
            "complexity": "medium",
            "acceptance_criteria": [
                {
                    "id": "AC1",
                    "description": f"Verify {effective_goal} is functional and project tests pass",
                    "verification": "test_result",
                }
            ],
        }

    # ── ActionPlan Construction ──────────────────────────────────────────

    def _build_action_plan(
        self,
        plan_dict: Dict[str, Any],
        planner_input: PlannerInput,
        raw_response: str,
        latency_ms: float,
    ) -> ActionPlan:
        """Construct a validated ActionPlan from parsed JSON + metadata."""
        # Safely extract fields with defaults
        goal = plan_dict.get("goal") or "Execute requested task"
        context = plan_dict.get("context")
        complexity_raw = plan_dict.get("complexity", "medium")

        # Validate complexity enum
        try:
            complexity = Complexity(complexity_raw.lower().strip())
        except (ValueError, AttributeError):
            complexity = Complexity.MEDIUM

        # Build requirements — support list of dicts OR list of strings
        requirements = []
        raw_reqs = plan_dict.get("requirements") or plan_dict.get("tasks") or plan_dict.get("steps") or []
        for r in raw_reqs:
            if isinstance(r, dict):
                desc = r.get("description") or r.get("title") or r.get("name") or r.get("task")
                if desc:
                    requirements.append({
                        "id": r.get("id", f"R{len(requirements) + 1}"),
                        "description": str(desc),
                        "priority": r.get("priority", "must"),
                    })
            elif isinstance(r, str) and r.strip():
                requirements.append({
                    "id": f"R{len(requirements) + 1}",
                    "description": r.strip(),
                    "priority": "must",
                })

        # Ensure requirements are never empty
        if not requirements:
            requirements.append({
                "id": "R1",
                "description": f"Implement core functionality for {goal}",
                "priority": "must",
            })

        # Build file refs
        files = []
        for f in plan_dict.get("files", []):
            if isinstance(f, dict) and "path" in f:
                files.append({
                    "path": f["path"],
                    "role": f.get("role", "context"),
                    "summary": f.get("summary"),
                })

        # Build constraints
        constraints = []
        for c in plan_dict.get("constraints", []):
            if isinstance(c, dict) and "description" in c:
                constraints.append({
                    "description": c["description"],
                    "reason": c.get("reason"),
                })

        # Build acceptance criteria — support list of dicts OR list of strings
        acceptance = []
        raw_ac = (
            plan_dict.get("acceptance_criteria")
            or plan_dict.get("acceptanceCriteria")
            or plan_dict.get("criteria")
            or plan_dict.get("test_criteria")
            or []
        )
        for ac in raw_ac:
            if isinstance(ac, dict):
                desc = ac.get("description") or ac.get("criterion") or ac.get("name") or ac.get("test")
                if desc:
                    acceptance.append({
                        "id": ac.get("id", f"AC{len(acceptance) + 1}"),
                        "description": str(desc),
                        "verification": ac.get("verification"),
                    })
            elif isinstance(ac, str) and ac.strip():
                acceptance.append({
                    "id": f"AC{len(acceptance) + 1}",
                    "description": ac.strip(),
                    "verification": "test_result",
                })

        # Ensure acceptance criteria are never empty
        if not acceptance:
            for i, req in enumerate(requirements):
                acceptance.append({
                    "id": f"AC{i + 1}",
                    "description": f"Verify {req['description']}",
                    "verification": "test_result",
                })

        tokens_used = estimate_tokens(raw_response)

        return ActionPlan(
            input_type=planner_input.input_type,
            goal=goal,
            context=context,
            requirements=requirements,
            files=files,
            constraints=constraints,
            complexity=complexity,
            acceptance_criteria=acceptance,
            model_used=self.model_name,
            raw_input=planner_input.text,
            image_analysis=plan_dict.get("image_analysis"),
            tokens_used=tokens_used,
            latency_ms=round(latency_ms, 2),
        )
