"""Tests for the Gemma-powered planner agent.

Tests cover:
- ActionPlan schema validation (text, images, files)
- PlannerInput type detection
- File summarizer (truncation, prioritization, budget)
- Planner JSON extraction (clean, fenced, fallback)
- End-to-end plan generation with mocked Ollama responses
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mileage.agents.file_summarizer import (
    build_file_context,
    estimate_tokens,
    get_file_priority,
    smart_truncate,
    summarize_file,
)
from mileage.agents.planner import PlannerAgent
from mileage.agents.planner_schemas import (
    AcceptanceCriterion,
    ActionPlan,
    Complexity,
    Constraint,
    InputType,
    PlannerFileRef,
    PlannerInput,
    Requirement,
)


# ── Schema Tests ────────────────────────────────────────────────────────────


class TestActionPlanSchema:
    """Validate ActionPlan Pydantic model."""

    def test_minimal_plan(self):
        plan = ActionPlan(goal="Add logging to the API")
        assert plan.goal == "Add logging to the API"
        assert plan.complexity == Complexity.MEDIUM
        assert plan.input_type == InputType.TEXT
        assert plan.requirements == []
        assert plan.files == []
        assert plan.constraints == []
        assert plan.acceptance_criteria == []
        assert plan.plan_id.startswith("plan_")

    def test_full_plan(self):
        plan = ActionPlan(
            goal="Refactor authentication module",
            context="Current auth is using basic JWT without refresh tokens",
            input_type=InputType.MIXED,
            complexity=Complexity.HIGH,
            requirements=[
                Requirement(id="R1", description="Add refresh token support", priority="must"),
                Requirement(id="R2", description="Add rate limiting", priority="should"),
            ],
            files=[
                PlannerFileRef(path="src/auth.py", role="target", summary="Main auth module"),
                PlannerFileRef(path="tests/test_auth.py", role="context"),
            ],
            constraints=[
                Constraint(description="Must not break existing sessions", reason="Production traffic"),
            ],
            acceptance_criteria=[
                AcceptanceCriterion(id="AC1", description="Refresh token endpoint returns 200"),
            ],
            model_used="gemma3:4b",
            tokens_used=512,
            latency_ms=1234.56,
        )
        assert plan.complexity == Complexity.HIGH
        assert len(plan.requirements) == 2
        assert len(plan.files) == 2
        assert plan.files[0].role == "target"
        assert plan.constraints[0].reason == "Production traffic"

    def test_requirement_priority_validation(self):
        req = Requirement(id="R1", description="Test", priority="MUST")
        assert req.priority == "must"

        req2 = Requirement(id="R2", description="Test", priority="invalid")
        assert req2.priority == "must"  # defaults to must

    def test_plan_serialization_roundtrip(self):
        plan = ActionPlan(
            goal="Test serialization",
            requirements=[Requirement(id="R1", description="Serialize", priority="must")],
        )
        json_str = plan.model_dump_json()
        loaded = ActionPlan.model_validate_json(json_str)
        assert loaded.goal == plan.goal
        assert len(loaded.requirements) == 1


class TestPlannerInput:
    """Validate PlannerInput type detection."""

    def test_text_only(self):
        inp = PlannerInput(text="Add a feature")
        assert inp.input_type == InputType.TEXT

    def test_image_only(self):
        inp = PlannerInput(image_paths=["screenshot.png"])
        assert inp.input_type == InputType.IMAGE

    def test_mixed_text_and_image(self):
        inp = PlannerInput(text="Implement this", image_paths=["ui.png"])
        assert inp.input_type == InputType.MIXED

    def test_file_only(self):
        inp = PlannerInput(file_paths=["src/main.py"])
        assert inp.input_type == InputType.FILE

    def test_mixed_text_and_files(self):
        inp = PlannerInput(text="Refactor", file_paths=["src/app.py"])
        assert inp.input_type == InputType.MIXED

    def test_empty_input(self):
        inp = PlannerInput()
        assert inp.input_type == InputType.TEXT


# ── File Summarizer Tests ───────────────────────────────────────────────────


class TestFileSummarizer:
    """Test intelligent file summarization and context building."""

    def test_estimate_tokens(self):
        assert estimate_tokens("") == 1
        assert estimate_tokens("hello world") >= 1
        assert estimate_tokens("a" * 400) == 100

    def test_get_file_priority(self):
        assert get_file_priority("main.py") == 10
        assert get_file_priority("app.ts") == 9
        assert get_file_priority("config.yaml") == 7
        assert get_file_priority("readme.md") == 5
        assert get_file_priority("unknown.xyz") == 1

    def test_smart_truncate_no_truncation(self):
        content = "short content"
        result, was_truncated = smart_truncate(content, max_tokens=100)
        assert result == content
        assert was_truncated is False

    def test_smart_truncate_with_truncation(self):
        # Create content that exceeds token limit
        lines = [f"line {i}: some content here\n" for i in range(200)]
        content = "".join(lines)
        result, was_truncated = smart_truncate(content, max_tokens=100)
        assert was_truncated is True
        assert "omitted" in result or "truncated" in result.lower()

    def test_summarize_file_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            test_file = root / "test.py"
            test_file.write_text("print('hello')\n", encoding="utf-8")

            result = summarize_file(str(test_file), root, max_tokens_per_file=500)
            assert result is not None
            assert "test.py" in result
            assert "print" in result

    def test_summarize_file_not_found(self):
        result = summarize_file("nonexistent.py", Path("."), max_tokens_per_file=500)
        assert result is None

    def test_build_file_context_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Create several files
            for i in range(5):
                f = root / f"file_{i}.py"
                f.write_text(f"# File {i}\nprint({i})\n", encoding="utf-8")

            paths = [f"file_{i}.py" for i in range(5)]
            context, tokens = build_file_context(paths, root, max_total_tokens=200)
            assert tokens > 0
            assert isinstance(context, str)

    def test_build_file_context_priority_ordering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Python files should come before markdown
            py_file = root / "app.py"
            py_file.write_text("import os\n", encoding="utf-8")
            md_file = root / "readme.md"
            md_file.write_text("# Readme\n", encoding="utf-8")

            context, _ = build_file_context(
                ["readme.md", "app.py"], root, max_total_tokens=1000
            )
            # Python should appear first due to higher priority
            py_pos = context.find("app.py")
            md_pos = context.find("readme.md")
            assert py_pos < md_pos


# ── Planner JSON Extraction Tests ───────────────────────────────────────────


class TestPlannerJSONExtraction:
    """Test the planner's JSON extraction from model responses."""

    def _make_planner(self) -> PlannerAgent:
        """Create a PlannerAgent with a mocked client."""
        mock_client = MagicMock()
        mock_client.start_daemon.return_value = True
        return PlannerAgent(
            ollama_client=mock_client,
            model_name="gemma3:4b",
        )

    def test_extract_clean_json(self):
        planner = self._make_planner()
        raw = json.dumps({"goal": "Test goal", "requirements": []})
        result = planner._extract_json(raw)
        assert result["goal"] == "Test goal"

    def test_extract_fenced_json(self):
        planner = self._make_planner()
        raw = '```json\n{"goal": "Fenced goal"}\n```'
        result = planner._extract_json(raw)
        assert result["goal"] == "Fenced goal"

    def test_extract_json_with_preamble(self):
        planner = self._make_planner()
        raw = 'Here is the plan:\n{"goal": "Embedded goal", "complexity": "low"}\nDone!'
        result = planner._extract_json(raw)
        assert result["goal"] == "Embedded goal"

    def test_extract_fallback_on_garbage(self):
        planner = self._make_planner()
        raw = "This is not JSON at all"
        result = planner._extract_json(raw)
        # Should produce a fallback plan
        assert "goal" in result


# ── End-to-End Planner Tests (Mocked Ollama) ────────────────────────────────


class TestPlannerEndToEnd:
    """Test complete plan generation with mocked Ollama responses."""

    def _mock_ollama_response(self, plan_dict: dict) -> MagicMock:
        """Create a mock OllamaClient that returns the given plan dict."""
        mock_client = MagicMock()
        mock_client.start_daemon.return_value = True

        # Mock the HTTP client
        mock_http_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": json.dumps(plan_dict)}
        }
        mock_http_client.post.return_value = mock_response
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)
        mock_client._get_http_client.return_value = mock_http_client

        return mock_client

    def test_plan_from_text(self):
        plan_dict = {
            "goal": "Add rate limiting to the API",
            "context": "API is currently unprotected",
            "requirements": [
                {"id": "R1", "description": "Add per-IP rate limiting", "priority": "must"},
                {"id": "R2", "description": "Add configurable limits", "priority": "should"},
            ],
            "files": [
                {"path": "src/api.py", "role": "target", "summary": "Main API routes"},
            ],
            "constraints": [
                {"description": "Must not affect existing tests", "reason": "CI/CD pipeline"},
            ],
            "complexity": "medium",
            "acceptance_criteria": [
                {"id": "AC1", "description": "429 returned after limit exceeded"},
            ],
        }
        mock_client = self._mock_ollama_response(plan_dict)
        planner = PlannerAgent(ollama_client=mock_client, model_name="gemma3:4b")

        plan = planner.plan_from_text("Add rate limiting to the API")

        assert plan.goal == "Add rate limiting to the API"
        assert plan.complexity == Complexity.MEDIUM
        assert len(plan.requirements) == 2
        assert plan.requirements[0].priority == "must"
        assert len(plan.files) == 1
        assert plan.files[0].role == "target"
        assert len(plan.constraints) == 1
        assert len(plan.acceptance_criteria) == 1
        assert plan.model_used == "gemma3:4b"
        assert plan.latency_ms > 0

    def test_plan_from_voice_transcript(self):
        plan_dict = {
            "goal": "Fix the login bug",
            "requirements": [{"id": "R1", "description": "Debug login flow", "priority": "must"}],
            "files": [],
            "constraints": [],
            "complexity": "low",
            "acceptance_criteria": [],
        }
        mock_client = self._mock_ollama_response(plan_dict)
        planner = PlannerAgent(ollama_client=mock_client, model_name="gemma3:4b")

        plan = planner.plan_from_voice("Fix the login bug")
        assert plan.input_type == InputType.VOICE_TRANSCRIPT
        assert plan.goal == "Fix the login bug"

    def test_plan_with_image(self):
        plan_dict = {
            "goal": "Implement the dashboard UI",
            "image_analysis": "Dashboard showing metrics with 3 cards and a chart",
            "requirements": [{"id": "R1", "description": "Create dashboard component", "priority": "must"}],
            "files": [{"path": "src/dashboard.tsx", "role": "output"}],
            "constraints": [],
            "complexity": "high",
            "acceptance_criteria": [],
        }
        mock_client = self._mock_ollama_response(plan_dict)
        planner = PlannerAgent(ollama_client=mock_client, model_name="gemma3:4b")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n")
            img_path = f.name

        plan = planner.plan_from_image(img_path, text="Implement this dashboard")
        assert plan.image_analysis is not None
        assert "dashboard" in plan.image_analysis.lower()

    def test_plan_with_file_context(self):
        plan_dict = {
            "goal": "Refactor the models module",
            "requirements": [{"id": "R1", "description": "Split models into separate files", "priority": "must"}],
            "files": [{"path": "src/models.py", "role": "target"}],
            "constraints": [],
            "complexity": "medium",
            "acceptance_criteria": [{"id": "AC1", "description": "All tests pass after refactor"}],
        }
        mock_client = self._mock_ollama_response(plan_dict)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src_file = root / "src" / "models.py"
            src_file.parent.mkdir(parents=True)
            src_file.write_text("class User:\n    pass\n", encoding="utf-8")

            workspace = MagicMock()
            workspace.root_dir = root
            workspace.get_stats.return_value = MagicMock(
                project_name="test", total_files=1, extension_counts={".py": 1}
            )

            planner = PlannerAgent(
                ollama_client=mock_client,
                model_name="gemma3:4b",
                workspace=workspace,
            )

            plan = planner.plan_from_files(
                ["src/models.py"], text="Refactor the models module"
            )
            assert plan.goal == "Refactor the models module"
            assert len(plan.acceptance_criteria) == 1

    def test_plan_handles_invalid_complexity(self):
        """Model returns an unrecognized complexity — should default to MEDIUM."""
        plan_dict = {
            "goal": "Test invalid complexity",
            "complexity": "super_hard",
            "requirements": [],
            "files": [],
            "constraints": [],
            "acceptance_criteria": [],
        }
        mock_client = self._mock_ollama_response(plan_dict)
        planner = PlannerAgent(ollama_client=mock_client, model_name="gemma3:4b")

        plan = planner.plan_from_text("Test invalid complexity")
        assert plan.complexity == Complexity.MEDIUM

    def test_plan_json_output_is_valid(self):
        """Ensure the plan can be serialized to valid JSON."""
        plan_dict = {
            "goal": "JSON roundtrip test",
            "requirements": [{"id": "R1", "description": "Test", "priority": "must"}],
            "files": [],
            "constraints": [],
            "complexity": "trivial",
            "acceptance_criteria": [],
        }
        mock_client = self._mock_ollama_response(plan_dict)
        planner = PlannerAgent(ollama_client=mock_client, model_name="gemma3:4b")

        plan = planner.plan_from_text("JSON roundtrip test")
        json_str = plan.model_dump_json(indent=2)
        parsed = json.loads(json_str)
        assert parsed["goal"] == "JSON roundtrip test"
        assert parsed["complexity"] == "trivial"
