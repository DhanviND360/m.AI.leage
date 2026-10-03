"""Agents subsystem for m.AI.leage."""

from mileage.agents.base import BaseAgent
from mileage.agents.coding_agent import CodingAgent
from mileage.agents.coding_schemas import (
    AgentEvent,
    AgentPhase,
    AgentStatus,
    CodingSessionRecord,
    CodingStepRecord,
)
from mileage.agents.escalation import CopilotEscalation
from mileage.agents.evaluator import EvaluatorAgent
from mileage.agents.evaluator_schemas import (
    AcceptanceCriterionResult,
    EscalationReason,
    EscalationRecord,
    EvaluationReport,
    RequirementResult,
    RequirementVerdict,
)
from mileage.agents.local_agent import LocalAgent
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
from mileage.agents.project_tester import ProjectTester, TestRunResult
from mileage.agents.stagnation import (
    StagnationDetector,
    StagnationReason,
    StagnationReport,
)

__all__ = [
    "BaseAgent",
    "LocalAgent",
    "PlannerAgent",
    "CodingAgent",
    "EvaluatorAgent",
    "CopilotEscalation",
    "AgentPhase",
    "AgentStatus",
    "AgentEvent",
    "CodingSessionRecord",
    "CodingStepRecord",
    "ProjectTester",
    "TestRunResult",
    "StagnationDetector",
    "StagnationReason",
    "StagnationReport",
    "ActionPlan",
    "Complexity",
    "InputType",
    "PlannerInput",
    "PlannerFileRef",
    "Requirement",
    "Constraint",
    "AcceptanceCriterion",
    "EvaluationReport",
    "EscalationRecord",
    "EscalationReason",
    "RequirementVerdict",
    "RequirementResult",
    "AcceptanceCriterionResult",
]
