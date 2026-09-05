"""Structured state for the SE-Agent workflow.

These models are the contract between stages. Each one is deliberately small:
they are stored in ADK session state, which is copied into every subsequent
prompt, so bulk data (file contents, full diffs, raw test logs) belongs in
artifacts, referenced from here by name.

Field limits are enforced rather than suggested -- an agent that returns forty
requirements would otherwise quietly blow up the context budget for every stage
downstream.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Shared aliases keep the caps visible at the point of use.
ShortText = Annotated[str, Field(max_length=400)]
MediumText = Annotated[str, Field(max_length=2000)]


class Phase(str, Enum):
    """Workflow stages, in execution order."""

    REQUIREMENTS = "requirements"
    ANALYSIS = "analysis"
    ARCHITECTURE = "architecture"
    PLANNING = "planning"
    CODING = "coding"
    TESTING = "testing"
    REVIEW = "review"
    FINAL = "final"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewVerdict(str, Enum):
    APPROVED = "APPROVED"
    CHANGES_REQUIRED = "CHANGES_REQUIRED"


class ChangeScope(str, Enum):
    """How far back a review finding sends the workflow."""

    NONE = "none"  # approved
    MINOR = "minor"  # back to the coder
    MAJOR = "major"  # back to the architect


class SEModel(BaseModel):
    """Base for every workflow model: reject unknown fields, strip whitespace."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ------------------------------------------------------------- requirements


class AcceptanceCriterion(SEModel):
    id: str = Field(max_length=32, description="Stable id, e.g. AC-1")
    description: ShortText
    verification: ShortText = Field(
        description="How this criterion is checked -- ideally a specific test"
    )


class Requirements(SEModel):
    """What the user asked for, made explicit and checkable."""

    summary: MediumText = Field(description="One-paragraph restatement of the request")
    functional: list[ShortText] = Field(default_factory=list, max_length=15)
    non_functional: list[ShortText] = Field(default_factory=list, max_length=10)
    constraints: list[ShortText] = Field(default_factory=list, max_length=10)
    acceptance_criteria: list[AcceptanceCriterion] = Field(
        default_factory=list, max_length=12
    )
    assumptions: list[ShortText] = Field(
        default_factory=list,
        max_length=10,
        description="Decisions made where the request was silent",
    )
    ambiguities: list[ShortText] = Field(
        default_factory=list,
        max_length=10,
        description="Genuine unknowns worth flagging to the user",
    )
    out_of_scope: list[ShortText] = Field(default_factory=list, max_length=10)


# -------------------------------------------------------- repository analysis


class RelevantFile(SEModel):
    path: str = Field(max_length=300)
    reason: ShortText = Field(description="Why this file matters to the request")
    role: str = Field(
        default="source",
        max_length=32,
        description="source | test | config | docs | entrypoint",
    )


class RepositoryAnalysis(SEModel):
    """A map of the repository, not a copy of it."""

    languages: list[str] = Field(default_factory=list, max_length=10)
    structure_summary: MediumText = Field(
        description="How the project is laid out and why"
    )
    architecture_notes: MediumText = ""
    relevant_files: list[RelevantFile] = Field(default_factory=list, max_length=25)
    existing_implementations: list[ShortText] = Field(
        default_factory=list,
        max_length=10,
        description="Code that already does part of this, to extend rather than duplicate",
    )
    test_layout: ShortText = ""
    test_command: str = Field(
        default="pytest -q",
        max_length=200,
        description="The command that actually runs this project's tests",
    )
    dependencies: list[str] = Field(default_factory=list, max_length=40)
    conventions: list[ShortText] = Field(
        default_factory=list,
        max_length=10,
        description="Patterns new code must match to look native here",
    )


# -------------------------------------------------------------- risk analysis


class Risk(SEModel):
    id: str = Field(max_length=32)
    category: str = Field(
        max_length=40,
        description="dependency | compatibility | security | migration | performance | regression",
    )
    description: ShortText
    severity: Severity = Severity.MEDIUM
    mitigation: ShortText = ""


class RiskAnalysis(SEModel):
    risks: list[Risk] = Field(default_factory=list, max_length=15)
    likely_failure_points: list[ShortText] = Field(default_factory=list, max_length=10)
    required_migrations: list[ShortText] = Field(default_factory=list, max_length=6)
    security_concerns: list[ShortText] = Field(default_factory=list, max_length=10)

    @property
    def blocking_risks(self) -> list[Risk]:
        return [r for r in self.risks if r.severity == Severity.CRITICAL]


# ------------------------------------------------------------- architecture


class Component(SEModel):
    name: str = Field(max_length=120)
    responsibility: ShortText
    files: list[str] = Field(default_factory=list, max_length=15)
    change_type: str = Field(
        default="modify", max_length=16, description="create | modify | delete"
    )


class ArchitectureDesign(SEModel):
    approach: MediumText = Field(description="The chosen design, stated plainly")
    components: list[Component] = Field(default_factory=list, max_length=15)
    interfaces: list[ShortText] = Field(
        default_factory=list,
        max_length=15,
        description="Function/class/endpoint signatures being added or changed",
    )
    data_flow: MediumText = ""
    tradeoffs: list[ShortText] = Field(default_factory=list, max_length=8)
    alternatives_rejected: list[ShortText] = Field(default_factory=list, max_length=5)
    implementation_risks: list[ShortText] = Field(default_factory=list, max_length=8)


# ---------------------------------------------------------------- planning


class Task(SEModel):
    task_id: str = Field(max_length=32, description="Stable id, e.g. T-1")
    objective: ShortText
    files: list[str] = Field(default_factory=list, max_length=10)
    dependencies: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="task_ids that must complete first",
    )
    implementation_notes: MediumText = ""
    validation_criteria: ShortText = Field(
        description="How to tell this task is actually done"
    )


class ImplementationPlan(SEModel):
    tasks: list[Task] = Field(default_factory=list, max_length=20)
    ordering_rationale: MediumText = ""

    @field_validator("tasks")
    @classmethod
    def _dependencies_must_resolve(cls, tasks: list[Task]) -> list[Task]:
        """A plan that references a nonexistent task cannot be executed."""
        known = {t.task_id for t in tasks}
        for task in tasks:
            unknown = set(task.dependencies) - known
            if unknown:
                raise ValueError(
                    f"task {task.task_id} depends on unknown task(s): "
                    f"{', '.join(sorted(unknown))}"
                )
        return tasks


# ----------------------------------------------------------------- testing


class TestReport(SEModel):
    """The result of actually running the tests. Never authored by an LLM."""

    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    test_count: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0)
    command: str = Field(default="", max_length=300)
    exit_code: int = 0
    failure_summary: list[ShortText] = Field(
        default_factory=list,
        max_length=20,
        description="One line per failing test: node id and the assertion",
    )
    output_artifact: str = Field(
        default="",
        max_length=200,
        description="Artifact name holding the full log; the log itself is not stored here",
    )
    execution_error: str = Field(
        default="",
        max_length=1000,
        description="Set when the suite could not be run at all (sandbox failure)",
    )

    @property
    def is_green(self) -> bool:
        """True only when tests genuinely ran and all of them passed."""
        return (
            not self.execution_error
            and self.exit_code == 0
            and self.failed == 0
            and self.errors == 0
            and self.test_count > 0
        )

    @property
    def ran(self) -> bool:
        return not self.execution_error and self.test_count > 0


# ------------------------------------------------------------------ review


class ReviewFinding(SEModel):
    id: str = Field(max_length=32)
    category: str = Field(
        max_length=40,
        description="requirements | correctness | architecture | maintainability "
        "| security | tests | regression",
    )
    severity: Severity
    file: str = Field(default="", max_length=300)
    line: int | None = None
    description: ShortText
    recommendation: ShortText
    scope: ChangeScope = Field(
        default=ChangeScope.MINOR,
        description="minor -> coder fixes it; major -> architecture must change",
    )


class CodeReview(SEModel):
    verdict: ReviewVerdict
    summary: MediumText
    findings: list[ReviewFinding] = Field(default_factory=list, max_length=20)
    requirements_covered: list[str] = Field(
        default_factory=list, max_length=12, description="Acceptance criterion ids met"
    )
    requirements_missed: list[str] = Field(default_factory=list, max_length=12)

    @property
    def routing_scope(self) -> ChangeScope:
        """Where the workflow goes next, derived from the findings themselves.

        Deterministic on purpose: the router reads this instead of asking a
        model where to go.
        """
        if self.verdict == ReviewVerdict.APPROVED:
            return ChangeScope.NONE
        if any(f.scope == ChangeScope.MAJOR for f in self.findings):
            return ChangeScope.MAJOR
        return ChangeScope.MINOR

    @field_validator("findings")
    @classmethod
    def _approved_reviews_have_no_blockers(
        cls, findings: list[ReviewFinding], info
    ) -> list[ReviewFinding]:
        verdict = (info.data or {}).get("verdict")
        if verdict == ReviewVerdict.APPROVED:
            blocking = [
                f
                for f in findings
                if f.severity in (Severity.HIGH, Severity.CRITICAL)
            ]
            if blocking:
                raise ValueError(
                    "a review cannot be APPROVED while it carries high/critical "
                    f"findings: {', '.join(f.id for f in blocking)}"
                )
        return findings


# ------------------------------------------------------------- final report


class FinalReport(SEModel):
    feature_request: MediumText
    outcome: str = Field(
        max_length=32, description="completed | partial | failed"
    )
    summary: MediumText
    files_changed: list[str] = Field(default_factory=list, max_length=40)
    tests: TestReport | None = None
    review_verdict: ReviewVerdict | None = None
    iterations_used: int = Field(default=0, ge=0)
    requirements_met: list[str] = Field(default_factory=list, max_length=12)
    requirements_unmet: list[str] = Field(default_factory=list, max_length=12)
    known_limitations: list[ShortText] = Field(default_factory=list, max_length=10)
    follow_up: list[ShortText] = Field(default_factory=list, max_length=10)
    diff_artifact: str = Field(default="", max_length=200)


# ------------------------------------------------------------ run telemetry


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageEvent(SEModel):
    """One row of the observable timeline the UI renders."""

    phase: Phase
    status: StageStatus
    agent: str = Field(default="", max_length=80)
    iteration: int = Field(default=0, ge=0)
    detail: ShortText = ""
    duration_seconds: float = Field(default=0.0, ge=0)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WorkflowState(SEModel):
    """The complete structured state of one SE-Agent run.

    This is what lives in ADK session state. Large outputs are referenced by
    artifact name, never inlined.
    """

    feature_request: str = Field(max_length=8000)
    phase: Phase = Phase.REQUIREMENTS
    iteration: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=5, ge=1, le=10)

    requirements: Requirements | None = None
    repository: RepositoryAnalysis | None = None
    risks: RiskAnalysis | None = None
    architecture: ArchitectureDesign | None = None
    plan: ImplementationPlan | None = None
    tests: TestReport | None = None
    review: CodeReview | None = None
    report: FinalReport | None = None

    completed_tasks: list[str] = Field(default_factory=list, max_length=20)
    files_changed: list[str] = Field(default_factory=list, max_length=40)
    timeline: list[StageEvent] = Field(default_factory=list, max_length=200)
    errors: list[ShortText] = Field(default_factory=list, max_length=30)

    @property
    def iterations_exhausted(self) -> bool:
        return self.iteration >= self.max_iterations
