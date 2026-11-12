# SE-Agent Architecture and Latent Defect Analysis

## 1. Overview
SE-Agent is an automated software engineering multi-agent system built on top of the Google Agent Development Kit (ADK) and backed primarily by Kimi K3 via NVIDIA NIM (with configurable support for Gemini). It automates the software engineering lifecycle from a natural-language feature request to a verified, code-reviewed implementation.

---

## 2. Core Architecture & Workflow Pipeline

SE-Agent supports two main operational modes: the **Full Pipeline** and the **Direct Run Mode**.

### A. Full Pipeline Flow
1. **Requirements Analyst (`requirements.py`)**
   - **Role:** Converts a natural-language feature request into an explicit, checkable specification (`Requirements` schema).
   - **Outputs:** Functional/non-functional requirements, acceptance criteria (with IDs like `AC-1`), assumptions, ambiguities, and out-of-scope declarations.

2. **Repository Analyst (`repository.py`) & Risk Analyst (`risk.py`)**
   - **Execution:** Run in parallel.
   - **Repository Analyst:** Maps the codebase structure, identifies relevant files, locates existing related implementations (to avoid duplication), extracts test commands, and identifies project conventions.
   - **Risk Analyst:** Predicts dependency, compatibility, security, migration, and regression risks, providing concrete mitigations and failure points.

3. **Software Architect (`architecture.py`)**
   - **Role:** Designs the change based on requirements, repository analysis, and risk predictions.
   - **Outputs:** Component list, real interfaces with signatures, change types (create/modify/delete), tradeoffs, and rejected alternatives.

4. **Implementation Planner (`planner.py`)**
   - **Role:** Breaks the architecture into small, ordered, verifiable tasks (`T-1`, `T-2`, etc.).
   - **Outputs:** Implementation plan with task objectives, target files, validation criteria, and dependencies.

5. **Implementation Coordinator & Coder (`coordinator.py` / `coder.py`)**
   - **Role:** Executes the plan against the real repository.
   - **Authority:** Full coding toolset (`edit_file`, `write_file`, `run_tests`, etc.).
   - **Specialists:** Wraps diagnostic specialists (`debugger_agent`, `security_agent`, `testing_agent`) as tools rather than sub-agents so control returns immediately after consultation.

6. **Tester Agent (`tester.py`)**
   - **Determinism:** Has **no LLM** by design. Test results are produced deterministically by executing the project's test suite inside an isolated sandbox (`Sandbox`), parsing process output and exit codes into a strict `TestReport`.
   - **Artifacts:** Large test execution logs are stored as session artifacts (`test_output_iteration_X.txt`) to keep session state concise.

7. **Code Reviewer (`reviewer.py`)**
   - **Role:** Independent, read-only evaluation of the finished change.
   - **Checks:** Requirements coverage, correctness, regressions, test quality, security, and maintainability.
   - **Verdicts:** `APPROVED` or `CHANGES_REQUIRED`.

8. **Report Agent (`coordinator.py`)**
   - **Role:** Compiles the final human-readable `FinalReport` detailing requirements met, files changed, test results, limitations, and follow-up actions.

### B. Direct Mode Flow
- Skips requirements, architecture, repository survey, and review phases.
- Uses `build_direct_planner_agent()` to inspect the repository directly, generate a streamlined implementation plan, and hand off directly to the coordinator/coder and tester.

---

## 3. Agent Tooling & Shared Discipline
- **Shared Preamble & Discipline (`common.py`):** Every agent runs with strict rules requiring grounding all claims in actual tool outputs (e.g., `read_file`, `search_code`), avoiding assumptions, and adhering to strict Pydantic schemas.
- **Specialists (`debugger.py`):**
  - `debugger_agent`: Read-only diagnosis of test failures (root cause, evidence, fix).
  - `security_agent`: Targeted analysis of injection, authz gaps, secret handling, and unsafe defaults.
  - `testing_agent`: Strategy advisor for coverage gaps and weak assertions.

---

## 4. Latent Defect & Failure Mode Inventory

Based on observed codebase patterns, error handling mechanisms, and workflow constraints, several latent risks and failure modes are managed or present in the system:

1. **Test Log Artifact Storage Failures**
   - *Pattern:* In `tester.py`, saving large test logs as artifacts via `artifact_service.save_artifact(...)` is wrapped in a robust `try...except` block with a warning log.
   - *Latent Risk:* If the artifact service fails or times out, the test run must still succeed without crashing the entire workflow iteration.

2. **State Contamination and Stale Repository Artifacts**
   - *Pattern:* ADK session state persists across invocations. If a previous run left files or state keys (`plan`, `feature_request`, `tests`), downstream agents might misattribute previous work to the current run.
   - *Mitigation:* Explicit fallback prompts (`NO_PLAN`, `NO_REQUEST`) and strict validation ensure agents report only on verifiable current state and real `git_diff`/`git_status`.

3. **LLM Hallucination of Test Results**
   - *Risk:* Language models asked to run tests or verify code can fabricate passing outcomes.
   - *Mitigation:* `TesterAgent` completely eschews an LLM, executing tests purely deterministically via sandbox process execution.

4. **Context Window Inflation & Token Budget Pressures**
   - *Risk:* Verbose test outputs or redundant file contents passed through session state can overwhelm context windows or cause upstream models to summarize/lose critical failure details.
   - *Mitigation:* Offloading large test output logs to artifacts and enforcing concise structured Pydantic schemas across all agents.

5. **Rigidity in Iteration Limits & Task Scope**
   - *Risk:* Under strict iteration limits or poorly scoped plans, agents may get stuck in infinite review/fix loops or produce partial outcomes.
   - *Mitigation:* Explicit outcome states (`completed`, `partial`, `failed`) and strict adherence to task ordering and file boundaries.
