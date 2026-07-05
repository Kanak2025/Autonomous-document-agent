"""
The autonomous agent itself.

Pipeline (mirrors a classic plan -> act -> reflect agent loop):

  1. GUARDRAILS   - reject empty/too-short/too-long/unsafe requests before
                     spending any LLM calls.
  2. PLAN         - one LLM call turns the free-text request into a structured
                     ExecutionPlan: document type, title, section list, a
                     concrete task list, and any assumptions the agent had
                     to make to fill gaps in an ambiguous request.
  3. EXECUTE      - one LLM call per planned task, generating the actual
                     prose/bullets for that section. Each call is
                     independently retried; if the LLM is unreachable we
                     fall back to a deterministic template so the pipeline
                     still produces a usable document instead of a 500.
  4. REFLECT      - *** the mandatory engineering improvement (self-check) ***
                     One extra LLM call re-reads the full draft against the
                     original request and flags anything missing, generic,
                     or contradictory. If it finds real issues, the agent
                     regenerates only the flagged sections (max one repair
                     pass, so we never loop indefinitely) before returning.
"""
import logging
import time
from typing import Dict, List, Tuple

from app.config import settings
from app.llm_client import chat, safe_json_parse, LLMError
from app.models import ExecutionPlan, TaskItem, ReflectionResult

logger = logging.getLogger("agent.planner")

BLOCKED_KEYWORDS = ["bomb making", "malware", "exploit code"]  # minimal guardrail example


class GuardrailError(Exception):
    pass


def run_guardrails(request_text: str) -> None:
    if len(request_text) < settings.MIN_REQUEST_CHARS:
        raise GuardrailError("Request is too short to act on. Please add more detail.")
    if len(request_text) > settings.MAX_REQUEST_CHARS:
        raise GuardrailError("Request is too long (max 4000 characters).")
    lowered = request_text.lower()
    for kw in BLOCKED_KEYWORDS:
        if kw in lowered:
            raise GuardrailError("Request contains disallowed content.")


# ---------------------------------------------------------------------------
# 1. PLAN
# ---------------------------------------------------------------------------
PLAN_SYSTEM_PROMPT = """You are an autonomous business-documentation agent.
Given a user's natural-language request, decide:
  - what KIND of business document best satisfies it (choose the closest fit:
    business_proposal, meeting_minutes, project_plan, business_report,
    technical_design, sop, product_spec, or "other" with your own label),
  - a clear TITLE for the document,
  - the ordered list of SECTIONS the document should contain,
  - a concrete TASK for producing each section,
  - and, importantly, if the request is ambiguous or missing information
    (dates, names, budget, audience, etc.), do NOT ask a clarifying
    question -- autonomously make the most reasonable business assumption
    and record it in "assumptions" so the user can see what you inferred.

Respond with ONLY a JSON object, no prose, no markdown fences, matching:
{
  "document_type": "string",
  "title": "string",
  "assumptions": ["string", ...],
  "sections": ["string", ...],
  "tasks": [{"id": 1, "name": "string", "section": "string"}, ...]
}
"tasks" must have exactly one task per section, in the same order as "sections".
"""


def plan_request(request_text: str) -> ExecutionPlan:
    messages = [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {"role": "user", "content": request_text},
    ]
    try:
        raw = chat(messages, json_mode=True, temperature=0.3)
        data = safe_json_parse(raw)
        tasks = [TaskItem(**t) for t in data["tasks"]]
        return ExecutionPlan(
            document_type=data.get("document_type", "business_report"),
            title=data.get("title", "Untitled Document"),
            assumptions=data.get("assumptions", []),
            sections=data.get("sections", []),
            tasks=tasks,
        )
    except (LLMError, KeyError, TypeError) as e:
        logger.error(f"Planning failed, using fallback generic plan: {e}")
        return _fallback_plan(request_text)


def _fallback_plan(request_text: str) -> ExecutionPlan:
    """Deterministic plan used only if the LLM is completely unreachable."""
    sections = ["Executive Summary", "Background", "Scope of Work",
                "Timeline", "Budget", "Risks & Mitigations", "Next Steps"]
    tasks = [TaskItem(id=i + 1, name=f"Draft {s}", section=s) for i, s in enumerate(sections)]
    return ExecutionPlan(
        document_type="business_report",
        title="Generated Business Document",
        assumptions=["LLM was unreachable; generic section structure used as a safe fallback."],
        sections=sections,
        tasks=tasks,
    )


# ---------------------------------------------------------------------------
# 2. EXECUTE
# ---------------------------------------------------------------------------
SECTION_SYSTEM_PROMPT = """You are drafting ONE section of a professional business document.
Write clear, concrete, well-organized content -- use realistic mock data
(names, dates, numbers) where the request doesn't supply real ones, since
mock data is explicitly allowed. Use "- " at the start of a line for bullet
points where a list is more readable than prose. Do not repeat the section
title in your answer. Do not add markdown headers. 120-220 words unless the
section is naturally a table/list (e.g. Timeline, Budget), in which case
prefer a bulleted list of line items.
"""


def execute_plan(request_text: str, plan: ExecutionPlan) -> Dict[str, str]:
    section_content: Dict[str, str] = {}
    for i, task in enumerate(plan.tasks):
        if i > 0:
            time.sleep(1.5)  # spread calls out; avoids bursting the free-tier TPM budget
        content = _generate_section(request_text, plan, task)
        section_content[task.section] = content
        task.status = "done"
    return section_content


def _generate_section(request_text: str, plan: ExecutionPlan, task: TaskItem) -> str:
    user_prompt = (
        f"Original user request: {request_text}\n\n"
        f"Document title: {plan.title}\n"
        f"Document type: {plan.document_type}\n"
        f"Assumptions already made: {plan.assumptions}\n\n"
        f"Write the content for the section titled: '{task.section}'."
    )
    messages = [
        {"role": "system", "content": SECTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return chat(messages, json_mode=False, temperature=0.5).strip()
    except LLMError as e:
        logger.error(f"Section '{task.section}' generation failed, using placeholder: {e}")
        task.status = "failed"
        return (f"[Auto-generated placeholder: content for '{task.section}' could not be "
                f"retrieved from the LLM after retries. Please regenerate this section.]")


# ---------------------------------------------------------------------------
# 3. REFLECT  (*** the mandatory engineering improvement ***)
# ---------------------------------------------------------------------------
REFLECTION_SYSTEM_PROMPT = """You are a meticulous editor reviewing a draft business document
against the request that produced it. Check whether:
  - every section is specific and on-topic (not generic filler),
  - nothing the user explicitly asked for is missing,
  - there are no direct contradictions between sections.

Respond with ONLY a JSON object, no prose:
{
  "passed": true/false,
  "issues": ["short description of issue 1", ...],
  "sections_to_revise": ["Section Name", ...]
}
If everything is fine, return passed=true and empty lists.
"""


def reflect_on_draft(request_text: str, plan: ExecutionPlan,
                      section_content: Dict[str, str]) -> ReflectionResult:
    draft_text = "\n\n".join(f"## {s}\n{section_content.get(s, '')}" for s in plan.sections)
    messages = [
        {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Original request: {request_text}\n\nDraft:\n{draft_text}"},
    ]
    try:
        raw = chat(messages, json_mode=True, temperature=0.0)
        data = safe_json_parse(raw)
        return ReflectionResult(
            passed=bool(data.get("passed", True)),
            issues=data.get("issues", []),
            revised_sections=data.get("sections_to_revise", []),
        )
    except LLMError as e:
        logger.warning(f"Reflection step failed, skipping self-check for this run: {e}")
        return ReflectionResult(passed=True, issues=[f"Reflection unavailable: {e}"], revised_sections=[])


def repair_sections(request_text: str, plan: ExecutionPlan,
                     section_content: Dict[str, str],
                     reflection: ReflectionResult) -> Dict[str, str]:
    """Regenerate only the sections flagged by reflection -- capped at one pass."""
    for task in plan.tasks:
        if task.section in reflection.revised_sections:
            logger.info(f"Repairing section flagged by self-check: {task.section}")
            section_content[task.section] = _generate_section(request_text, plan, task)
    return section_content


def run_agent_pipeline(request_text: str) -> Tuple[ExecutionPlan, Dict[str, str], ReflectionResult]:
    """Full plan -> act -> reflect -> (repair) loop used by the API layer."""
    run_guardrails(request_text)

    plan = plan_request(request_text)
    section_content = execute_plan(request_text, plan)

    reflection = reflect_on_draft(request_text, plan, section_content)
    if not reflection.passed and reflection.revised_sections:
        section_content = repair_sections(request_text, plan, section_content, reflection)
        # re-check once so the returned reflection reflects the repaired draft
        reflection = reflect_on_draft(request_text, plan, section_content)

    return plan, section_content, reflection
