"""Post-call report generation for the Mars settler screening agent.

Everything the agent does *after* the call ends lives here:

  * `build_transcript`        — flatten a LiveKit AgentSession history to text
  * `evaluate_candidate`      — send the transcript to the eval LLM and
                                 parse its structured JSON response
  * `render_markdown_report`  — turn the evaluation + transcript into a
                                 rich markdown report
  * `save_report`             — write the markdown to ./reports/
  * `generate_and_save`       — orchestrates the whole pipeline and also
                                 appends a row to Google Sheets if
                                 configured; this is what the agent's
                                 shutdown callback calls

Keeping this out of interview_agent.py keeps the agent file focused on
the voice loop: STT/LLM/TTS wiring, prompt, entrypoint.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from livekit.agents import AgentSession, llm
from openai import AsyncOpenAI

from google_sheets import sheets_service

logger = logging.getLogger("mars-recruiter.report")

# ---------------------------------------------------------------------------
# Config + constants
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPORTS_DIR = HERE / "reports"

COMPANY = "Mars Recruitment Services"
INTERVIEWER_NAME = "Nova"

SIMPLISMART_API_KEY = os.environ.get("SIMPLISMART_API_KEY")

# Post-call evaluation LLM. OpenAI-compatible endpoint; defaults to
# Simplismart's gpt-oss-120b. Override either via .env.local.
EVAL_LLM_URL = os.environ.get("EVAL_LLM_URL", "https://api.simplismart.live")
EVAL_LLM_MODEL = os.environ.get("EVAL_LLM_MODEL", "openai/gpt-oss-120b")


# ---------------------------------------------------------------------------
# Evaluation prompts
# ---------------------------------------------------------------------------

EVALUATION_SYSTEM_PROMPT = f"""
You are an expert recruiter at {COMPANY} producing a written evaluation
of a first-round phone screening interview for a Mars settler role.
Evaluate the candidate strictly based on what was said in the transcript
provided in the next message.

Respond with ONLY a single JSON object — no prose, no markdown code
fences, no commentary before or after. The JSON must have exactly these
keys:

{{
  "candidate_name": "<first name from the transcript, or 'Unknown'>",
  "selected_role": "<the mission role they chose, or 'Unknown'>",
  "selection_notes": "<1-2 sentences summarizing what the candidate said about their interests / why this role>",
  "score": <integer 1-10 reflecting overall suitability>,
  "recommendation": "<one of: Proceed | On Hold | Reject>",
  "strengths": "<2-4 sentence summary of strongest points, citing specifics>",
  "areas_for_improvement": "<2-4 sentence summary of gaps, concerns, or red flags>",
  "motivation_assessment": "<2-3 sentences on motivation and commitment to a multi-year Mars deployment>",
  "summary": "<3-5 sentence overall assessment for the mission selection board>",
  "answers": [
    {{
      "topic": "<short topic label, e.g. 'Background' or 'Motivation for Mars'>",
      "question_asked": "<the exact question the interviewer asked>",
      "response_summary": "<1-2 sentence neutral summary of the candidate's answer>",
      "strengths": "<specific strengths cited from the answer, or empty string>",
      "concerns": "<specific concerns or gaps, or empty string>"
    }}
  ]
}}

The "answers" array should contain one entry per interview question
actually asked and answered in the transcript (typically two).

SCORING RUBRIC:
- 8-10 → Strong fit, proceed to next round
- 5-7  → Mixed signals, on hold / needs further assessment
- 1-4  → Poor fit or insufficient information, reject
""".strip()


EVALUATION_USER_PROMPT_TEMPLATE = (
    "Transcript of the screening interview:\n\n{transcript}\n\n"
    "Now produce the JSON evaluation."
)


# ---------------------------------------------------------------------------
# State derived from the evaluation JSON
# ---------------------------------------------------------------------------


@dataclass
class CapturedAnswer:
    topic: str
    question_asked: str
    response_summary: str
    strengths: str = ""
    concerns: str = ""


@dataclass
class CapturedState:
    candidate_name: str = ""
    selected_role: str = ""
    selection_notes: str = ""
    answers: list[CapturedAnswer] = field(default_factory=list)


def state_from_evaluation(evaluation: dict[str, Any]) -> CapturedState:
    """Populate a CapturedState from the post-call evaluation JSON."""
    answers_raw = evaluation.get("answers") or []
    answers: list[CapturedAnswer] = []
    for a in answers_raw:
        if not isinstance(a, dict):
            continue
        answers.append(
            CapturedAnswer(
                topic=str(a.get("topic") or "").strip() or "Untitled",
                question_asked=str(a.get("question_asked") or "").strip(),
                response_summary=str(a.get("response_summary") or "").strip(),
                strengths=str(a.get("strengths") or "").strip(),
                concerns=str(a.get("concerns") or "").strip(),
            )
        )
    return CapturedState(
        candidate_name=str(evaluation.get("candidate_name") or "").strip(),
        selected_role=str(evaluation.get("selected_role") or "").strip(),
        selection_notes=str(evaluation.get("selection_notes") or "").strip(),
        answers=answers,
    )


# ---------------------------------------------------------------------------
# Transcript + evaluation LLM call
# ---------------------------------------------------------------------------


def build_transcript(session: AgentSession) -> str:
    """Flatten a LiveKit AgentSession history into a plaintext transcript."""
    lines: list[str] = []
    for item in session.history.items:
        if not isinstance(item, llm.ChatMessage):
            continue
        text = (item.text_content or "").strip()
        if not text:
            continue
        role_label = "Candidate" if item.role == "user" else INTERVIEWER_NAME
        lines.append(f"{role_label}: {text}")
    return "\n".join(lines)


async def evaluate_candidate(transcript: str) -> tuple[dict[str, Any], str]:
    """Evaluate the transcript via EVAL_LLM_URL / EVAL_LLM_MODEL.

    Uses non-streaming chat completions against an OpenAI-compatible
    endpoint. Non-streaming is important for reasoning models like
    gpt-oss that can emit nothing on the visible stream channel.
    """
    if not transcript.strip():
        logger.warning("empty transcript — skipping evaluation")
        return {}, "Empty transcript — the session ended before anything was said."

    # Short transcripts aren't worth a full evaluation call.
    turn_count = sum(1 for line in transcript.splitlines() if line.strip())
    if len(transcript) < 200 or turn_count < 3:
        msg = (
            f"Transcript too short to evaluate "
            f"({len(transcript)} chars, {turn_count} turn(s)). "
            "The candidate likely disconnected before the screening began."
        )
        logger.warning(msg)
        return {}, msg

    provider = f"'{EVAL_LLM_MODEL}' ({EVAL_LLM_URL})"
    logger.info(
        "sending transcript to %s for evaluation (%d chars)",
        provider, len(transcript),
    )

    try:
        client = AsyncOpenAI(api_key=SIMPLISMART_API_KEY, base_url=EVAL_LLM_URL)
        completion = await client.chat.completions.create(
            model=EVAL_LLM_MODEL,
            messages=[
                {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": EVALUATION_USER_PROMPT_TEMPLATE.format(transcript=transcript),
                },
            ],
            temperature=0.1,
            max_tokens=4096,
            stream=False,
        )
    except Exception as e:
        logger.error("%s call raised %s: %s", provider, type(e).__name__, e, exc_info=True)
        return {}, f"{provider} raised {type(e).__name__}: {e}"

    choice = completion.choices[0] if completion.choices else None
    finish_reason = getattr(choice, "finish_reason", None) if choice else None
    result_text = (choice.message.content or "").strip() if choice else ""
    logger.info(
        "%s returned %d chars, finish_reason=%s, usage=%s",
        provider, len(result_text), finish_reason, completion.usage,
    )

    if not result_text:
        logger.error(
            "%s returned empty content; choice=%s",
            provider, choice.model_dump() if choice else None,
        )
        return {}, (
            f"{provider} returned empty content (finish_reason={finish_reason})."
        )

    cleaned = re.sub(r"```(?:json)?\s*", "", result_text).strip()
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not json_match:
        logger.error(
            "%s response had no JSON; first 500 chars: %s",
            provider, result_text[:500],
        )
        return {}, f"{provider} response did not contain JSON."

    try:
        return json.loads(json_match.group()), ""
    except json.JSONDecodeError as e:
        logger.error(
            "%s JSON parse failed: %s; first 500 chars: %s",
            provider, e, json_match.group()[:500],
        )
        return {}, f"{provider} JSON failed to parse: {e}"


# ---------------------------------------------------------------------------
# Markdown rendering + disk save
# ---------------------------------------------------------------------------


def _slug(s: str, max_len: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", s or "").strip("-")
    return (s[:max_len] or "candidate").lower()


def render_markdown_report(
    *,
    session_id: str,
    timestamp: datetime,
    state: CapturedState,
    evaluation: dict[str, Any],
    transcript: str,
) -> str:
    """Render a rich markdown evaluation combining captured state + LLM summary."""
    name = state.candidate_name or evaluation.get("candidate_name") or "Unknown"
    role = state.selected_role or evaluation.get("selected_role") or "Unknown"
    score = evaluation.get("score", "n/a")
    rec = evaluation.get("recommendation", "n/a")
    date_str = timestamp.strftime("%B %d, %Y %H:%M")

    parts: list[str] = []
    parts.append(f"# Candidate Evaluation — {name}")
    parts.append("")
    parts.append(
        "<img src=\"../res/operation_settle_mars_transparent.png\" "
        "width=\"240\" alt=\"Operation Settle Mars\">"
    )
    parts.append("")
    parts.append("## Executive summary")
    parts.append(str(evaluation.get("summary") or "(not generated)"))
    parts.append("")

    parts.append("| Field | Value |")
    parts.append("| --- | --- |")
    parts.append(f"| Candidate | {name} |")
    parts.append(f"| Role | {role} |")
    parts.append(f"| Recommendation | **{rec}** |")
    parts.append(f"| Score | {score}/10 |")
    parts.append(f"| Interview date | {date_str} |")
    parts.append(f"| Session ID | `{session_id}` |")
    parts.append("")

    if state.selection_notes:
        parts.append("## Candidate's stated interests")
        parts.append(state.selection_notes)
        parts.append("")

    parts.append("## Strengths")
    parts.append(str(evaluation.get("strengths") or "(none noted)"))
    parts.append("")

    parts.append("## Areas for improvement")
    parts.append(str(evaluation.get("areas_for_improvement") or "(none noted)"))
    parts.append("")

    parts.append("## Motivation assessment")
    parts.append(str(evaluation.get("motivation_assessment") or "(not provided)"))
    parts.append("")

    parts.append("## Per-question evaluation")
    if state.answers:
        for i, ans in enumerate(state.answers, start=1):
            parts.append(f"### Q{i} — {ans.topic}")
            if ans.question_asked:
                parts.append(f"**Question asked:** {ans.question_asked}")
                parts.append("")
            if ans.response_summary:
                parts.append(f"**Response summary:** {ans.response_summary}")
                parts.append("")
            if ans.strengths:
                parts.append(f"**Strengths:** {ans.strengths}")
                parts.append("")
            if ans.concerns:
                parts.append(f"**Concerns:** {ans.concerns}")
                parts.append("")
    else:
        parts.append(
            "_No per-question evaluation was produced — either the call "
            "ended before the interview questions, or the evaluation LLM "
            "did not return `answers`. See the transcript below._"
        )
        parts.append("")

    parts.append("## Full transcript")
    parts.append("```")
    parts.append(transcript or "(empty)")
    parts.append("```")

    return "\n".join(parts)


def save_report(
    session_id: str,
    timestamp: datetime,
    markdown: str,
) -> Path | None:
    """Write the markdown to ./reports/interview_<date>_<time>_<sessionId>.md."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = timestamp.strftime("%Y-%m-%d_%H%M%S")
    sid = _slug(session_id)
    path = REPORTS_DIR / f"interview_{stamp}_{sid}.md"
    try:
        path.write_text(markdown, encoding="utf-8")
        logger.info("saved interview report to %s", path)
        return path
    except Exception:
        logger.exception("failed to save interview report")
        return None


def _empty_evaluation(reason: str) -> dict[str, Any]:
    return {
        "candidate_name": "Unknown",
        "selected_role": "Unknown",
        "selection_notes": "",
        "score": "",
        "recommendation": "",
        "strengths": "",
        "areas_for_improvement": "",
        "motivation_assessment": "",
        "summary": reason or "Evaluation could not be generated.",
        "answers": [],
    }


# ---------------------------------------------------------------------------
# Public orchestrator — called from the agent's shutdown callback
# ---------------------------------------------------------------------------


async def generate_and_save(session: AgentSession, session_id: str) -> None:
    """Full post-call pipeline: transcript → eval → markdown → disk → Sheets."""
    logger.info("session '%s' closed — running post-interview evaluation", session_id)
    transcript = build_transcript(session)

    if not transcript:
        logger.warning("no transcript captured — nothing to evaluate")
        return

    evaluation, error_reason = await evaluate_candidate(transcript)
    if not evaluation:
        evaluation = _empty_evaluation(error_reason)

    state = state_from_evaluation(evaluation)
    timestamp = datetime.now()
    markdown = render_markdown_report(
        session_id=session_id,
        timestamp=timestamp,
        state=state,
        evaluation=evaluation,
        transcript=transcript,
    )
    save_report(session_id, timestamp, markdown)

    if sheets_service.worksheet is not None:
        sheet_row = dict(evaluation)
        sheet_row["conversation_transcript"] = transcript
        await sheets_service.save_interview_result(session_id, sheet_row)
    else:
        logger.info("Google Sheets not initialized — report saved locally only")
