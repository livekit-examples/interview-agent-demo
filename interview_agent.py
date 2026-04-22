"""
Mars Recruitment Services — Settler Screening Agent.

A prompt-driven voice agent that screens candidates for open Mars settler
roles. No mid-call function tools — small open-source LLMs (Llama 3.1 8B,
Gemma 3 4B) tend to crash their serving proxies on tool-call continuation
turns, so all structured capture happens post-call. After the call ends
the full transcript is sent to the LLM once to produce a rich JSON
evaluation: executive summary, recommendation, score, per-question notes,
and strengths/concerns. That evaluation is rendered into a markdown
report under ./reports/ and appended as a row to a Google Sheet.

Model stack is env-driven (per-service URLs + model names) so you can swap
STT, LLM, or TTS independently via .env.local without touching this file.
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

from dotenv import load_dotenv

from livekit import api
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
    inference,
    llm,
    room_io,
    text_transforms,
)
from livekit.plugins import openai, silero, simplismart
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from google_sheets import sheets_service

logger = logging.getLogger("mars-recruiter")
logging.basicConfig(level=logging.INFO)

HERE = Path(__file__).resolve().parent
load_dotenv()
load_dotenv(HERE / ".env.local", override=True)

JOBS_FILE = HERE / "mars_jobs.json"
REPORTS_DIR = HERE / "reports"
COMPANY = "Mars Recruitment Services"
INTERVIEWER_NAME = "Nova"


# ---------------------------------------------------------------------------
# Model stack — per-service env vars. Override any one in .env.local to
# swap just that model.
# ---------------------------------------------------------------------------

SIMPLISMART_API_KEY = os.environ.get("SIMPLISMART_API_KEY")

STT_URL = os.environ.get("STT_URL", "https://api.simplismart.live/predict")
STT_MODEL = os.environ.get("STT_MODEL", "openai/whisper-large-v3-turbo")

LLM_URL = os.environ.get("LLM_URL", "https://api.simplismart.live")
LLM_MODEL = os.environ.get("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")

TTS_URL = os.environ.get("TTS_URL", "https://api.simplismart.live/tts")
TTS_MODEL = os.environ.get("TTS_MODEL", "canopylabs/orpheus-3b-0.1-ft")
TTS_VOICE = os.environ.get("TTS_VOICE", "") or None

# Post-call evaluation runs through LiveKit Inference (no Simplismart
# dependency) so the structured JSON + markdown report doesn't depend on
# whatever model drives the conversation.
EVALUATION_LLM_MODEL = os.environ.get("EVALUATION_LLM_MODEL", "openai/gpt-4.1-mini")


def build_stt_llm_tts():
    """Return (stt, llm, tts) instances for the configured endpoints.

    TTS plugin is chosen by URL path:
      * /audio/speech → openai.TTS (OpenAI-compatible)
      * otherwise     → simplismart.TTS (native /tts)
    """
    if not SIMPLISMART_API_KEY:
        logger.warning("SIMPLISMART_API_KEY not set — model calls will fail.")

    logger.info(
        "Model stack — STT: %s (%s), LLM: %s (%s), TTS: %s (%s, voice=%s)",
        STT_URL, STT_MODEL, LLM_URL, LLM_MODEL, TTS_URL, TTS_MODEL, TTS_VOICE,
    )

    stt_inst = simplismart.STT(
        base_url=STT_URL,
        api_key=SIMPLISMART_API_KEY,
        model=STT_MODEL,
    )

    llm_inst = openai.LLM(
        model=LLM_MODEL,
        api_key=SIMPLISMART_API_KEY,
        base_url=LLM_URL,
    )

    tts_kwargs: dict[str, Any] = dict(
        model=TTS_MODEL,
        api_key=SIMPLISMART_API_KEY,
        base_url=TTS_URL,
    )
    if TTS_VOICE:
        tts_kwargs["voice"] = TTS_VOICE
    tts_inst = simplismart.TTS(**tts_kwargs)

    return stt_inst, llm_inst, tts_inst


# ---------------------------------------------------------------------------
# Job catalog
# ---------------------------------------------------------------------------


def _strip_html(html: str | None) -> str:
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    return re.sub(r"\s+", " ", text).strip()


def load_jobs() -> list[dict[str, Any]]:
    """Load mission roles from mars_jobs.json, lightly flattened."""
    if not JOBS_FILE.exists():
        logger.warning("mission catalog not found at %s", JOBS_FILE)
        return []

    with JOBS_FILE.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    out = []
    for j in payload.get("jobs", []):
        plain = j.get("description_plain") or _strip_html(j.get("description_html"))
        plain = re.sub(r"\s+", " ", plain).strip()
        if len(plain) > 300:
            plain = plain[:300].rsplit(" ", 1)[0] + "…"
        out.append(
            {
                "title": (j.get("title") or "").strip(),
                "department": (j.get("department") or "").strip(),
                "team": (j.get("team") or "").strip(),
                "location": (j.get("location") or "").strip(),
                "summary": plain,
            }
        )
    return out


def format_jobs_for_prompt(jobs: list[dict[str, Any]]) -> str:
    lines = []
    for j in jobs:
        lines.append(
            f"- {j['title']} ({j['department']} / {j['team']}, {j['location']}): {j['summary']}"
        )
    return "\n".join(lines) if lines else "(no roles available)"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


INTERVIEW_PROMPT_TEMPLATE = f"""
You are {INTERVIEWER_NAME}, a senior recruiter at {COMPANY}. You conduct
first-round phone screenings for candidates applying to live and work on
Mars. Stay warm, professional, and curious — never harsh, never
sycophantic. Speak naturally: no markdown, no bullets, no emojis, no
asterisks. Keep turns short so the call stays conversational.

INTERVIEW FLOW (one step at a time, wait for the candidate each time):

1. Greet the candidate, introduce yourself as {INTERVIEWER_NAME} from
   {COMPANY}, and ask for their first name so you can personalize the
   call.

2. Briefly tell them what to expect: you'll confirm which mission role
   they want to be considered for, then walk through two short interview
   questions, and the mission selection board will follow up after.

3. Ask what kind of Mars work excites them — background, area of
   interest, location preference (Olympus Base, Utopia Planitia, Jezero
   Crater, Gale Science Station, cis-lunar comms post). Based on what
   they say, describe two or three relevant roles from the catalog below
   in plain spoken style — mention title, team, location, and a
   one-sentence sense of the work. Ask which one they'd like to be
   considered for and confirm it.

4. Ask interview question 1 — Background: invite them to walk through
   their background and the experience most relevant to performing this
   specific role on Mars. Ask one concise follow-up if the answer is
   very short or avoids the question.

5. Ask interview question 2 — Motivation for Mars: why do they want to
   live and work on Mars, and why this role specifically? Probe the
   seriousness of commitment (multi-year deployment, limited return
   windows, family impact). One follow-up if needed.

6. Thank them warmly, let them know the mission selection board at
   {COMPANY} will review the screening and follow up within a week.
   Wish them well.

GUIDELINES:
- Never share scores, evaluations, or hiring signals during the call —
  those are internal to the selection board.
- Acknowledge each answer briefly (e.g. "Thanks, that helps") before
  moving on.
- If the candidate clearly asks to end the call early, thank them, wrap
  up warmly, and stop asking further questions.

MISSION ROLE CATALOG:
{{job_catalog}}
""".strip()


EVALUATION_PROMPT_TEMPLATE = f"""
You are an expert recruiter at {COMPANY}. Below is the full transcript of
a first-round phone screening interview for a Mars settler role.
Evaluate the candidate strictly based on what was said in the transcript.

TRANSCRIPT:
{{transcript}}

Produce a JSON object with exactly these keys (no extra keys, no
markdown code fences):
{{{{
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
    {{{{
      "topic": "<short topic label, e.g. 'Background' or 'Motivation for Mars'>",
      "question_asked": "<the exact question the interviewer asked>",
      "response_summary": "<1-2 sentence neutral summary of the candidate's answer>",
      "strengths": "<specific strengths cited from the answer, or empty string>",
      "concerns": "<specific concerns or gaps, or empty string>"
    }}}}
  ]
}}}}

The "answers" array should contain one entry per interview question
actually asked and answered in the transcript (typically two).

SCORING RUBRIC:
8-10 → Strong fit, proceed to next round
5-7  → Mixed signals, on hold / needs further assessment
1-4  → Poor fit or insufficient information, reject

Return ONLY the raw JSON object. Do not include any explanation outside
the JSON.
""".strip()


# ---------------------------------------------------------------------------
# Captured state + Agent
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


class InterviewAssistant(Agent):
    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        prompt = INTERVIEW_PROMPT_TEMPLATE.format(
            job_catalog=format_jobs_for_prompt(jobs)
        )
        super().__init__(instructions=prompt)


# ---------------------------------------------------------------------------
# Post-session evaluation
# ---------------------------------------------------------------------------


def _build_transcript(session: AgentSession) -> str:
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


async def _evaluate_candidate(transcript: str) -> tuple[dict[str, Any], str]:
    """Run the evaluation LLM. Returns (evaluation_dict, error_reason).

    evaluation_dict is empty on failure. error_reason is "" on success,
    or a short human-readable string describing why evaluation failed —
    useful for surfacing the cause in the fallback report.
    """
    if not transcript.strip():
        logger.warning("empty transcript — skipping evaluation")
        return {}, "Empty transcript — the session ended before anything was said."

    # Short transcripts aren't worth a full evaluation call — the LLM will
    # just return empty "Unknown" fields anyway. Short-circuit with a clear
    # note explaining what happened.
    turn_count = sum(1 for line in transcript.splitlines() if line.strip())
    if len(transcript) < 200 or turn_count < 3:
        msg = (
            f"Transcript too short to evaluate "
            f"({len(transcript)} chars, {turn_count} turn(s)). "
            "The candidate likely disconnected before the screening began."
        )
        logger.warning(msg)
        return {}, msg

    prompt = EVALUATION_PROMPT_TEMPLATE.format(transcript=transcript)
    logger.info(
        "sending transcript to LiveKit Inference '%s' for evaluation (%d chars)",
        EVALUATION_LLM_MODEL, len(transcript),
    )

    try:
        eval_llm = inference.LLM(EVALUATION_LLM_MODEL)
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="user", content=prompt)
        chunks: list[str] = []
        async for chunk in eval_llm.chat(chat_ctx=chat_ctx):
            if chunk.delta and chunk.delta.content:
                chunks.append(chunk.delta.content)
        result_text = "".join(chunks).strip()
    except Exception as e:
        logger.error("evaluation LLM call failed: %s", e, exc_info=True)
        return {}, f"Evaluation LLM call raised {type(e).__name__}: {e}"

    if not result_text:
        logger.error("evaluation LLM returned empty response")
        return {}, "Evaluation LLM returned empty response."

    cleaned = re.sub(r"```(?:json)?\s*", "", result_text).strip()
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not json_match:
        logger.error("no JSON object found in evaluation response: %s", result_text[:500])
        return {}, "Evaluation LLM response did not contain JSON."

    try:
        return json.loads(json_match.group()), ""
    except json.JSONDecodeError as e:
        logger.error("failed to parse evaluation JSON: %s; raw: %s", e, json_match.group()[:500])
        return {}, f"Evaluation JSON failed to parse: {e}"


def _slug(s: str, max_len: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", s or "").strip("-")
    return (s[:max_len] or "candidate").lower()


def _render_markdown_report(
    *,
    session_id: str,
    timestamp: datetime,
    state: CapturedState,
    evaluation: dict[str, Any],
    transcript: str,
) -> str:
    """Render a rich markdown evaluation combining captured state + LLM summary."""
    # Prefer tool-captured values over LLM-extracted ones.
    name = state.candidate_name or evaluation.get("candidate_name") or "Unknown"
    role = state.selected_role or evaluation.get("selected_role") or "Unknown"
    score = evaluation.get("score", "n/a")
    rec = evaluation.get("recommendation", "n/a")
    date_str = timestamp.strftime("%B %d, %Y %H:%M")

    parts: list[str] = []
    parts.append(f"# Candidate Evaluation — {name}")
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
    """Write the evaluation markdown to ./reports/interview_<date>_<time>_<sessionId>.md."""
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


async def on_session_close(
    session: AgentSession,
    session_id: str,
) -> None:
    logger.info("session '%s' closed — running post-interview evaluation", session_id)
    transcript = _build_transcript(session)

    if not transcript:
        logger.warning("no transcript captured — nothing to evaluate")
        return

    evaluation, error_reason = await _evaluate_candidate(transcript)
    if not evaluation:
        evaluation = {
            "candidate_name": "Unknown",
            "selected_role": "Unknown",
            "selection_notes": "",
            "score": "",
            "recommendation": "",
            "strengths": "",
            "areas_for_improvement": "",
            "motivation_assessment": "",
            "summary": error_reason or "Evaluation could not be generated.",
            "answers": [],
        }

    state = state_from_evaluation(evaluation)

    timestamp = datetime.now()
    markdown = _render_markdown_report(
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


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
# shutdown_process_timeout default is 10s — not enough for the post-call
# LLM evaluation + Sheets write. Give it a full minute.
server = AgentServer(shutdown_process_timeout=60.0)

def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


@server.rtc_session()
#@server.rtc_session(agent_name="mars-recruiter")
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("starting agent in room %s", ctx.room.name)

    # Initialize Google Sheets. If it fails, sheets saving is skipped — the
    # local JSON/markdown files are always written.
    try:
        sheets_service.initialize()
    except Exception as e:
        logger.warning(
            "Google Sheets not initialized — results will only be saved locally: %s", e
        )

    jobs = load_jobs()
    stt_inst, llm_inst, tts_inst = build_stt_llm_tts()

    session = AgentSession(
        stt=stt_inst,
        llm=llm_inst,
        tts=tts_inst,
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
        ),
        tts_text_transforms=[
            "filter_emoji",
            "filter_markdown",
            text_transforms.replace({"LiveKit": "<<ˈ|l|aɪ|v>> <<ˈ|k|ɪ|t>>"}),
        ],
    )

    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev: ConversationItemAddedEvent) -> None:
        if not isinstance(ev.item, llm.ChatMessage):
            return
        if ev.item.role != "assistant":
            return
        m = ev.item.metrics or {}
        e2e = m.get("e2e_latency")
        if e2e is not None:
            logger.info(
                "assistant turn — ttft=%.3fs ttfb=%.3fs e2e=%.3fs",
                m.get("llm_node_ttft") or 0.0,
                m.get("tts_node_ttfb") or 0.0,
                e2e,
            )

    session_id = ctx.job.id or ctx.room.name or "unknown"
    agent = InterviewAssistant(jobs)

    async def _on_shutdown() -> None:
        for model_usage in session.usage.model_usage:
            logger.info("usage: %s", model_usage)
        await on_session_close(session, session_id)

    ctx.add_shutdown_callback(_on_shutdown)

    await session.start(
        room=ctx.room,
        agent=agent,
        room_options=room_io.RoomOptions(),
    )

    await session.generate_reply(
        instructions=(
            f"Greet the candidate warmly, introduce yourself as {INTERVIEWER_NAME} "
            f"from {COMPANY}, and ask for their first name."
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
server.setup_fnc = prewarm

if __name__ == "__main__":
    LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY")
    LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET")
    LIVEKIT_URL = os.environ.get("LIVEKIT_URL")

    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        print("Missing LIVEKIT_API_KEY or LIVEKIT_API_SECRET in .env file")
        print("Get these from your LiveKit Cloud dashboard: https://cloud.livekit.io/")
    else:
        token = (
            api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity("test_user")
            .with_grants(api.VideoGrants(room_join=True, room="test_room"))
        )
        jwt_token = token.to_jwt()

        print("\n\nLiveKit Interview Agent Ready!\033[0m\n\033[94m" + "=" * 50 + "\033[0m")
        print(f"\033[94mConnect at: https://agents-playground.livekit.io/\033[0m\n\033[94m")
        print(f"\033[94mURL: {LIVEKIT_URL}\033[0m")
        print(f"\033[94mToken: {jwt_token}\033[0m\n" + "=" * 50 + "\033[0m")

    cli.run_app(server)
