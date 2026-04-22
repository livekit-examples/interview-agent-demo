"""
Mars Recruitment Services — Settler Screening Agent.

A prompt-driven voice agent that screens candidates for open Mars settler
roles. During the call, the agent uses three lightweight function tools
to capture canonical state: candidate name, confirmed role, and per-question
evaluation notes. After the call ends, the full transcript is sent to an
LLM once more to produce an executive summary, a recommendation, and a
score. Everything is rendered into a rich markdown report under ./reports/
and also appended as a row to a Google Sheet.

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
from openai import AsyncOpenAI

from livekit import api
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    JobProcess,
    RunContext,
    SessionUsageUpdatedEvent,
    TurnHandlingOptions,
    WorkerOptions,
    cli,
    llm,
    room_io,
    text_transforms,
)
from livekit.agents.llm import function_tool
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

TOOL USE (internal — these capture notes for the selection board, never
speak the tool names or their contents aloud):
- As soon as the candidate states their first name, call
  `record_candidate_name`.
- As soon as the candidate confirms which mission role they want to be
  considered for, call `record_selected_role` with the exact role title
  and a short note summarizing their stated interests.
- After each of the two interview questions — once you've gathered
  enough to evaluate the answer — call `record_interview_answer` with
  the topic label, the exact question you asked, a neutral 1–2 sentence
  summary of their response, specific strengths, and specific concerns.
  These notes are not shared with the candidate.

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
  "candidate_name": "<first name extracted from transcript, or 'Unknown'>",
  "selected_role": "<the mission role they chose, as stated in the transcript, or 'Unknown'>",
  "score": <integer 1-10 reflecting overall suitability>,
  "recommendation": "<one of: Proceed | On Hold | Reject>",
  "strengths": "<2-4 sentence summary of the candidate's strongest points, citing specifics>",
  "areas_for_improvement": "<2-4 sentence summary of gaps, concerns, or red flags>",
  "motivation_assessment": "<2-3 sentences on the candidate's motivation and commitment to a multi-year Mars deployment>",
  "summary": "<3-5 sentence overall assessment suitable for the mission selection board>"
}}}}

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


class InterviewAssistant(Agent):
    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        prompt = INTERVIEW_PROMPT_TEMPLATE.format(
            job_catalog=format_jobs_for_prompt(jobs)
        )
        super().__init__(instructions=prompt)
        self.state = CapturedState()

    @function_tool()
    async def record_candidate_name(self, context: RunContext, name: str) -> str:
        """Record the candidate's first name once they've said it aloud.

        Args:
            name: The candidate's first name as they introduced themselves.
        """
        name = (name or "").strip()
        if not name:
            return "No name captured."
        self.state.candidate_name = name
        logger.info("tool record_candidate_name: %s", name)
        return "Noted."

    @function_tool()
    async def record_selected_role(
        self,
        context: RunContext,
        title: str,
        interests_notes: str = "",
    ) -> str:
        """Record the mission role the candidate has confirmed interest in.

        Args:
            title: The role title, ideally as listed in the mission catalog.
            interests_notes: Short summary of what they said about their
                interests, background, or preferences relevant to the role.
        """
        title = (title or "").strip()
        if not title:
            return "No role captured."
        self.state.selected_role = title
        self.state.selection_notes = (interests_notes or "").strip()
        logger.info("tool record_selected_role: %s", title)
        return "Noted."

    @function_tool()
    async def record_interview_answer(
        self,
        context: RunContext,
        topic: str,
        question_asked: str,
        response_summary: str,
        strengths: str = "",
        concerns: str = "",
    ) -> str:
        """Record an internal evaluation of one interview answer.

        Call this once after each of the two interview questions, when
        you have enough signal to evaluate the answer. The notes feed
        directly into the mission selection board's report — be
        specific and evidence-based. Never speak these notes aloud.

        Args:
            topic: Short topic label, e.g. "Background" or "Motivation".
            question_asked: The exact question you ended up asking,
                tailored to the role.
            response_summary: 1–2 sentence neutral summary of what the
                candidate said in response.
            strengths: Specific strengths, citing what they said.
                Empty string if nothing particularly strong.
            concerns: Specific gaps, red flags, or weak reasoning.
                Empty string if no concerns.
        """
        entry = CapturedAnswer(
            topic=(topic or "").strip() or "Untitled",
            question_asked=(question_asked or "").strip(),
            response_summary=(response_summary or "").strip(),
            strengths=(strengths or "").strip(),
            concerns=(concerns or "").strip(),
        )
        self.state.answers.append(entry)
        logger.info("tool record_interview_answer: topic=%s", entry.topic)
        return "Noted."


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


async def _evaluate_candidate(transcript: str) -> dict[str, Any]:
    if not transcript.strip():
        logger.warning("empty transcript — skipping evaluation")
        return {}

    prompt = EVALUATION_PROMPT_TEMPLATE.format(transcript=transcript)
    logger.info("sending transcript to LLM for evaluation (%d chars)", len(transcript))

    try:
        client = AsyncOpenAI(api_key=SIMPLISMART_API_KEY, base_url=LLM_URL)
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        result_text = response.choices[0].message.content or ""
    except Exception as e:
        logger.error("evaluation LLM call failed: %s", e, exc_info=True)
        return {}

    cleaned = re.sub(r"```(?:json)?\s*", "", result_text).strip()
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not json_match:
        logger.error("no JSON object found in evaluation response: %s", result_text[:500])
        return {}

    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError as e:
        logger.error("failed to parse evaluation JSON: %s", e)
        return {}


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
            "_No per-question notes were captured via `record_interview_answer` "
            "during this call. See transcript below for the raw exchange._"
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
    agent: InterviewAssistant,
    session_id: str,
) -> None:
    logger.info("session '%s' closed — running post-interview evaluation", session_id)
    transcript = _build_transcript(session)
    state = agent.state

    if not transcript:
        logger.warning("no transcript captured — nothing to evaluate")
        return

    evaluation = await _evaluate_candidate(transcript)
    if not evaluation:
        evaluation = {
            "candidate_name": state.candidate_name or "Unknown",
            "selected_role": state.selected_role or "Unknown",
            "score": "",
            "recommendation": "",
            "strengths": "",
            "areas_for_improvement": "",
            "motivation_assessment": "",
            "summary": "Evaluation could not be generated.",
        }

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
        # Prefer tool-captured name/role for the sheet too.
        if state.candidate_name:
            sheet_row["candidate_name"] = state.candidate_name
        if state.selected_role:
            sheet_row["selected_role"] = state.selected_role
        sheet_row["conversation_transcript"] = transcript
        await sheets_service.save_interview_result(session_id, sheet_row)
    else:
        logger.info("Google Sheets not initialized — report saved locally only")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
server = AgentServer()

def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


# @server.rtc_session()
@server.rtc_session(agent_name="mars-recruiter")
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
            interruption={"mode": "adaptive"},
        ),
        tts_text_transforms=[
            "filter_emoji",
            "filter_markdown",
            text_transforms.replace({"LiveKit": "<<ˈ|l|aɪ|v>> <<ˈ|k|ɪ|t>>"}),
        ],
    )

    @session.on("session_usage_updated")
    def _on_session_usage_updated(ev: SessionUsageUpdatedEvent) -> None:
        for model_usage in ev.usage.model_usage:
            logger.info("usage: %s", model_usage)

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
        await on_session_close(session, agent, session_id)

    ctx.add_shutdown_callback(_on_shutdown)

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=room_io.RoomInputOptions(),
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
