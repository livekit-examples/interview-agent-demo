"""
Mars Recruitment Services — Settler Screening Agent (simplified).

A single prompt-driven voice agent that screens candidates for open Mars
settler roles. No function tools, no AgentTasks. After the call ends, the
full transcript is sent to an LLM once to produce a structured JSON
evaluation and a short markdown report, saved under ./output/ and
./reports/.

Model stack is env-driven (per-service URLs + model names) so you can swap
STT, LLM, or TTS independently via .env.local without touching this file.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from livekit import api
from livekit.agents import (
    Agent,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    JobProcess,
    SessionUsageUpdatedEvent,
    TurnHandlingOptions,
    WorkerOptions,
    cli,
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
# Agent
# ---------------------------------------------------------------------------


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


async def on_session_close(session: AgentSession, session_id: str) -> None:
    logger.info("session '%s' closed — running post-interview evaluation", session_id)
    transcript = _build_transcript(session)
    if not transcript:
        logger.warning("no transcript captured — nothing to evaluate")
        return

    evaluation = await _evaluate_candidate(transcript)
    if not evaluation:
        evaluation = {
            "candidate_name": "Unknown",
            "selected_role": "Unknown",
            "score": "",
            "recommendation": "",
            "strengths": "",
            "areas_for_improvement": "",
            "motivation_assessment": "",
            "summary": "Evaluation could not be generated.",
        }

    if sheets_service.worksheet is None:
        logger.warning(
            "Google Sheets not initialized — dropping evaluation (no local save configured)"
        )
        return

    evaluation["conversation_transcript"] = transcript
    await sheets_service.save_interview_result(session_id, evaluation)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


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

    async def _on_shutdown() -> None:
        await on_session_close(session, session_id)

    ctx.add_shutdown_callback(_on_shutdown)

    await session.start(
        room=ctx.room,
        agent=InterviewAssistant(jobs),
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

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
