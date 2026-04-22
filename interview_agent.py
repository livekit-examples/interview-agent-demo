"""
Mars Recruitment Services — Settler Screening Agent.

A voice agent that conducts a first-round phone screening for candidates
applying to live and work on Mars. The agent picks up where a glossy
careers page leaves off: it learns what the candidate wants, pairs them
with a real opening from the mission catalog, asks two structured
interview questions, captures evaluation notes on each answer, and writes
a written evaluation report for the mission selection board.

Persona:
    Nova, a senior recruiter at Mars Recruitment Services.
    Warm, professional, curious. Asks hard questions without being harsh.
    Does NOT share scores or evaluation notes with the candidate during the
    call — that's internal. At the end, thanks the candidate and tells
    them the selection board will follow up.

Flow:
    IntroTask          — greet, collect candidate name, frame the call
    JobSelectionTask   — gather interests, recommend matches, confirm pick
    2 x InterviewQuestionTask — background & fit; motivation for Mars
                                 & commitment to the role
    Report generation  — LLM-written evaluation saved to ./reports/; structured
                         results saved to ./output/<date>_<time>_<session_id>.json
    Wrap-up            — thank candidate, hang up
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AgentTask,
    ConversationItemAddedEvent,
    JobContext,
    JobProcess,
    RunContext,
    SessionUsageUpdatedEvent,
    TurnHandlingOptions,
    cli,
    inference,
    llm,
    room_io,
    text_transforms,

)
from livekit.agents.beta.workflows import TaskGroup
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero, simplismart
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit import api

logger = logging.getLogger("mars-recruiter")

HERE = Path(__file__).resolve().parent
# Load default env first, then allow local overrides.
load_dotenv()
load_dotenv(HERE / ".env.local", override=True)

JOBS_FILE = HERE / "mars_jobs.json"
REPORTS_DIR = HERE / "reports"
OUTPUT_DIR = HERE / "output"
COMPANY = "Mars Recruitment Services"
INTERVIEWER_NAME = "Nova"


# ---------------------------------------------------------------------------
# Model stack toggle — LiveKit inference (default) or Simplismart
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# Per-service endpoints + models. Defaults point at the original
# api.simplismart.live stack; override any one via env to swap just that model.
STT_URL = os.environ.get("STT_URL", "https://api.simplismart.live/predict")
STT_MODEL = os.environ.get("STT_MODEL", "openai/whisper-large-v3-turbo")

LLM_URL = os.environ.get("LLM_URL", "https://api.simplismart.live")
LLM_MODEL = os.environ.get("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")

TTS_URL = os.environ.get("TTS_URL", "https://api.simplismart.live/tts")
TTS_MODEL = os.environ.get("TTS_MODEL", "canopylabs/orpheus-3b-0.1-ft")
TTS_VOICE = os.environ.get("TTS_VOICE", "") or None


def build_stt_llm_tts():
    """Return (stt, llm, tts) instances using the configured endpoints.

    TTS plugin is chosen by URL path:
      * /v1/audio/speech (or …/audio/speech) → openai.TTS (OpenAI-compatible)
      * anything else                        → simplismart.TTS (native /tts)
    This lets you swap the TTS endpoint independently of STT/LLM.
    """
    api_key = os.environ.get("SIMPLISMART_API_KEY")
    if not api_key:
        logger.warning(
            "SIMPLISMART_API_KEY not set — STT, LLM, and TTS calls will fail."
        )
    logger.info(
        "Model stack — STT: %s (%s), LLM: %s (%s), TTS: %s (%s, voice=%s)",
        STT_URL, STT_MODEL, LLM_URL, LLM_MODEL, TTS_URL, TTS_MODEL, TTS_VOICE,
    )

    #stt_inst=inference.STT("deepgram/nova-3", language="multi")
    stt_inst = simplismart.STT(
        base_url=STT_URL,
        api_key=api_key,
        model=STT_MODEL,
    )

    #llm_inst=inference.LLM("openai/gpt-4.1-mini")
    llm_inst = openai.LLM(
        model=LLM_MODEL,
        api_key=api_key,
        base_url=LLM_URL,
    )

    #tts_inst=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")
    tts_inst = simplismart.TTS(
        base_url=TTS_URL,
        api_key=api_key,
        model="canopylabs/orpheus-3b-0.1-ft"
    )

    return stt_inst, llm_inst, tts_inst



# ---------------------------------------------------------------------------
# Job catalog loading
# ---------------------------------------------------------------------------


def _strip_html(html: str | None) -> str:
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _compact_description(job: dict[str, Any], max_chars: int = 500) -> str:
    """Pull a short plain-text blurb out of the job record."""
    plain = job.get("description_plain") or _strip_html(job.get("description_html"))
    plain = re.sub(r"\s+", " ", plain).strip()
    if len(plain) > max_chars:
        plain = plain[:max_chars].rsplit(" ", 1)[0] + "…"
    return plain


def load_jobs() -> list[dict[str, Any]]:
    """Load and lightly flatten the mission role catalog.

    Reads mars_jobs.json. If the file is missing, logs a warning and returns
    an empty list — the agent will detect the empty catalog on startup and
    politely end the call rather than crash.
    """
    if not JOBS_FILE.exists():
        logger.warning(
            "mission catalog not found at %s — cannot run the interview",
            JOBS_FILE,
        )
        return []

    with JOBS_FILE.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    jobs = []
    for j in payload.get("jobs", []):
        comp = j.get("compensation") or {}
        min_salary, max_salary, currency = _extract_salary(comp)
        jobs.append(
            {
                "id": j.get("id"),
                "title": (j.get("title") or "").strip(),
                "department": (j.get("department") or "").strip(),
                "team": (j.get("team") or "").strip(),
                "location": (j.get("location") or "").strip(),
                "is_remote": bool(j.get("is_remote")),
                "employment_type": j.get("employment_type") or "",
                "compensation": comp.get("compensationTierSummary") or "",
                "min_salary": min_salary,
                "max_salary": max_salary,
                "currency": currency,
                "job_url": j.get("job_url") or "",
                "application_url": j.get("application_url") or "",
                "summary": _compact_description(j, max_chars=400),
            }
        )
    return jobs


def _extract_salary(comp: dict[str, Any]) -> tuple[int | None, int | None, str | None]:
    """Pull the first salary component (annual) from the Ashby compensation blob.

    Returns (min, max, currency_code) or (None, None, None) if unavailable.
    """
    for component in comp.get("summaryComponents") or []:
        if component.get("compensationType") == "Salary":
            return (
                component.get("minValue"),
                component.get("maxValue"),
                component.get("currencyCode"),
            )
    return (None, None, None)


def format_job_catalog_for_llm(jobs: list[dict[str, Any]]) -> str:
    """Produce a compact, deterministic menu the LLM can reason over."""
    lines = []
    for j in jobs:
        lines.append(
            f"- id={j['id']} | {j['title']} | {j['department']} / {j['team']} "
            f"| {j['location']}{' (remote)' if j['is_remote'] else ''}\n"
            f"  Summary: {j['summary']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared session state
# ---------------------------------------------------------------------------


@dataclass
class SelectedJob:
    id: str
    title: str
    department: str
    team: str
    location: str
    summary: str
    compensation: str
    job_url: str


@dataclass
class QuestionResult:
    topic: str
    question: str
    response_summary: str
    strengths: str
    concerns: str
    score: int  # 1-10, the interviewer's evaluation on this question
    # False for placeholder rows when the candidate ends before a question is scored.
    include_in_scoring: bool = True


@dataclass
class Userdata:
    jobs: list[dict[str, Any]]
    candidate_name: str = ""
    interests_notes: str = ""
    selected_job: SelectedJob | None = None
    question_results: list[QuestionResult] = field(default_factory=list)
    # Set in rtc entrypoint from JobContext (used for output filenames and JSON).
    session_id: str = ""
    room_name: str = ""
    room_sid: str = ""
    candidate_requested_early_end: bool = False
    # Filled by ensure_minimum_report_state when we had to invent data for filing.
    report_used_fallback_role: bool = False
    report_used_placeholder_scoring: bool = False


# ---------------------------------------------------------------------------
# Task result types
# ---------------------------------------------------------------------------


@dataclass
class IntroResult:
    name: str


@dataclass
class JobSelectionResult:
    job: SelectedJob
    interests_notes: str


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


class IntroTask(AgentTask[IntroResult]):
    """Greet candidate warmly and collect their name."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                f"You are {INTERVIEWER_NAME}, a senior recruiter at {COMPANY}. "
                "The candidate has just joined a voice call for a first-round "
                "screening interview for a Mars settler position. Welcome them "
                "warmly, introduce yourself briefly, and set expectations: you'll "
                "confirm which mission role they want to be considered for, then "
                "walk through two short interview questions, and the mission "
                "selection board will follow up after the call. Ask for their "
                "first name so you can personalize the conversation. Keep it "
                "concise, professional, and human — no bullet points, no "
                "markdown, plain spoken English."
            ),
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                f"Greet the candidate warmly, introduce yourself as "
                f"{INTERVIEWER_NAME} from {COMPANY}, and in one or two "
                "sentences frame the call: we'll confirm which mission role "
                "they want to be considered for, then run two short interview "
                "questions. Ask for their first name."
            ),
        )

    @function_tool()
    async def record_name(self, context: RunContext[Userdata], name: str) -> str:
        """Record the candidate's preferred name once they provide it.

        Args:
            name: The candidate's preferred first name.
        """
        name = (name or "").strip()
        if not name:
            return "I didn't catch a name — please ask again."
        context.userdata.candidate_name = name
        self.complete(IntroResult(name=name))
        return f"Great — recorded name as {name}."

    @function_tool()
    async def test_write_report(self, context: RunContext[Userdata]) -> str:
        """Testing only: call when the user says the phrase 'test write report'.

        Writes sample report files to reports/ and output/ next to the agent
        script so you can verify the pipeline without finishing an interview.
        """
        return await invoke_test_write_report(context)


class JobSelectionTask(AgentTask[JobSelectionResult]):
    """Ask about interests, recommend matching jobs, confirm a pick."""

    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        self._jobs = jobs
        self._by_id = {j["id"]: j for j in jobs}
        self._by_title = {j["title"].lower(): j for j in jobs}

        # Lightweight at-a-glance map so the LLM knows what's available
        # without burning instruction tokens on full descriptions.
        # Use the facet/filter tools below for details.
        quick_list = "\n".join(
            f"- {j['title']}  ({j['department']} / {j['team']})" for j in jobs
        )

        super().__init__(
            instructions=(
                f"You are {INTERVIEWER_NAME}, a senior recruiter at {COMPANY}. "
                "Your job in this stage is to pair the candidate with one open "
                "mission role to be considered for.\n\n"
                "Conversational flow (spoken, not bullet points):\n"
                "  1. Ask what kind of work excites them on Mars — area, "
                "     background, preferred base, surface work vs. cis-lunar, "
                "     pay expectations, anything relevant.\n"
                "  2. Use the query tools to narrow the catalog efficiently "
                "     rather than guessing: list_departments, list_teams, "
                "     list_locations, list_compensation_ranges, "
                "     summarize_remote, filter_jobs, and get_job_details. "
                "     Combine filters when the candidate gives multiple "
                "     criteria.\n"
                "  3. Describe the two or three most relevant roles in a "
                "     natural spoken style — no ids, no lists, no markdown. "
                "     Mention title, team, posting location (e.g. Olympus "
                "     Base, Jezero Crater), and a one-line sense of the work. "
                "     Mention pay range only if useful.\n"
                "  4. Ask which one they'd like to be considered for. If none "
                "     fit, keep exploring with the tools.\n"
                "  5. As soon as they confirm a role, you MUST call exactly one of: "
                "     confirm_spoken_role (best when they say things like "
                "     'robotics engineer' or 'the hydroponics job' — it resolves "
                "     to the catalog automatically), OR record_selection with the "
                "     exact job id from the tools. Until one of these succeeds, "
                "     this stage is NOT finished — do not run screening questions "
                "     and do not improvise a multi-question interview here.\n\n"
                "Tone: upbeat, encouraging, conversational. Avoid markdown, "
                "asterisks, emojis, and numbered lists in spoken replies.\n\n"
                "Quick list of open roles (titles only — use tools for "
                "details and to narrow down):\n"
                f"{quick_list}"
            ),
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Ask the candidate which mission role they are interested in "
                "and why. Remind them briefly that the catalog spans "
                "engineering, life sciences, operations, and habitability "
                "roles across Olympus Base, Utopia Planitia, Jezero Crater, "
                "Gale Science Station, and one cis-lunar communications post."
            ),
        )

    def _lookup(self, job_id: str | None, job_title: str | None) -> dict[str, Any] | None:
        if job_id and job_id in self._by_id:
            return self._by_id[job_id]
        if job_title and job_title.lower() in self._by_title:
            return self._by_title[job_title.lower()]
        # fuzzy fallback on title
        if job_title:
            needle = job_title.lower()
            for j in self._jobs:
                if needle in j["title"].lower() or j["title"].lower() in needle:
                    return j
        return None

    def _find_best_job_match(self, phrase: str) -> dict[str, Any] | None:
        """Resolve plain-language role phrases (e.g. 'robotics engineer') to a job.

        The catalog titles rarely match spoken phrases exactly; we score token
        overlap on title + summary so 'robotics' matches 'Rover & Robotics Technician'.
        """
        phrase = (phrase or "").strip()
        if not phrase:
            return None
        direct = self._lookup(None, phrase)
        if direct is not None:
            return direct

        tokens = [t for t in re.split(r"\W+", phrase.lower()) if len(t) > 2]
        if not tokens:
            return None

        ranked: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
        for idx, j in enumerate(self._jobs):
            title = (j.get("title") or "").lower()
            summary = (j.get("summary") or "").lower()
            score = 0
            title_hits = 0
            longest_title_match = 0
            for t in tokens:
                if t in title:
                    score += 3
                    title_hits += 1
                    longest_title_match = max(longest_title_match, len(t))
                elif t in summary:
                    score += 1
            if score > 0:
                # Prefer higher score, then more title hits, then stronger title keyword
                # (so "robotics engineer" beats plain "engineer" when "robotics" matches).
                key = (score, title_hits, longest_title_match, -idx)
                ranked.append((key, j))
        if not ranked:
            return None
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked[0][1]

    def _apply_job_selection(
        self,
        context: RunContext[Userdata],
        job: dict[str, Any],
        interests_notes: str,
    ) -> str:
        selected = SelectedJob(
            id=job["id"],
            title=job["title"],
            department=job["department"],
            team=job["team"],
            location=job["location"],
            summary=job["summary"],
            compensation=job["compensation"],
            job_url=job["job_url"],
        )
        context.userdata.selected_job = selected
        context.userdata.interests_notes = (interests_notes or "").strip()
        self.complete(
            JobSelectionResult(
                job=selected, interests_notes=context.userdata.interests_notes
            )
        )
        return f"Recorded selection: {selected.title}."

    # --- internal helpers used by both the facet tools and filter_jobs -----

    def _counted_distinct(self, field: str) -> list[dict[str, Any]]:
        """Return [{value, count}] with distinct values sorted by count desc."""
        counts: dict[str, int] = {}
        for j in self._jobs:
            val = (j.get(field) or "").strip()
            if not val:
                continue
            counts[val] = counts.get(val, 0) + 1
        return [
            {"value": v, "count": c}
            for v, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ]

    def _compact(self, job: dict[str, Any]) -> dict[str, Any]:
        """Shape a single job for tool return values — compact, LLM-friendly."""
        return {
            "id": job["id"],
            "title": job["title"],
            "department": job["department"],
            "team": job["team"],
            "location": job["location"],
            "is_remote": job["is_remote"],
            "compensation": job["compensation"] or None,
            "min_salary": job.get("min_salary"),
            "max_salary": job.get("max_salary"),
            "currency": job.get("currency"),
        }

    @staticmethod
    def _ci_eq(a: str | None, b: str | None) -> bool:
        return (a or "").strip().lower() == (b or "").strip().lower()

    @staticmethod
    def _ci_contains(haystack: str | None, needle: str | None) -> bool:
        if not needle:
            return True
        return (needle or "").strip().lower() in (haystack or "").lower()

    # --- facet tools -------------------------------------------------------

    @function_tool()
    async def list_departments(self) -> list[dict[str, Any]]:
        """List the distinct departments with open roles, and how many roles
        each has. Use this first when the candidate hasn't narrowed their
        interests yet so you can ask an informed follow-up.
        """
        return self._counted_distinct("department")

    @function_tool()
    async def list_teams(
        self, department: str | None = None
    ) -> list[dict[str, Any]]:
        """List the distinct teams with open roles, optionally scoped to a
        department. Returns [{value, count}] sorted by count desc.

        Args:
            department: Optional department to filter teams by (e.g. "R&D",
                "S&M"). Case-insensitive. Omit for all teams.
        """
        counts: dict[str, int] = {}
        for j in self._jobs:
            if department and not self._ci_eq(j.get("department"), department):
                continue
            team = (j.get("team") or "").strip()
            if not team:
                continue
            counts[team] = counts.get(team, 0) + 1
        return [
            {"value": v, "count": c}
            for v, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ]

    @function_tool()
    async def list_locations(self) -> list[dict[str, Any]]:
        """List the distinct locations with open roles and how many roles
        each has. Locations are as Ashby presents them (e.g. "Remote, U.S").
        """
        return self._counted_distinct("location")

    @function_tool()
    async def list_employment_types(self) -> list[dict[str, Any]]:
        """List the distinct employment types (e.g. FullTime) with counts."""
        return self._counted_distinct("employment_type")

    @function_tool()
    async def list_compensation_ranges(self) -> list[dict[str, Any]]:
        """List every role's salary/compensation summary.

        Returns one entry per role so the candidate can compare. Each entry
        is {job_id, title, compensation_summary, min_salary, max_salary,
        currency}. Salary values may be None when Ashby hasn't posted them.
        """
        out = []
        for j in self._jobs:
            out.append(
                {
                    "job_id": j["id"],
                    "title": j["title"],
                    "compensation_summary": j["compensation"] or None,
                    "min_salary": j.get("min_salary"),
                    "max_salary": j.get("max_salary"),
                    "currency": j.get("currency"),
                }
            )
        return out

    @function_tool()
    async def summarize_remote(self) -> dict[str, Any]:
        """Summarize how many roles are remote vs. on-site and list the
        remote regions represented. Handy when the candidate has a strong
        location preference.
        """
        remote = [j for j in self._jobs if j["is_remote"]]
        not_remote = [j for j in self._jobs if not j["is_remote"]]
        remote_regions: dict[str, int] = {}
        for j in remote:
            loc = j["location"] or "Unknown"
            remote_regions[loc] = remote_regions.get(loc, 0) + 1
        return {
            "remote_count": len(remote),
            "on_site_count": len(not_remote),
            "remote_regions": [
                {"location": k, "count": v}
                for k, v in sorted(remote_regions.items(), key=lambda kv: (-kv[1], kv[0]))
            ],
        }

    # --- filter + details -------------------------------------------------

    @function_tool()
    async def filter_jobs(
        self,
        department: str | None = None,
        team: str | None = None,
        location_contains: str | None = None,
        is_remote: bool | None = None,
        min_salary: int | None = None,
        max_salary: int | None = None,
        keyword: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the subset of jobs that match every provided filter.

        Each returned entry is compact: {id, title, department, team,
        location, is_remote, compensation, min_salary, max_salary, currency}.
        Pass only the filters that matter — omit the rest.

        Args:
            department: Case-insensitive department match (e.g. "R&D").
            team: Case-insensitive team match (e.g. "Engineering", "Design").
            location_contains: Case-insensitive substring match against the
                job's location (e.g. "U.S", "India", "San Francisco").
            is_remote: True to keep only remote roles, False for on-site only.
            min_salary: Minimum annual salary the candidate will accept. Jobs
                whose max_salary is below this are dropped. Jobs with unknown
                salary are kept.
            max_salary: Maximum annual salary cap. Jobs whose min_salary is
                above this are dropped. Jobs with unknown salary are kept.
            keyword: Case-insensitive keyword searched in title and summary
                (e.g. "python", "infra", "telephony").
        """
        matches = []
        for j in self._jobs:
            if department and not self._ci_eq(j.get("department"), department):
                continue
            if team and not self._ci_eq(j.get("team"), team):
                continue
            if location_contains and not self._ci_contains(
                j.get("location"), location_contains
            ):
                continue
            if is_remote is not None and bool(j.get("is_remote")) != bool(is_remote):
                continue
            if min_salary is not None:
                max_s = j.get("max_salary")
                if max_s is not None and max_s < min_salary:
                    continue
            if max_salary is not None:
                min_s = j.get("min_salary")
                if min_s is not None and min_s > max_salary:
                    continue
            if keyword:
                needle = keyword.strip().lower()
                hay = f"{j.get('title','')} {j.get('summary','')}".lower()
                if needle and needle not in hay:
                    continue
            matches.append(self._compact(j))
        return matches

    @function_tool()
    async def get_job_details(self, job_id: str) -> dict[str, Any]:
        """Return the full details of one role by id.

        Includes the summary paragraph, job URL, and compensation. Use this
        after filter_jobs to describe a specific role to the candidate.

        Args:
            job_id: The id returned by filter_jobs or one of the list tools.
        """
        job = self._by_id.get(job_id)
        if job is None:
            return {"error": f"no job with id {job_id}"}
        details = self._compact(job)
        details["summary"] = job["summary"]
        details["job_url"] = job.get("job_url") or None
        details["application_url"] = job.get("application_url") or None
        details["employment_type"] = job.get("employment_type") or None
        return details

    @function_tool()
    async def record_selection(
        self,
        context: RunContext[Userdata],
        job_id: str,
        job_title: str,
        interests_notes: str,
    ) -> str:
        """Record which role the candidate wants to interview for.

        Only call this after the candidate has explicitly confirmed their choice.

        Args:
            job_id: The exact id of the chosen role from the job catalog.
            job_title: The job's title — used as a fallback if the id is off.
            interests_notes: A short summary of what the candidate said they
                are interested in (areas, skills, preferences).
        """
        job = self._lookup(job_id, job_title)
        if job is None:
            return (
                "I couldn't match that to a role in the catalog. Please confirm "
                "the title with the candidate and try again."
            )
        return self._apply_job_selection(context, job, interests_notes)

    @function_tool()
    async def confirm_spoken_role(
        self,
        context: RunContext[Userdata],
        what_they_chose: str,
        interests_notes: str = "",
    ) -> str:
        """Record the mission role using the candidate's own words.

        Call this as soon as they confirm a role in plain language (for example
        'robotics engineer', 'something in robotics', 'the hydroponics lead').
        This resolves to the closest catalog job by keyword match — use
        record_selection instead if you already have the exact job id from tools.

        Args:
            what_they_chose: Short phrase for what they want (title-like words).
            interests_notes: Optional summary of interests they mentioned.
        """
        job = self._find_best_job_match(what_they_chose)
        if job is None:
            return (
                "Could not match that phrase to a catalog role. Use filter_jobs "
                "or list tools, then call record_selection with the exact job_id, "
                "or ask the candidate to pick from the titles you list."
            )
        return self._apply_job_selection(context, job, interests_notes)

    @function_tool()
    async def test_write_report(self, context: RunContext[Userdata]) -> str:
        """Testing only: call when the user says the phrase 'test write report'.

        Writes sample report files to reports/ and output/ next to the agent
        script so you can verify the pipeline without finishing an interview.
        """
        return await invoke_test_write_report(context)


class InterviewQuestionTask(AgentTask[QuestionResult]):
    """One question of the screening interview, tailored to the selected role."""

    def __init__(
        self,
        *,
        topic: str,
        prompt_seed: str,
        question_index: int,
        total_questions: int,
        role: SelectedJob,
    ) -> None:
        self._topic = topic
        self._question_index = question_index
        self._total_questions = total_questions
        self._role = role
        self._captured_question: str | None = None

        super().__init__(
            instructions=(
                f"You are {INTERVIEWER_NAME}, a senior recruiter at {COMPANY} "
                f"conducting a first-round screening interview for the "
                f"'{role.title}' role ({role.department} / {role.team}, "
                f"posted at {role.location}).\n\n"
                f"Role summary (for your reference): {role.summary}\n\n"
                f"This is question {question_index} of {total_questions}. Topic: "
                f"{topic}. The spirit of the question is: {prompt_seed}\n\n"
                "Guidelines:\n"
                f"  - Ask ONE clear, tailored version of this question, phrased "
                f"    naturally for the {role.title} role and the realities of "
                f"    living on Mars.\n"
                "  - Let the candidate respond. If their answer is very short "
                "    or avoids the question, you may ask ONE concise follow-up; "
                "    otherwise move on.\n"
                "  - Acknowledge the answer briefly (e.g. 'Thanks, that helps') "
                "    but do NOT share your evaluation, score, or hiring "
                "    signals — those are internal to the selection board.\n"
                "  - When you have enough to judge the answer, call "
                "    record_response with your evaluation: a response summary, "
                "    specific strengths, any concerns or red flags, and an "
                "    integer score from 1 to 10 for this answer.\n"
                "  - If the candidate clearly asks to end or wrap up the "
                "    screening early (time, fatigue, discomfort), call "
                "    end_interview_now — do not continue with further "
                "    questions after that.\n"
                "  - Stay professional, warm, and curious — not harsh, not "
                "    sycophantic. Keep it conversational with no bullets, no "
                "    markdown, short turns."
            ),
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                f"Ask question {self._question_index} of {self._total_questions} "
                f"on the topic of {self._topic}, tailored to the "
                f"{self._role.title} role on Mars. Keep it to one or two "
                f"sentences."
            ),
        )

    @function_tool()
    async def record_response(
        self,
        context: RunContext[Userdata],
        question_asked: str,
        response_summary: str,
        strengths: str,
        concerns: str,
        score: int,
    ) -> str:
        """Record your evaluation of the candidate's answer.

        Call this once you have enough signal to score the answer. Your
        notes feed directly into the mission selection board's report — be
        specific, evidence-based, and honest.

        Args:
            question_asked: The exact question you ended up asking (tailored
                to the role).
            response_summary: Concise, neutral summary of what the candidate
                said.
            strengths: What they did well — evidence from their answer, not
                generic praise. Empty string if nothing particularly strong.
            concerns: Any red flags, gaps, weak reasoning, or unanswered
                parts. Empty string if no concerns.
            score: Your evaluation of this answer as an integer from 1 to 10.
                1-3 = poor fit / disqualifying for this question.
                4-6 = mixed, unclear, or thin evidence either way.
                7-8 = clearly competent, convincing answer.
                9-10 = exceptional, strongly positive signal.
        """
        try:
            score_int = int(score)
        except (TypeError, ValueError):
            score_int = 5
        score_int = max(1, min(10, score_int))

        result = QuestionResult(
            topic=self._topic,
            question=question_asked.strip() or "(question not captured)",
            response_summary=response_summary.strip(),
            strengths=strengths.strip(),
            concerns=concerns.strip(),
            score=score_int,
        )
        context.userdata.question_results.append(result)
        self.complete(result)

        if self._question_index < self._total_questions:
            return "Recorded. Move to the next question."
        return "Recorded — that's the last question."

    @function_tool()
    async def end_interview_now(self, context: RunContext[Userdata]) -> str:
        """End the screening early because the candidate asked to stop or wrap up.

        Call this when they clearly want to finish before remaining questions.
        If they already gave an answer to this question, call record_response
        first, then call this tool. Do not disclose scores or ratings to them.
        """
        context.userdata.candidate_requested_early_end = True
        placeholder = QuestionResult(
            topic=self._topic,
            question="(interview ended at candidate request — not scored)",
            response_summary=(
                "Candidate requested to end the interview before this question "
                "was completed and scored."
            ),
            strengths="",
            concerns="",
            score=0,
            include_in_scoring=False,
        )
        context.userdata.question_results.append(placeholder)
        self.complete(placeholder)
        return (
            "Interview will end after results are saved. Do not ask more "
            "interview questions."
        )

    @function_tool()
    async def test_write_report(self, context: RunContext[Userdata]) -> str:
        """Testing only: call when the user says the phrase 'test write report'.

        Writes sample report files to reports/ and output/ next to the agent
        script so you can verify the pipeline without finishing an interview.
        """
        return await invoke_test_write_report(context)


# ---------------------------------------------------------------------------
# Question plan — topics + prompts; tasks tailor them to the selected role
# ---------------------------------------------------------------------------


QUESTION_PLAN: list[tuple[str, str]] = [
    (
        "Background",
        "Invite the candidate to walk through their background and the "
        "experience most relevant to performing this specific role on Mars.",
    ),
    (
        "Motivation for Mars",
        "Why do they want to live and work on Mars — and why this role "
        "specifically? Probe the seriousness of the commitment (multi-year "
        "deployment, limited return windows, family impact).",
    ),
]


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _slug(s: str, max_len: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", s or "").strip("-")
    return (s[:max_len] or "candidate").lower()


def _aggregate_score(userdata: Userdata) -> tuple[float | None, str]:
    """Compute the average per-question score and a recommendation bucket.

    Buckets:
        >= 7.5 → Proceed to next round
        5.0 to 7.5 → Hold / needs further assessment
        < 5.0 → Reject

    Returns (average_score, recommendation). average_score is None if no
    questions were scored.
    """
    scored = [
        r.score
        for r in userdata.question_results
        if r.include_in_scoring and isinstance(r.score, int)
    ]
    if not scored:
        return None, "Hold / needs further assessment"
    avg = sum(scored) / len(scored)
    if avg >= 7.5:
        rec = "Proceed to next round"
    elif avg >= 5.0:
        rec = "Hold / needs further assessment"
    else:
        rec = "Reject"
    return avg, rec


def _build_report_prompt(userdata: Userdata) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the evaluation report LLM call."""
    job = userdata.selected_job
    assert job is not None
    avg, rec = _aggregate_score(userdata)
    avg_str = f"{avg:.1f} / 10" if avg is not None else "n/a"

    round_lines: list[str] = []
    scored_i = 0
    for r in userdata.question_results:
        if not r.include_in_scoring:
            round_lines.append(
                f"Early termination — {r.topic}\n"
                f"  {r.response_summary}"
            )
            continue
        scored_i += 1
        round_lines.append(
            f"Q{scored_i} — {r.topic} (score: {r.score}/10)\n"
            f"  Question asked: {r.question}\n"
            f"  Candidate's answer (summary): {r.response_summary}\n"
            f"  Strengths noted: {r.strengths or '(none noted)'}\n"
            f"  Concerns noted: {r.concerns or '(none noted)'}"
        )
    rounds_blob = "\n\n".join(round_lines)

    system_prompt = (
        f"You are {INTERVIEWER_NAME}, a senior recruiter at {COMPANY}. Write a "
        "written candidate evaluation for the mission selection board based "
        "on your first-round screening call. Be professional, specific, "
        "and evidence-based — cite the candidate's actual answers. Never "
        "invent facts. The report is internal and never shown to the "
        "candidate.\n\n"
        "Output requirements:\n"
        "  - Pure markdown. No code fences around the whole thing.\n"
        "  - Start with an H1 title, then a metadata table (candidate, role, "
        "    team, posting location, date, average score, recommendation).\n"
        "  - 'Recommendation' section: 1-2 sentences restating the bucket "
        "    (Proceed / Hold / Reject) with the single most important reason.\n"
        "  - 'Summary for hiring manager' section: a 3-5 sentence synthesis "
        "    a busy mission lead can read in 30 seconds.\n"
        "  - 'Strengths' section: 3-5 bullets grounded in specific answers.\n"
        "  - 'Concerns' section: 3-5 bullets grounded in specific answers. "
        "    If no concerns, write 'None surfaced in this screening.'\n"
        "  - 'Role-fit assessment' section: one short paragraph on how the "
        "    candidate maps to this specific Mars role's demands.\n"
        "  - 'Question-by-question evaluation' section: one H3 per question "
        "    with the question asked, a one-line takeaway, the score, and "
        "    any decisive quote or detail.\n"
        "  - End with a short 'Interviewer notes' paragraph from your "
        "    perspective (observed demeanor, rapport, overall impression).\n"
        "  - Do NOT invent hire decisions beyond the recommendation bucket.\n"
        "  - Do NOT praise or encourage the candidate — this is an internal "
        "    evaluation, not feedback to them."
    )

    user_prompt = (
        f"Candidate: {userdata.candidate_name or 'Candidate'}\n"
        f"Role: {job.title} ({job.department} / {job.team})\n"
        f"Posting location: {job.location}\n"
        f"Compensation note: {job.compensation or 'not listed'}\n"
        f"Role summary: {job.summary}\n"
        f"Candidate's stated interests: {userdata.interests_notes or '(not captured)'}\n"
        f"Date: {datetime.now().strftime('%B %d, %Y')}\n"
        f"Average question score: {avg_str}\n"
        f"Recommendation bucket: {rec}\n\n"
        f"Interview notes:\n{rounds_blob}\n\n"
        "Now write the full markdown evaluation report."
    )
    return system_prompt, user_prompt


async def generate_report_markdown(session_llm: llm.LLM, userdata: Userdata) -> str:
    system_prompt, user_prompt = _build_report_prompt(userdata)

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content=system_prompt)
    chat_ctx.add_message(role="user", content=user_prompt)

    chunks: list[str] = []
    async for chunk in session_llm.chat(chat_ctx=chat_ctx):
        if chunk.delta and chunk.delta.content:
            chunks.append(chunk.delta.content)
    return "".join(chunks).strip()


def _fallback_report(userdata: Userdata) -> str:
    """Deterministic evaluation used if the LLM report call fails."""
    job = userdata.selected_job
    assert job is not None
    now = datetime.now().strftime("%B %d, %Y")
    name = userdata.candidate_name or "Candidate"
    avg, rec = _aggregate_score(userdata)
    avg_str = f"{avg:.1f} / 10" if avg is not None else "n/a"

    lines: list[str] = []
    lines.append(f"# Candidate Evaluation — {name}")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Candidate | {name} |")
    lines.append(f"| Role | {job.title} |")
    lines.append(f"| Team | {job.department} / {job.team} |")
    lines.append(f"| Posting location | {job.location} |")
    lines.append(f"| Date | {now} |")
    lines.append(f"| Average score | {avg_str} |")
    lines.append(f"| Recommendation | {rec} |")
    lines.append("")
    n_scored = sum(1 for r in userdata.question_results if r.include_in_scoring)
    lines.append("## Recommendation")
    lines.append(
        f"{rec} for the {job.title} role based on an average question score "
        f"of {avg_str} across {n_scored} scored questions."
    )
    lines.append("")
    lines.append("## Summary for hiring manager")
    lines.append(
        f"{name} was screened by {INTERVIEWER_NAME} for the {job.title} "
        f"posting at {job.location}. See the per-question evaluation below "
        "for evidence."
    )
    lines.append("")
    lines.append("## Strengths")
    strengths_any = False
    si = 0
    for r in userdata.question_results:
        if not r.include_in_scoring:
            continue
        si += 1
        if r.strengths:
            lines.append(f"- Q{si} ({r.topic}): {r.strengths}")
            strengths_any = True
    if not strengths_any:
        lines.append("- None noted in this screening.")
    lines.append("")
    lines.append("## Concerns")
    concerns_any = False
    ci = 0
    for r in userdata.question_results:
        if not r.include_in_scoring:
            continue
        ci += 1
        if r.concerns:
            lines.append(f"- Q{ci} ({r.topic}): {r.concerns}")
            concerns_any = True
    if not concerns_any:
        lines.append("- None surfaced in this screening.")
    lines.append("")
    lines.append("## Question-by-question evaluation")
    qi = 0
    for r in userdata.question_results:
        if not r.include_in_scoring:
            lines.append(f"### Early termination — {r.topic}")
            lines.append(r.response_summary)
            lines.append("")
            continue
        qi += 1
        lines.append(f"### Q{qi} — {r.topic}  (score: {r.score}/10)")
        lines.append(f"**Question:** {r.question}")
        lines.append("")
        lines.append(f"**Candidate's answer (summary):** {r.response_summary}")
        lines.append("")
        if r.strengths:
            lines.append(f"**Strengths:** {r.strengths}")
            lines.append("")
        if r.concerns:
            lines.append(f"**Concerns:** {r.concerns}")
            lines.append("")
    lines.append("## Interviewer notes")
    lines.append(
        f"{INTERVIEWER_NAME} conducted a two-question first-round screening. "
        f"Per-question scores and evidence above."
    )
    return "\n".join(lines)


def save_report(userdata: Userdata, markdown: str) -> Path:
    """Write the evaluation markdown to ``reports/`` and return the file path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    job = userdata.selected_job
    assert job is not None

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    filename = f"{stamp}_{_slug(job.title)}_{_slug(userdata.candidate_name)}.md"
    path = REPORTS_DIR / filename
    path.write_text(markdown, encoding="utf-8")
    return path


def _safe_session_filename_part(session_id: str) -> str:
    if not (session_id or "").strip():
        return "unknown"
    return re.sub(r"[^\w.\-]+", "_", session_id.strip())


def build_interview_results_payload(
    userdata: Userdata,
    report_markdown: str,
    *,
    partial: bool,
) -> dict[str, Any]:
    job = userdata.selected_job
    assert job is not None
    avg, rec = _aggregate_score(userdata)
    scoring_n = sum(1 for r in userdata.question_results if r.include_in_scoring)
    return {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "partial_screening": partial,
        "candidate_requested_early_end": userdata.candidate_requested_early_end,
        "report_used_fallback_role": userdata.report_used_fallback_role,
        "report_used_placeholder_scoring": userdata.report_used_placeholder_scoring,
        "questions_scored": scoring_n,
        "livekit": {
            "session_id": userdata.session_id,
            "room_name": userdata.room_name,
            "room_sid": userdata.room_sid,
        },
        "candidate": {
            "name": userdata.candidate_name,
            "interests_notes": userdata.interests_notes,
        },
        "role": asdict(job),
        "question_results": [asdict(q) for q in userdata.question_results],
        "scores": {
            "average": avg,
            "recommendation_bucket": rec,
        },
        "report_markdown": report_markdown,
    }


def save_interview_results_json(
    userdata: Userdata, report_markdown: str, partial: bool
) -> Path:
    """Write structured interview results to output/<date>_<time>_<session_id>.json."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    sid = _safe_session_filename_part(userdata.session_id)
    path = OUTPUT_DIR / f"{stamp}_{sid}.json"
    payload = build_interview_results_payload(
        userdata, report_markdown, partial=partial
    )
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _dict_to_selected_job(job: dict[str, Any]) -> SelectedJob:
    return SelectedJob(
        id=str(job["id"]),
        title=job["title"],
        department=job["department"],
        team=job["team"],
        location=job["location"],
        summary=job["summary"],
        compensation=job["compensation"] or "",
        job_url=job.get("job_url") or "",
    )


def _synthetic_unrecorded_job() -> SelectedJob:
    """Placeholder role row when the catalog is empty or unavailable."""
    return SelectedJob(
        id="unrecorded",
        title="Role not recorded",
        department="—",
        team="—",
        location="—",
        summary=(
            "No catalog role was confirmed via record_selection or "
            "confirm_spoken_role before the session ended. This row exists only "
            "so the interview report can be filed."
        ),
        compensation="",
        job_url="",
    )


def ensure_minimum_report_state(userdata: Userdata) -> None:
    """Guarantee selected_job + at least structural question rows so reports always write."""
    if userdata.selected_job is None:
        userdata.report_used_fallback_role = True
        if userdata.jobs:
            userdata.selected_job = _dict_to_selected_job(userdata.jobs[0])
            logger.info(
                "no catalog role recorded — using first catalog listing for "
                "required report (use record_selection or confirm_spoken_role "
                "to record the candidate's choice)"
            )
        else:
            userdata.selected_job = _synthetic_unrecorded_job()
            logger.warning(
                "mission catalog empty — report will use a synthetic "
                "unrecorded-role row"
            )

    scoring_n = sum(1 for r in userdata.question_results if r.include_in_scoring)
    if scoring_n == 0:
        userdata.report_used_placeholder_scoring = True
        userdata.question_results = [
            QuestionResult(
                topic=topic,
                question=(
                    "(Structured scoring not captured — record_response was not "
                    "called for this topic, or the session ended before scoring.)"
                ),
                response_summary=(
                    "No scored rubric row was persisted; see session transcript "
                    "if available."
                ),
                strengths="",
                concerns="",
                score=0,
                include_in_scoring=False,
            )
            for topic, _ in QUESTION_PLAN
        ]


def _seed_test_report_data(userdata: Userdata) -> None:
    """Fill minimal catalog + answers so report writers can run (local testing)."""
    userdata.candidate_requested_early_end = False
    if not userdata.candidate_name.strip():
        userdata.candidate_name = "Test Candidate"
    if not userdata.interests_notes.strip():
        userdata.interests_notes = "(test run — invoke test_write_report tool)"
    if userdata.selected_job is None and userdata.jobs:
        userdata.selected_job = _dict_to_selected_job(userdata.jobs[0])
    scoring_n = sum(1 for r in userdata.question_results if r.include_in_scoring)
    if scoring_n == 0:
        userdata.question_results = [
            QuestionResult(
                topic=topic,
                question=f"(test) Screening question: {topic}",
                response_summary="Synthetic answer for pipeline test.",
                strengths="Clear communication in test scenario.",
                concerns="",
                score=8,
                include_in_scoring=True,
            )
            for topic, _ in QUESTION_PLAN
        ]


async def persist_interview_reports(
    session: AgentSession[Userdata], userdata: Userdata
) -> tuple[Path | None, Path | None]:
    """Generate markdown + JSON under reports/ and output/. Returns paths if saved."""
    ensure_minimum_report_state(userdata)

    scoring_n = sum(1 for r in userdata.question_results if r.include_in_scoring)
    total = len(QUESTION_PLAN)
    partial = scoring_n < total
    if partial:
        logger.info(
            "saving partial evaluation report (%d/%d scored questions)",
            scoring_n,
            total,
        )

    try:
        session_llm = session.llm
        if isinstance(session_llm, llm.LLM):
            markdown = await generate_report_markdown(session_llm, userdata)
        else:
            markdown = _fallback_report(userdata)
    except Exception:
        logger.exception("report generation failed; falling back to template")
        markdown = _fallback_report(userdata)

    if not markdown.strip():
        markdown = _fallback_report(userdata)

    prefix_notes = ""
    if userdata.report_used_fallback_role:
        if userdata.jobs:
            prefix_notes += (
                "> **Note:** No mission role was confirmed via catalog tools — "
                "this report uses the **first catalog listing** as a filing "
                "default. Verify the role with the candidate.\n\n"
            )
        else:
            prefix_notes += (
                "> **Note:** Mission catalog was empty — the role block is a "
                "**synthetic placeholder** for filing only.\n\n"
            )
    if userdata.report_used_placeholder_scoring:
        prefix_notes += (
            "> **Note:** No `record_response` scores were persisted — question "
            "rows below are **placeholders**, not rubric evaluations. Use the "
            "session transcript where available.\n\n"
        )
    if prefix_notes:
        markdown = prefix_notes + markdown

    if partial:
        if userdata.candidate_requested_early_end:
            if scoring_n:
                note = (
                    f"> ⚠️ Partial screening — candidate requested to end after "
                    f"{scoring_n} of {total} scored questions.\n\n"
                )
            else:
                note = (
                    f"> ⚠️ Partial screening — candidate requested to end before "
                    f"any scored questions.\n\n"
                )
        else:
            note = (
                f"> ⚠️ Partial screening — interview interrupted after "
                f"{scoring_n} of {total} scored questions.\n\n"
            )
        markdown = note + markdown

    md_path: Path | None = None
    json_path: Path | None = None
    try:
        md_path = save_report(userdata, markdown)
        logger.info(
            "saved %s evaluation report to %s",
            "partial" if partial else "full",
            md_path,
        )
    except Exception:
        logger.exception("failed to save evaluation report")

    try:
        json_path = save_interview_results_json(userdata, markdown, partial)
        logger.info(
            "saved %s interview results JSON to %s",
            "partial" if partial else "full",
            json_path,
        )
    except Exception:
        logger.exception("failed to save interview results JSON")

    return md_path, json_path


async def invoke_test_write_report(context: RunContext[Userdata]) -> str:
    """Shared handler for the test_write_report tool across agents/tasks."""
    _seed_test_report_data(context.userdata)
    md_path, json_path = await persist_interview_reports(
        context.session, context.userdata
    )
    if md_path is None:
        return (
            "test_write_report: could not write files. Ensure mars_jobs.json "
            "exists and lists at least one role (the first role is used when "
            "none is selected)."
        )
    if json_path is None:
        return (
            f"test_write_report: wrote markdown to {md_path} but JSON save "
            "failed — see logs."
        )
    return (
        f"Wrote markdown to {md_path} and JSON to {json_path}. "
        "These paths are next to interview_agent.py, not under .venv."
    )


# ---------------------------------------------------------------------------
# Top-level interviewer agent
# ---------------------------------------------------------------------------


class MarsInterviewAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                f"You are {INTERVIEWER_NAME}, a senior recruiter at {COMPANY}. "
                "You conduct first-round screening interviews for prospective "
                "Mars settlers. Speak naturally — no markdown, no bullet "
                "points, no emojis, no asterisks. Keep responses concise so "
                "the call stays conversational. Stay professional and curious, "
                "never harsh, never sycophantic. Do not share evaluations, "
                "scores, or hiring signals with the candidate during the call."
            ),
        )

    async def on_enter(self) -> None:
        userdata: Userdata = self.session.userdata
        jobs = userdata.jobs

        if not jobs:
            await self.session.generate_reply(
                instructions=(
                    "Apologize briefly — the mission role catalog could not be "
                    "loaded, so you can't run the screening right now. Invite "
                    "the candidate to try again later, then end the call."
                ),
            )
            self.session.shutdown()
            return

        # The whole flow below is wrapped so that a mid-call candidate
        # disconnect (which the framework signals as a ToolError cancelling
        # the in-flight AgentTask) is handled cleanly instead of surfacing as
        # an unhandled error. If we got at least some interview data before
        # the disconnect, save a partial evaluation report.
        disconnect_during_interview = False
        try:
            # Phase 1: intake — name + mission role selection
            intake_group = TaskGroup()
            intake_group.add(
                lambda: IntroTask(), id="intro", description="Collect candidate name."
            )
            intake_group.add(
                lambda: JobSelectionTask(jobs=jobs),
                id="job_selection",
                description="Match candidate to one open mission role and confirm.",
            )
            intake_results = await intake_group

            selection: JobSelectionResult | None = intake_results.task_results.get("job_selection")  # type: ignore[assignment]
            if selection is None or not isinstance(selection, JobSelectionResult):
                # selection didn't complete — graceful exit
                await self.session.generate_reply(
                    instructions=(
                        "Thank the candidate warmly and explain that since a "
                        "role wasn't confirmed we'll pause here. Say goodbye."
                    ),
                )
                self.session.shutdown()
                return

            role = selection.job

            # Phase 2: transition into the formal interview
            await self.session.generate_reply(
                instructions=(
                    f"Confirm the choice: you'll move into the formal screening "
                    f"for the {role.title} role — two short questions, "
                    f"conversational. The mission selection board will follow "
                    f"up after the call. Ask if they're ready to begin. Keep "
                    f"it to one or two sentences."
                ),
            )

            # Phase 3: two interview questions, one at a time (allows early end).
            total = len(QUESTION_PLAN)
            for i, (topic, prompt_seed) in enumerate(QUESTION_PLAN, start=1):
                if userdata.candidate_requested_early_end:
                    break
                qtask = InterviewQuestionTask(
                    topic=topic,
                    prompt_seed=prompt_seed,
                    question_index=i,
                    total_questions=total,
                    role=role,
                )
                await qtask
                if userdata.candidate_requested_early_end:
                    break
        except llm.ToolError as e:
            disconnect_during_interview = True
            # The framework raises ToolError("AgentTask X is cancelled") when
            # the candidate disconnects during a running AgentTask. Not our
            # error — just log at info level and fall through to save
            # whatever partial data we captured.
            logger.info(
                "interview interrupted before completion "
                "(candidate likely disconnected): %s",
                e,
            )

        # Phase 4: persist scores and report files, then say goodbye (unless
        # the candidate already dropped). Phase 5: hang up.
        await self._write_report(userdata)

        if userdata.selected_job is not None and not disconnect_during_interview:
            try:
                if userdata.candidate_requested_early_end:
                    await self.session.generate_reply(
                        instructions=(
                            f"The candidate asked to wrap up before all "
                            f"questions. Thank {userdata.candidate_name or 'them'} "
                            f"warmly for their time. Say the mission selection "
                            f"board at {COMPANY} will review what was covered "
                            f"and follow up. Wish them well briefly. Do NOT "
                            f"mention numeric scores, ratings, or a written report."
                        ),
                    )
                else:
                    await self.session.generate_reply(
                        instructions=(
                            f"Thank {userdata.candidate_name or 'the candidate'} for "
                            f"their time. Let them know the mission selection board at "
                            f"{COMPANY} will review their screening and follow up "
                            f"within a week. Wish them well. Keep it warm and brief — "
                            f"one or two sentences. Do NOT mention a report, scores, "
                            f"or recommendations."
                        ),
                    )
            except Exception:
                logger.exception("goodbye reply failed")

        try:
            self.session.shutdown()
        except Exception:
            # The session may already be closing (e.g. candidate disconnect).
            pass

    async def _write_report(self, userdata: Userdata) -> None:
        """Generate the evaluation markdown and save it to reports/.

        Safe to call with any amount of captured data. If nothing was
        captured (no selected role), logs a note and skips the write.
        """
        await persist_interview_reports(self.session, userdata)

    @function_tool()
    async def test_write_report(self, context: RunContext[Userdata]) -> str:
        """Testing only: call when the user says the phrase 'test write report'.

        Writes sample report files to reports/ and output/ next to the agent
        script so you can verify the pipeline without finishing an interview.
        """
        return await invoke_test_write_report(context)

    @function_tool()
    async def end_interview(self, context: RunContext[Userdata]) -> None:
        """Hang up the call. Prefer letting the interview flow finish; the session
        ends automatically after goodbye. Use only if you must force hangup."""
        context.session.shutdown()


# ---------------------------------------------------------------------------
# Server wiring
# ---------------------------------------------------------------------------


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    jobs = load_jobs()
    if not jobs:
        logger.warning(
            "no mission roles loaded from %s — check that the file exists",
            JOBS_FILE,
        )

    # See build_stt_llm_tts() above. Default: LiveKit inference proxy.
    stt_inst, llm_inst, tts_inst = build_stt_llm_tts()

    userdata = Userdata(jobs=jobs)
    jr = ctx.job.room
    userdata.session_id = ctx.job.id or ""
    userdata.room_name = jr.name or ""
    userdata.room_sid = jr.sid or ""

    session: AgentSession[Userdata] = AgentSession[Userdata](
        userdata=userdata,
        stt=stt_inst,
        llm=llm_inst,
        tts=tts_inst,
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
            # See more at https://docs.livekit.io/agents/build/turns
            turn_detection=MultilingualModel(),
            interruption={
                "mode": "adaptive",
            },
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
        m = ev.item.metrics or {}
        if ev.item.role == "assistant":
            ttft = m.get("llm_node_ttft")
            ttfb = m.get("tts_node_ttfb")
            e2e = m.get("e2e_latency")
            logger.info(
                "assistant turn latency — ttft=%.3fs ttfb=%.3fs e2e=%.3fs",
                ttft or 0.0, ttfb or 0.0, e2e or 0.0,
            )

    async def log_usage():
        logger.info(f"Usage: {session.usage}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=MarsInterviewAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # uncomment to enable Krisp BVC noise cancellation
                # noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )


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
