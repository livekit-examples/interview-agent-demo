# Mars Recruitment Services — Voice Screening Agent

A LiveKit Agents demo that conducts a first-round phone screening for
prospective Mars settlers. The agent greets the candidate, discusses
open mission roles from the catalog, runs two short interview
questions (background, motivation for Mars), and closes warmly. After
the call ends, the full transcript is sent to an LLM once to produce a
structured JSON evaluation and a short markdown report. Results are
written locally and, optionally, appended to a Google Sheet.

The agent is intentionally simple — a single `Agent` with one system
prompt, no function tools, no task groups. All structured capture
happens post-call by re-prompting the LLM with the transcript.

## Project layout

```
interview_agent.py   # the agent (entrypoint)
google_sheets.py     # optional Google Sheets integration (ADC-auth)
mars_jobs.json       # catalog of open Mars mission roles
reports/             # per-interview markdown evaluations (gitignored)
output/              # per-interview structured JSON (gitignored)
env.example          # sample .env — copy to .env.local and fill in
```

## Prerequisites

- Python 3.11+ (Python 3.13 tested)
- A LiveKit Cloud project — credentials at https://cloud.livekit.io/projects/p_/settings/keys
- A Simplismart API key for STT / LLM / TTS — https://app.simplismart.ai/settings?tab=2
- (Optional) `gcloud` CLI if you want to write results to a Google Sheet

## Setup

### 1. Python virtual env

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirement.txt
```

### 2. Environment variables

Copy `env.example` to `.env.local` and fill in the required values:

```bash
cp env.example .env.local
```

Required:

```
SIMPLISMART_API_KEY=eyJhb...
LIVEKIT_URL=wss://<your-project>.livekit.cloud
LIVEKIT_API_KEY=API...
LIVEKIT_API_SECRET=...
```

Optional — per-service model overrides. Set only the ones you want to
change; unset vars fall back to the defaults in `interview_agent.py`.

```
STT_URL=https://api.simplismart.live/predict
STT_MODEL=openai/whisper-large-v3-turbo

LLM_URL=https://api.simplismart.live
LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct

TTS_URL=https://api.simplismart.live/tts
TTS_MODEL=canopylabs/orpheus-3b-0.1-ft
TTS_VOICE=                # optional, depends on the TTS model

GOOGLE_SHEETS_SPREADSHEET_ID=   # optional, see Google Sheets section
```

The TTS plugin is chosen from the URL: endpoints ending in
`/audio/speech` use `openai.TTS` (OpenAI-compatible); everything else
uses `simplismart.TTS` (native `/tts`). Switch TTS stacks just by
flipping `TTS_URL`.

## Running

Dev mode (auto-reload, prints a playground connection URL + token on
startup):

```bash
python interview_agent.py dev
```

Console mode (run the agent in the terminal without LiveKit rooms, for
quick local testing):

```bash
python interview_agent.py console
```

Production-style start (no reload):

```bash
python interview_agent.py start
```

Connect the test client at https://agents-playground.livekit.io/ using
the URL + token printed on startup.

## Outputs

Every completed call writes two local files:

- `output/<date>_<time>_<session_id>.json` — structured evaluation + full transcript
- `reports/<date>_<time>_<role>_<name>.md` — human-readable markdown report

Both directories are gitignored.

## Google Sheets (optional)

Results are additionally appended to a Google Sheet when configured.
Authentication uses a **Google Cloud service account JSON key**, picked
up automatically via `GOOGLE_APPLICATION_CREDENTIALS`. `google.auth.default()`
detects the env var and hands the service-account credentials to gspread.

> **Note:** We tried user-based ADC (`gcloud auth application-default login`)
> first, but Google now blocks the Sheets and Drive scopes for the default
> gcloud OAuth client. A service account avoids that and is stable for
> headless/deployed setups too.

### One-time setup

You'll need a Google Cloud project you have **admin** on — org-owned
projects (like `livekit-cloud-site`) often won't let you enable APIs or
create service accounts. If you don't have a personal one, create one at
https://console.cloud.google.com/projectcreate (or reuse an existing
personal project).

#### 1. Install and authenticate gcloud

```bash
# macOS
brew install --cask google-cloud-sdk
# or follow https://cloud.google.com/sdk/docs/install

gcloud auth login     # opens a browser — sign in as yourself
gcloud projects list  # confirm your project appears
gcloud config set project <your-project-id>   # e.g. chriswilson-geminitest
```

> Do **not** use the project *number* here — use the project **id**
> (the string-y one in the first column of `gcloud projects list`).

#### 2. Enable the Sheets + Drive APIs on that project

```bash
gcloud services enable sheets.googleapis.com drive.googleapis.com
```

Or via the console:
- https://console.cloud.google.com/apis/library/sheets.googleapis.com
- https://console.cloud.google.com/apis/library/drive.googleapis.com

Confirm the project selector in the top bar matches your project before
clicking **ENABLE**.

#### 3. Create a service account and download its key

```bash
PROJECT_ID=$(gcloud config get-value project)

gcloud iam service-accounts create mars-interview \
  --display-name="Mars Interview Writer"

SA_EMAIL="mars-interview@${PROJECT_ID}.iam.gserviceaccount.com"

mkdir -p ~/.config/gcp
gcloud iam service-accounts keys create ~/.config/gcp/mars-interview-sa.json \
  --iam-account="$SA_EMAIL"

echo "Share your sheet with: $SA_EMAIL"
```

The key file lives at `~/.config/gcp/mars-interview-sa.json`. Treat it
like a password — it's not checked in, but don't paste it anywhere.

#### 4. Share the target Google Sheet with the service account

Service accounts act as themselves, so the sheet must explicitly grant
them access:

1. Open the sheet you want results written to.
2. Click **Share** in the top right.
3. Paste the `mars-interview@…iam.gserviceaccount.com` email from the
   previous step.
4. Role: **Editor**.
5. Uncheck "Notify people" (service accounts can't read email).
6. Click **Share**.

#### 5. Configure this project

Add to `.env.local`:

```
GOOGLE_APPLICATION_CREDENTIALS=/Users/<you>/.config/gcp/mars-interview-sa.json
GOOGLE_SHEETS_SPREADSHEET_ID=<the long segment from the sheet URL>
```

The spreadsheet id is the long alphanumeric part of
`https://docs.google.com/spreadsheets/d/<THIS_PART>/edit`.

#### 6. Verify

```bash
.venv/bin/python scripts/check_sheets.py
```

Expected output: `read ok — first tab: Sheet1` (or whatever your first
tab is named).

On the next `python interview_agent.py dev` run you should see
`Google Sheets service initialized successfully` in the logs. A
worksheet named `Interview Results` is created automatically on first
write with columns: Timestamp, Session ID, Candidate Name, Score (1-10),
Recommendation, Strengths, Areas for Improvement, Summary, Full
Conversation.

If Google Sheets isn't set up, the agent logs a warning and continues —
the local JSON/markdown files in `output/` and `reports/` are always
written.

### Troubleshooting

- **403 "Google Sheets API has not been used in project X…"** — The
  Sheets API isn't enabled on the project that owns the service account.
  Re-run step 2 for that project.
- **403 "The caller does not have permission"** — The sheet isn't shared
  with the service-account email, or it was shared as Viewer instead of
  Editor. Re-run step 4.
- **"serviceusage.services.list" permission denied** — You don't have
  admin on that project (typical for org-managed projects like
  `livekit-cloud-site`). Use a personal project instead.
- **`gcloud` is acting as a service account** (`gcloud auth list` shows a
  `*.iam.gserviceaccount.com` as ACTIVE) — run `gcloud auth login` again
  as your user.

## Swapping models

Change one `*_URL` / `*_MODEL` / `*_VOICE` line in `.env.local` and
restart the agent. Each service is independent. The post-call
evaluation uses `LLM_URL` + `LLM_MODEL`, so swapping the LLM swaps
both the conversation and the evaluation.

If you need a model that isn't OpenAI-compatible, you'll have to edit
`build_stt_llm_tts()` in `interview_agent.py` to use a different
plugin.

## LiveKit docs

This project follows the [LiveKit Agents](https://docs.livekit.io/agents/)
conventions. The `lk docs` CLI and the LiveKit Docs MCP server are the
fastest way to look up current API details — see `AGENTS.md` for
install instructions and usage.
