# Mars Recruitment Services — Voice Screening Agent

A demo voice agent that conducts first-round phone screenings for
prospective Mars settlers. The candidate picks one open mission role from
the catalog, answers two structured interview questions tailored to that
role, and the agent writes a written evaluation report for the mission
selection board.


## Setup Environment Variables

This requires the following environment variables

[Simplismart Settings](https://app.simplismart.ai/settings?tab=2)

```
export SIMPLISMART_API_KEY=eyJhb...
SIMPLISMART_BASE_URL=https://api.simplismart.live
```

[LiveKit Setting](https://cloud.livekit.io/projects/p_/settings/keys)

```
LIVEKIT_URL=wss://xyz.livekit.cloud
LIVEKIT_API_KEY=API...
LIVEKIT_API_SECRET=rXy...
```

## Setup Python Virtual Env

```
python -m venv .venv
source .venv/bin/activate
pip install -U pip

pip install -r requirements.txt

python interview_agent.py download-files

```


## Run Agent Console

This is for local testing 

```
python interview_agent.py console

```


## Run Agent Shared

```
python interview_agent.py

```


## Using Other Models

Go to Simplismart [Market Place](https://app.simplismart.ai/model-marketplace) and find you model. Update code with model and URL.


