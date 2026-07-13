# bee-bug-hunter

Autonomous bug-hunting crew built on the [BeeAI framework](https://github.com/i-am-bee/beeai-framework) —
a port of `crew-bug-hunter` (CrewAI) with the same features:

- An **Investigation Manager** supervisor agent delegates (via handoff tools, never running
  domain tools itself) to five workers: API Flow Runner (Playwright / requests), Docker Log
  Capturer, DB Query Agent, Bug Analyst, and SQL Performance Agent.
- Flows are YAML files in `bee_bug_hunter/flows/`; the batch to monitor is listed in
  `bee_bug_hunter/flows_manifest.yaml`.
- Switchable LLM providers via `LLM_PROVIDER` in `.env`: `ollama`, `openai`, `anthropic`, or
  `claude_cli` (shells to `claude -p`, reusing your Claude Code login — no API key).
- Structured JSONL logging with a per-run `run_id` correlation id; one markdown report per
  flow run in `reports/`.
- A Textual TUI (`python -m bee_bug_hunter.tui`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env  # fill in LLM_PROVIDER, MySQL creds, etc.
```

## Running

```bash
python -m bee_bug_hunter.main --once     # one pass over every flow, then exit
python -m bee_bug_hunter.main            # continuous monitor
python -m bee_bug_hunter.tui             # TUI
```

## Local test target: `demo_app/`

Dockerized Flask + MySQL app with two seeded issues (a login handler referencing a
nonexistent `passwd` column causing every login to 500, and an N+1 query on
`/api/orders/<user_id>`):

```bash
cd demo_app && docker compose up --build -d
cd .. && python -m bee_bug_hunter.main --once
```
