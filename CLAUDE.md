# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env  # fill in LLM_PROVIDER, MySQL creds, etc.
```

```bash
pip install -r requirements-dev.txt
python -m pytest tests/                          # unit tests (no LLM/docker needed)
```

## Running

```bash
python -m bee_bug_hunter.main --once             # one pass over every flow in the manifest
python -m bee_bug_hunter.main                    # continuous monitor (poll_interval_seconds)
python -m bee_bug_hunter.main --manifest path/to/other_manifest.yaml
python -m bee_bug_hunter.tui                     # Textual TUI
```

Flows are YAML files in `bee_bug_hunter/flows/` (steps: `goto`, `click`, `fill`,
`wait_for_selector`, `wait_for_response`). The batch to monitor is listed in
`bee_bug_hunter/flows_manifest.yaml` — each entry names a flow file (or a registered
`api_flows.py` function for `kind: api`) and the Docker containers backing it.

## Local test target: `demo_app/`

Throwaway dockerized Flask + MySQL app with two seeded issues: a login handler referencing a
nonexistent `passwd` column (real column is `password`, every login 500s), and an N+1 query on
`/api/orders/<user_id>` (only reliably measurable past ~3000 rows).

```bash
cd demo_app && docker compose up --build -d
cd .. && python -m bee_bug_hunter.main --once
```

## Architecture

This is a BeeAI-framework port of `~/workspace/crew-bug-hunter` (CrewAI). Per flow, per poll
cycle, `manager.build_supervisor()` builds an **Investigation Manager** `RequirementAgent`
whose only tools are `CapturingHandoffTool`s (subclass of BeeAI's `HandoffTool`) targeting
five worker `RequirementAgent`s defined in `agents.py`:

1. **API Flow Runner** — Playwright (async API) or requests flows, records every request/response.
2. **Docker Log Capturer** — `docker logs -f --since 5m` (window deliberately 5m, not 0s:
   workers run sequentially, so the flow's output predates capture start).
3. **DB Query Agent** — finds SQL in logs, runs read-only equivalents (`MySQLQueryTool`
   rejects non-SELECT; `EXPLAIN` passes).
4. **Bug Analyst** — root-cause synthesis; optional `check_anomalies` heuristic tool and a
   private `ContextStore` scratchpad (not shared with other agents or the manager).
5. **SQL Performance Agent** — `EXPLAIN`-backed index/query fixes; own private scratchpad.

The manager-only-delegates rule is structural here (its tool list is only handoffs), unlike
CrewAI's hierarchical-process `agent=` gotcha. Cross-delegation context comes free from
`HandoffTool`, which propagates the conversation-so-far into each worker's memory.

`delegation_capture.py` records each worker's own returned text keyed by the `run_id`
contextvar (the manager's final answer is a synthesized summary, not the workers' raw
output); `orchestrator.run_flow_once` uses it to compute deterministic anomaly signals
(`anomaly_detector.py`) and pull the Bug Analyst / SQL Performance reports for
`reports/*.md` and the TUI, independent of what the manager quoted.

## LLM providers (`llm.py`)

`LLM_PROVIDER` in `.env`: `ollama` (default, native tool calling, no API key), `openai`,
or `anthropic`.

There used to be a `claude_cli` provider (`claude_cli_llm.py`) that shelled to
`claude -p --safe-mode --tools none` per reasoning step, bridging tool calls through a
hand-rolled `{"tool": ..., "args": {...}}` JSON text protocol since the CLI's own tool use
was disabled. Removed: `RequirementAgent`'s per-step forced `tool_choice` plus a
subprocess-per-step, no-prompt-caching text bridge is a poor fit for a frontier model with
native tool calling and a large cacheable context — it produced a ~36% wasted-call rate
(`missed_required_tool_call`, the model not reliably following the text convention) and no
prompt-cache reuse across steps. This decomposed-into-many-forced-tool-choice-steps
architecture is a better match for small/local models (`ollama`) that need heavy
scaffolding to stay on track turn by turn.

## Async notes

- Tools are async. Playwright uses the **async** API directly; blocking work (docker
  subprocess wait, pymysql, the CLI subprocess) is wrapped in `asyncio.to_thread`.
- `orchestrator.run_flow_once` is sync and calls `asyncio.run(supervisor.run(...))`;
  `asyncio.run` copies the current context, so the `run_id` contextvar set by
  `new_run_context()` is visible to tool-level log lines. The TUI runs `run_flow_once`
  in a worker thread so the supervisor's loop never shares Textual's own loop.

## Observability

JSONL structured logging (stdout + rotating `logs/bee_bug_hunter.jsonl`, `LOG_FILE`/
`LOG_LEVEL` in `.env`), every line carrying `run_id` via `new_run_context()`. grep one run:

```bash
grep '"run_id": "<id>"' logs/bee_bug_hunter.jsonl | jq .
```

MySQL logging is query text + row count + timing only — row contents are never logged
(may hold customer/PII data).
