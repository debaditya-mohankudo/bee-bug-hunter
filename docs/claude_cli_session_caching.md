# claude_cli: session-based caching and tool_choice reliability

This documents a debugging/design arc on `bee_bug_hunter/claude_cli_llm.py` (the
`claude_cli` LLM provider, which shells to `claude -p` and bridges BeeAI tool
calling through a hand-rolled JSON-in-text protocol since `--tools none`
disables the CLI's own tool use).

## The problem

A single `example_login_api` investigation (5 workers + manager) made **25
`claude -p` subprocess calls**, each one a cold, sessionless CLI invocation that
re-flattened and resent the *entire* growing conversation history as a brand-new
prompt every time. Two consequences:

1. **No prompt-cache reuse.** BeeAI's `cache_control_injection_points` machinery
   only wires through the `litellm` adapter — never implemented in the
   hand-rolled `ClaudeCLIChatModel`. A fresh, sessionless subprocess every call
   has no identical/growing prefix for Anthropic's cache to hit against, so
   every call was full-price.
2. **A ~36% `missed_required_tool_call` rate.** `RequirementAgent` forces
   `tool_choice="required"` on nearly every step; the model frequently replied
   in prose (or the wrong JSON shape) instead, triggering BeeAI's own retry path
   — a wasted subprocess call and, worse, a full re-send of the propagated
   conversation context per miss.

This led to briefly removing `claude_cli` entirely in favor of `ollama`
(`qwen3:4b`) as the default provider — a heavily-scaffolded, forced-tool-choice
framework like `RequirementAgent` is a better match for small/local models that
need step-by-step guidance than for a frontier model with native tool calling
and large cacheable context. `ollama` remains the `.env` default. `claude_cli`
was then rebuilt properly as a documented, working alternative — this doc
covers that rebuild.

## Fix 1: root+fork session topology

`claude -p` supports real session persistence: `--session-id <uuid>` mints a
session, `-r/--resume <uuid>` continues it, `--fork-session` (used with
`--resume`) branches a **new**, independent session off an existing one,
inheriting its context up to the fork point without polluting either the
original or sibling branches. Verified empirically (see git history) before
building on it: a root session taught the word BANANA; forking it produced a
new session_id that still knew BANANA; adding PINEAPPLE to the fork afterward
did not leak back into the root.

This maps directly onto why per-role session isolation matters here: each
role's (worker's) system prompt carries a *different* tool schema
(`_describe_tools(input.tools)`). A single session shared across all roles
would leave stale tool schemas from other roles sitting in the history,
confusing the model about which tools are actually callable a given turn.
Forking from a common root instead gives every role the same shared starting
context (cache-friendly) while keeping each role's own tool-specific
continuation isolated.

**Topology**: one root session per flow investigation, seeded with only
generic, role-agnostic framing (flow name, containers — no tool schemas,
`_ensure_root_session`). Each of the 6 roles (Investigation Manager + 5
workers) forks its own session off that root on its first-ever call
(`ClaudeCLIChatModel.for_role`), then reuses that fork (`--resume <its_id>`) on
every later call. All forks are direct children of the root — never
fork-of-a-fork.

**Isolation is per-`(flow_name, role)`**, not global — `ClaudeCLIChatModel._instances`
is keyed by `(flow_key, role_key)`. `HandoffTool` clones the target agent
(and its llm) on every delegation; `ClaudeCLIChatModel.clone()` deliberately
returns `self` rather than a fresh instance, so repeat delegations to the same
role resume that role's one ongoing session instead of going cold. This is
only correct because delegations are sequential (`allow_parallel_tool_calls=False`
for every provider) — concurrent delegations to the same role would race on
`_sent_message_count`/`_session_started`.

**Persistence survives process restarts.** The session_id must never be lost —
an in-memory `_instances` dict alone resets on every fresh `--once` invocation
or crash. `.claude_cli_sessions.json` (gitignored) persists
`{flow_name: {root, roles: {role: session_id}}}` to disk; `for_role()` always
checks it before minting, and a known id is always resumed, never re-forked.
Verified by simulating a fresh process (clearing `_instances`) and confirming
`for_role()` correctly resumed the persisted session and the model still
recalled prior context.

**Deltas, not full history.** `_create()` only flattens
`input.messages[self._sent_message_count:]` once a session is underway — the
CLI session already holds everything sent on prior turns.

Measured effect: real investigations went from **269–280s** (flat, no-session
design) to **147–191s** (root+fork) at similar call counts — the win is
per-call cost/latency from cache reuse, not fewer calls.

## Fix 2: the `missed_required_tool_call` root cause

Diagnosed by adding `raw_preview` (the actual CLI response text) to the
`missed_required_tool_call` log line — previously only the failure *outcome*
was logged, not the response that caused it, so the actual cause was
unknown. The evidence was decisive: the JSON was **not malformed**. The model
was replying with well-formed `{"final_answer": "..."}` on turns where
`tool_choice="required"` demanded a tool call.

Root cause: BeeAI's "required" usually still permits finishing — via calling
the `final_answer` *tool* (`final_answer_as_tool=True` forces the final answer
to arrive as a tool call too) — but the old prompt instruction told the model
outright: *"do NOT use the final_answer format this turn"*. That's false
whenever `final_answer` genuinely was allowed, so the model was being punished
for legitimately deciding it was done, with no sanctioned way to say so.

Two changes:

1. **Honest instructions** — `final_answer_allowed = any(t.name == "final_answer"
   for t in (input.tools or []))`. When true, swap in
   `_REQUIRED_TOOL_CHOICE_WITH_FINAL_ANSWER_INSTRUCTIONS`, which sanctions
   finishing *as a tool call*: `{"tool": "final_answer", "args": {"response": ...}}`.
   The strict "no final_answer at all" wording only applies when that tool
   truly isn't offered.
2. **Local rescue** — even so, if the model still slips and replies bare
   `{"final_answer": ...}` on an allowed turn, `_create()` rewrites it in place
   into the equivalent tool call before the "did the model call a tool?" check,
   logged as `claude_cli_final_answer_converted`. Costs nothing extra — one CLI
   call in, one valid tool call out, reshaped in Python.

Effect: 35% (7/20) → 19% (3/16) miss rate.

## Fix 3: genuinely malformed JSON

The residual 3 misses (all the Bug Analyst, whose responses are the longest)
had `raw_preview` showing the model *had* replied in the now-correct
`{"tool": "final_answer", ...}` shape — yet still failed, meaning the full JSON
itself wasn't parsing (a long markdown-heavy `response` string with an
unescaped quote or raw newline breaking strict `json.loads`). Fixed by adding
`beeai_framework.backend.utils.parse_broken_json` (the same `json_repair`
library BeeAI's own runner uses for this identical problem) as a last-resort
fallback in `_extract_json_object`, after the balanced-brace scanner.

Effect: 19% (3/16) → **0% (0/17)**.

## Incidental fix: `_extract_json_object`'s extraction bug

Along the way, the original greedy `\{.*\}` regex (spanning first-`{` to
last-`}` in the whole response) was replaced with a proper balanced-brace
scanner (`_find_balanced_json_objects`, string/escape-aware) that requires a
candidate to actually have a `"tool"` or `"final_answer"` key before accepting
it — the greedy regex, and later a naive "first parseable" heuristic, both
failed on responses where surrounding prose happened to quote its own
(possibly valid-JSON) braces, e.g. an HTTP error body appearing in the model's
reasoning text.

## Net result

| Metric | Before | After |
|---|---|---|
| Miss rate | ~36% | 0% |
| Elapsed (real run) | 269–280s | 147–191s |
| Prompt cache reuse | none (cold every call) | per-role session, growing prefix |
| Session survives process restart | no | yes (disk-persisted) |

Also unrelated but adjacent work from this session: `ConditionalRequirement`
structural enforcement on the Investigation Manager (workers can't be delegated
to before `Docker Log Capturer` has run) and the Bug Analyst (`check_anomalies`
forced at step 1), plus a real pytest suite (`tests/test_agents.py`,
`tests/test_manager.py`) exercising both without needing a live LLM.
