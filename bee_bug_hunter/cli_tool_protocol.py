"""Shared text-bridge tool-calling protocol for CLI-backed ChatModels
(claude_cli_llm.py, copilot_cli_llm.py): both shell out to a coding-agent CLI
with its own native tool use disabled, and both need the same hand-rolled
{"tool": ..., "args": {...}} / {"final_answer": ...} JSON convention so
BeeAI's own agent loop can drive tool calls instead of the CLI's. This module
holds the CLI-agnostic half of that bridge -- prompt text, JSON extraction,
and message flattening -- so neither backend duplicates it.
"""
import json
import re

from beeai_framework.backend.message import MessageToolCallContent, MessageToolResultContent
from beeai_framework.backend.utils import parse_broken_json

CODE_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

TOOL_PROTOCOL_INSTRUCTIONS = """
You have access to tools. To call one, respond with ONLY this JSON (no other text):
{{"tool": "<tool_name>", "args": {{...}}}}

When you have the final answer and don't need any more tools, respond with ONLY this JSON:
{{"final_answer": "<your answer>"}}

Available tools:
{tool_descriptions}
""".strip()

# A model is sometimes forced (tool_choice="required" or a specific Tool) to guarantee
# it takes an action this step. Left unhandled, the model often just answers in prose
# since the base protocol above always offers a final_answer escape hatch -- BeeAI's own
# Retryable then catches the resulting ChatModelToolCallError and re-asks with a
# corrective nudge, which works but costs a wasted CLI subprocess call (real latency) per
# miss. Stating the constraint up front avoids that.
#
# When BeeAI says "required" it usually *includes* its final_answer tool in the allowed
# list (final_answer_as_tool forces the final answer to arrive as a tool call too), so
# "required" does not mean "you may not finish" -- it means "your reply must be a tool
# call, and finishing is done by calling the final_answer tool". A previous version of
# this instruction forbade the final_answer format outright, which punished the model for
# legitimately being done: raw_preview logging (claude_cli_llm.py) showed the misses were
# well-formed {"final_answer": ...} replies, i.e. the model wanting to finish and the
# protocol giving it no sanctioned way to say so. Hence two variants, picked by whether
# final_answer is actually allowed.
REQUIRED_TOOL_CHOICE_INSTRUCTIONS = """
IMPORTANT: A tool call is required this turn. You MUST respond with ONLY the
{{"tool": "<tool_name>", "args": {{...}}}} JSON for one of the tools listed above.
Do NOT respond with plain text and do NOT use the final_answer format this turn.
""".strip()

REQUIRED_TOOL_CHOICE_WITH_FINAL_ANSWER_INSTRUCTIONS = """
IMPORTANT: A tool call is required this turn. You MUST respond with ONLY the
{{"tool": "<tool_name>", "args": {{...}}}} JSON for one of the tools listed above.
Do NOT respond with plain text. If you are done and want to give your final answer,
do it AS A TOOL CALL: {{"tool": "final_answer", "args": {{"response": "<your answer>"}}}}
""".strip()

FORCED_TOOL_INSTRUCTIONS = """
IMPORTANT: You MUST call the '{tool_name}' tool this turn. Respond with ONLY this JSON
(no other text): {{"tool": "{tool_name}", "args": {{...}}}}
""".strip()


def find_balanced_json_objects(text: str) -> list[str]:
    """Scans for every top-level {...} span via brace-depth counting (string/escape
    aware), instead of a regex spanning first-'{' to last-'}' -- that greedy-regex
    approach swallows the whole response into one unparseable blob whenever the
    model's surrounding prose happens to quote braces of its own (e.g. an HTTP
    error body), which was silently producing 'no tool call' failures."""
    spans = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append(text[start:i + 1])
    return spans


def extract_json_object(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = CODE_FENCE_PATTERN.search(text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Several brace spans can be in play at once -- the real tool-call JSON plus
    # prose that happens to quote its own (possibly also-valid-JSON) braces, e.g.
    # an HTTP error body. Parsing "first span that merely parses" is not enough:
    # prose braces are often valid JSON in their own right. Require the object to
    # actually look like our protocol (has a "tool" or "final_answer" key) before
    # accepting it; only fall back to "first parseable" if nothing matches the
    # protocol shape.
    parsed_candidates = []
    for candidate in find_balanced_json_objects(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and ("tool" in parsed or "final_answer" in parsed):
            return parsed
        parsed_candidates.append(parsed)
    if parsed_candidates:
        return parsed_candidates[0]

    # Last resort: genuinely malformed JSON (not just extra surrounding prose) --
    # e.g. a long markdown-heavy final_answer response with an unescaped quote or
    # raw newline breaking strict json.loads on every balanced-brace candidate
    # above. Use beeai_framework's own repair (json_repair, the same library its
    # runner uses for the identical problem) rather than hand-rolling another parser.
    repaired = parse_broken_json(text, fallback=None)
    if isinstance(repaired, dict) and ("tool" in repaired or "final_answer" in repaired):
        return repaired
    return None


def describe_tools(tools) -> str:
    if not tools:
        return "(none)"
    lines = []
    for t in tools:
        try:
            schema = json.dumps(t.input_schema.model_json_schema())
        except Exception:
            schema = "{}"
        lines.append(f"- {t.name}: {t.description}\n  parameters schema: {schema}")
    return "\n".join(lines)


def flatten_messages(messages) -> tuple[str, str]:
    """Returns (system_prompt, conversation_prompt) — CLI backends here take one
    system string and one user string per invocation, so the given slice of
    message history is flattened into role-prefixed text. Callers pass only the
    messages new since the last invocation once a session is underway, not the
    full history every time."""
    system_parts = []
    convo_parts = []
    for m in messages:
        role = getattr(m, "role", "user")
        role = role.value if hasattr(role, "value") else str(role)
        if role == "system":
            system_parts.append(m.text)
            continue
        chunks = []
        for c in m.content:
            if isinstance(c, MessageToolResultContent):
                chunks.append(f"Tool '{c.tool_name}' result:\n{c.result}")
            elif isinstance(c, MessageToolCallContent):
                chunks.append(f'{{"tool": "{c.tool_name}", "args": {c.args}}}')
            else:
                chunks.append(getattr(c, "text", ""))
        convo_parts.append(f"{role.upper()}: " + "\n".join(chunks))
    return "\n\n".join(system_parts), "\n\n".join(convo_parts)
