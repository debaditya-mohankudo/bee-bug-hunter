"""LLM provider abstraction — swap backend via LLM_PROVIDER env var."""
import os

from beeai_framework.backend import ChatModel

from bee_bug_hunter.claude_cli_llm import ClaudeCLIChatModel
from bee_bug_hunter.config import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_CLAUDE_CLI_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_NUM_CTX,
    DEFAULT_OPENAI_MODEL,
)


# False is ChatModel's own default (beeai_framework.backend.chat.ChatModel.__init__),
# but every model here sets it explicitly rather than relying on that default holding:
# the Investigation Manager's sequential-delegation design (see manager.py's prompt and
# DockerLogCaptureTool's `--since 5m` window comment) assumes the model requests one
# handoff tool call per turn. If a model were allowed to request two in the same turn,
# beeai_framework.agents._utils.run_tools() executes them concurrently via asyncio.gather
# -- e.g. Docker Log Capturer starting before API Flow Runner has produced anything to
# capture. Async tool execution itself doesn't cause that (async just avoids blocking the
# event loop for one tool call); this flag is the actual guarantee.
_SEQUENTIAL_TOOL_CALLS = {"allow_parallel_tool_calls": False}


def get_chat_model() -> ChatModel:
    provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).lower()

    if provider == "claude_cli":
        # Belt-and-suspenders: ClaudeCLIChatModel's own _create() only ever parses one
        # {"tool": ...} JSON object per CLI call, so it's structurally incapable of
        # emitting >1 tool call per turn regardless of this flag -- set for consistency
        # with the other providers, not because this backend actually needs it.
        return ClaudeCLIChatModel(
            model=os.getenv("CLAUDE_CLI_MODEL", DEFAULT_CLAUDE_CLI_MODEL), **_SEQUENTIAL_TOOL_CALLS,
        )
    if provider == "ollama":
        from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel

        return OllamaChatModel(
            os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            # Ollama-only context window size; other providers ignore/never see this.
            settings={"num_ctx": int(os.getenv("OLLAMA_NUM_CTX", DEFAULT_OLLAMA_NUM_CTX))},
            **_SEQUENTIAL_TOOL_CALLS,
        )
    if provider == "openai":
        return ChatModel.from_name(
            f"openai:{os.getenv('OPENAI_MODEL', DEFAULT_OPENAI_MODEL)}", **_SEQUENTIAL_TOOL_CALLS,
        )
    if provider == "anthropic":
        return ChatModel.from_name(
            f"anthropic:{os.getenv('ANTHROPIC_MODEL', DEFAULT_ANTHROPIC_MODEL)}", **_SEQUENTIAL_TOOL_CALLS,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
