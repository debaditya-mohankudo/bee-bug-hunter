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


def get_chat_model() -> ChatModel:
    provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).lower()

    if provider == "claude_cli":
        return ClaudeCLIChatModel(model=os.getenv("CLAUDE_CLI_MODEL", DEFAULT_CLAUDE_CLI_MODEL))
    if provider == "ollama":
        from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel

        return OllamaChatModel(
            os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            # Ollama-only context window size; other providers ignore/never see this.
            settings={"num_ctx": int(os.getenv("OLLAMA_NUM_CTX", DEFAULT_OLLAMA_NUM_CTX))},
        )
    if provider == "openai":
        return ChatModel.from_name(f"openai:{os.getenv('OPENAI_MODEL', DEFAULT_OPENAI_MODEL)}")
    if provider == "anthropic":
        return ChatModel.from_name(f"anthropic:{os.getenv('ANTHROPIC_MODEL', DEFAULT_ANTHROPIC_MODEL)}")

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
