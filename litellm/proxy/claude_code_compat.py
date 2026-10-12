"""Claude Code model-discovery compatibility helpers.

Claude Code's gateway discovery only keeps model IDs containing ``claude`` or
``anthropic``.  This module provides opt-in aliases for other LiteLLM models
without changing their normal public names or routing targets.
"""

import os
import re
from typing import Any, Mapping, Optional


CLAUDE_CODE_COMPAT_PREFIX = "claude-compat/"
_CLAUDE_CODE_USER_AGENT = re.compile(r"^claude-code/", re.IGNORECASE)


def claude_code_compat_enabled(general_settings: Optional[Mapping[str, Any]] = None) -> bool:
    """Return whether synthetic Claude Code discovery IDs are enabled."""
    env_value = os.getenv("LITELLM_CLAUDE_CODE_COMPAT_MODE", "")
    if env_value:
        return env_value.strip().lower() in {"1", "true", "yes", "on"}
    return bool((general_settings or {}).get("claude_code_compat_mode", False))


def is_claude_code_user_agent(user_agent: Optional[str]) -> bool:
    """Recognize Claude Code's discovery/request User-Agent."""
    return bool(user_agent and _CLAUDE_CODE_USER_AGENT.search(user_agent.strip()))


def to_claude_code_model_id(model_id: str) -> str:
    """Expose a model under an ID Claude Code's discovery filter accepts."""
    if model_id.startswith(CLAUDE_CODE_COMPAT_PREFIX):
        return model_id
    # Claude Code filters discovery IDs to those containing either token.
    # Preserve native Claude/Anthropic IDs and expose every other route under
    # the compatibility prefix, including Fusion and wildcard routes.
    if re.search(r"(claude|anthropic)", model_id, re.IGNORECASE):
        return model_id
    return f"{CLAUDE_CODE_COMPAT_PREFIX}{model_id}"


def claude_code_model_metadata(model_id: str, llm_router: Any = None) -> dict[str, str]:
    """Return picker-friendly metadata for a Claude Code discovery entry.

    LiteLLM's OpenAI-compatible ``/v1/models`` serializer omits deployment
    ``display_name`` and ``description`` fields.  Claude Code accepts both,
    so expose the configured values for compatibility responses and provide
    readable fallbacks for aliases that do not define them.
    """
    configured: Mapping[str, Any] = {}
    if llm_router is not None:
        try:
            deployments = llm_router.get_model_list(model_name=model_id) or []
        except Exception:
            deployments = []
        if deployments:
            candidate = deployments[0].get("model_info") or {}
            if isinstance(candidate, Mapping):
                configured = candidate

    display_name = configured.get("display_name")
    description = configured.get("description")
    if isinstance(display_name, str):
        result = {"display_name": display_name}
        if isinstance(description, str):
            result["description"] = description
        return result

    normalized = model_id.removeprefix("chatgpt/")
    if model_id.lower().startswith("chatgpt/"):
        label = normalized.replace("-", " ").replace(".", ".")
        return {
            "display_name": display_name if isinstance(display_name, str) else f"{label} (ChatGPT)",
            "description": description if isinstance(description, str) else f"{label} via the ChatGPT gateway.",
        }

    if model_id == "deepseek-flash-max":
        fallback_name = "DeepSeek V4.1 Flash Max"
        fallback_description = "DeepSeek V4.1 Flash with maximum reasoning effort."
    elif model_id == "deepseek-flash-no-thinking":
        fallback_name = "DeepSeek V4.1 Flash (No Thinking)"
        fallback_description = "DeepSeek V4.1 Flash with reasoning disabled for faster responses."
    elif model_id.startswith("deepseek/"):
        model_suffix = model_id.removeprefix("deepseek/")
        fallback_name = "DeepSeek " + model_suffix.replace("-", " ").title()
        fallback_description = f"{fallback_name} routed through the gateway."
    else:
        # No configured override: keep the source ID readable and, crucially,
        # do not expose the synthetic ``claude-compat/`` prefix as its label.
        fallback_name = model_id
        fallback_description = ""

    return {
        "display_name": display_name if isinstance(display_name, str) else fallback_name,
        "description": description if isinstance(description, str) else fallback_description,
    }


def from_claude_code_model_id(model_id: str) -> str:
    """Map a synthetic discovery ID back to the configured LiteLLM model."""
    if model_id.startswith(CLAUDE_CODE_COMPAT_PREFIX):
        return model_id[len(CLAUDE_CODE_COMPAT_PREFIX) :]
    return model_id
