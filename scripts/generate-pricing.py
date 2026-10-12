#!/usr/bin/env python3
"""
Generate a LiteLLM-compatible model_prices_and_context_window.json.

Fetches live pricing and capabilities from OpenRouter API, preserves rich
metadata from the upstream LiteLLM pricing file, and adds hardcoded entries
for DeepSeek (official pricing) and ChatGPT (subscription, $0 rates).

Usage:
  uv run scripts/generate-pricing.py -o custom-pricing.json
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
OR_API = "https://openrouter.ai/api/v1/models"

DEEPSEEK_PRICING = {
    "deepseek-v4-flash": {
        "input_cost_per_token": 0.00000014,
        "cache_creation_input_token_cost": 0.00000014,
        "cache_read_input_token_cost": 0.0000000028,
        "output_cost_per_token": 0.00000028,
        "max_input_tokens": 1000000,
        "max_output_tokens": 384000,
        "max_tokens": 1000000,
        "source": "https://api-docs.deepseek.com/quick_start/pricing",
    },
    "deepseek-v4-pro": {
        "input_cost_per_token": 0.000000435,
        "cache_creation_input_token_cost": 0.000000435,
        "cache_read_input_token_cost": 0.000000003625,
        "output_cost_per_token": 0.00000087,
        "max_input_tokens": 1000000,
        "max_output_tokens": 384000,
        "max_tokens": 1000000,
        "source": "https://api-docs.deepseek.com/quick_start/pricing",
    },
}

CHATGPT_MODELS = {
    "chatgpt/gpt-5.5": {
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "chatgpt",
        "mode": "responses",
        "max_input_tokens": 200000,
        "max_output_tokens": 32000,
        "max_tokens": 200000,
        "supports_vision": True,
        "supports_function_calling": True,
        "supports_tool_choice": True,
        "supports_reasoning": True,
        "supports_prompt_caching": False,
        "supports_system_messages": True,
        "supports_parallel_function_calling": True,
        "supported_modalities": ["text", "image"],
    },
    "chatgpt/gpt-5.4": {
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "chatgpt",
        "mode": "responses",
        "max_input_tokens": 200000,
        "max_output_tokens": 32000,
        "max_tokens": 200000,
        "supports_vision": True,
        "supports_function_calling": True,
        "supports_tool_choice": True,
        "supports_system_messages": True,
        "supports_parallel_function_calling": True,
        "supported_modalities": ["text", "image"],
    },
    "chatgpt/gpt-5.4-mini": {
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "chatgpt",
        "mode": "responses",
        "max_input_tokens": 200000,
        "max_output_tokens": 16000,
        "max_tokens": 200000,
        "supports_vision": True,
        "supports_function_calling": True,
        "supports_tool_choice": True,
        "supports_system_messages": True,
        "supports_parallel_function_calling": True,
        "supported_modalities": ["text", "image"],
    },
}

# Clone OpenRouter → native provider key mappings.
# Entry for "openrouter/deepseek/deepseek-v4-pro" gets cloned as
# "deepseek/deepseek-v4-pro" with litellm_provider="deepseek".
_PROVIDER_MAP = {
    "deepseek": "deepseek",
}

COST_FIELDS_TO_UPDATE = frozenset({
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
    "input_cost_per_token_cache_hit",
})

_PARAM_TO_CAP = {
    "tools": ("supports_function_calling", "supports_tool_choice", "supports_parallel_function_calling"),
    "tool_choice": ("supports_tool_choice",),
    "reasoning": ("supports_reasoning",),
    "structured_outputs": ("supports_response_schema",),
    "response_format": ("supports_response_schema",),
    "include_reasoning": ("supports_reasoning",),
    "prompt_caching": ("supports_prompt_caching",),
    "image": ("supports_vision",),
    "stop": ("supports_system_messages",),
    "temperature": ("supports_system_messages",),
}

_MODALITY_TO_CAP = {
    "image": ("supports_vision", "supports_image_input"),
    "text": (),
    "audio": ("supports_audio_input",),
    "video": ("supports_video_input",),
    "file": ("supports_pdf_input",),
}

_or_cache = None


def _safe_float(s, default=0.0):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _or_models_raw():
    global _or_cache
    if _or_cache is None:
        print(f"Fetching OpenRouter models from {OR_API}...", file=sys.stderr)
        req = urllib.request.Request(OR_API)
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=30) as resp:
            _or_cache = json.loads(resp.read().decode())["data"]
        print(f"  → {len(_or_cache)} models", file=sys.stderr)
    return _or_cache


def _build_litellm_entry(or_model: dict) -> dict:
    model_id = or_model["id"]
    pricing = or_model.get("pricing", {})
    arch = or_model.get("architecture", {})
    params = set(or_model.get("supported_parameters", []) or [])
    limits = or_model.get("per_request_limits") or {}
    top = or_model.get("top_provider", {})

    entry = {
        "input_cost_per_token": _safe_float(pricing.get("prompt")),
        "output_cost_per_token": _safe_float(pricing.get("completion")),
        "litellm_provider": "openrouter",
        "mode": "chat",
        "max_input_tokens": or_model.get("context_length", 0),
        "max_output_tokens": top.get("max_completion_tokens", 0),
        "max_tokens": top.get("max_completion_tokens") or or_model.get("context_length", 0),
        "source": f"https://openrouter.ai/{model_id}",
        "supported_endpoints": ["/v1/chat/completions", "/v1/completions"],
    }

    input_mods = arch.get("input_modalities", [])
    output_mods = arch.get("output_modalities", [])
    if input_mods:
        entry["supported_modalities"] = list(input_mods)
    if output_mods:
        entry["supported_output_modalities"] = list(output_mods)

    for param in params:
        for cap in _PARAM_TO_CAP.get(param, ()):
            entry[cap] = True
    for mod in input_mods:
        for cap in _MODALITY_TO_CAP.get(mod, ()):
            entry[cap] = True

    if limits.get("rpm") or limits.get("requests_per_interval"):
        entry["rpm"] = limits.get("rpm") or limits.get("requests_per_interval")
    if limits.get("tpm") or limits.get("tokens_per_interval"):
        entry["tpm"] = limits.get("tpm") or limits.get("tokens_per_interval")

    if pricing.get("input_cache_read"):
        entry["cache_read_input_token_cost"] = _safe_float(pricing["input_cache_read"])
    if pricing.get("input_cache_write"):
        entry["cache_creation_input_token_cost"] = _safe_float(pricing["input_cache_write"])

    return entry


def download_base_pricing():
    print(f"Downloading base pricing from {LITELLM_PRICING_URL}...", file=sys.stderr)
    req = urllib.request.Request(LITELLM_PRICING_URL)
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def extract_openrouter_entries(base):
    result = {}
    for key, value in base.items():
        if key.startswith("openrouter/"):
            result[key] = value
    print(f"  → {len(result)} openrouter entries in base", file=sys.stderr)
    return result


def update_pricing(base_entries, live_models):
    updated = 0
    new_models = 0
    for model in live_models:
        key = f"openrouter/{model['id']}"
        live_entry = _build_litellm_entry(model)
        if key in base_entries:
            for field in COST_FIELDS_TO_UPDATE:
                if field in live_entry:
                    base_entries[key][field] = live_entry[field]
            updated += 1
        else:
            base_entries[key] = live_entry
            new_models += 1
    print(f"  Updated pricing for {updated} models, added {new_models} new", file=sys.stderr)
    return base_entries


def apply_deepseek_overrides(entries):
    for model_name, pricing in DEEPSEEK_PRICING.items():
        key = f"openrouter/deepseek/{model_name}"
        if key in entries:
            for field, value in pricing.items():
                entries[key][field] = value
            print(
                f"  DS override: {key} "
                f"(${pricing['input_cost_per_token']}/input)",
                file=sys.stderr,
            )
        else:
            entries[key] = dict(
                litellm_provider="openrouter",
                mode="chat",
                supports_function_calling=True,
                supports_tool_choice=True,
                supports_reasoning=True,
                supports_assistant_prefill=True,
                **pricing,
            )
    return entries


def clone_native_provider_entries(entries, base):
    """Clone openrouter/ entries to native provider keys for direct API use.
    E.g. openrouter/deepseek/deepseek-v4-pro → deepseek/deepseek-v4-pro"""
    count = 0
    for or_key, entry in list(entries.items()):
        if not or_key.startswith("openrouter/"):
            continue
        path = or_key.split("/", 1)[1]  # "deepseek/deepseek-v4-pro"

        # Determine native provider
        native_provider = None
        for prefix, provider in _PROVIDER_MAP.items():
            if path.startswith(f"{prefix}/"):
                native_provider = provider
                break
        if not native_provider:
            continue

        native_key = f"{native_provider}/{path.split('/', 1)[1]}"

        # Prefer base entry for native key (richer metadata), fall back to clone
        if native_key in base:
            native_entry = dict(base[native_key])
            for field in COST_FIELDS_TO_UPDATE:
                if field in entry and field in native_entry:
                    native_entry[field] = entry[field]
        else:
            native_entry = dict(entry)
            native_entry["litellm_provider"] = native_provider
            if "source" in native_entry and "openrouter.ai" in native_entry["source"]:
                native_entry.pop("source", None)

        entries[native_key] = native_entry
        count += 1

    print(f"  Cloned {count} native provider entries", file=sys.stderr)
    return entries


def load_overlay(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded overlay: {len(data)} entries from {path}", file=sys.stderr)
    return data


def main():
    p = argparse.ArgumentParser(description="Generate LiteLLM pricing JSON")
    p.add_argument("-o", "--output", default="custom-pricing.json")
    p.add_argument("--overlay", default=None)
    p.add_argument("--base-url", default=LITELLM_PRICING_URL)
    args = p.parse_args()
    overlay_path = args.overlay or os.environ.get(
        "PRICING_OVERLAY",
        str(Path(__file__).resolve().parent / "pricing-overlay.json"),
    )

    base = download_base_pricing()

    for k in list(base.keys()):
        if k.startswith("openrouter/") or k.startswith("deepseek/"):
            del base[k]
    print(f"  Stripped openrouter/deepseek, remaining: {len(base)} entries", file=sys.stderr)

    live = _or_models_raw()
    for model in live:
        key = f"openrouter/{model['id']}"
        base[key] = _build_litellm_entry(model)

    apply_deepseek_overrides(base)
    clone_native_provider_entries(base, base)
    base.update(CHATGPT_MODELS)

    overlay = load_overlay(overlay_path)
    base.update(overlay)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(base, f, indent=2)
    print(f"\nWrote {len(base)} entries to {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
