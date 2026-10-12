"""
Handler for the Anthropic v1/messages -> OpenAI Responses API path.

Used when the target model is an OpenAI or Azure model.
"""

import json
from typing import Any, AsyncIterator, Coroutine, Dict, List, Optional, Union

import litellm
from litellm.types.llms.anthropic import AnthropicMessagesRequest
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.llms.openai import ResponsesAPIResponse

from .streaming_iterator import AnthropicResponsesStreamWrapper
from .transformation import LiteLLMAnthropicToResponsesAPIAdapter
from ..messages.fake_stream_iterator import FakeAnthropicMessagesStreamIterator


def _tool_search_kind(tool: Dict[str, Any]) -> Optional[str]:
    tool_type = str(tool.get("type", ""))
    if tool_type.startswith("tool_search_tool_regex"):
        return "regex"
    if tool_type.startswith("tool_search_tool_bm25"):
        return "bm25"
    return None


def _response_item_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _native_tool_search_events(
    response: ResponsesAPIResponse, search_tool_type: str
) -> List[Dict[str, Any]]:
    """Translate OpenAI Responses tool-search items to Anthropic content blocks."""
    events: List[Dict[str, Any]] = []
    search_name = (
        "tool_search_tool_regex"
        if "regex" in search_tool_type
        else "tool_search_tool_bm25"
    )
    search_id = "srvtoolu_openai_tool_search"
    for item in response.output:
        item_type = _response_item_value(item, "type")
        if item_type not in {"tool_search_call", "tool_search_output"}:
            continue
        call_id = _response_item_value(item, "call_id") or _response_item_value(
            item, "id"
        )
        if item_type == "tool_search_call":
            search_id = call_id or search_id
            arguments = _response_item_value(item, "arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"query": arguments}
            events.append(
                {
                    "type": "server_tool_use",
                    "id": search_id,
                    "name": search_name,
                    "input": arguments if isinstance(arguments, dict) else {},
                }
            )
        else:
            loaded_tools = _response_item_value(item, "tools", []) or []
            references = []
            for tool in loaded_tools:
                name = _response_item_value(tool, "name")
                function = _response_item_value(tool, "function")
                if not name and function is not None:
                    name = _response_item_value(function, "name")
                if name:
                    references.append(
                        {"type": "tool_reference", "tool_name": name}
                    )
            events.append(
                {
                    "type": "tool_search_tool_result",
                    "tool_use_id": search_id,
                    "content": {
                        "type": "tool_search_tool_search_result",
                        "tool_references": references,
                    },
                }
            )
    return events

_ADAPTER = LiteLLMAnthropicToResponsesAPIAdapter()


def _build_responses_kwargs(
    *,
    max_tokens: int,
    messages: List[Dict],
    model: str,
    context_management: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    output_config: Optional[Dict] = None,
    stop_sequences: Optional[List[str]] = None,
    stream: Optional[bool] = False,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    thinking: Optional[Dict] = None,
    tool_choice: Optional[Dict] = None,
    tools: Optional[List[Dict]] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    output_format: Optional[Dict] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the kwargs dict to pass directly to litellm.responses() / litellm.aresponses().
    """
    # Build a typed AnthropicMessagesRequest for the adapter
    request_data: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if context_management:
        request_data["context_management"] = context_management
    if output_config:
        request_data["output_config"] = output_config
    if metadata:
        request_data["metadata"] = metadata
    if system:
        request_data["system"] = system
    if temperature is not None:
        request_data["temperature"] = temperature
    if thinking:
        request_data["thinking"] = thinking
    if tool_choice:
        request_data["tool_choice"] = tool_choice
    if tools:
        request_data["tools"] = tools
    if top_p is not None:
        request_data["top_p"] = top_p
    if output_format:
        request_data["output_format"] = output_format

    anthropic_request = AnthropicMessagesRequest(**request_data)  # type: ignore[typeddict-item]
    responses_kwargs = _ADAPTER.translate_request(anthropic_request)

    # Normalize reasoning effort based on model capabilities
    # (e.g. "max" → "xhigh"/"high", "minimal" → "low" if unsupported)
    reasoning = responses_kwargs.get("reasoning")
    if isinstance(reasoning, dict) and "effort" in reasoning:
        from litellm.llms.anthropic.experimental_pass_through.utils import (
            normalize_reasoning_effort_value,
        )

        effort = reasoning["effort"]
        normalized = normalize_reasoning_effort_value(
            effort,
            model=model,
            custom_llm_provider=(extra_kwargs or {}).get("custom_llm_provider"),
        )
        if normalized != effort:
            responses_kwargs["reasoning"] = {**reasoning, "effort": normalized}

    if stream:
        responses_kwargs["stream"] = True

    # Forward litellm-specific kwargs (api_key, api_base, logging obj, etc.)
    excluded = {"anthropic_messages"}
    for key, value in (extra_kwargs or {}).items():
        if key == "litellm_logging_obj" and value is not None:
            from litellm.litellm_core_utils.litellm_logging import (
                Logging as LiteLLMLoggingObject,
            )
            from litellm.types.utils import CallTypes

            if isinstance(value, LiteLLMLoggingObject):
                # Reclassify as acompletion so the success handler doesn't try to
                # validate the Responses API event as an AnthropicResponse.
                # (Mirrors the pattern used in LiteLLMMessagesToCompletionTransformationHandler.)
                setattr(value, "call_type", CallTypes.anthropic_messages.value)
            responses_kwargs[key] = value
        elif key not in excluded and key not in responses_kwargs and value is not None:
            responses_kwargs[key] = value

    return responses_kwargs


class LiteLLMMessagesToResponsesAPIHandler:
    """
    Handles Anthropic /v1/messages requests for OpenAI / Azure models by
    calling litellm.responses() / litellm.aresponses() directly and translating
    the response back to Anthropic format.
    """

    @staticmethod
    async def async_anthropic_messages_handler(
        max_tokens: int,
        messages: List[Dict],
        model: str,
        context_management: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        output_config: Optional[Dict] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking: Optional[Dict] = None,
        tool_choice: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        output_format: Optional[Dict] = None,
        **kwargs,
    ) -> Union[AnthropicMessagesResponse, AsyncIterator]:
        tool_search = next(
            (tool for tool in (tools or []) if _tool_search_kind(tool)), None
        )
        search_kind = _tool_search_kind(tool_search) if tool_search else None
        search_events: List[Dict[str, Any]] = []

        responses_kwargs = _build_responses_kwargs(
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            context_management=context_management,
            metadata=metadata,
            output_config=output_config,
            stop_sequences=stop_sequences,
            stream=stream,
            system=system,
            temperature=temperature,
            thinking=thinking,
            tool_choice=tool_choice,
            tools=tools,
            top_k=top_k,
            top_p=top_p,
            output_format=output_format,
            extra_kwargs=kwargs,
        )

        request_stream = bool(stream)
        if search_kind:
            # Native OpenAI hosted tool search emits the search call/output and
            # the resulting model output in one Responses result.  Buffer it so
            # we can translate those items into Anthropic server-tool blocks.
            responses_kwargs["stream"] = False
        result = await litellm.aresponses(**responses_kwargs)

        if search_kind and isinstance(result, ResponsesAPIResponse):
            search_events = _native_tool_search_events(
                result, tool_search.get("type", "") if tool_search else ""
            )

        if request_stream and search_events and isinstance(result, ResponsesAPIResponse):
            response = _ADAPTER.translate_response(result)
            response["content"] = search_events + list(response.get("content", []))
            return FakeAnthropicMessagesStreamIterator(response)

        if stream:
            wrapper = AnthropicResponsesStreamWrapper(
                responses_stream=result, model=model
            )
            return wrapper.async_anthropic_sse_wrapper()

        if not isinstance(result, ResponsesAPIResponse):
            raise ValueError(f"Expected ResponsesAPIResponse, got {type(result)}")

        return _ADAPTER.translate_response(result)

    @staticmethod
    def anthropic_messages_handler(
        max_tokens: int,
        messages: List[Dict],
        model: str,
        context_management: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        output_config: Optional[Dict] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking: Optional[Dict] = None,
        tool_choice: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        output_format: Optional[Dict] = None,
        _is_async: bool = False,
        **kwargs,
    ) -> Union[
        AnthropicMessagesResponse,
        AsyncIterator[Any],
        Coroutine[Any, Any, Union[AnthropicMessagesResponse, AsyncIterator[Any]]],
    ]:
        if _is_async:
            return (
                LiteLLMMessagesToResponsesAPIHandler.async_anthropic_messages_handler(
                    max_tokens=max_tokens,
                    messages=messages,
                    model=model,
                    context_management=context_management,
                    metadata=metadata,
                    output_config=output_config,
                    stop_sequences=stop_sequences,
                    stream=stream,
                    system=system,
                    temperature=temperature,
                    thinking=thinking,
                    tool_choice=tool_choice,
                    tools=tools,
                    top_k=top_k,
                    top_p=top_p,
                    output_format=output_format,
                    **kwargs,
                )
            )

        # Sync path
        responses_kwargs = _build_responses_kwargs(
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            context_management=context_management,
            metadata=metadata,
            output_config=output_config,
            stop_sequences=stop_sequences,
            stream=stream,
            system=system,
            temperature=temperature,
            thinking=thinking,
            tool_choice=tool_choice,
            tools=tools,
            top_k=top_k,
            top_p=top_p,
            output_format=output_format,
            extra_kwargs=kwargs,
        )

        result = litellm.responses(**responses_kwargs)

        if stream:
            wrapper = AnthropicResponsesStreamWrapper(
                responses_stream=result, model=model
            )
            return wrapper.async_anthropic_sse_wrapper()

        if not isinstance(result, ResponsesAPIResponse):
            raise ValueError(f"Expected ResponsesAPIResponse, got {type(result)}")

        return _ADAPTER.translate_response(result)
