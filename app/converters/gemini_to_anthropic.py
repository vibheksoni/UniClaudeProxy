import json
import logging
import re
import time
import uuid
from typing import Any, AsyncIterator
from urllib.parse import quote, unquote

debug_logger = logging.getLogger("anyclaude.debug")

THOUGHT_SIG_SEP = "__ts__"

ToolParamIndex = dict[str, list[str]]


def build_tool_param_index(tools: list[Any] | None) -> ToolParamIndex:
    """Build an index of tool_name -> list of property names from Anthropic tool defs.

    Args:
        tools: list[Any] | None - Anthropic tool definitions.

    Returns:
        ToolParamIndex - Mapping of tool name to its parameter property names.
    """
    index: ToolParamIndex = {}
    if not tools:
        return index
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name", "")
            schema = tool.get("input_schema", {})
        else:
            name = getattr(tool, "name", "")
            schema = getattr(tool, "input_schema", {})
            if hasattr(schema, "model_dump"):
                schema = schema.model_dump()
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        index[name] = list(props.keys())
    return index


_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _camel_to_snake(name: str) -> str:
    """Convert camelCase or PascalCase to snake_case.

    Args:
        name: str - The camelCase/PascalCase string.

    Returns:
        str - The snake_case equivalent.
    """
    return _CAMEL_RE.sub("_", name).lower()


def _fix_tool_args(
    tool_name: str,
    args: dict[str, Any],
    param_index: ToolParamIndex,
) -> dict[str, Any]:
    """Remap model-sent args to match the tool's expected parameter names.

    Handles camelCase->snake_case conversion and 1:1 fuzzy matching
    for any remaining unmatched parameters.

    Args:
        tool_name: str - The function/tool name.
        args: dict[str, Any] - The args the model sent.
        param_index: ToolParamIndex - Tool param index from build_tool_param_index.

    Returns:
        dict[str, Any] - Corrected args dict.
    """
    if not param_index or tool_name not in param_index or not args:
        return args

    expected_params = param_index[tool_name]
    if not expected_params:
        return args

    expected_set = set(expected_params)
    snake_lookup = {_camel_to_snake(p): p for p in expected_params}

    fixed: dict[str, Any] = {}
    unmatched_args: dict[str, Any] = {}

    for sent_name, value in args.items():
        if sent_name in expected_set:
            fixed[sent_name] = value
        elif _camel_to_snake(sent_name) in snake_lookup:
            real_name = snake_lookup[_camel_to_snake(sent_name)]
            if real_name not in fixed:
                debug_logger.info("PARAM FIX: %s: %s -> %s (camel)", tool_name, sent_name, real_name)
                fixed[real_name] = value
            else:
                unmatched_args[sent_name] = value
        else:
            unmatched_args[sent_name] = value

    if unmatched_args:
        remaining_expected = [p for p in expected_params if p not in fixed]
        if len(unmatched_args) == len(remaining_expected):
            for (sent_name, value), expected_name in zip(unmatched_args.items(), remaining_expected):
                debug_logger.info("PARAM FIX: %s: %s -> %s (positional)", tool_name, sent_name, expected_name)
                fixed[expected_name] = value
        else:
            fixed.update(unmatched_args)

    return fixed


def _generate_message_id() -> str:
    """Generate a unique Anthropic-style message ID.

    Returns:
        str - A message ID in the format 'msg_<uuid>'.
    """
    return f"msg_{uuid.uuid4().hex[:24]}"


def _generate_tool_id() -> str:
    """Generate a unique Anthropic-style tool use ID.

    Returns:
        str - A tool use ID in the format 'toolu_<uuid>'.
    """
    return f"toolu_{uuid.uuid4().hex[:24]}"


def _map_finish_reason(finish_reason: str | None) -> str:
    """Map a Gemini finishReason to an Anthropic stop_reason.

    Args:
        finish_reason: str | None - Gemini finish reason string.

    Returns:
        str - Anthropic-compatible stop reason.
    """
    mapping = {
        "STOP": "end_turn",
        "MAX_TOKENS": "max_tokens",
        "SAFETY": "end_turn",
        "RECITATION": "end_turn",
        "OTHER": "end_turn",
        "TOOL_CODE": "tool_use",
    }
    return mapping.get(finish_reason or "STOP", "end_turn")


def from_gemini_response(
    response_data: dict[str, Any],
    anthropic_model: str,
    param_index: ToolParamIndex | None = None,
) -> dict[str, Any]:
    """Convert a non-streaming Gemini response to Anthropic format.

    Args:
        response_data: dict[str, Any] - Raw Gemini generateContent response.
        anthropic_model: str - The original Anthropic model name.

    Returns:
        dict[str, Any] - Anthropic Messages API compatible response.
    """
    msg_id = _generate_message_id()
    content: list[dict[str, Any]] = []
    stop_reason = "end_turn"
    has_tool_use = False

    candidates = response_data.get("candidates", [])
    if candidates:
        candidate = candidates[0]
        finish_reason = candidate.get("finishReason")
        stop_reason = _map_finish_reason(finish_reason)

        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            if "text" in part and not part.get("thought"):
                content.append({
                    "type": "text",
                    "text": part["text"],
                })
            elif part.get("thought") and "text" in part:
                content.append({
                    "type": "thinking",
                    "thinking": part["text"],
                })
            elif "functionCall" in part:
                has_tool_use = True
                fc = part["functionCall"]
                tool_id = _generate_tool_id()
                sig = part.get("thoughtSignature", "")
                if sig:
                    tool_id = f"{tool_id}{THOUGHT_SIG_SEP}{quote(sig, safe='')}"
                fc_name = fc.get("name", "")
                fc_args = fc.get("args", {})
                if param_index:
                    fc_args = _fix_tool_args(fc_name, fc_args, param_index)
                content.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": fc_name,
                    "input": fc_args,
                })

    if not content:
        content.append({"type": "text", "text": ""})

    if has_tool_use:
        stop_reason = "tool_use"

    usage_meta = response_data.get("usageMetadata", {})
    input_tokens = usage_meta.get("promptTokenCount", 0)
    output_tokens = usage_meta.get("candidatesTokenCount", 0)

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": anthropic_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def _sse_event(event_type: str, data: dict[str, Any]) -> str:
    """Format a single SSE event string.

    Args:
        event_type: str - The SSE event type.
        data: dict[str, Any] - The event data payload.

    Returns:
        str - Formatted SSE event string.
    """
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _build_message_start_event(anthropic_model: str, msg_id: str, input_tokens: int = 0) -> str:
    """Build the Anthropic message_start SSE event.

    Args:
        anthropic_model: str - The Anthropic model name.
        msg_id: str - The message ID.
        input_tokens: int - Number of input tokens.

    Returns:
        str - Formatted message_start SSE event.
    """
    return _sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": anthropic_model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    })


def _build_content_block_start_event(index: int, block_type: str = "text", **kwargs: Any) -> str:
    """Build the Anthropic content_block_start SSE event.

    Args:
        index: int - The content block index.
        block_type: str - The block type ('text', 'tool_use', 'thinking').
        **kwargs: Any - Additional fields for the content block.

    Returns:
        str - Formatted content_block_start SSE event.
    """
    if block_type == "text":
        content_block = {"type": "text", "text": ""}
    elif block_type == "thinking":
        content_block = {"type": "thinking", "thinking": ""}
    elif block_type == "tool_use":
        content_block = {
            "type": "tool_use",
            "id": kwargs.get("tool_id", _generate_tool_id()),
            "name": kwargs.get("name", ""),
            "input": {},
        }
    else:
        content_block = {"type": "text", "text": ""}

    return _sse_event("content_block_start", {
        "type": "content_block_start",
        "index": index,
        "content_block": content_block,
    })


def _build_text_delta_event(index: int, text: str) -> str:
    """Build the Anthropic content_block_delta SSE event for text.

    Args:
        index: int - The content block index.
        text: str - The text delta.

    Returns:
        str - Formatted content_block_delta SSE event.
    """
    return _sse_event("content_block_delta", {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    })


def _build_thinking_delta_event(index: int, thinking: str) -> str:
    """Build the Anthropic content_block_delta SSE event for thinking.

    Args:
        index: int - The content block index.
        thinking: str - The thinking text delta.

    Returns:
        str - Formatted content_block_delta SSE event.
    """
    return _sse_event("content_block_delta", {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "thinking_delta", "thinking": thinking},
    })


def _build_input_json_delta_event(index: int, partial_json: str) -> str:
    """Build the Anthropic content_block_delta SSE event for tool input JSON.

    Args:
        index: int - The content block index.
        partial_json: str - The partial JSON string.

    Returns:
        str - Formatted content_block_delta SSE event.
    """
    return _sse_event("content_block_delta", {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": partial_json},
    })


def _build_content_block_stop_event(index: int) -> str:
    """Build the Anthropic content_block_stop SSE event.

    Args:
        index: int - The content block index.

    Returns:
        str - Formatted content_block_stop SSE event.
    """
    return _sse_event("content_block_stop", {
        "type": "content_block_stop",
        "index": index,
    })


def _build_message_delta_event(stop_reason: str, output_tokens: int = 0) -> str:
    """Build the Anthropic message_delta SSE event.

    Args:
        stop_reason: str - The stop reason string.
        output_tokens: int - Number of output tokens.

    Returns:
        str - Formatted message_delta SSE event.
    """
    return _sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"type": "message_delta", "stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })


def _build_message_stop_event() -> str:
    """Build the Anthropic message_stop SSE event.

    Returns:
        str - Formatted message_stop SSE event.
    """
    return _sse_event("message_stop", {"type": "message_stop"})


def _build_ping_event() -> str:
    """Build the Anthropic ping SSE event.

    Returns:
        str - Formatted ping SSE event.
    """
    return _sse_event("ping", {"type": "ping"})


async def stream_gemini_to_anthropic(
    response_stream: AsyncIterator[bytes],
    anthropic_model: str,
    param_index: ToolParamIndex | None = None,
) -> AsyncIterator[str]:
    """Convert a streaming Gemini response to Anthropic SSE events.

    Args:
        response_stream: AsyncIterator[bytes] - Raw SSE byte stream from Gemini.
        anthropic_model: str - The original Anthropic model name.

    Yields:
        str - Anthropic-formatted SSE event strings.
    """
    msg_id = _generate_message_id()
    next_index = 0
    text_block_index = -1
    text_block_started = False
    thinking_block_index = -1
    thinking_block_started = False
    started = False
    output_tokens = 0
    input_tokens = 0
    finish_reason = "STOP"
    has_tool_use = False

    yield _build_message_start_event(anthropic_model, msg_id)
    yield _build_ping_event()

    buffer = ""

    async for chunk in response_stream:
        text = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        buffer += text

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()

            if not line:
                continue

            if not line.startswith("data: "):
                continue

            json_str = line[6:]
            if json_str == "[DONE]":
                break

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                debug_logger.warning("Failed to parse Gemini SSE: %s", json_str[:200])
                continue

            usage_meta = data.get("usageMetadata")
            if usage_meta:
                output_tokens = usage_meta.get("candidatesTokenCount", output_tokens)
                input_tokens = usage_meta.get("promptTokenCount", input_tokens)

            candidates = data.get("candidates", [])
            if not candidates:
                continue

            candidate = candidates[0]
            fr = candidate.get("finishReason")
            if fr:
                finish_reason = fr

            parts = candidate.get("content", {}).get("parts", [])
            for part in parts:
                if part.get("thought") and "text" in part:
                    if text_block_started:
                        yield _build_content_block_stop_event(text_block_index)
                        text_block_started = False

                    if not thinking_block_started:
                        thinking_block_index = next_index
                        next_index += 1
                        yield _build_content_block_start_event(thinking_block_index, "thinking")
                        thinking_block_started = True
                        started = True

                    yield _build_thinking_delta_event(thinking_block_index, part["text"])

                elif "text" in part and not part.get("thought"):
                    if thinking_block_started:
                        yield _build_content_block_stop_event(thinking_block_index)
                        thinking_block_started = False

                    if not text_block_started:
                        text_block_index = next_index
                        next_index += 1
                        yield _build_content_block_start_event(text_block_index, "text")
                        text_block_started = True
                        started = True

                    yield _build_text_delta_event(text_block_index, part["text"])

                elif "functionCall" in part:
                    has_tool_use = True

                    if thinking_block_started:
                        yield _build_content_block_stop_event(thinking_block_index)
                        thinking_block_started = False
                    if text_block_started:
                        yield _build_content_block_stop_event(text_block_index)
                        text_block_started = False

                    fc = part["functionCall"]
                    tool_id = _generate_tool_id()
                    sig = part.get("thoughtSignature", "")
                    if sig:
                        tool_id = f"{tool_id}{THOUGHT_SIG_SEP}{quote(sig, safe='')}"
                    tool_name = fc.get("name", "")
                    tool_args = fc.get("args", {})
                    if param_index:
                        tool_args = _fix_tool_args(tool_name, tool_args, param_index)

                    block_idx = next_index
                    next_index += 1

                    yield _build_content_block_start_event(
                        block_idx,
                        "tool_use",
                        tool_id=tool_id,
                        name=tool_name,
                    )
                    started = True

                    args_json = json.dumps(tool_args)
                    yield _build_input_json_delta_event(block_idx, args_json)
                    yield _build_content_block_stop_event(block_idx)

    if thinking_block_started:
        yield _build_content_block_stop_event(thinking_block_index)

    if text_block_started:
        yield _build_content_block_stop_event(text_block_index)

    if not started:
        yield _build_content_block_start_event(0, "text")
        yield _build_content_block_stop_event(0)

    stop_reason = "tool_use" if has_tool_use else _map_finish_reason(finish_reason)
    yield _build_message_delta_event(stop_reason, output_tokens)
    yield _build_message_stop_event()
