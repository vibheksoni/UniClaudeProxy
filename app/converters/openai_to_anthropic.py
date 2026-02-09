import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Optional

debug_logger = logging.getLogger("anyclaude.debug")


def _fc_to_toolu(fc_id: str) -> str:
    """Convert OpenAI Responses API function call ID to Anthropic tool ID.

    Args:
        fc_id: str - OpenAI function call ID (e.g. 'fc_abc123').

    Returns:
        str - Anthropic tool_use ID with 'toolu_' prefix.
    """
    if fc_id.startswith("toolu_"):
        return fc_id
    if fc_id.startswith("fc_"):
        return f"toolu_{fc_id[3:]}"
    if fc_id.startswith("call_"):
        return f"toolu_{fc_id[5:]}"
    return f"toolu_{fc_id}"


def _generate_message_id() -> str:
    """Generate a unique Anthropic-style message ID.

    Returns:
        str - A message ID in the format 'msg_<uuid>'.
    """
    return f"msg_{uuid.uuid4().hex[:24]}"


def _generate_content_block_id() -> str:
    """Generate a unique content block ID for tool use blocks.

    Returns:
        str - A tool use ID in the format 'toolu_<uuid>'.
    """
    return f"toolu_{uuid.uuid4().hex[:24]}"


def _map_finish_reason_to_stop_reason(finish_reason: Optional[str]) -> str:
    """Map an OpenAI finish_reason to an Anthropic stop_reason.

    Args:
        finish_reason: Optional[str] - OpenAI's finish reason string.

    Returns:
        str - Anthropic-compatible stop reason.
    """
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "max_tokens": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "end_turn",
    }
    return mapping.get(finish_reason or "stop", "end_turn")


def _map_responses_status_to_stop_reason(status: Optional[str]) -> str:
    """Map an OpenAI Responses API status to an Anthropic stop_reason.

    Args:
        status: Optional[str] - OpenAI Responses API status string.

    Returns:
        str - Anthropic-compatible stop reason.
    """
    mapping = {
        "completed": "end_turn",
        "incomplete": "max_tokens",
        "failed": "end_turn",
        "cancelled": "end_turn",
    }
    return mapping.get(status or "completed", "end_turn")


def from_openai_chat_response(
    response_data: dict[str, Any],
    anthropic_model: str,
) -> dict[str, Any]:
    """Convert an OpenAI Chat Completions response to Anthropic Messages format.

    Args:
        response_data: dict[str, Any] - The raw OpenAI Chat Completions response.
        anthropic_model: str - The original Anthropic model name from the request.

    Returns:
        dict[str, Any] - Anthropic Messages API compatible response.
    """
    content_blocks: list[dict[str, Any]] = []
    finish_reason = "stop"

    choices = response_data.get("choices", [])
    if choices:
        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")

        text = message.get("content")
        if text:
            content_blocks.append({"type": "text", "text": text})

        tool_calls = message.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", {})
            try:
                args = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}

            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", _generate_content_block_id()),
                "name": func.get("name", ""),
                "input": args,
            })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    usage = response_data.get("usage", {})

    return {
        "id": _generate_message_id(),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": anthropic_model,
        "stop_reason": _map_finish_reason_to_stop_reason(finish_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def from_openai_responses_response(
    response_data: dict[str, Any],
    anthropic_model: str,
) -> dict[str, Any]:
    """Convert an OpenAI Responses API response to Anthropic Messages format.

    Args:
        response_data: dict[str, Any] - The raw OpenAI Responses API response.
        anthropic_model: str - The original Anthropic model name from the request.

    Returns:
        dict[str, Any] - Anthropic Messages API compatible response.
    """
    content_blocks: list[dict[str, Any]] = []
    status = response_data.get("status", "completed")

    output_items = response_data.get("output", [])
    for item in output_items:
        item_type = item.get("type", "")

        if item_type == "reasoning":
            summary_parts = item.get("summary", [])
            thinking_text = ""
            for sp in summary_parts:
                if sp.get("type") == "summary_text":
                    thinking_text += sp.get("text", "")
            if thinking_text:
                content_blocks.append({
                    "type": "thinking",
                    "thinking": thinking_text,
                })

        elif item_type == "message" and item.get("role") == "assistant":
            for part in item.get("content", []):
                part_type = part.get("type", "")
                if part_type == "output_text":
                    content_blocks.append({
                        "type": "text",
                        "text": part.get("text", ""),
                    })
                elif part_type == "refusal":
                    content_blocks.append({
                        "type": "text",
                        "text": part.get("refusal", ""),
                    })

        elif item_type == "function_call":
            try:
                args = json.loads(item.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}

            content_blocks.append({
                "type": "tool_use",
                "id": _fc_to_toolu(item.get("call_id", _generate_content_block_id())),
                "name": item.get("name", ""),
                "input": args,
            })

    has_tool_use = any(b.get("type") == "tool_use" for b in content_blocks)

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    stop_reason = _map_responses_status_to_stop_reason(status)
    if has_tool_use and stop_reason == "end_turn":
        stop_reason = "tool_use"

    usage = response_data.get("usage", {})

    return {
        "id": _generate_message_id(),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": anthropic_model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def _sse_event(event_type: str, data: dict[str, Any]) -> str:
    """Format a single SSE event string.

    Args:
        event_type: str - The SSE event name.
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
        input_tokens: int - Number of input tokens consumed.

    Returns:
        str - Formatted message_start SSE event.
    """
    return _sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": anthropic_model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": 1,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
    })


def _build_content_block_start_event(index: int, block_type: str = "text", **kwargs: Any) -> str:
    """Build the Anthropic content_block_start SSE event.

    Args:
        index: int - Content block index.
        block_type: str - Type of content block ("text" or "tool_use").
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
            "id": kwargs.get("tool_id", _generate_content_block_id()),
            "name": kwargs.get("name", ""),
            "input": {},
        }
    else:
        content_block = {"type": block_type}

    return _sse_event("content_block_start", {
        "type": "content_block_start",
        "index": index,
        "content_block": content_block,
    })


def _build_text_delta_event(index: int, text: str) -> str:
    """Build the Anthropic content_block_delta SSE event for text.

    Args:
        index: int - Content block index.
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
        index: int - Content block index.
        thinking: str - The thinking delta text.

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
        index: int - Content block index.
        partial_json: str - Partial JSON string for tool arguments.

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
        index: int - Content block index.

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
        stop_reason: str - The reason generation stopped.
        output_tokens: int - Total output tokens generated.

    Returns:
        str - Formatted message_delta SSE event.
    """
    return _sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
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


async def stream_openai_chat_to_anthropic(
    response_stream: AsyncIterator[bytes],
    anthropic_model: str,
) -> AsyncIterator[str]:
    """Convert a streaming OpenAI Chat Completions response to Anthropic SSE events.

    Args:
        response_stream: AsyncIterator[bytes] - Raw SSE byte stream from OpenAI.
        anthropic_model: str - The original Anthropic model name.

    Yields:
        str - Anthropic-formatted SSE event strings.
    """
    msg_id = _generate_message_id()
    next_index = 0
    text_block_index = -1
    tool_index_map: dict[int, int] = {}
    tool_args_buffer: dict[int, str] = {}
    started = False
    text_block_started = False
    output_tokens = 0
    finish_reason = "stop"

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

            if line == "data: [DONE]":
                break

            if not line.startswith("data: "):
                continue

            json_str = line[6:]
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                continue

            usage_data = data.get("usage")
            if usage_data:
                output_tokens = usage_data.get("completion_tokens", output_tokens)

            choices = data.get("choices", [])
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta", {})
            fr = choice.get("finish_reason")
            if fr:
                finish_reason = fr

            delta_content = delta.get("content")
            if delta_content:
                if not text_block_started:
                    text_block_index = next_index
                    next_index += 1
                    yield _build_content_block_start_event(text_block_index, "text")
                    text_block_started = True
                    started = True
                yield _build_text_delta_event(text_block_index, delta_content)

            tool_calls = delta.get("tool_calls", [])
            for tc in tool_calls:
                tc_index = tc.get("index", 0)

                if tc_index not in tool_index_map:
                    if text_block_started:
                        yield _build_content_block_stop_event(text_block_index)
                        text_block_started = False

                    tool_id = tc.get("id", _generate_content_block_id())
                    tool_name = tc.get("function", {}).get("name", "")
                    block_idx = next_index
                    next_index += 1
                    tool_index_map[tc_index] = block_idx
                    tool_args_buffer[tc_index] = ""

                    yield _build_content_block_start_event(
                        block_idx,
                        "tool_use",
                        tool_id=tool_id,
                        name=tool_name,
                    )
                    started = True

                args_delta = tc.get("function", {}).get("arguments", "")
                if args_delta:
                    tool_args_buffer[tc_index] += args_delta
                    yield _build_input_json_delta_event(
                        tool_index_map[tc_index],
                        args_delta,
                    )

    if text_block_started:
        yield _build_content_block_stop_event(text_block_index)

    if not started:
        yield _build_content_block_start_event(0, "text")
        yield _build_content_block_stop_event(0)

    for tc_idx in sorted(tool_index_map.keys()):
        yield _build_content_block_stop_event(tool_index_map[tc_idx])

    stop_reason = _map_finish_reason_to_stop_reason(finish_reason)
    yield _build_message_delta_event(stop_reason, output_tokens)
    yield _build_message_stop_event()


def _extract_shell_call_input(item: dict) -> str:
    """Extract command from a shell_call or local_shell_call output item as JSON string.

    Args:
        item: dict - The shell_call or local_shell_call output item.

    Returns:
        str - JSON string of the tool input for Claude Code's Bash tool.
    """
    action = item.get("action", {})
    if isinstance(action, dict):
        commands = action.get("command", action.get("commands", []))
        if isinstance(commands, list):
            command_str = " && ".join(commands) if commands else ""
        else:
            command_str = str(commands)
    else:
        command_str = str(action)
    return json.dumps({"command": command_str, "timeout": action.get("timeout_ms", 120000) if isinstance(action, dict) else 120000})


async def stream_openai_responses_to_anthropic(
    response_stream: AsyncIterator[bytes],
    anthropic_model: str,
    tool_mapping: dict[str, str] | None = None,
) -> AsyncIterator[str]:
    """Convert a streaming OpenAI Responses API response to Anthropic SSE events.

    Args:
        response_stream: AsyncIterator[bytes] - Raw SSE byte stream from OpenAI Responses API.
        anthropic_model: str - The original Anthropic model name.
        tool_mapping: dict[str, str] | None - Map upstream tool types to Claude Code tool names.

    Yields:
        str - Anthropic-formatted SSE event strings.
    """
    tool_mapping = tool_mapping or {}
    msg_id = _generate_message_id()
    content_index = 0
    text_block_started = False
    thinking_block_started = False
    thinking_item_id: str = ""
    started = False
    output_tokens = 0
    input_tokens = 0
    stop_reason = "end_turn"

    tool_blocks: dict[str, int] = {}
    tool_args_buffer: dict[str, str] = {}
    active_tool_item_ids: list[str] = []

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

            if line.startswith("event: "):
                continue

            if not line.startswith("data: "):
                continue

            json_str = line[6:]
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                continue

            event_type = data.get("type", "")
            debug_logger.info("  [RESPONSES] event_type=%s", event_type)

            if event_type == "response.output_text.delta":
                delta_text = data.get("delta", "")
                if delta_text:
                    if thinking_block_started:
                        yield _build_content_block_stop_event(content_index)
                        content_index += 1
                        thinking_block_started = False
                    if not text_block_started:
                        debug_logger.info("  -> EMIT content_block_start(text, idx=%d)", content_index)
                        yield _build_content_block_start_event(content_index, "text")
                        text_block_started = True
                        started = True
                    yield _build_text_delta_event(content_index, delta_text)

            elif event_type == "response.output_text.done":
                if text_block_started:
                    debug_logger.info("  -> EMIT content_block_stop(idx=%d)", content_index)
                    yield _build_content_block_stop_event(content_index)
                    content_index += 1
                    text_block_started = False

            elif event_type == "response.reasoning_summary_part.added":
                if not thinking_block_started:
                    debug_logger.info("  -> EMIT content_block_start(thinking, idx=%d)", content_index)
                    yield _build_content_block_start_event(content_index, "thinking")
                    thinking_block_started = True
                    thinking_item_id = data.get("item_id", "")
                    started = True

            elif event_type == "response.reasoning_summary_text.delta":
                delta_text = data.get("delta", "")
                if delta_text:
                    if not thinking_block_started:
                        debug_logger.info("  -> EMIT content_block_start(thinking, idx=%d) [from delta]", content_index)
                        yield _build_content_block_start_event(content_index, "thinking")
                        thinking_block_started = True
                        thinking_item_id = data.get("item_id", "")
                        started = True
                    yield _build_thinking_delta_event(content_index, delta_text)

            elif event_type in ("response.reasoning_summary_text.done", "response.reasoning_summary_part.done"):
                pass

            elif event_type == "response.function_call_arguments.delta":
                item_id = data.get("item_id", "")
                args_delta = data.get("delta", "")
                debug_logger.info("  [RESPONSES] func_call_args.delta item_id=%s delta=%s", item_id, args_delta[:100] if args_delta else "")

                if item_id not in tool_blocks:
                    debug_logger.info("  [RESPONSES] WARNING: item_id=%s not in tool_blocks! keys=%s", item_id, list(tool_blocks.keys()))
                elif args_delta:
                    tool_args_buffer.setdefault(item_id, "")
                    tool_args_buffer[item_id] += args_delta
                    yield _build_input_json_delta_event(tool_blocks[item_id], args_delta)

            elif event_type == "response.function_call_arguments.done":
                item_id = data.get("item_id", "")
                debug_logger.info("  [RESPONSES] func_call_args.done item_id=%s", item_id)
                if item_id in tool_blocks:
                    yield _build_content_block_stop_event(tool_blocks[item_id])

            elif event_type == "response.output_item.added":
                item = data.get("item", {})
                item_type = item.get("type", "")
                debug_logger.info("  [RESPONSES] output_item.added type=%s item=%s", item_type, json.dumps(item, default=str)[:300])

                if item_type == "function_call":
                    if thinking_block_started:
                        yield _build_content_block_stop_event(content_index)
                        content_index += 1
                        thinking_block_started = False
                    if text_block_started:
                        yield _build_content_block_stop_event(content_index)
                        content_index += 1
                        text_block_started = False

                    item_id = item.get("id", "")
                    call_id = _fc_to_toolu(item.get("call_id", _generate_content_block_id()))
                    func_name = item.get("name", "")

                    debug_logger.info("  -> EMIT content_block_start(tool_use, idx=%d, name=%s, call_id=%s, item_id=%s)", content_index, func_name, call_id, item_id)

                    tool_blocks[item_id] = content_index
                    tool_args_buffer[item_id] = ""
                    active_tool_item_ids.append(item_id)

                    yield _build_content_block_start_event(
                        content_index,
                        "tool_use",
                        tool_id=call_id,
                        name=func_name,
                    )
                    content_index += 1
                    started = True

                elif item_type in tool_mapping:
                    mapped_name = tool_mapping[item_type]
                    debug_logger.info("  [RESPONSES] TOOL MAPPING: %s -> %s", item_type, mapped_name)

                    if thinking_block_started:
                        yield _build_content_block_stop_event(content_index)
                        content_index += 1
                        thinking_block_started = False
                    if text_block_started:
                        yield _build_content_block_stop_event(content_index)
                        content_index += 1
                        text_block_started = False

                    item_id = item.get("id", "")
                    call_id = _fc_to_toolu(item.get("call_id", _generate_content_block_id()))
                    tool_input_json = _extract_shell_call_input(item)

                    debug_logger.info("  -> EMIT content_block_start(tool_use, idx=%d, name=%s, mapped from %s)", content_index, mapped_name, item_type)

                    tool_blocks[item_id] = content_index
                    active_tool_item_ids.append(item_id)

                    yield _build_content_block_start_event(
                        content_index,
                        "tool_use",
                        tool_id=call_id,
                        name=mapped_name,
                    )
                    yield _build_input_json_delta_event(content_index, tool_input_json)
                    yield _build_content_block_stop_event(content_index)
                    content_index += 1
                    started = True

            elif event_type in ("response.shell_call.completed", "response.local_shell_call.completed"):
                debug_logger.info("  [RESPONSES] shell call completed event=%s", event_type)

            elif event_type == "response.output_item.done":
                item = data.get("item", {})
                item_type = item.get("type", "")
                item_id = item.get("id", "")
                debug_logger.info("  [RESPONSES] output_item.done type=%s id=%s", item_type, item_id)

                if item_type == "reasoning" and thinking_block_started:
                    debug_logger.info("  -> EMIT content_block_stop(thinking, idx=%d)", content_index)
                    yield _build_content_block_stop_event(content_index)
                    content_index += 1
                    thinking_block_started = False

                if item_type in tool_mapping and item_id not in tool_blocks:
                    mapped_name = tool_mapping[item_type]
                    debug_logger.info("  [RESPONSES] LATE TOOL MAPPING: %s -> %s (from output_item.done)", item_type, mapped_name)

                    if text_block_started:
                        yield _build_content_block_stop_event(content_index)
                        content_index += 1
                        text_block_started = False

                    call_id = _fc_to_toolu(item.get("call_id", _generate_content_block_id()))
                    tool_input_json = _extract_shell_call_input(item)

                    tool_blocks[item_id] = content_index
                    active_tool_item_ids.append(item_id)

                    yield _build_content_block_start_event(
                        content_index,
                        "tool_use",
                        tool_id=call_id,
                        name=mapped_name,
                    )
                    yield _build_input_json_delta_event(content_index, tool_input_json)
                    yield _build_content_block_stop_event(content_index)
                    content_index += 1
                    started = True

            elif event_type == "response.completed":
                resp = data.get("response", {})
                usage = resp.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                status = resp.get("status", "completed")
                stop_reason = _map_responses_status_to_stop_reason(status)
                debug_logger.info("  [RESPONSES] completed status=%s stop_reason=%s tokens=%d/%d", status, stop_reason, input_tokens, output_tokens)

                output_items = resp.get("output", [])
                for oi in output_items:
                    oi_type = oi.get("type", "?")
                    debug_logger.info("  [RESPONSES] completed output item: type=%s", oi_type)

                    if oi_type in tool_mapping and oi.get("id", "") not in tool_blocks:
                        mapped_name = tool_mapping[oi_type]
                        oi_id = oi.get("id", "")
                        call_id = _fc_to_toolu(oi.get("call_id", _generate_content_block_id()))
                        tool_input_json = _extract_shell_call_input(oi)

                        debug_logger.info("  [RESPONSES] FINAL TOOL MAPPING: %s -> %s (from completed)", oi_type, mapped_name)

                        if text_block_started:
                            yield _build_content_block_stop_event(content_index)
                            content_index += 1
                            text_block_started = False

                        tool_blocks[oi_id] = content_index
                        yield _build_content_block_start_event(
                            content_index,
                            "tool_use",
                            tool_id=call_id,
                            name=mapped_name,
                        )
                        yield _build_input_json_delta_event(content_index, tool_input_json)
                        yield _build_content_block_stop_event(content_index)
                        content_index += 1
                        started = True

                has_tool_use = len(tool_blocks) > 0
                if has_tool_use and stop_reason == "end_turn":
                    stop_reason = "tool_use"

            elif event_type == "response.incomplete":
                stop_reason = "max_tokens"
                debug_logger.info("  [RESPONSES] incomplete -> max_tokens")

            elif event_type == "response.failed":
                stop_reason = "end_turn"
                debug_logger.info("  [RESPONSES] failed -> end_turn, data=%s", json.dumps(data, default=str)[:500])

    if text_block_started:
        yield _build_content_block_stop_event(content_index)
        content_index += 1

    if not started:
        yield _build_content_block_start_event(content_index, "text")
        yield _build_content_block_stop_event(content_index)

    yield _build_message_delta_event(stop_reason, output_tokens)
    yield _build_message_stop_event()
