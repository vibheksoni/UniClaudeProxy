import json
import logging
import re
import uuid
from typing import Any, AsyncIterator

from app.react.parser import _generate_tool_id, has_tool_call, parse_tool_calls, split_text_and_tool_calls
from app.react.prompt import build_react_system, format_observation_xml, format_tool_call_xml

logger = logging.getLogger("anyclaude")
debug_logger = logging.getLogger("anyclaude.debug")

STOP_SEQ = "</tool_call>"


def transform_request(body: dict[str, Any]) -> dict[str, Any]:
    """Transform a raw Anthropic request body for ReAct-style tool calling.

    Strips native tools, injects XML tool descriptions into the system prompt,
    converts tool_use/tool_result history blocks to XML text, and adds
    </tool_call> as a stop sequence. System replacements are applied upstream
    in main.py before this function is called.

    Args:
        body: dict[str, Any] - The raw Anthropic request body.

    Returns:
        dict[str, Any] - Modified request body with ReAct transformations applied.
    """
    tools = body.get("tools")
    if not tools:
        return body

    body = dict(body)

    original_system = body.get("system")
    react_system = build_react_system(tools, original_system)
    body["system"] = react_system

    body.pop("tools", None)
    body.pop("tool_choice", None)

    stop_seqs = list(body.get("stop_sequences") or [])
    if STOP_SEQ not in stop_seqs:
        stop_seqs.append(STOP_SEQ)
    body["stop_sequences"] = stop_seqs

    body["messages"] = _convert_message_history(body.get("messages", []))

    debug_logger.info("REACT: transformed request — tools injected into system prompt, %d messages converted", len(body["messages"]))
    return body


def _convert_message_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert message history: tool_use blocks → XML text, tool_result blocks → observation XML.

    Args:
        messages: list[dict[str, Any]] - Original Anthropic messages.

    Returns:
        list[dict[str, Any]] - Converted messages with XML text instead of native tool blocks.
    """
    converted: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            converted.append({"role": role, "content": str(content)})
            continue

        if role == "assistant":
            converted.append(_convert_assistant_message(msg))
        elif role == "user":
            converted.append(_convert_user_message(msg))
        else:
            converted.append(msg)

    return converted


def _convert_assistant_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert assistant message: keep text blocks, convert tool_use to XML text.

    Args:
        msg: dict[str, Any] - An assistant message with content blocks.

    Returns:
        dict[str, Any] - Converted assistant message.
    """
    content = msg.get("content", [])
    if isinstance(content, str):
        return msg

    text_parts: list[str] = []

    for block in content:
        if not isinstance(block, dict):
            block = block.model_dump() if hasattr(block, "model_dump") else dict(block)

        btype = block.get("type", "")

        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            name = block.get("name", "")
            args = block.get("input", {})
            text_parts.append(format_tool_call_xml(name, args))
        elif btype == "thinking":
            pass

    combined = "\n\n".join(p for p in text_parts if p)
    combined = _strip_think_tags(combined)
    return {"role": "assistant", "content": combined or ""}


def _convert_user_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert user message: keep text/image blocks, convert tool_result to observation XML.

    Args:
        msg: dict[str, Any] - A user message with content blocks.

    Returns:
        dict[str, Any] - Converted user message.
    """
    content = msg.get("content", [])
    if isinstance(content, str):
        return msg

    text_parts: list[str] = []
    other_blocks: list[dict[str, Any]] = []

    for block in content:
        if not isinstance(block, dict):
            block = block.model_dump() if hasattr(block, "model_dump") else dict(block)

        btype = block.get("type", "")

        if btype == "tool_result":
            tool_content = block.get("content", "")
            if isinstance(tool_content, list):
                result_text = " ".join(
                    b.get("text", "") for b in tool_content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            elif isinstance(tool_content, str):
                result_text = tool_content
            else:
                result_text = str(tool_content)

            is_error = block.get("is_error", False)
            tool_use_id = block.get("tool_use_id", "?")
            debug_logger.info("REACT OBS: tool_use_id=%s is_error=%s content=%s", tool_use_id, is_error, result_text[:300])
            if is_error:
                result_text = f"ERROR: {result_text}"

            text_parts.append(format_observation_xml(result_text))
        elif btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "image":
            other_blocks.append(block)

    combined_text = "\n\n".join(p for p in text_parts if p)

    if other_blocks and combined_text:
        new_content: list[dict[str, Any]] = [{"type": "text", "text": combined_text}]
        new_content.extend(other_blocks)
        return {"role": "user", "content": new_content}

    return {"role": "user", "content": combined_text or ""}


def transform_response(response: dict[str, Any], anthropic_model: str) -> dict[str, Any]:
    """Parse XML tool calls from a non-streaming Anthropic response.

    If the response text contains <tool_call> XML, splits it into
    text + tool_use content blocks and sets stop_reason to "tool_use".

    Args:
        response: dict[str, Any] - The Anthropic-formatted response.
        anthropic_model: str - The original Anthropic model name.

    Returns:
        dict[str, Any] - Modified response with tool_use blocks if tool calls found.
    """
    content = response.get("content", [])
    if not content:
        return response

    full_text = ""
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            full_text += block.get("text", "")

    full_text = _strip_think_tags(full_text)

    if not full_text.strip() and not has_tool_call(full_text):
        response = dict(response)
        response["content"] = [{"type": "text", "text": ""}]
        return response

    if not has_tool_call(full_text):
        response = dict(response)
        response["content"] = [{"type": "text", "text": full_text.strip()}]
        return response

    debug_logger.info("REACT: parsing tool calls from response text")
    new_blocks = split_text_and_tool_calls(full_text)

    has_tool = any(b.get("type") == "tool_use" for b in new_blocks)
    response = dict(response)
    response["content"] = new_blocks
    if has_tool:
        response["stop_reason"] = "tool_use"

    return response


_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_SENTINEL = "<tool_call"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_PARTIAL_RE = re.compile(r"<think>.*$", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Strip all <think>...</think> blocks from text.

    Handles both complete and unclosed think blocks.

    Args:
        text: str - Raw text potentially containing think tags.

    Returns:
        str - Text with think blocks removed.
    """
    text = _THINK_RE.sub("", text)
    text = _THINK_PARTIAL_RE.sub("", text)
    return text


def _process_raw(raw: str, in_think: bool) -> tuple[str, str, str, bool]:
    """Process raw text in real-time, separating clean text from think content.

    Handles partial tags at the end of the buffer by holding them back.

    Args:
        raw: str - Unprocessed raw text buffer.
        in_think: bool - Whether we're currently inside a think block.

    Returns:
        tuple[str, str, str, bool] - (clean_text, think_text, remaining_tail, in_think_state).
    """
    clean = ""
    think = ""
    pos = 0

    while pos < len(raw):
        if in_think:
            close_idx = raw.find(_THINK_CLOSE, pos)
            if close_idx >= 0:
                think += raw[pos:close_idx]
                pos = close_idx + len(_THINK_CLOSE)
                in_think = False
            else:
                for i in range(min(len(_THINK_CLOSE) - 1, len(raw) - pos), 0, -1):
                    if raw[pos:].endswith(_THINK_CLOSE[:i]):
                        think += raw[pos:len(raw) - i]
                        return clean, think, raw[len(raw) - i:], True
                think += raw[pos:]
                return clean, think, "", True
        else:
            open_idx = raw.find(_THINK_OPEN, pos)
            if open_idx >= 0:
                clean += raw[pos:open_idx]
                pos = open_idx + len(_THINK_OPEN)
                in_think = True
            else:
                remainder = raw[pos:]
                for i in range(min(len(_THINK_OPEN) - 1, len(remainder)), 0, -1):
                    if remainder.endswith(_THINK_OPEN[:i]):
                        clean += remainder[:-i]
                        return clean, think, remainder[-i:], False
                clean += remainder
                return clean, think, "", False

    return clean, think, "", in_think


async def transform_stream(
    upstream_events: AsyncIterator[str],
    anthropic_model: str,
) -> AsyncIterator[str]:
    """Stream text live to the client, buffering only when <tool_call> is detected.

    Text content is forwarded in real-time. When a <tool_call tag appears
    in the accumulating text, the stream switches to buffer mode, collects
    the complete XML block, parses it, and emits a proper tool_use block.

    Args:
        upstream_events: AsyncIterator[str] - The upstream Anthropic SSE event strings.
        anthropic_model: str - The original Anthropic model name.

    Yields:
        str - Anthropic SSE event strings with live text and parsed tool_use blocks.
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    input_tokens = 0
    output_tokens = 0
    original_stop_reason = "end_turn"

    accum = ""
    flushed_up_to = 0
    block_idx = 0
    text_block_open = False
    think_block_open = False
    header_sent = False
    found_tool = False
    in_think = False
    raw_tail = ""

    async for event_str in upstream_events:
        for line in event_str.strip().split("\n"):
            if not line.startswith("data: "):
                continue
            try:
                evt = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            evt_type = evt.get("type", "")

            if evt_type == "message_start":
                msg = evt.get("message", {})
                message_id = msg.get("id", message_id)
                usage = msg.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)

                if not header_sent:
                    header_sent = True
                    yield _sse("message_start", {
                        "type": "message_start",
                        "message": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": anthropic_model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                        },
                    })

            elif evt_type == "content_block_delta":
                delta = evt.get("delta", {})
                if delta.get("type") == "text_delta":
                    raw_tail += delta.get("text", "")
                    clean, think_text, raw_tail, in_think = _process_raw(raw_tail, in_think)

                    if think_text:
                        if text_block_open:
                            yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_idx})
                            block_idx += 1
                            text_block_open = False
                        if not think_block_open:
                            think_block_open = True
                            yield _sse("content_block_start", {
                                "type": "content_block_start",
                                "index": block_idx,
                                "content_block": {"type": "thinking", "thinking": ""},
                            })
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {"type": "thinking_delta", "thinking": think_text},
                        })

                    if clean and not in_think:
                        if think_block_open:
                            yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_idx})
                            block_idx += 1
                            think_block_open = False

                    accum += clean

                    safe_end = _safe_flush_point(accum, flushed_up_to)
                    if safe_end > flushed_up_to:
                        chunk = accum[flushed_up_to:safe_end]
                        if not text_block_open:
                            text_block_open = True
                            yield _sse("content_block_start", {
                                "type": "content_block_start",
                                "index": block_idx,
                                "content_block": {"type": "text", "text": ""},
                            })
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {"type": "text_delta", "text": chunk},
                        })
                        flushed_up_to = safe_end

            elif evt_type == "message_delta":
                delta = evt.get("delta", {})
                original_stop_reason = delta.get("stop_reason", original_stop_reason)
                usage = evt.get("usage", {})
                output_tokens = usage.get("output_tokens", output_tokens)

    if think_block_open:
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_idx})
        block_idx += 1
        think_block_open = False

    if not header_sent:
        header_sent = True
        yield _sse("message_start", {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": anthropic_model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        })

    remaining = accum[flushed_up_to:]
    tool_calls = parse_tool_calls(remaining) if _TOOL_CALL_SENTINEL in remaining else []

    if tool_calls:
        pre_text = remaining[:tool_calls[0]["start"]].strip()
        if pre_text:
            if not text_block_open:
                text_block_open = True
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": block_idx,
                    "content_block": {"type": "text", "text": ""},
                })
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": block_idx,
                "delta": {"type": "text_delta", "text": pre_text},
            })

        if text_block_open:
            yield _sse("content_block_stop", {
                "type": "content_block_stop",
                "index": block_idx,
            })
            block_idx += 1
            text_block_open = False

        for tc in tool_calls:
            found_tool = True
            tool_id = _generate_tool_id()

            yield _sse("content_block_start", {
                "type": "content_block_start",
                "index": block_idx,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tc["name"],
                },
            })
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": block_idx,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(tc["arguments"])},
            })
            yield _sse("content_block_stop", {
                "type": "content_block_stop",
                "index": block_idx,
            })
            block_idx += 1

        post_text = remaining[tool_calls[-1]["end"]:].strip()
        if post_text:
            yield _sse("content_block_start", {
                "type": "content_block_start",
                "index": block_idx,
                "content_block": {"type": "text", "text": ""},
            })
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": block_idx,
                "delta": {"type": "text_delta", "text": post_text},
            })
            yield _sse("content_block_stop", {
                "type": "content_block_stop",
                "index": block_idx,
            })
            block_idx += 1

    else:
        if remaining.strip():
            if not text_block_open:
                text_block_open = True
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": block_idx,
                    "content_block": {"type": "text", "text": ""},
                })
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": block_idx,
                "delta": {"type": "text_delta", "text": remaining.strip()},
            })

        if text_block_open:
            yield _sse("content_block_stop", {
                "type": "content_block_stop",
                "index": block_idx,
            })
            block_idx += 1
            text_block_open = False

    if block_idx == 0:
        yield _sse("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        yield _sse("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": ""},
        })
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": 0,
        })

    stop_reason = "tool_use" if found_tool else original_stop_reason

    debug_logger.info("REACT STREAM: %d blocks, tool_use=%s, stop=%s", block_idx, found_tool, stop_reason)

    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })

    yield _sse("message_stop", {"type": "message_stop"})


def _safe_flush_point(accum: str, flushed_up_to: int) -> int:
    """Determine the safe point up to which text can be flushed live.

    Flushes all text up to but not including any partial <tool_call tag.
    If no sentinel is detected in the unflushed region, flushes everything.

    Args:
        accum: str - The full accumulated text so far.
        flushed_up_to: int - The index up to which text has already been flushed.

    Returns:
        int - The safe index up to which text can be flushed.
    """
    unflushed = accum[flushed_up_to:]

    idx = unflushed.find(_TOOL_CALL_SENTINEL)
    if idx >= 0:
        return flushed_up_to + idx

    for i in range(min(len(_TOOL_CALL_SENTINEL) - 1, len(unflushed)), 0, -1):
        if unflushed.endswith(_TOOL_CALL_SENTINEL[:i]):
            return len(accum) - i

    return len(accum)


def _sse(event_type: str, data: dict[str, Any]) -> str:
    """Format a single SSE event string.

    Args:
        event_type: str - The SSE event type.
        data: dict[str, Any] - The event data payload.

    Returns:
        str - Formatted SSE event string.
    """
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
