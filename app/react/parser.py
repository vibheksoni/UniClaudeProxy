import json
import logging
import re
import uuid
from typing import Any

debug_logger = logging.getLogger("anyclaude.debug")

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<name>\s*(.*?)\s*</name>\s*<parameters>\s*(.*?)\s*</parameters>\s*</tool_call>",
    re.DOTALL,
)

PARTIAL_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<name>\s*(.*?)\s*</name>\s*<parameters>\s*(.*?)\s*$",
    re.DOTALL,
)


def _generate_tool_id() -> str:
    """Generate a unique Anthropic-style tool use ID.

    Returns:
        str - ID in the format 'toolu_<hex>'.
    """
    return f"toolu_{uuid.uuid4().hex[:24]}"


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse all <tool_call> XML blocks from model output text.

    Args:
        text: str - The raw model output text potentially containing XML tool calls.

    Returns:
        list[dict[str, Any]] - List of parsed tool call dicts with keys:
            name (str), arguments (dict), start (int), end (int).
    """
    results: list[dict[str, Any]] = []

    for match in TOOL_CALL_RE.finditer(text):
        name = match.group(1).strip()
        params_raw = match.group(2).strip()

        try:
            arguments = json.loads(params_raw)
        except json.JSONDecodeError:
            debug_logger.warning("REACT PARSE: invalid JSON in tool_call params for %s: %s", name, params_raw[:200])
            arguments = {}

        results.append({
            "name": name,
            "arguments": arguments,
            "start": match.start(),
            "end": match.end(),
        })

    if not results:
        partial = PARTIAL_TOOL_CALL_RE.search(text)
        if partial:
            name = partial.group(1).strip()
            params_raw = partial.group(2).strip()
            if not params_raw.endswith("}"):
                params_raw += "}"

            try:
                arguments = json.loads(params_raw)
            except json.JSONDecodeError:
                arguments = {}

            if name:
                debug_logger.info("REACT PARSE: recovered partial tool_call for %s", name)
                results.append({
                    "name": name,
                    "arguments": arguments,
                    "start": partial.start(),
                    "end": partial.end(),
                })

    return results


def split_text_and_tool_calls(text: str) -> list[dict[str, Any]]:
    """Split model output text into alternating text and tool_use content blocks.

    Args:
        text: str - The raw model output text.

    Returns:
        list[dict[str, Any]] - List of Anthropic content blocks:
            {"type": "text", "text": "..."} or
            {"type": "tool_use", "id": "...", "name": "...", "input": {...}}.
    """
    tool_calls = parse_tool_calls(text)
    if not tool_calls:
        return [{"type": "text", "text": text}] if text.strip() else []

    blocks: list[dict[str, Any]] = []
    cursor = 0

    for tc in tool_calls:
        pre_text = text[cursor:tc["start"]].strip()
        if pre_text:
            blocks.append({"type": "text", "text": pre_text})

        blocks.append({
            "type": "tool_use",
            "id": _generate_tool_id(),
            "name": tc["name"],
            "input": tc["arguments"],
        })
        cursor = tc["end"]

    post_text = text[cursor:].strip()
    if post_text:
        blocks.append({"type": "text", "text": post_text})

    return blocks


def has_tool_call(text: str) -> bool:
    """Check if text contains a tool call XML block.

    Args:
        text: str - Text to check.

    Returns:
        bool - True if a tool_call XML block is detected.
    """
    return bool(TOOL_CALL_RE.search(text)) or bool(PARTIAL_TOOL_CALL_RE.search(text))
