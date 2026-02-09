import json as _json
from typing import Any
from urllib.parse import unquote

from app.converters.gemini_to_anthropic import THOUGHT_SIG_SEP
from app.models import AnthropicRequest
from app.utils.images import detect_media_type


def _extract_system_prompt(request: AnthropicRequest) -> str | None:
    """Extract system prompt from an Anthropic request.

    Args:
        request: AnthropicRequest - The incoming Anthropic-formatted request.

    Returns:
        str | None - The extracted system prompt text, or None.
    """
    system = request.system
    if system is None:
        return None
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict):
                text = block.get("text", "")
            else:
                text = getattr(block, "text", "")
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else None
    return None


_STRIP_SCHEMA_KEYS = frozenset({
    "$schema", "$id", "$ref", "$comment", "$defs",
    "additionalProperties", "propertyNames", "patternProperties",
    "definitions", "examples", "default", "const",
    "if", "then", "else", "not",
    "anyOf", "any_of", "oneOf", "one_of", "allOf", "all_of",
    "minProperties", "maxProperties",
    "minItems", "maxItems", "uniqueItems",
    "minLength", "maxLength", "pattern",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "multipleOf", "contentMediaType", "contentEncoding",
    "title",
})

_GEMINI_ALLOWED_KEYS = frozenset({
    "type", "description", "properties", "required",
    "items", "enum", "format", "nullable",
})


def _clean_schema(schema: Any) -> dict[str, Any]:
    """Recursively strip non-Gemini JSON Schema fields.

    Args:
        schema: Any - A JSON Schema dict (or sub-dict).

    Returns:
        dict[str, Any] - Cleaned schema safe for Gemini functionDeclarations.
    """
    if not isinstance(schema, dict):
        return schema

    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _STRIP_SCHEMA_KEYS:
            continue
        if key not in _GEMINI_ALLOWED_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {k: _clean_schema(v) for k, v in value.items()}
        elif isinstance(value, dict):
            cleaned[key] = _clean_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [_clean_schema(item) if isinstance(item, dict) else item for item in value]
        else:
            cleaned[key] = value

    if "required" in cleaned and "properties" in cleaned:
        valid_props = set(cleaned["properties"].keys())
        cleaned["required"] = [r for r in cleaned["required"] if r in valid_props]
        if not cleaned["required"]:
            del cleaned["required"]

    if "type" not in cleaned and "properties" in cleaned:
        cleaned["type"] = "object"

    return cleaned


def _convert_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert Anthropic tool definitions to Gemini functionDeclarations.

    Args:
        tools: list[Any] - Anthropic tool definitions.

    Returns:
        list[dict[str, Any]] - Gemini tools array with functionDeclarations.
    """
    declarations: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, dict):
            tool_type = tool.get("type", "custom")
        else:
            tool_type = getattr(tool, "type", "custom")
            tool = tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)

        if tool_type in ("computer_20241022", "text_editor_20241022", "bash_20241022"):
            continue

        name = tool.get("name", "")
        description = tool.get("description", "")
        input_schema = tool.get("input_schema", {})

        schema = _clean_schema(input_schema)

        declarations.append({
            "name": name,
            "description": description,
            "parameters": schema,
        })

    if not declarations:
        return []

    return [{"functionDeclarations": declarations}]


def _build_user_parts(content: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build Gemini parts from Anthropic user message content.

    Returns a tuple of (user_parts, tool_response_messages) since Gemini
    requires tool responses as separate messages with role "user".

    Args:
        content: Any - String or list of Anthropic content blocks.

    Returns:
        tuple[list[dict[str, Any]], list[dict[str, Any]]] - (regular parts, tool response messages).
    """
    if isinstance(content, str):
        return [{"text": content}], []

    if not isinstance(content, list):
        return [{"text": str(content)}], []

    parts: list[dict[str, Any]] = []
    tool_responses: list[dict[str, Any]] = []

    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type", "")
        else:
            block_type = getattr(block, "type", "")
            block = block.model_dump() if hasattr(block, "model_dump") else dict(block)

        if block_type == "text":
            text = block.get("text", "")
            if text:
                parts.append({"text": text})

        elif block_type == "image":
            source = block.get("source", {})
            src_type = source.get("type", "")

            if src_type == "base64":
                declared = source.get("media_type", "")
                b64_data = source.get("data", "")
                media_type = detect_media_type(b64_data, declared or "image/png")
                parts.append({
                    "inlineData": {
                        "mimeType": media_type,
                        "data": b64_data,
                    }
                })

        elif block_type == "tool_result":
            tool_name = block.get("tool_use_id", "")
            tool_content = block.get("content", "")

            if isinstance(tool_content, str):
                result_text = tool_content
            elif isinstance(tool_content, list):
                result_text = " ".join(
                    b.get("text", "") for b in tool_content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                result_text = str(tool_content)

            tool_responses.append({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": tool_name,
                        "response": {"result": result_text},
                    }
                }],
            })

    return parts, tool_responses


def _build_assistant_parts(content: Any) -> list[dict[str, Any]]:
    """Build Gemini parts from Anthropic assistant message content.

    Args:
        content: Any - String or list of Anthropic content blocks.

    Returns:
        list[dict[str, Any]] - Gemini parts array.
    """
    if isinstance(content, str):
        return [{"text": content}]

    if not isinstance(content, list):
        return [{"text": str(content)}]

    parts: list[dict[str, Any]] = []

    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type", "")
        else:
            block_type = getattr(block, "type", "")
            block = block.model_dump() if hasattr(block, "model_dump") else dict(block)

        if block_type == "text":
            text = block.get("text", "")
            if text:
                parts.append({"text": text})

        elif block_type == "tool_use":
            name = block.get("name", "")
            input_data = block.get("input", {})
            raw_id = block.get("id", "")

            part: dict[str, Any] = {
                "functionCall": {
                    "name": name,
                    "args": input_data,
                }
            }

            if THOUGHT_SIG_SEP in raw_id:
                _, encoded_sig = raw_id.split(THOUGHT_SIG_SEP, 1)
                part["thoughtSignature"] = unquote(encoded_sig)

            parts.append(part)

        elif block_type == "thinking":
            pass

    return parts


def _build_contents(request: AnthropicRequest) -> list[dict[str, Any]]:
    """Build Gemini contents array from Anthropic messages.

    Args:
        request: AnthropicRequest - The incoming Anthropic-formatted request.

    Returns:
        list[dict[str, Any]] - Gemini contents array.
    """
    contents: list[dict[str, Any]] = []
    tool_id_to_name: dict[str, str] = {}

    for msg in request.messages:
        msg_dict = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
        role = msg_dict.get("role", "user")
        content = msg_dict.get("content", "")

        if role == "assistant":
            if isinstance(content, list):
                for block in content:
                    b = block if isinstance(block, dict) else (block.model_dump() if hasattr(block, "model_dump") else dict(block))
                    if b.get("type") == "tool_use":
                        raw_id = b.get("id", "")
                        tool_id_to_name[raw_id] = b.get("name", "")

            parts = _build_assistant_parts(content)
            if parts:
                contents.append({"role": "model", "parts": parts})

        elif role == "user":
            user_parts, tool_responses = _build_user_parts(content)

            if tool_responses:
                merged_parts: list[dict[str, Any]] = []
                for tr in tool_responses:
                    for part in tr["parts"]:
                        if "functionResponse" in part:
                            fr = part["functionResponse"]
                            raw_tool_use_id = fr["name"]
                            if raw_tool_use_id in tool_id_to_name:
                                fr["name"] = tool_id_to_name[raw_tool_use_id]
                        merged_parts.append(part)
                contents.append({"role": "user", "parts": merged_parts})

            if user_parts:
                contents.append({"role": "user", "parts": user_parts})

    return contents


def to_gemini_request(
    request: AnthropicRequest,
    model_id: str,
) -> dict[str, Any]:
    """Convert an Anthropic request to a Gemini generateContent request body.

    Args:
        request: AnthropicRequest - The incoming Anthropic-formatted request.
        model_id: str - The Gemini model identifier.

    Returns:
        dict[str, Any] - Gemini generateContent compatible request body.
    """
    body: dict[str, Any] = {
        "contents": _build_contents(request),
    }

    system_prompt = _extract_system_prompt(request)
    if system_prompt:
        body["systemInstruction"] = {
            "parts": [{"text": system_prompt}],
        }

    gen_config: dict[str, Any] = {}
    if request.max_tokens:
        gen_config["maxOutputTokens"] = request.max_tokens
    if request.temperature is not None:
        gen_config["temperature"] = request.temperature
    if request.top_p is not None:
        gen_config["topP"] = request.top_p
    if request.stop_sequences:
        gen_config["stopSequences"] = request.stop_sequences

    if gen_config:
        body["generationConfig"] = gen_config

    if request.tools:
        gemini_tools = _convert_tools(request.tools)
        if gemini_tools:
            body["tools"] = gemini_tools

    return body
