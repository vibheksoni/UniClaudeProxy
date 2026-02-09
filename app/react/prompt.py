import json
from typing import Any, Optional, Union


REACT_SYSTEM_TEMPLATE = """# TOOL CALLING — YOU MUST USE THIS EXACT XML FORMAT

To call a tool, you MUST output this EXACT XML block — nothing else works:

<tool_call>
<name>TOOL_NAME</name>
<parameters>
{{"param1": "value1"}}
</parameters>
</tool_call>

CRITICAL RULES:
1. You MUST use the <tool_call> XML block above to call tools. Do NOT describe or narrate tool calls in plain text.
2. Output ONLY ONE tool call per response.
3. Parameters MUST be valid JSON.
4. STOP writing immediately after </tool_call> — no text after it.
5. Wait for <observation> before continuing.

EXAMPLE — calling a tool named "Bash" with parameter "command":

I need to list the files in the current directory.

<tool_call>
<name>Bash</name>
<parameters>
{{"command": "ls -la"}}
</parameters>
</tool_call>

## Available Tools

{tool_definitions}"""


def _compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip verbose keys from JSON schema, keeping only what's needed for tool calls.

    Args:
        schema: dict[str, Any] - Full JSON schema.

    Returns:
        dict[str, Any] - Minimal schema with type, properties, required, items, enum.
    """
    keep = {"type", "properties", "required", "items", "enum", "description"}
    out: dict[str, Any] = {}
    for k, v in schema.items():
        if k not in keep:
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _compact_schema(pv) for pk, pv in v.items()}
        elif k == "items" and isinstance(v, dict):
            out[k] = _compact_schema(v)
        elif k == "description" and isinstance(v, str):
            out[k] = v[:100]
        else:
            out[k] = v
    return out


def _format_single_tool(tool: dict[str, Any]) -> str:
    """Format one tool definition with name, description, and parameter schema.

    Args:
        tool: dict[str, Any] - Anthropic tool definition dict.

    Returns:
        str - Formatted tool definition string.
    """
    name = tool.get("name", "unknown")
    desc = tool.get("description", "")
    schema = tool.get("input_schema", {})

    short_desc = desc[:200].split("\n")[0] if desc else ""
    lines = [f"### {name}"]
    if short_desc:
        lines.append(short_desc)
    if schema.get("properties"):
        compact = _compact_schema(schema)
        lines.append(f"Parameters: {json.dumps(compact, separators=(',', ':'))}")
    return "\n".join(lines)


def build_react_system(
    tools: list[Any],
    original_system: Optional[Union[str, list[dict[str, Any]]]] = None,
) -> str:
    """Build the full system prompt with ReAct XML tool calling instructions.

    Args:
        tools: list[Any] - Anthropic tool definitions (dicts or Pydantic models).
        original_system: Optional[Union[str, list[dict[str, Any]]]] - The original system prompt.

    Returns:
        str - Combined system prompt with ReAct instructions and tool definitions.
    """
    sys_text = ""
    if isinstance(original_system, str):
        sys_text = original_system
    elif isinstance(original_system, list):
        parts = []
        for block in original_system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        sys_text = "\n".join(parts)

    tool_defs: list[str] = []
    for tool in tools:
        td = tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)
        tool_type = td.get("type")
        if tool_type and tool_type in _BUILTIN_TYPES:
            continue
        tool_defs.append(_format_single_tool(td))

    react_block = REACT_SYSTEM_TEMPLATE.format(
        tool_definitions="\n\n".join(tool_defs),
    )

    if sys_text:
        return f"{sys_text}\n\n{react_block}"
    return react_block


def format_tool_call_xml(name: str, arguments: dict[str, Any]) -> str:
    """Format a tool call as XML text (used for converting history).

    Args:
        name: str - Tool name.
        arguments: dict[str, Any] - Tool arguments.

    Returns:
        str - XML formatted tool call string.
    """
    args_json = json.dumps(arguments, ensure_ascii=False)
    return f"<tool_call>\n<name>{name}</name>\n<parameters>\n{args_json}\n</parameters>\n</tool_call>"


def format_observation_xml(content: str) -> str:
    """Format a tool result as an XML observation block.

    Args:
        content: str - The tool result text.

    Returns:
        str - XML formatted observation string.
    """
    return f"<observation>\n{content}\n</observation>"


_BUILTIN_TYPES = {
    "computer_20241022",
    "text_editor_20241022",
    "bash_20241022",
    "computer_20250124",
    "text_editor_20250124",
    "bash_20250124",
}
