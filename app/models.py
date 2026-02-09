from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class AnthropicTextContent(BaseModel):
    """A text content block in an Anthropic message.

    Attributes:
        type: str - Always "text".
        text: str - The text content.
    """

    model_config = ConfigDict(extra="allow")

    type: str = "text"
    text: str = ""


class AnthropicImageSource(BaseModel):
    """Source data for an image content block.

    Attributes:
        type: str - Source type, e.g. "base64" or "url".
        media_type: Optional[str] - MIME type of the image.
        data: Optional[str] - Base64-encoded image data.
        url: Optional[str] - URL of the image.
    """

    model_config = ConfigDict(extra="allow")

    type: str = "base64"
    media_type: Optional[str] = None
    data: Optional[str] = None
    url: Optional[str] = None


class AnthropicImageContent(BaseModel):
    """An image content block in an Anthropic message.

    Attributes:
        type: str - Always "image".
        source: AnthropicImageSource - The image source data.
    """

    model_config = ConfigDict(extra="allow")

    type: str = "image"
    source: AnthropicImageSource


class AnthropicToolUseContent(BaseModel):
    """A tool_use content block in an Anthropic message.

    Attributes:
        type: str - Always "tool_use".
        id: str - Unique identifier for this tool use.
        name: str - Name of the tool being called.
        input: dict[str, Any] - Arguments passed to the tool.
    """

    model_config = ConfigDict(extra="allow")

    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict[str, Any] = Field(default_factory=dict)


class AnthropicToolResultContent(BaseModel):
    """A tool_result content block in an Anthropic message.

    Attributes:
        type: str - Always "tool_result".
        tool_use_id: str - ID of the tool_use this result corresponds to.
        content: Union[str, list[Any]] - The result content.
        is_error: bool - Whether the tool call resulted in an error.
    """

    model_config = ConfigDict(extra="allow")

    type: str = "tool_result"
    tool_use_id: str = ""
    content: Union[str, list[Any]] = ""
    is_error: bool = False


class AnthropicThinkingContent(BaseModel):
    """A thinking content block in an Anthropic message.

    Attributes:
        type: str - Always "thinking".
        thinking: str - The thinking text.
        signature: str - Verification signature for the thinking block.
    """

    model_config = ConfigDict(extra="allow")

    type: str = "thinking"
    thinking: str = ""
    signature: str = ""


ContentBlock = Union[
    AnthropicTextContent,
    AnthropicImageContent,
    AnthropicToolUseContent,
    AnthropicToolResultContent,
    AnthropicThinkingContent,
    dict[str, Any],
]


class AnthropicMessage(BaseModel):
    """A single message in an Anthropic conversation.

    Attributes:
        role: str - The role: "user" or "assistant".
        content: Union[str, list[ContentBlock]] - Message content.
    """

    model_config = ConfigDict(extra="allow")

    role: str
    content: Union[str, list[ContentBlock]]


class AnthropicToolDef(BaseModel):
    """Tool definition in an Anthropic request.

    Attributes:
        name: str - Tool name.
        description: str - Tool description.
        input_schema: dict[str, Any] - JSON Schema for tool input.
        type: Optional[str] - Tool type identifier.
    """

    model_config = ConfigDict(extra="allow")

    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    type: Optional[str] = None


class AnthropicRequest(BaseModel):
    """Incoming request in Anthropic Messages API format.

    Attributes:
        model: str - The model to use.
        messages: list[AnthropicMessage] - Conversation messages.
        max_tokens: int - Maximum tokens to generate.
        system: Optional[Union[str, list[dict[str, Any]]]] - System prompt.
        stream: bool - Whether to stream the response.
        temperature: Optional[float] - Sampling temperature.
        top_p: Optional[float] - Nucleus sampling parameter.
        top_k: Optional[int] - Top-k sampling parameter.
        stop_sequences: Optional[list[str]] - Custom stop sequences.
        tools: Optional[list[AnthropicToolDef]] - Available tools.
        tool_choice: Optional[dict[str, Any]] - Tool selection strategy.
        metadata: Optional[dict[str, Any]] - Request metadata.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = 4096
    system: Optional[Union[str, list[dict[str, Any]]]] = None
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[list[str]] = None
    tools: Optional[list[AnthropicToolDef]] = None
    tool_choice: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None


class AnthropicUsage(BaseModel):
    """Token usage information in an Anthropic response.

    Attributes:
        input_tokens: int - Number of input tokens consumed.
        output_tokens: int - Number of output tokens generated.
        cache_creation_input_tokens: int - Tokens used to create cache.
        cache_read_input_tokens: int - Tokens read from cache.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class AnthropicResponse(BaseModel):
    """Non-streaming response in Anthropic Messages API format.

    Attributes:
        id: str - Unique message identifier.
        type: str - Always "message".
        role: str - Always "assistant".
        content: list[ContentBlock] - Response content blocks.
        model: str - Model that generated the response.
        stop_reason: Optional[str] - Why generation stopped.
        stop_sequence: Optional[str] - Which stop sequence was hit.
        usage: AnthropicUsage - Token usage information.
    """

    id: str = ""
    type: str = "message"
    role: str = "assistant"
    content: list[ContentBlock] = Field(default_factory=list)
    model: str = ""
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)
