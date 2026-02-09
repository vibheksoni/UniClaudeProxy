import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a specific model within a provider.

    Attributes:
        name: str - Human-readable model name.
        upstream_model_id: str | None - Override model ID sent to provider (when config key differs).
        responses: bool - Whether to use the /v1/responses endpoint.
        use_react: bool - Whether the model supports ReAct-style tool use.
        inject_context: bool - Inject system prompt and tool summary into input messages.
        force_stream: bool - Provider always returns SSE; consume internally for non-stream.
        upstream_system: bool - Provider forces its own system prompt and tools.
        tool_mapping: dict[str, str] - Map upstream tool names to Claude Code tool names.
        reasoning: dict[str, Any] - Reasoning config e.g. {"effort": "high", "summary": "auto"}.
        truncation: str | None - Truncation strategy ("auto", "disabled", or None).
        text: dict[str, str] - Text config e.g. {"verbosity": "low"}.
        max_output_tokens: int | None - Max output tokens override for this model.
        parallel_tool_calls: bool | None - Enable parallel tool calls.
        image_mode: str - How to handle images: "input_image", "save_and_ref", or "strip".
        image_dir: str | None - Directory to save images when using save_and_ref mode.
        system_replacements: dict[str, str] - String replacements applied to system prompt (key=target, value=replacement).
    """

    name: str
    upstream_model_id: Optional[str] = None
    responses: bool = False
    use_react: bool = False
    inject_context: bool = False
    force_stream: bool = False
    upstream_system: bool = False
    tool_mapping: dict[str, str] = Field(default_factory=dict)
    reasoning: dict[str, Any] = Field(default_factory=dict)
    truncation: Optional[str] = None
    text: dict[str, str] = Field(default_factory=dict)
    max_output_tokens: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    image_mode: str = "input_image"
    image_dir: Optional[str] = None
    system_replacements: dict[str, str] = Field(default_factory=dict)


class ProviderConfig(BaseModel):
    """Configuration for a provider (OpenAI, Gemini, etc.).

    Attributes:
        provider_type: str - The type of provider (openai, claude, gemini).
        api_key: str - API key for authentication.
        base_url: str - Base URL for the provider's API.
        headers: dict[str, str] - Extra headers to include in requests.
        models: dict[str, ModelConfig] - Available models for this provider.
    """

    provider_type: str
    api_key: str
    base_url: str
    headers: dict[str, str] = Field(default_factory=dict)
    models: dict[str, ModelConfig] = Field(default_factory=dict)


class ServerConfig(BaseModel):
    """Server binding configuration.

    Attributes:
        host: str - Host address to bind to.
        port: int - Port number to listen on.
        local_only: bool - Restrict access to localhost connections only.
    """

    host: str = "127.0.0.1"
    port: int = 9223
    local_only: bool = True


class AppConfig(BaseModel):
    """Root application configuration.

    Attributes:
        server: ServerConfig - Server binding settings.
        models: dict[str, str] - Mapping of Anthropic model names to provider/model identifiers.
        providers: dict[str, ProviderConfig] - Provider configurations keyed by provider name.
    """

    server: ServerConfig = Field(default_factory=ServerConfig)
    models: dict[str, str] = Field(default_factory=dict)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)


class ResolvedRoute:
    """Resolved routing information for a request.

    Attributes:
        provider_name: str - Name of the provider in config.
        provider: ProviderConfig - Full provider configuration.
        provider_type: str - The provider type (openai, gemini, claude).
        model_id: str - The model identifier to send to the provider.
        model_config: ModelConfig - Model-specific configuration flags.
        use_responses: bool - Whether to use the /v1/responses endpoint.
        endpoint_url: str - Fully resolved endpoint URL (OpenAI providers only).
    """

    def __init__(
        self,
        provider_name: str,
        provider: ProviderConfig,
        model_id: str,
        model_config: ModelConfig,
    ):
        """Initialize a ResolvedRoute.

        Args:
            provider_name: str - Name of the provider in config.
            provider: ProviderConfig - Full provider configuration.
            model_id: str - The model identifier to send to the provider.
            model_config: ModelConfig - Model-specific configuration flags.
        """
        self.provider_name = provider_name
        self.provider = provider
        self.provider_type = provider.provider_type
        self.model_id = model_config.upstream_model_id or model_id
        self.model_config = model_config
        self.use_responses = model_config.responses
        self.inject_context = model_config.inject_context
        self.force_stream = model_config.force_stream
        self.upstream_system = model_config.upstream_system
        self.tool_mapping = model_config.tool_mapping
        self.reasoning = model_config.reasoning
        self.truncation = model_config.truncation
        self.text = model_config.text
        self.max_output_tokens = model_config.max_output_tokens
        self.parallel_tool_calls = model_config.parallel_tool_calls
        self.image_mode = model_config.image_mode
        self.image_dir = model_config.image_dir

        base = provider.base_url.rstrip("/")
        if self.provider_type == "claude":
            self.endpoint_url = f"{base}/messages"
        elif self.provider_type == "gemini":
            self.endpoint_url = base
        elif self.use_responses:
            self.endpoint_url = f"{base}/responses"
        else:
            self.endpoint_url = f"{base}/chat/completions"

    def build_headers(self) -> dict[str, str]:
        """Build the full set of headers for the outgoing request.

        Returns:
            dict[str, str] - Merged headers with authorization.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.provider.api_key}",
        }
        headers.update(self.provider.headers)
        return headers


_config: Optional[AppConfig] = None
_config_path: Optional[str] = None


def config_path() -> str:
    """Return the resolved path to config.json.

    Returns:
        str - Absolute path to the config file.
    """
    return _config_path or str(Path(__file__).parent.parent / "config.json")


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load and parse the application configuration from disk.

    Args:
        path: Optional[str] - Path to config.json. Defaults to project root.

    Returns:
        AppConfig - Parsed application configuration.
    """
    global _config, _config_path
    if _config is not None:
        return _config

    if path is None:
        path = str(Path(__file__).parent.parent / "config.json")

    _config_path = path

    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    _config = AppConfig(**raw)
    return _config


def reload_config() -> AppConfig:
    """Force-reload configuration from disk, replacing the cached config.

    Returns:
        AppConfig - Freshly parsed application configuration.
    """
    global _config
    _config = None
    return load_config(_config_path)


def resolve_route(anthropic_model: str) -> ResolvedRoute:
    """Resolve an Anthropic model name to a provider route.

    Args:
        anthropic_model: str - The model name from the incoming Anthropic request.

    Returns:
        ResolvedRoute - Resolved routing information.

    Raises:
        ValueError: If the model or provider is not found in config.
    """
    cfg = load_config()

    mapping = cfg.models.get(anthropic_model)
    if mapping is None:
        raise ValueError(f"Model '{anthropic_model}' not found in config models mapping")

    parts = mapping.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid model mapping format: '{mapping}'. Expected 'provider/model'")

    provider_name, model_id = parts

    provider = cfg.providers.get(provider_name)
    if provider is None:
        raise ValueError(f"Provider '{provider_name}' not found in config providers")

    model_cfg = provider.models.get(model_id)
    if model_cfg is None:
        raise ValueError(f"Model '{model_id}' not found in provider '{provider_name}' models")

    return ResolvedRoute(
        provider_name=provider_name,
        provider=provider,
        model_id=model_id,
        model_config=model_cfg,
    )
