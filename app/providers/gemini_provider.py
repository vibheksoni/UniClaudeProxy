import json
import logging
from typing import Any, AsyncIterator

import httpx

from app.config import ResolvedRoute
from app.converters.anthropic_to_gemini import to_gemini_request
from app.models import AnthropicRequest

logger = logging.getLogger("anyclaude.provider")
debug_logger = logging.getLogger("anyclaude.debug")

_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:
    """Get or create the shared async HTTP client for Gemini.

    Returns:
        httpx.AsyncClient - The shared HTTP client instance.
    """
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0))
    return _client


async def close_client() -> None:
    """Close the shared HTTP client."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
    _client = None


def _build_endpoint_url(route: ResolvedRoute, stream: bool) -> str:
    """Build the full Gemini endpoint URL with model in path.

    Args:
        route: ResolvedRoute - Resolved routing information.
        stream: bool - Whether this is a streaming request.

    Returns:
        str - Full endpoint URL.
    """
    base = route.provider.base_url.rstrip("/")
    model = route.model_id
    if stream:
        return f"{base}/models/{model}:streamGenerateContent?alt=sse"
    return f"{base}/models/{model}:generateContent"


def _build_headers(route: ResolvedRoute) -> dict[str, str]:
    """Build headers for Gemini request.

    Args:
        route: ResolvedRoute - Resolved routing information.

    Returns:
        dict[str, str] - Request headers.
    """
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {route.provider.api_key}",
    }
    headers.update(route.provider.headers)
    return headers


async def send_non_streaming(
    request: AnthropicRequest,
    route: ResolvedRoute,
) -> dict[str, Any]:
    """Send a non-streaming request to a Gemini-compatible provider.

    Args:
        request: AnthropicRequest - The incoming Anthropic-formatted request.
        route: ResolvedRoute - Resolved routing information.

    Returns:
        dict[str, Any] - The raw JSON response from the provider.

    Raises:
        httpx.HTTPStatusError: If the provider returns a non-2xx status.
    """
    body = to_gemini_request(request, route.model_id)
    url = _build_endpoint_url(route, stream=False)
    headers = _build_headers(route)
    client = await get_client()

    debug_logger.info(">>> GEMINI REQUEST (non-streaming) to %s", url)
    debug_logger.info(">>> BODY:\n%s", json.dumps(body, indent=2, default=str)[:5000])

    resp = await client.post(url, json=body, headers=headers)
    if resp.status_code >= 400:
        debug_logger.error("<<< GEMINI ERROR (status=%d):\n%s", resp.status_code, resp.text[:3000])
    resp.raise_for_status()

    return resp.json()


async def send_streaming(
    request: AnthropicRequest,
    route: ResolvedRoute,
) -> AsyncIterator[bytes]:
    """Send a streaming request to a Gemini-compatible provider.

    Args:
        request: AnthropicRequest - The incoming Anthropic-formatted request.
        route: ResolvedRoute - Resolved routing information.

    Yields:
        bytes - Raw SSE line chunks from the provider.

    Raises:
        httpx.HTTPStatusError: If the provider returns a non-2xx status.
    """
    body = to_gemini_request(request, route.model_id)
    url = _build_endpoint_url(route, stream=True)
    headers = _build_headers(route)
    headers["Accept"] = "text/event-stream"
    client = await get_client()

    debug_logger.info(">>> GEMINI REQUEST (streaming) to %s", url)
    debug_logger.info(">>> BODY:\n%s", json.dumps(body, indent=2, default=str)[:5000])

    async with client.stream("POST", url, json=body, headers=headers) as resp:
        if resp.status_code >= 400:
            error_body = await resp.aread()
            debug_logger.error("<<< GEMINI ERROR (status=%d):\n%s", resp.status_code, error_body.decode()[:3000])
            resp.raise_for_status()

        async for line in resp.aiter_lines():
            debug_logger.debug("<<< GEMINI SSE: %s", line[:200] if line else "")
            yield (line + "\n").encode("utf-8")
