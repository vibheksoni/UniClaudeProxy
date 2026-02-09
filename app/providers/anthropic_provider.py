import json
import logging
from typing import Any, AsyncIterator

import httpx

from app.config import ResolvedRoute

logger = logging.getLogger("anyclaude.provider")
debug_logger = logging.getLogger("anyclaude.debug")

_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:
    """Get or create the shared async HTTP client for Anthropic passthrough.

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


def _build_headers(route: ResolvedRoute) -> dict[str, str]:
    """Build headers for Anthropic passthrough request.

    Args:
        route: ResolvedRoute - Resolved routing information.

    Returns:
        dict[str, str] - Request headers.
    """
    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    headers.update(route.provider.headers)
    return headers


def _build_body(raw_body: dict[str, Any], route: ResolvedRoute) -> dict[str, Any]:
    """Build the request body for Anthropic passthrough.

    Replaces the model name with the upstream model ID.

    Args:
        raw_body: dict[str, Any] - The original Anthropic request body.
        route: ResolvedRoute - Resolved routing information.

    Returns:
        dict[str, Any] - The modified request body.
    """
    body = dict(raw_body)
    body["model"] = route.model_id
    return body


async def send_non_streaming(
    raw_body: dict[str, Any],
    route: ResolvedRoute,
) -> dict[str, Any]:
    """Send a non-streaming passthrough request to an Anthropic-compatible provider.

    Args:
        raw_body: dict[str, Any] - The raw Anthropic request body.
        route: ResolvedRoute - Resolved routing information.

    Returns:
        dict[str, Any] - The raw JSON response from the provider.

    Raises:
        httpx.HTTPStatusError: If the provider returns a non-2xx status.
    """
    body = _build_body(raw_body, route)
    url = route.endpoint_url
    headers = _build_headers(route)
    client = await get_client()

    debug_logger.info(">>> ANTHROPIC PASSTHROUGH (non-streaming) to %s", url)

    resp = await client.post(url, json=body, headers=headers)
    if resp.status_code >= 400:
        debug_logger.error("<<< ANTHROPIC ERROR (status=%d):\n%s", resp.status_code, resp.text[:3000])
    resp.raise_for_status()

    return resp.json()


async def send_streaming(
    raw_body: dict[str, Any],
    route: ResolvedRoute,
) -> AsyncIterator[bytes]:
    """Send a streaming passthrough request to an Anthropic-compatible provider.

    Args:
        raw_body: dict[str, Any] - The raw Anthropic request body.
        route: ResolvedRoute - Resolved routing information.

    Yields:
        bytes - Raw SSE line chunks from the provider, passed through as-is.

    Raises:
        httpx.HTTPStatusError: If the provider returns a non-2xx status.
    """
    body = _build_body(raw_body, route)
    body["stream"] = True
    url = route.endpoint_url
    headers = _build_headers(route)
    client = await get_client()

    debug_logger.info(">>> ANTHROPIC PASSTHROUGH (streaming) to %s", url)

    async with client.stream("POST", url, json=body, headers=headers) as resp:
        if resp.status_code >= 400:
            error_body = await resp.aread()
            debug_logger.error("<<< ANTHROPIC ERROR (status=%d):\n%s", resp.status_code, error_body.decode()[:3000])
            resp.raise_for_status()

        async for chunk in resp.aiter_bytes():
            yield chunk
