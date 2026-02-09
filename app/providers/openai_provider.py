import json
import logging
from typing import Any, AsyncIterator

import httpx

from app.config import ResolvedRoute
from app.converters.anthropic_to_openai import (
    to_openai_chat_request,
    to_openai_responses_request,
)
from app.models import AnthropicRequest

logger = logging.getLogger("anyclaude.provider")
debug_logger = logging.getLogger("anyclaude.debug")

_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:
    """Get or create the shared async HTTP client.

    Returns:
        httpx.AsyncClient - The shared HTTP client instance.
    """
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0))
    return _client


async def close_client() -> None:
    """Close the shared async HTTP client."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None


def _build_request_body(request: AnthropicRequest, route: ResolvedRoute) -> dict[str, Any]:
    """Build the provider request body based on route configuration.

    Args:
        request: AnthropicRequest - The incoming Anthropic-formatted request.
        route: ResolvedRoute - Resolved routing information.

    Returns:
        dict[str, Any] - The provider-specific request body.
    """
    if route.use_responses:
        return to_openai_responses_request(
            request,
            route.model_id,
            inject_context=route.inject_context,
            upstream_system=route.upstream_system,
            reasoning=route.reasoning or None,
            truncation=route.truncation,
            text=route.text or None,
            max_output_tokens=route.max_output_tokens,
            parallel_tool_calls=route.parallel_tool_calls,
            image_mode=route.image_mode,
            image_dir=route.image_dir,
        )
    return to_openai_chat_request(
        request,
        route.model_id,
        max_output_tokens=route.max_output_tokens,
    )


async def send_non_streaming(
    request: AnthropicRequest,
    route: ResolvedRoute,
) -> dict[str, Any]:
    """Send a non-streaming request to an OpenAI-compatible provider.

    Args:
        request: AnthropicRequest - The incoming Anthropic-formatted request.
        route: ResolvedRoute - Resolved routing information.

    Returns:
        dict[str, Any] - The raw JSON response from the provider.

    Raises:
        httpx.HTTPStatusError: If the provider returns a non-2xx status.
    """
    body = _build_request_body(request, route)
    headers = route.build_headers()
    client = await get_client()

    debug_logger.info(">>> OUTGOING REQUEST (non-streaming) to %s", route.endpoint_url)
    debug_logger.info(">>> BODY:\n%s", json.dumps(body, indent=2, default=str)[:5000])

    resp = await client.post(
        route.endpoint_url,
        json=body,
        headers=headers,
    )
    if resp.status_code >= 400:
        debug_logger.error("<<< ERROR RESPONSE (status=%d):\n%s", resp.status_code, resp.text[:3000])
    resp.raise_for_status()

    raw_text = resp.text.strip()
    debug_logger.info("<<< RESPONSE (status=%d):\n%s", resp.status_code, raw_text[:3000])

    if not raw_text:
        raise ValueError(f"Provider returned empty response (status {resp.status_code})")

    try:
        return resp.json()
    except Exception as e:
        logger.error("Failed to parse provider response: %s | Body: %s", e, raw_text[:500])
        raise ValueError(f"Provider returned non-JSON response: {raw_text[:200]}") from e


async def send_streaming(
    request: AnthropicRequest,
    route: ResolvedRoute,
) -> AsyncIterator[bytes]:
    """Send a streaming request to an OpenAI-compatible provider.

    The async generator keeps the HTTP connection alive for the duration
    of iteration. The httpx stream context manager closes when the
    generator is fully consumed or garbage collected.

    Args:
        request: AnthropicRequest - The incoming Anthropic-formatted request.
        route: ResolvedRoute - Resolved routing information.

    Yields:
        bytes - Raw SSE line chunks from the provider (each line terminated with newline).

    Raises:
        httpx.HTTPStatusError: If the provider returns a non-2xx status.
    """
    body = _build_request_body(request, route)
    headers = route.build_headers()
    headers["Accept"] = "text/event-stream"
    client = await get_client()

    debug_logger.info(">>> OUTGOING REQUEST (streaming) to %s", route.endpoint_url)
    outgoing_tools = body.get("tools", [])
    debug_logger.info(">>> OUTGOING TOOLS COUNT: %d", len(outgoing_tools))
    if outgoing_tools:
        for oti, ot in enumerate(outgoing_tools[:5]):
            debug_logger.info(">>>   TOOL[%d] name=%s type=%s", oti, ot.get("name", ot.get("function", {}).get("name", "?")), ot.get("type", "?"))
    debug_logger.info(">>> BODY:\n%s", json.dumps(body, indent=2, default=str)[:5000])

    async with client.stream(
        "POST",
        route.endpoint_url,
        json=body,
        headers=headers,
    ) as resp:
        if resp.status_code >= 400:
            error_body = await resp.aread()
            debug_logger.error("<<< STREAM ERROR (status=%d):\n%s", resp.status_code, error_body.decode("utf-8", errors="replace")[:3000])
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if line.strip():
                debug_logger.info("<<< SSE LINE: %s", line[:500])
            yield (line + "\n").encode("utf-8")
