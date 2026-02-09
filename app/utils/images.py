import base64
import os
import tempfile
import uuid
from typing import Any


_IMAGE_SIGNATURES = {
    b"\x89PNG": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"RIFF": "image/webp",
}

_MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/svg+xml": ".svg",
}

_IMAGE_DIR: str | None = None


def detect_media_type(b64_data: str, fallback: str = "image/png") -> str:
    """Detect image media type from base64-encoded data header bytes.

    Args:
        b64_data: str - Base64-encoded image data.
        fallback: str - Fallback media type if detection fails.

    Returns:
        str - Detected MIME type string.
    """
    try:
        raw = base64.b64decode(b64_data[:32])
    except Exception:
        return fallback

    for sig, mime in _IMAGE_SIGNATURES.items():
        if raw[:len(sig)] == sig:
            if mime == "image/webp" and len(raw) >= 12:
                if raw[8:12] != b"WEBP":
                    continue
            return mime

    return fallback


def save_image(b64_data: str, media_type: str, image_dir: str | None = None) -> str:
    """Save a base64-encoded image to disk and return the file path.

    Args:
        b64_data: str - Base64-encoded image data.
        media_type: str - MIME type of the image.
        image_dir: str | None - Directory to save images in. Uses temp dir if None.

    Returns:
        str - Absolute path to the saved image file.
    """
    global _IMAGE_DIR

    if image_dir:
        target_dir = image_dir
    elif _IMAGE_DIR:
        target_dir = _IMAGE_DIR
    else:
        target_dir = os.path.join(tempfile.gettempdir(), "anyclaude_images")
        _IMAGE_DIR = target_dir

    os.makedirs(target_dir, exist_ok=True)

    ext = _MIME_TO_EXT.get(media_type, ".png")
    filename = f"{uuid.uuid4().hex[:16]}{ext}"
    filepath = os.path.join(target_dir, filename)

    try:
        raw = base64.b64decode(b64_data)
        with open(filepath, "wb") as f:
            f.write(raw)
    except Exception:
        return ""

    return filepath


def build_image_parts(
    source: dict[str, Any],
    image_mode: str = "input_image",
    image_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Build OpenAI Responses API content parts for an image block.

    Args:
        source: dict[str, Any] - Anthropic image source with type, media_type, data/url.
        image_mode: str - How to handle images: "input_image", "save_and_ref", or "strip".
        image_dir: str | None - Directory for saving images (save_and_ref mode).

    Returns:
        list[dict[str, Any]] - List of content parts to append to the message.
    """
    if image_mode == "strip":
        return [{"type": "input_text", "text": "[Image was attached but image support is disabled for this model]"}]

    source_type = source.get("type", "")
    declared_type = source.get("media_type", "")
    b64_data = source.get("data", "")
    url = source.get("url", "")

    if source_type == "base64" and b64_data:
        media_type = detect_media_type(b64_data, declared_type or "image/png")
        parts: list[dict[str, Any]] = []

        parts.append({
            "type": "input_image",
            "detail": "auto",
            "image_url": f"data:{media_type};base64,{b64_data}",
        })

        if image_mode == "save_and_ref":
            save_image(b64_data, media_type, image_dir)

        return parts

    elif url:
        parts = [{
            "type": "input_image",
            "detail": "auto",
            "image_url": url,
        }]

        if image_mode == "save_and_ref":
            pass

        return parts

    return [{"type": "input_text", "text": "[Image block with no data]"}]
