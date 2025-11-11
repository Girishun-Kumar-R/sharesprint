"""Preview image caching utilities shared by app and indexer."""
from __future__ import annotations

import os
from typing import Optional

import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image

PREVIEW_MAX = int(os.getenv("PREVIEW_MAX_PX", "1024"))
PREVIEW_CACHE_DIR = os.getenv("PREVIEW_CACHE_DIR", "preview_cache")
_CACHE_EXT = ".jpg"


def _cache_path(file_id: str) -> str:
    return os.path.join(PREVIEW_CACHE_DIR, f"{file_id}{_CACHE_EXT}")


def _ensure_dir() -> None:
    os.makedirs(PREVIEW_CACHE_DIR, exist_ok=True)


def decode_to_image(image_bytes: bytes) -> Image.Image:
    """Decode raw image bytes into a resized Pillow Image."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Unable to decode image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image.thumbnail((PREVIEW_MAX, PREVIEW_MAX))
    return image


def write_cache(file_id: str, image: Image.Image) -> None:
    """Persist a preview image to disk for later reuse."""
    _ensure_dir()
    cache_image = image.convert("RGB")
    cache_image.save(_cache_path(file_id), format="JPEG", quality=85)


def cache_from_bytes(file_id: str, image_bytes: bytes) -> Image.Image:
    """Decode bytes, resize, write cache, and return an RGBA image."""
    image = decode_to_image(image_bytes)
    write_cache(file_id, image)
    return image.convert("RGBA")


def load_cached(file_id: str) -> Optional[Image.Image]:
    """Load a cached preview if present; return RGBA image."""
    path = _cache_path(file_id)
    if not os.path.exists(path):
        return None
    try:
        with Image.open(path) as cached:
            return cached.convert("RGBA")
    except OSError:
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def delete_cache(file_id: str) -> None:
    """Remove cached preview for a file_id, ignoring missing files."""
    path = _cache_path(file_id)
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
