"""Shared DeepFace embedding utilities for ShareSprint."""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Iterable, List, Sequence, Union

import cv2  # type: ignore
import numpy as np
from deepface import DeepFace  # type: ignore

EMBEDDING_DIM: int = 512
MIN_CONFIDENCE: float = float(os.getenv("FACE_MIN_CONF", "0"))  # callers may override

_LOGGER = logging.getLogger(__name__)

BytesLike = Union[bytes, bytearray, memoryview]
ImageInput = Union[str, BytesLike]


def safe_detector_backend_chain(primary: str) -> List[str]:
    """Return ordered detector backends with graceful fallbacks."""
    options = [primary, "mtcnn", "opencv"]
    seen = set()
    chain: List[str] = []
    for option in options:
        if option and option not in seen:
            chain.append(option)
            seen.add(option)
    return chain


def represent_from_bytes(
    image_bytes: BytesLike,
    detector_backend: str,
    model_name: str,
    allow_normalize_kw: bool = True,
) -> dict:
    """Compute an embedding for raw image bytes using DeepFace via a temp JPEG file."""
    path = _bytes_to_temp_jpeg(image_bytes)
    try:
        return represent_from_file(
            path,
            detector_backend=detector_backend,
            model_name=model_name,
            allow_normalize_kw=allow_normalize_kw,
        )
    finally:
        _safe_delete(path)


def represent_from_file(
    file_path: str,
    detector_backend: str,
    model_name: str,
    allow_normalize_kw: bool = True,
) -> dict:
    """Compute an embedding from an image file path."""
    representations = _call_deepface(
        file_path,
        detector_backend=detector_backend,
        model_name=model_name,
        allow_normalize_kw=allow_normalize_kw,
    )
    return _select_best_representation(representations)


def represent_with_fallbacks(
    image_path_or_bytes: ImageInput,
    detector_backend: str,
    model_name: str,
    allow_normalize_kw: bool = True,
) -> dict:
    """Try DeepFace embeddings across detector fallbacks."""
    last_error: Exception | None = None
    cleanup_path: str | None = None
    if isinstance(image_path_or_bytes, (bytes, bytearray, memoryview)):
        cleanup_path = _bytes_to_temp_jpeg(image_path_or_bytes)
        path = cleanup_path
    else:
        path = image_path_or_bytes

    try:
        for backend in safe_detector_backend_chain(detector_backend):
            try:
                return represent_from_file(
                    path,
                    detector_backend=backend,
                    model_name=model_name,
                    allow_normalize_kw=allow_normalize_kw,
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                last_error = exc
                _LOGGER.debug("DeepFace backend %s failed: %s", backend, exc)
        if last_error:
            raise last_error
        raise ValueError("No face representations returned")
    finally:
        if cleanup_path:
            _safe_delete(cleanup_path)


def _call_deepface(
    img_path: str,
    detector_backend: str,
    model_name: str,
    allow_normalize_kw: bool,
) -> Iterable[dict]:
    kwargs = {
        "img_path": img_path,
        "detector_backend": detector_backend,
        "model_name": model_name,
        "enforce_detection": False,
        "align": True,
    }
    if allow_normalize_kw:
        kwargs["normalize"] = True
    try:
        result = DeepFace.represent(**kwargs)
    except TypeError:
        if not allow_normalize_kw:
            raise
        _LOGGER.debug("DeepFace.represent rejected normalize kwarg; retrying without it")
        result = DeepFace.represent(
            img_path=img_path,
            detector_backend=detector_backend,
            model_name=model_name,
            enforce_detection=False,
            align=True,
        )
    if isinstance(result, dict):
        return [result]
    return result


def _select_best_representation(representations: Iterable[dict]) -> dict:
    reps_list = list(representations)
    if not reps_list:
        raise ValueError("No face detected")
    best = max(reps_list, key=lambda rep: float(rep.get("face_confidence", 0.0)))
    confidence = float(best.get("face_confidence") or 0.0)
    if confidence < MIN_CONFIDENCE:
        raise ValueError(
            f"Face confidence {confidence:.3f} below threshold {MIN_CONFIDENCE:.3f}"
        )
    embedding = np.asarray(best.get("embedding"), dtype="float32")
    if embedding.ndim != 1:
        embedding = embedding.reshape(-1)
    if embedding.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Unexpected embedding size {embedding.shape[0]}")
    facial_area = best.get("facial_area") or best.get("region") or {}
    return {"embedding": embedding, "facial_area": facial_area, "confidence": confidence}


def _bytes_to_temp_jpeg(image_bytes: BytesLike) -> str:
    arr = np.frombuffer(bytes(image_bytes), dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Unable to decode image bytes")
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError("Failed to encode JPEG")
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_path = tmp.name
    tmp.close()
    with open(tmp_path, "wb") as handle:
        handle.write(encoded.tobytes())
    return tmp_path


def _safe_delete(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
