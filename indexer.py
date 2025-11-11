"""Drive index builder and watcher for ShareSprint with shared embedding utils.

Fixes / improvements:
- Adds missing DB helpers: `db()` context manager and `ensure_schema()`.
- Safer, idempotent schema + indices; WAL for perf; FK ON for cascade deletes.
- Embedding pipeline shares thresholds with app via env; logs clearly.
- Robust page-loop handling for Drive build + watch flows.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
import traceback
from contextlib import contextmanager
from typing import Iterable, List, Optional, Tuple, Union

from dotenv import load_dotenv

import face_embed
import drive
from vector_index import FaceIndex
import preview_cache

# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
load_dotenv()  # read .env from CWD

DB_PATH = os.getenv("FACE_DB_PATH", "face_index.sqlite")
INDEX_PATH = os.getenv("FACE_INDEX_PATH", "faces.index")
IDMAP_PATH = os.getenv("FACE_IDMAP_PATH", "faces.idmap.json")

# Accept both legacy FACE_* and app-level vars
DETECTOR_BACKEND = os.getenv("FACE_DETECTOR") or os.getenv("DETECTOR_BACKEND", "retinaface")
MODEL_NAME = os.getenv("FACE_MODEL") or os.getenv("MODEL_NAME", "ArcFace")
MIN_CONFIDENCE = float(
    os.getenv("FACE_MIN_CONF")
    or os.getenv("MIN_CONFIDENCE")
    or os.getenv("MIN_SCORE")
    or "0.85"
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)

face_embed.MIN_CONFIDENCE = MIN_CONFIDENCE
DETECTOR_CHAIN = face_embed.safe_detector_backend_chain(DETECTOR_BACKEND)

logging.info(
    "Face pipeline config model=%s detectors=%s min_conf=%.2f",
    MODEL_NAME,
    " -> ".join(DETECTOR_CHAIN),
    MIN_CONFIDENCE,
)

# ----------------------------------------------------------------------------
# SQLite helpers (missing before)
# ----------------------------------------------------------------------------

@contextmanager
def db(path: str | None = None):
    """Yield a SQLite connection with sane defaults.

    - WAL journal for concurrent reads
    - NORMAL synchronous for speed on Windows
    - foreign_keys ON for cascade cleanup
    """
    conn = sqlite3.connect(path or DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
        conn.commit()
    finally:
        conn.close()


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create required tables/indices if they don't exist."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS files (
            file_id   TEXT PRIMARY KEY,
            name      TEXT,
            mime      TEXT,
            modified  TEXT
        );

        CREATE TABLE IF NOT EXISTS faces (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL,
            bbox    TEXT,
            vec     BLOB NOT NULL,
            FOREIGN KEY(file_id) REFERENCES files(file_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_faces_file_id ON faces(file_id);
        """
    )


# ----------------------------------------------------------------------------
# Face pipeline helpers
# ----------------------------------------------------------------------------
BytesLike = Union[bytes, bytearray, memoryview]
DrivePayload = Union[BytesLike, Tuple[BytesLike, str]]


def _to_bytes(obj: DrivePayload) -> bytes:
    """Normalize different payloads to raw bytes."""
    if isinstance(obj, tuple):
        obj = obj[0]
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if isinstance(obj, memoryview):
        return obj.tobytes()
    raise TypeError(f"Unsupported image payload type: {type(obj)}")


def _face_embeddings(image_payload: DrivePayload) -> List[dict]:
    data = _to_bytes(image_payload)
    last_error: Exception | None = None
    for backend in DETECTOR_CHAIN:
        try:
            result = face_embed.represent_from_bytes(
                data,
                detector_backend=backend,
                model_name=MODEL_NAME,
            )
            logging.info(
                "Embedding created with backend %s (confidence=%.3f)",
                backend,
                result["confidence"],
            )
            bbox = json.dumps(result.get("facial_area", {}))
            return [
                {
                    "embedding": result["embedding"],
                    "bbox": bbox,
                    "confidence": result["confidence"],
                }
            ]
        except ValueError as exc:
            last_error = exc
            message = str(exc)
            if "Unexpected embedding size" in message:
                logging.warning("Skipping embedding from backend %s: %s", backend, message)
            else:
                logging.info("Backend %s produced no valid face: %s", backend, message)
        except Exception as exc:  # defensive: DeepFace/TF hiccups
            last_error = exc
            logging.warning("DeepFace error on backend %s: %s", backend, exc)
    if last_error:
        logging.debug("No embeddings generated: %s", last_error)
    return []


def _store_file(conn: sqlite3.Connection, meta: dict) -> None:
    conn.execute(
        """
        INSERT INTO files(file_id, name, mime, modified)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(file_id) DO UPDATE SET
            name=excluded.name,
            mime=excluded.mime,
            modified=excluded.modified
        """,
        (
            meta.get("id"),
            meta.get("name"),
            meta.get("mimeType"),
            meta.get("modifiedTime"),
        ),
    )


def _replace_faces(conn: sqlite3.Connection, file_id: str, faces: List[dict]) -> None:
    conn.execute("DELETE FROM faces WHERE file_id = ?", (file_id,))
    for face in faces:
        conn.execute(
            "INSERT INTO faces(file_id, bbox, vec) VALUES(?, ?, ?)",
            (file_id, face["bbox"], face["embedding"].tobytes()),
        )


def _get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


# ----------------------------------------------------------------------------
# Build & watch
# ----------------------------------------------------------------------------

def build_index(folder_id: str) -> None:
    logging.info("Starting full index build for folder %s", folder_id)
    index = FaceIndex(DB_PATH, INDEX_PATH, IDMAP_PATH)
    with db() as conn:
        ensure_schema(conn)

    page_token: Optional[str] = None
    processed = 0
    vector_count = 0

    while True:
        files, page_token = drive.list_event_images(folder_id, page_token=page_token)
        if not files:
            break
        with db() as conn:
            ensure_schema(conn)
            for meta in files:
                file_id = meta["id"]
                try:
                    content = drive.get_media(file_id)
                    try:
                        preview_cache.cache_from_bytes(file_id, content)
                    except ValueError:
                        logging.warning("Skipping preview cache for %s: undecodable image", meta.get("name"))
                    faces = _face_embeddings(content)
                    if not faces:
                        logging.info("No faces detected for %s", meta.get("name"))
                        continue
                    _store_file(conn, meta)
                    _replace_faces(conn, file_id, faces)
                    processed += 1
                    vector_count += len(faces)
                except Exception as exc:  # pragma: no cover - logging
                    logging.error("Failed to process %s: %s", meta.get("name"), exc)
                    logging.debug(traceback.format_exc())
        if not page_token:
            break

    logging.info(
        "Prepared %s vectors from %s files; rebuilding FAISS index",
        vector_count,
        processed,
    )
    index.rebuild_from_db()
    logging.info("Indexed %s files", processed)


def _process_change(file_meta: dict, index: FaceIndex) -> None:
    file_id = file_meta.get("id")
    mime = file_meta.get("mimeType", "")
    if not mime.startswith("image/"):
        logging.debug("Skipping non-image change %s", file_id)
        return
    content = drive.get_media(file_id)
    try:
        preview_cache.cache_from_bytes(file_id, content)
    except ValueError:
        logging.warning("Skipping preview cache for %s: undecodable image", file_meta.get("name"))
    faces = _face_embeddings(content)
    with db() as conn:
        ensure_schema(conn)
        _store_file(conn, file_meta)
        if faces:
            _replace_faces(conn, file_id, faces)
        else:
            conn.execute("DELETE FROM faces WHERE file_id=?", (file_id,))
            logging.info("No valid faces for %s; cleared stored embeddings", file_meta.get("name"))
    index.rebuild_from_db()
    logging.info("Updated index for %s", file_meta.get("name"))


def watch(folder_id: str, interval: int = 45) -> None:
    logging.info("Starting Drive changes watch loop")
    svc = drive.build()
    index = FaceIndex(DB_PATH, INDEX_PATH, IDMAP_PATH)

    while True:
        with db() as conn:
            ensure_schema(conn)
            start_token = _get_meta(conn, "drive_start_page_token")
            if not start_token:
                start_token = (
                    svc.changes()
                    .getStartPageToken(supportsAllDrives=True)
                    .execute()
                    .get("startPageToken")
                )
                _set_meta(conn, "drive_start_page_token", start_token)
        page_token = start_token

        while page_token:
            response = (
                svc.changes()
                .list(
                    pageToken=page_token,
                    spaces="drive",
                    fields=(
                        "nextPageToken,newStartPageToken,"
                        "changes(fileId,file(name,mimeType,modifiedTime,parents),removed)"
                    ),
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                )
                .execute()
            )
            for change in response.get("changes", []):
                file_obj = change.get("file")
                removed = change.get("removed")
                file_id = change.get("fileId")
                if removed:
                    with db() as conn:
                        conn.execute("DELETE FROM faces WHERE file_id=?", (file_id,))
                        conn.execute("DELETE FROM files WHERE file_id=?", (file_id,))
                    preview_cache.delete_cache(file_id)
                    index.rebuild_from_db()
                    continue
                if not file_obj:
                    continue
                parents = file_obj.get("parents", [])
                if folder_id not in parents:
                    continue
                file_meta = drive.get_meta(file_id)
                _process_change(file_meta, index)

            page_token = response.get("nextPageToken")
            new_token = response.get("newStartPageToken")
            if new_token:
                with db() as conn:
                    _set_meta(conn, "drive_start_page_token", new_token)

        time.sleep(interval)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Drive face index maintenance tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build", help="Perform a full rebuild")
    watch_parser = subparsers.add_parser("watch", help="Monitor Drive and incrementally update the index")
    watch_parser.add_argument(
        "--interval",
        type=int,
        default=45,
        help="Poll interval in seconds between Drive change checks",
    )

    args = parser.parse_args()

    folder_id = os.getenv("FOLDER_ID")
    if not folder_id:
        raise SystemExit("FOLDER_ID environment variable is required")

    if args.command == "build":
        build_index(folder_id)
    else:
        watch(folder_id, interval=getattr(args, "interval", 45))


if __name__ == "__main__":
    main()
