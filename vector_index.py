"""FAISS backed face vector index helpers."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import faiss  # type: ignore
import numpy as np

EMBED_DIM = 512


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype("float32", copy=True)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vec / norms


@dataclass
class FaceRecord:
    face_id: int
    file_id: str
    bbox: str
    name: str
    mime: str
    modified: str


class FaceIndex:
    """Thin wrapper around a cosine FAISS index and SQLite metadata."""

    def __init__(
        self,
        db_path: str = "face_index.sqlite",
        index_path: str = "faces.index",
        idmap_path: str = "faces.idmap.json",
    ) -> None:
        self.db_path = db_path
        self.index_path = index_path
        self.idmap_path = idmap_path
        self.index: faiss.Index = faiss.IndexFlatIP(EMBED_DIM)
        self.id_map: List[int] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.idmap_path):
            try:
                loaded_index = faiss.read_index(self.index_path)
                if loaded_index.d != EMBED_DIM:
                    logging.warning(
                        "Existing FAISS index dim=%s but expected %s; rebuilding from DB",
                        loaded_index.d,
                        EMBED_DIM,
                    )
                    self.index = faiss.IndexFlatIP(EMBED_DIM)
                    self.id_map = []
                    self.rebuild_from_db()
                    return
                self.index = loaded_index
                with open(self.idmap_path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                self.id_map = [int(i) for i in payload.get("face_ids", [])]
                logging.info("Loaded FAISS index with %s vectors", len(self.id_map))
            except Exception as exc:  # pragma: no cover - defensive logging
                logging.error("Failed to load existing FAISS index: %s", exc)
                self.index = faiss.IndexFlatIP(EMBED_DIM)
                self.id_map = []
        else:
            logging.warning("FAISS index not found; starting with empty index")
            self.index = faiss.IndexFlatIP(EMBED_DIM)
            self.id_map = []

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.idmap_path, "w", encoding="utf-8") as fh:
            json.dump({"face_ids": self.id_map}, fh)

    def rebuild_from_db(self) -> None:
        logging.info("Rebuilding FAISS index from database")
        index, id_map = build_from_db(self.db_path)
        self.index = index
        self.id_map = id_map
        self.save()

    def add(self, face_id: int, vector: Sequence[float]) -> None:
        arr = np.asarray(vector, dtype="float32").reshape(1, -1)
        if arr.shape[1] != EMBED_DIM:
            raise ValueError(f"Expected vector dim {EMBED_DIM}, got {arr.shape[1]}")
        arr = _normalize(arr)
        self.index.add(arr)
        self.id_map.append(int(face_id))
        self.save()

    def search(self, vector: Sequence[float], top_k: int = 12) -> List[Tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        arr = np.asarray(vector, dtype="float32").reshape(1, -1)
        if arr.shape[1] != EMBED_DIM:
            raise ValueError(f"Expected vector dim {EMBED_DIM}, got {arr.shape[1]}")
        arr = _normalize(arr)
        scores, indices = self.index.search(arr, top_k)
        result = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.id_map):
                continue
            result.append((self.id_map[idx], float(score)))
        return result

    def resolve_faces(self, face_ids: Iterable[int]) -> List[FaceRecord]:
        ids = [int(fid) for fid in face_ids]
        if not ids:
            return []
        placeholders = ",".join(["?"] * len(ids))
        query = f"""
            SELECT faces.face_id, faces.file_id, faces.bbox,
                   files.name, files.mime, files.modified
            FROM faces
            JOIN files ON files.file_id = faces.file_id
            WHERE faces.face_id IN ({placeholders})
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, ids).fetchall()
        records = [
            FaceRecord(
                face_id=row[0],
                file_id=row[1],
                bbox=row[2],
                name=row[3],
                mime=row[4],
                modified=row[5],
            )
            for row in rows
        ]
        record_map = {rec.face_id: rec for rec in records}
        ordered = [record_map[fid] for fid in ids if fid in record_map]
        return ordered


def build_from_db(db_path: str) -> Tuple[faiss.Index, List[int]]:
    if not os.path.exists(db_path):
        logging.warning("Database %s not found; returning empty index", db_path)
        return faiss.IndexFlatIP(EMBED_DIM), []
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT face_id, vec FROM faces ORDER BY face_id").fetchall()
    if not rows:
        return faiss.IndexFlatIP(EMBED_DIM), []
    face_ids = []
    vectors = []
    for face_id, blob in rows:
        vec = np.frombuffer(blob, dtype="float32")
        if vec.shape[0] != EMBED_DIM:
            logging.warning("Skipping face_id %s due to incorrect dim %s", face_id, vec.shape)
            continue
        face_ids.append(int(face_id))
        vectors.append(vec)
    if not vectors:
        return faiss.IndexFlatIP(EMBED_DIM), []
    arr = np.stack(vectors, axis=0).astype("float32")
    arr = _normalize(arr)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(arr)
    logging.info("Built FAISS index with %s vectors", index.ntotal)
    return index, face_ids
