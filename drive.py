"""Google Drive client helpers for ShareSprint app."""
from __future__ import annotations

import io
import logging
import os
from typing import Dict, List, Optional, Tuple

from google.oauth2 import service_account
from googleapiclient.discovery import build as google_build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
]

_drive_client = None
_drive_credentials = None


class DriveConfigError(RuntimeError):
    """Raised when Drive configuration is missing."""


def _load_credentials():
    global _drive_credentials
    if _drive_credentials is not None:
        return _drive_credentials
    json_path = os.getenv("SERVICE_ACCOUNT_JSON")
    if not json_path:
        raise DriveConfigError("SERVICE_ACCOUNT_JSON env var is required")
    if not os.path.exists(json_path):
        raise DriveConfigError(f"Service account json not found: {json_path}")
    _drive_credentials = service_account.Credentials.from_service_account_file(
        json_path, scopes=SCOPES
    )
    return _drive_credentials


def build(force: bool = False):
    """Return a cached Drive API client."""
    global _drive_client
    if _drive_client is not None and not force:
        return _drive_client
    creds = _load_credentials()
    _drive_client = google_build("drive", "v3", credentials=creds, cache_discovery=False)
    return _drive_client


def list_event_images(
    folder_id: str,
    page_token: Optional[str] = None,
    page_size: int = 100,
    order_by: str = "modifiedTime desc",
) -> Tuple[List[Dict], Optional[str]]:
    """List image files within the provided Drive folder."""
    service = build()
    query = (
        f"'{folder_id}' in parents and mimeType contains 'image/' and trashed = false"
    )
    try:
        resp = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                orderBy=order_by,
                pageToken=page_token,
                pageSize=page_size,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            .execute()
        )
    except HttpError as exc:
        logging.error("Drive list error: %s", exc)
        raise
    files = resp.get("files", [])
    return files, resp.get("nextPageToken")


def get_meta(file_id: str) -> Dict:
    service = build()
    try:
        metadata = (
            service.files()
            .get(
                fileId=file_id,
                fields="id, name, mimeType, modifiedTime, size",
                supportsAllDrives=True,
            )
            .execute()
        )
    except HttpError as exc:
        logging.error("Drive metadata error for %s: %s", file_id, exc)
        raise
    return metadata


def get_media(file_id: str) -> bytes:
    service = build()
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    file_buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(file_buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return file_buffer.getvalue()
