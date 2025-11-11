"""Google Sheets helper functions."""
from __future__ import annotations

import datetime as dt
import logging
import os
from typing import Iterable, Sequence

import pytz
from google.oauth2 import service_account
from googleapiclient.discovery import build as google_build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
_sheets_service = None
_credentials = None
_header_checked = False


class SheetsConfigError(RuntimeError):
    pass


def _load_credentials():
    global _credentials
    if _credentials is not None:
        return _credentials
    json_path = os.getenv("SERVICE_ACCOUNT_JSON")
    if not json_path:
        raise SheetsConfigError("SERVICE_ACCOUNT_JSON env var is required")
    if not os.path.exists(json_path):
        raise SheetsConfigError(f"Service account json not found: {json_path}")
    _credentials = service_account.Credentials.from_service_account_file(json_path, scopes=SCOPES)
    return _credentials


def _service():
    global _sheets_service
    if _sheets_service is not None:
        return _sheets_service
    creds = _load_credentials()
    _sheets_service = google_build("sheets", "v4", credentials=creds, cache_discovery=False)
    return _sheets_service


def append_row(row: Sequence[object]) -> None:
    sheet_id = os.getenv("SHEET_ID")
    sheet_tab = os.getenv("SHEET_TAB", "Posts")
    if not sheet_id:
        raise SheetsConfigError("SHEET_ID is required to append rows")
    body = {"values": [list(row)]}
    try:
        (
            _service()
            .spreadsheets()
            .values()
            .append(
                spreadsheetId=sheet_id,
                range=f"{sheet_tab}!A:Z",
                valueInputOption="RAW",
                insertDataOption="INSERT_ROWS",
                body=body,
            )
            .execute()
        )
    except HttpError as exc:
        logging.error("Sheets append error: %s", exc)
        raise


def ensure_header(header: Sequence[object]) -> None:
    sheet_id = os.getenv("SHEET_ID")
    sheet_tab = os.getenv("SHEET_TAB", "Posts")
    if not sheet_id:
        raise SheetsConfigError("SHEET_ID is required to ensure header")
    global _header_checked
    if _header_checked:
        return
    header_list = list(header)
    resp = (
        _service()
        .spreadsheets()
        .values()
        .get(spreadsheetId=sheet_id, range=f"{sheet_tab}!1:1")
        .execute()
    )
    existing = resp.get("values", [])
    if not existing or existing[0] != header_list:
        body = {"values": [header_list]}
        (
            _service()
            .spreadsheets()
            .values()
            .update(
                spreadsheetId=sheet_id,
                range=f"{sheet_tab}!A1",
                valueInputOption="RAW",
                body=body,
            )
            .execute()
        )
    _header_checked = True


def fetch_all_rows(range_a1: str | None = None) -> list[list[str]]:
    sheet_id = os.getenv("SHEET_ID")
    sheet_tab = os.getenv("SHEET_TAB", "Posts")
    if not sheet_id:
        raise SheetsConfigError("SHEET_ID is required to read rows")
    read_range = range_a1 or f"{sheet_tab}!A:Z"
    resp = (
        _service()
        .spreadsheets()
        .values()
        .get(spreadsheetId=sheet_id, range=read_range)
        .execute()
    )
    return resp.get("values", [])


def batch_update(range_a1: str, values: Iterable[Sequence[object]]) -> None:
    sheet_id = os.getenv("SHEET_ID")
    if not sheet_id:
        raise SheetsConfigError("SHEET_ID is required for batch update")
    body = {"values": [list(row) for row in values]}
    (
        _service()
        .spreadsheets()
        .values()
        .update(
            spreadsheetId=sheet_id,
            range=range_a1,
            valueInputOption="RAW",
            body=body,
        )
        .execute()
    )


def timestamp_ist() -> str:
    tz = pytz.timezone("Asia/Kolkata")
    return dt.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
