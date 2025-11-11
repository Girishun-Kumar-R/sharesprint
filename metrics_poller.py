"""LinkedIn engagement metrics poller."""
from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import requests
from urllib.parse import quote

import sheets

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)

REQUIRED_HEADERS = ["post_urn", "reactions", "comments", "reshares", "last_checked"]


def _column_letter(idx: int) -> str:
    """Convert 1-indexed column idx to A1 style letter."""
    result = ""
    while idx > 0:
        idx, remainder = divmod(idx - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _find_columns(header: List[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for idx, label in enumerate(header, start=1):
        key = label.strip().lower()
        if key in REQUIRED_HEADERS:
            mapping[key] = idx
    missing = [h for h in REQUIRED_HEADERS if h not in mapping]
    if missing:
        raise RuntimeError(f"Missing required sheet headers: {missing}")
    return mapping


def _fetch_metrics(token: str, post_urn: str) -> Tuple[int, int, int]:
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Restli-Protocol-Version": "2.0.0",
        "Linkedin-Version": "202504",
    }
    url = f"https://api.linkedin.com/rest/socialActions/{quote(post_urn, safe=':')}"
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"LinkedIn API error {resp.status_code}: {resp.text}")
    data = resp.json()
    reactions = (
        data.get("likesSummary", {}).get("totalLikes")
        or data.get("reactionsSummary", {}).get("total")
        or 0
    )
    comments = data.get("commentsSummary", {}).get("totalFirstLevelComments") or 0
    reshares = data.get("reshares", {}).get("shareCount") or 0
    return int(reactions), int(comments), int(reshares)


def main() -> None:
    token = os.getenv("LI_ACCESS_TOKEN")
    if not token:
        raise SystemExit("LI_ACCESS_TOKEN env var required for metrics polling")

    rows = sheets.fetch_all_rows()
    if not rows:
        logging.info("Sheet is empty; nothing to do")
        return

    header = rows[0]
    mapping = _find_columns([h.lower() for h in header])
    sheet_tab = os.getenv("SHEET_TAB", "Posts")
    updates: List[Tuple[str, List[str]]] = []
    for idx, row in enumerate(rows[1:], start=2):
        post_col = mapping["post_urn"]
        if len(row) < post_col:
            continue
        post_urn = row[post_col - 1].strip()
        if not post_urn:
            continue
        try:
            reactions, comments, reshares = _fetch_metrics(token, post_urn)
        except Exception as exc:  # pragma: no cover - network error logging
            logging.error("Failed to fetch metrics for %s: %s", post_urn, exc)
            continue
        row_values = list(row)
        while len(row_values) < max(mapping.values()):
            row_values.append("")
        row_values[mapping["reactions"] - 1] = str(reactions)
        row_values[mapping["comments"] - 1] = str(comments)
        row_values[mapping["reshares"] - 1] = str(reshares)
        row_values[mapping["last_checked"] - 1] = sheets.timestamp_ist()
        start_col = min(
            mapping["reactions"], mapping["comments"], mapping["reshares"], mapping["last_checked"]
        )
        end_col = max(
            mapping["reactions"], mapping["comments"], mapping["reshares"], mapping["last_checked"]
        )
        start_letter = _column_letter(start_col)
        end_letter = _column_letter(end_col)
        range_a1 = f"{sheet_tab}!{start_letter}{idx}:{end_letter}{idx}"
        updates.append((range_a1, [
            row_values[mapping["reactions"] - 1],
            row_values[mapping["comments"] - 1],
            row_values[mapping["reshares"] - 1],
            row_values[mapping["last_checked"] - 1],
        ]))

    for rng, values in updates:
        sheets.batch_update(rng, [values])
        logging.info("Updated %s", rng)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - cli guard
        logging.error("Metrics poller failed: %s", exc)
        sys.exit(1)
