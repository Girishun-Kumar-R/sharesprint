"""ShareSprint Flask application with resilient DeepFace selfie handling."""
from __future__ import annotations

import io
import logging
import os
import secrets
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote, urlencode

import requests
from flask import (
    Flask,
    abort,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from googleapiclient.errors import HttpError
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from PIL import Image, ImageDraw, ImageFont
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

import face_embed
import drive
import sheets
import preview_cache
from vector_index import FaceIndex, FaceRecord

try:
    import pillow_avif_plugin  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    pass

load_dotenv()

ALLOW_DEV = os.getenv("ALLOW_DEV_DEFAULTS") == "1"
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    if not ALLOW_DEV:
        raise RuntimeError(
            "SECRET_KEY is required. Set ALLOW_DEV_DEFAULTS=1 to auto-generate for dev."
        )
    SECRET_KEY = secrets.token_hex(32)

# Optional dev event id (only used when ALLOW_DEV_DEFAULTS=1 and session lacks event_id)
DEV_EVENT_ID = os.getenv("DEV_EVENT_ID")

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=False,
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

DB_PATH = os.getenv("FACE_DB_PATH", "face_index.sqlite")
INDEX_PATH = os.getenv("FACE_INDEX_PATH", "faces.index")
IDMAP_PATH = os.getenv("FACE_IDMAP_PATH", "faces.idmap.json")
FOLDER_ID = os.getenv("FOLDER_ID")
PREVIEW_MAX = int(os.getenv("PREVIEW_MAX_PX", "1024"))
WATERMARK_PREFIX = os.getenv("WATERMARK_PREFIX", "ShareSprint Preview")
# Accept both old and new variable names
DETECTOR_BACKEND = os.getenv("FACE_DETECTOR") or os.getenv("DETECTOR_BACKEND", "retinaface")
MODEL_NAME = os.getenv("FACE_MODEL") or os.getenv("MODEL_NAME", "ArcFace")
MIN_CONFIDENCE = float(
    os.getenv("FACE_MIN_CONF")
    or os.getenv("MIN_CONFIDENCE")
    or os.getenv("MIN_SCORE", "0.85")
)
MATCH_MIN_SCORE = float(os.getenv("MATCH_MIN_SCORE", "0.35"))
MATCH_TOP_K = max(1, int(os.getenv("MATCH_TOP_K", "12")))
MATCH_DISPLAY_LIMIT = max(1, int(os.getenv("MATCH_DISPLAY_LIMIT", "6")))
MATCH_FALLBACK_RESULTS = max(1, int(os.getenv("MATCH_FALLBACK_RESULTS", "3")))
LAST_POST_SESSION_KEY = "last_post_result"

face_embed.MIN_CONFIDENCE = MIN_CONFIDENCE
DETECTOR_CHAIN = face_embed.safe_detector_backend_chain(DETECTOR_BACKEND)

face_index = FaceIndex(DB_PATH, INDEX_PATH, IDMAP_PATH)
_preview_serializer = URLSafeTimedSerializer(app.secret_key, salt="preview-token")

APP_START_TS = time.time()


def _parse_id_list(raw: str) -> List[str]:
    result: List[str] = []
    seen = set()
    for chunk in (raw or "").split(","):
        trimmed = chunk.strip()
        if not trimmed or trimmed in seen:
            continue
        seen.add(trimmed)
        result.append(trimmed)
    return result


def get_mandatory_file_ids(event_id: Optional[str]) -> List[str]:
    """Return ordered list of mandatory Drive file ids for the event."""
    scoped = (event_id or "").strip().upper()
    if scoped:
        override = os.getenv(f"MANDATORY_FILE_IDS_{scoped}")
        if override:
            return _parse_id_list(override)
    base = os.getenv("MANDATORY_FILE_IDS", "")
    return _parse_id_list(base)


def _normalize_file_ids(raw_values: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for value in raw_values:
        candidate = (value or "").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def _merge_with_mandatory(selected: List[str], mandatory: List[str]) -> List[str]:
    if not mandatory:
        return selected
    seen = set(selected)
    merged = list(selected)
    for fid in mandatory:
        if fid and fid not in seen:
            merged.append(fid)
            seen.add(fid)
    return merged


def _is_truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


SHEETS_HEADER = [
    "event_id",
    "timestamp_ist",
    "linkedin_person_urn",
    "post_urn",
    "post_url",
    "drive_file_ids",
    "drive_file_names",
    "template_id",
    "caption_text",
    "status",
]

logger.info(
    "Face pipeline config model=%s detectors=%s min_conf=%.2f",
    MODEL_NAME,
    " -> ".join(DETECTOR_CHAIN),
    MIN_CONFIDENCE,
)


def _sqlite_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# --------------------------- LinkedIn helpers -------------------------------

def _linkedin_token_valid() -> bool:
    expires_at = session.get("li_expires_at")
    token = session.get("li_access_token")
    if not token or not expires_at:
        return False
    return time.time() < expires_at - 60


def _require_linkedin_auth() -> None:
    if not _linkedin_token_valid():
        session["post_auth_redirect"] = request.path
        abort(401)


# --------------------------- Event context helpers -------------------------

def _ensure_event_id() -> str:
    """Ensure an event_id exists in session; seed from query/form; allow DEV_EVENT_ID."""
    ev = session.get("event_id")
    # seed from incoming request (query or form)
    if not ev:
        candidate = request.args.get("event") or request.form.get("event_id")
        if candidate:
            ev = secure_filename(candidate)
            session["event_id"] = ev
    # allow dev default for local testing if provided
    if not ev and ALLOW_DEV and DEV_EVENT_ID:
        ev = secure_filename(DEV_EVENT_ID)
        session["event_id"] = ev
    if not ev:
        abort(400, description="Missing event context. Start from QR (…/start?event=YOUR_EVENT).")
    return ev


# Also seed event context automatically on every request when provided
@app.before_request
def _seed_event_from_request():
    candidate = request.args.get("event") or request.form.get("event_id")
    if candidate:
        session["event_id"] = secure_filename(candidate)


# --------------------------- Preview token helpers -------------------------

def _generate_preview_token(file_id: str, ttl: int = 300) -> str:
    return _preview_serializer.dumps({"file_id": file_id, "ttl": ttl})


def _verify_preview_token(file_id: str, token: str) -> None:
    try:
        data = _preview_serializer.loads(token, max_age=300)
    except (SignatureExpired, BadSignature):
        abort(403)
    if data.get("file_id") != file_id:
        abort(403)


# --------------------------- Image / face helpers --------------------------

def _selfie_embedding(image_bytes: bytes) -> Dict:
    """Return 512-d embedding for the strongest face using shared utilities."""
    logger.debug("Computing selfie embedding with detector chain: %s", " -> ".join(DETECTOR_CHAIN))
    result = face_embed.represent_with_fallbacks(
        image_bytes,
        detector_backend=DETECTOR_BACKEND,
        model_name=MODEL_NAME,
    )
    logger.info("Selfie embedding confidence %.3f", result["confidence"])
    return result


# --------------------------- DB helpers ------------------------------------

def _fetch_file_record(file_id: str) -> Optional[sqlite3.Row]:
    with _sqlite_conn() as conn:
        row = conn.execute(
            "SELECT file_id, name, mime, modified FROM files WHERE file_id = ?",
            (file_id,),
        ).fetchone()
    return row


def _fetch_file_records(file_ids: Iterable[str]) -> Dict[str, sqlite3.Row]:
    unique_ids = []
    seen = set()
    for fid in file_ids:
        if not fid or fid in seen:
            continue
        seen.add(fid)
        unique_ids.append(fid)
    if not unique_ids:
        return {}
    placeholders = ",".join(["?"] * len(unique_ids))
    query = f"SELECT file_id, name, mime, modified FROM files WHERE file_id IN ({placeholders})"
    with _sqlite_conn() as conn:
        rows = conn.execute(query, unique_ids).fetchall()
    return {row["file_id"]: row for row in rows}


def _fetch_recent_files(page: int, page_size: int = 48) -> List[sqlite3.Row]:
    offset = max(page - 1, 0) * page_size
    with _sqlite_conn() as conn:
        rows = conn.execute(
            "SELECT file_id, name, mime, modified FROM files ORDER BY datetime(modified) DESC LIMIT ? OFFSET ?",
            (page_size, offset),
        ).fetchall()
    return rows


# --------------------------- Rate limit ------------------------------------

def _rate_limit_post() -> bool:
    window_seconds = 300
    max_attempts = 3
    now = time.time()
    attempts = session.get("post_attempts", [])
    attempts = [t for t in attempts if now - t < window_seconds]
    if len(attempts) >= max_attempts:
        return False
    attempts.append(now)
    session["post_attempts"] = attempts
    session.modified = True
    return True


# --------------------------- Sheets logging --------------------------------

def _append_sheet_row(
    event_id: str,
    post_urn: str,
    post_url: str,
    files: List[dict],
    caption: str,
    template_id: Optional[str],
) -> None:
    sheets.ensure_header(SHEETS_HEADER)
    file_ids = ";".join(filter(None, [meta.get("id") for meta in files]))
    file_names = ";".join(filter(None, [meta.get("name") for meta in files]))
    row = [
        event_id,
        sheets.timestamp_ist(),
        session.get("li_person_urn"),
        post_urn,
        post_url,
        file_ids,
        file_names,
        template_id or "custom",
        caption,
        "posted",
    ]
    sheets.append_row(row)


# --------------------------- Flask hooks & errors --------------------------

@app.before_request
def _session_keepalive():
    session.permanent = True
    session["last_active"] = time.time()


@app.after_request
def _apply_security_headers(response):
    csp = (
        "default-src 'self'; "
        "img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; "
        "frame-ancestors 'none'"
    )
    response.headers.setdefault("Content-Security-Policy", csp)
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("X-Frame-Options", "DENY")
    return response


@app.errorhandler(HTTPException)
def _http_error(err: HTTPException):
    payload = {"error": err.name, "details": err.description}
    wants_json = (
        request.accept_mimetypes.best == "application/json" or request.path.startswith("/post/")
    )
    if wants_json:
        return jsonify(payload), err.code
    return f"{err.name}: {err.description}", err.code


# --------------------------- Routes ----------------------------------------

@app.route("/health")
def health():
    with _sqlite_conn() as conn:
        face_count = conn.execute("SELECT COUNT(1) FROM faces").fetchone()[0]
    uptime = time.time() - APP_START_TS
    return jsonify(
        {
            "uptime": round(uptime, 2),
            "index_loaded": face_index.index.ntotal,
            "db_rows": face_count,
            "drive_folder_id": FOLDER_ID,
        }
    )


@app.route("/start")
def start_flow():
    # Seed event id if provided
    ev = request.args.get("event")
    if ev:
        session["event_id"] = secure_filename(ev)
        session.modified = True
    if not _linkedin_token_valid():
        session["post_auth_redirect"] = url_for("upload_form")
        return redirect(url_for("linkedin_login"))
    return redirect(url_for("upload_form"))


@app.route("/reset")
def reset_flow():
    keys_to_clear = [
        "flow_step",
        "caption_from",
        "last_matches_payload",
        "post_form",
        "post_error",
        "post_attempts",
        LAST_POST_SESSION_KEY,
    ]
    for key in keys_to_clear:
        session.pop(key, None)
    session.modified = True
    event_id = request.args.get("event") or session.get("event_id")
    if event_id:
        return redirect(url_for("upload_form", event=event_id))
    return redirect(url_for("start_flow"))


@app.route("/upload")
def upload_form():
    if not _linkedin_token_valid():
        session["post_auth_redirect"] = url_for("upload_form")
        return redirect(url_for("linkedin_login"))
    event_id = _ensure_event_id()
    _set_flow_step(1)
    return render_template(
        "upload.html",
        event_id=event_id,
        current_step=1,
        max_step=session.get("flow_step", 1),
    )


@app.route("/match", methods=["POST"])
def match_faces():
    if not _linkedin_token_valid():
        abort(401)
    event_id = _ensure_event_id()
    if "selfie" not in request.files:
        abort(400, description="No selfie uploaded")
    selfie = request.files["selfie"]
    image_bytes = selfie.read()
    if not image_bytes:
        abort(400, description="Empty upload")
    logger.info(
        "Received /match selfie for event %s (%d bytes)",
        event_id,
        len(image_bytes),
    )
    try:
        result = _selfie_embedding(image_bytes)
    except ValueError as exc:
        logger.warning("Selfie embedding failed: %s", exc)
        abort(400, description=str(exc))
    matches = face_index.search(result["embedding"], top_k=MATCH_TOP_K)
    logger.info("FAISS search returned %d matches", len(matches))
    records = face_index.resolve_faces([face_id for face_id, _ in matches])
    best_by_file: Dict[str, Tuple[FaceRecord, float]] = {}
    for record, (_, score) in zip(records, matches):
        if record is None:
            continue
        existing = best_by_file.get(record.file_id)
        if existing and score <= existing[1]:
            continue
        best_by_file[record.file_id] = (record, float(score))

    sorted_candidates = sorted(best_by_file.values(), key=lambda item: item[1], reverse=True)
    strong_candidates = [item for item in sorted_candidates if item[1] >= MATCH_MIN_SCORE]
    low_confidence = False

    if strong_candidates:
        chosen_candidates = strong_candidates
        if len(sorted_candidates) > len(strong_candidates):
            logger.info(
                "Filtered out %d low-similarity matches below %.2f",
                len(sorted_candidates) - len(strong_candidates),
                MATCH_MIN_SCORE,
            )
    else:
        fallback_limit = MATCH_FALLBACK_RESULTS
        chosen_candidates = sorted_candidates[:fallback_limit]
        low_confidence = bool(chosen_candidates)
        if low_confidence:
            logger.info(
                "No matches met the %.2f threshold; returning top %d fallback result(s)",
                MATCH_MIN_SCORE,
                len(chosen_candidates),
            )

    display_candidates = chosen_candidates[:MATCH_DISPLAY_LIMIT] if strong_candidates or low_confidence else []
    ranked_rows = [
        {
            "file_id": record.file_id,
            "name": record.name,
            "score": round(score, 3),
        }
        for record, score in display_candidates
    ]

    if not ranked_rows:
        logger.info("No usable matches found for selfie; prompt user to browse full gallery.")
    else:
        logger.info("Returning %d match cards after dedupe and filtering", len(ranked_rows))

    session["last_matches_payload"] = {
        "matches": ranked_rows,
        "low_confidence": low_confidence,
        "timestamp": time.time(),
    }
    session.modified = True
    _set_flow_step(2)
    context = _build_match_view(ranked_rows, event_id, low_confidence)
    return render_template("choose_photo.html", **context)


@app.route("/matches")
def resume_matches():
    if not _linkedin_token_valid():
        abort(401)
    payload = session.get("last_matches_payload")
    if not payload or not payload.get("matches"):
        return redirect(url_for("upload_form"))
    event_id = _ensure_event_id()
    _set_flow_step(2)
    context = _build_match_view(
        payload.get("matches", []),
        event_id,
        bool(payload.get("low_confidence")),
    )
    return render_template("choose_photo.html", **context)


@app.route("/all")
def all_photos():
    if not _linkedin_token_valid():
        abort(401)
    event_id = _ensure_event_id()
    page = int(request.args.get("p", "1"))
    files = _fetch_recent_files(page)
    items = [
        {
            "file_id": row["file_id"],
            "name": row["name"],
            "preview": url_for("preview_image", file_id=row["file_id"], t=_generate_preview_token(row["file_id"])),
        }
        for row in files
    ]
    mandatory_ids = get_mandatory_file_ids(event_id)
    present = set()
    for photo in items:
        locked = photo["file_id"] in mandatory_ids
        photo["is_mandatory"] = locked
        if locked:
            present.add(photo["file_id"])
    missing_required = [fid for fid in mandatory_ids if fid not in present]
    required_cards = []
    if missing_required:
        required_meta = _fetch_file_records(missing_required)
        for fid in mandatory_ids:
            if fid in present:
                continue
            row = required_meta.get(fid)
            if not row:
                logger.warning("Mandatory file %s missing from DB for event %s", fid, event_id)
                continue
            required_cards.append(
                {
                    "file_id": fid,
                    "name": row["name"],
                    "preview": url_for("preview_image", file_id=fid, t=_generate_preview_token(fid)),
                    "is_mandatory": True,
                }
            )
    _set_flow_step(3)
    return render_template(
        "all_photos.html",
        photos=items,
        page=page,
        event_id=event_id,
        mandatory_cards=required_cards,
        mandatory_ids=mandatory_ids,
        current_step=3,
        max_step=session.get("flow_step", 1),
    )


CAPTION_TEMPLATES = [
    {"id": "cheer", "label": "Celebrating the ShareSprint event!"},
    {"id": "team", "label": "Grateful for this amazing community."},
    {"id": "learn", "label": "Learning, sharing, sprinting forward."},
]


@app.route("/choose_caption")
def choose_caption():
    if not _linkedin_token_valid():
        abort(401)
    event_id = _ensure_event_id()
    origin_param = (request.args.get("origin") or "").strip().lower()
    if origin_param and origin_param not in {"all", "matches", "upload"}:
        origin_param = None
    if origin_param:
        _set_caption_origin(origin_param)
        origin = origin_param
    else:
        origin = session.get("caption_from")
        if not origin:
            origin = "all"
            _set_caption_origin(origin)
    skip_only = request.args.get("skip") == "1"
    raw_ids = [] if skip_only else request.args.getlist("file_id")
    if not raw_ids and not skip_only:
        abort(400, description="Missing file id")
    selected_ids = _normalize_file_ids(raw_ids)
    if skip_only:
        selected_ids = []
    mandatory_ids = get_mandatory_file_ids(event_id)
    merged_ids = _merge_with_mandatory(selected_ids, mandatory_ids)
    text_only_mode = skip_only and not merged_ids
    if not merged_ids and not text_only_mode:
        abort(400, description="At least one photo must be selected")
    records = _fetch_file_records(merged_ids) if merged_ids else {}
    preview_items = []
    for fid in merged_ids:
        row = records.get(fid)
        if not row:
            abort(404, description=f"Image not indexed yet: {fid}")
        preview_items.append(
            {
                "file_id": fid,
                "name": row["name"],
                "preview": url_for("preview_image", file_id=fid, t=_generate_preview_token(fid)),
                "is_mandatory": fid in mandatory_ids,
            }
        )
    primary = preview_items[0] if preview_items else None
    post_error = session.pop("post_error", None)
    form_state = session.pop("post_form", None) or {}
    selected_template = form_state.get("template_id") or "custom"
    if selected_template not in {tpl["id"] for tpl in CAPTION_TEMPLATES}:
        selected_template = "custom"
    prefill_caption = form_state.get("caption", "")
    _set_flow_step(4)
    return render_template(
        "choose_caption.html",
        file_ids=[item["file_id"] for item in preview_items],
        images=preview_items,
        primary_image=primary,
        text_only_mode=text_only_mode,
        caption_templates=CAPTION_TEMPLATES,
        event_id=event_id,
        post_error=post_error,
        selected_template=selected_template,
        prefill_caption=prefill_caption,
        origin=origin,
        current_step=4,
        max_step=session.get("flow_step", 1),
    )


@app.route("/post/linkedin", methods=["POST"])
def post_linkedin():
    _require_linkedin_auth()
    event_id = _ensure_event_id()

    expects_json = request.is_json or request.accept_mimetypes.best == "application/json"
    payload = request.get_json(silent=True) if request.is_json else request.form
    raw_caption = payload.get("caption", "") or ""
    template_id = payload.get("template_id")
    template_labels = {tpl["id"]: tpl["label"] for tpl in CAPTION_TEMPLATES}
    template_id = (template_id or "custom").strip() or "custom"
    if template_id not in template_labels:
        template_id = "custom"

    def _extract_file_ids() -> List[str]:
        if request.is_json:
            data = payload.get("file_ids")
            if isinstance(data, str):
                return _normalize_file_ids(data.split(","))
            if isinstance(data, list):
                return _normalize_file_ids(data)
            single = payload.get("file_id")
            if isinstance(single, list):
                return _normalize_file_ids(single)
            if isinstance(single, str):
                return _normalize_file_ids([single])
            return []
        getlist = getattr(payload, "getlist", None)
        if callable(getlist):
            values = payload.getlist("file_id")
            if values:
                return _normalize_file_ids(values)
            values = payload.getlist("file_ids")
            if values:
                return _normalize_file_ids(values)
        single = payload.get("file_ids") or payload.get("file_id")
        if isinstance(single, list):
            return _normalize_file_ids(single)
        if isinstance(single, str):
            if "," in single:
                return _normalize_file_ids(single.split(","))
            return _normalize_file_ids([single])
        return []

    selected_ids = _extract_file_ids()
    mandatory_ids = get_mandatory_file_ids(event_id)
    file_ids = _merge_with_mandatory(selected_ids, mandatory_ids)
    allow_text_only = _is_truthy(payload.get("allow_text_only"))
    text_only_mode = not file_ids and allow_text_only

    def _resolve_caption(raw_text: str) -> str:
        cleaned = (raw_text or "").strip()
        if cleaned:
            return cleaned
        if template_id != "custom":
            return template_labels.get(template_id, "")
        return ""

    def _persist_form_error(message: str, details: Optional[str] = None, status: int = 400):
        combined = f"{message}: {details}" if details else message
        if expects_json:
            body = {"error": message}
            if details:
                body["details"] = details
            return jsonify(body), status
        session["post_error"] = combined
        session["post_form"] = {
            "caption": _resolve_caption(raw_caption),
            "template_id": template_id,
            "file_ids": file_ids,
            "allow_text_only": allow_text_only,
        }
        session.modified = True
        if not file_ids:
            if allow_text_only:
                origin = session.get("caption_from")
                if origin:
                    return redirect(url_for("choose_caption", skip="1", origin=origin))
                return redirect(url_for("choose_caption", skip="1"))
            return redirect(url_for("upload_form"))
        params = [("file_id", fid) for fid in file_ids]
        origin = session.get("caption_from")
        if origin:
            params.append(("origin", origin))
        query = urlencode(params)
        return redirect(f"{url_for('choose_caption')}?{query}")

    if not _rate_limit_post():
        return _persist_form_error("Too many attempts", status=429)

    if not file_ids and not text_only_mode:
        return _persist_form_error("At least one photo must be selected", status=400)

    final_caption_text = _resolve_caption(raw_caption)
    post_caption = final_caption_text or " "
    uploads: List[Dict[str, object]] = []
    for fid in file_ids:
        try:
            meta = drive.get_meta(fid)
            image_bytes = drive.get_media(fid)
        except HttpError as exc:
            logger.error("Drive error during post for %s: %s", fid, exc)
            return _persist_form_error("Drive fetch failed", getattr(exc, "reason", str(exc)), 502)
        uploads.append({"file_id": fid, "meta": meta, "bytes": image_bytes})

    token = session.get("li_access_token")
    person_urn = session.get("li_person_urn")
    if not token or not person_urn:
        return _persist_form_error("Not authorized", status=401)

    headers = {"Authorization": f"Bearer {token}"}

    def _register_and_upload(single_meta: dict, image_bytes: bytes) -> str:
        mime = single_meta.get("mimeType", "image/jpeg") or "image/jpeg"
        register_body = {
            "registerUploadRequest": {
                "owner": person_urn,
                "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
                "serviceRelationships": [
                    {"relationshipType": "OWNER", "identifier": "urn:li:userGeneratedContent"}
                ],
            }
        }
        reg_resp = requests.post(
            "https://api.linkedin.com/v2/assets?action=registerUpload",
            headers={**headers, "Content-Type": "application/json"},
            json=register_body,
            timeout=20,
        )
        if reg_resp.status_code >= 300:
            raise RuntimeError(f"LinkedIn register failed: {reg_resp.text}")
        reg_data = reg_resp.json().get("value", {})
        upload_mech = (
            reg_data.get("uploadMechanism", {}).get("com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest", {})
        )
        upload_url = upload_mech.get("uploadUrl")
        asset_urn = reg_data.get("asset")
        if not upload_url or not asset_urn:
            raise RuntimeError("LinkedIn register response incomplete")
        put_headers = {"Authorization": f"Bearer {token}", "Content-Type": mime}
        put_resp = requests.put(upload_url, headers=put_headers, data=image_bytes, timeout=30)
        if put_resp.status_code >= 300:
            raise RuntimeError(f"LinkedIn upload failed: {put_resp.text}")
        return asset_urn

    media_entries = []
    for item in uploads:
        try:
            asset_urn = _register_and_upload(item["meta"], item["bytes"])
        except RuntimeError as exc:
            logger.error("LinkedIn upload sequence failed for %s: %s", item["file_id"], exc)
            return _persist_form_error("LinkedIn upload failed", str(exc), 502)
        media_entries.append(
            {
                "status": "READY",
                "media": asset_urn,
                "title": {"text": item["meta"].get("name") or "Event photo"},
            }
        )

    # Step 2: create UGC post referencing the asset URN
    share_content = {
        "shareCommentary": {"text": post_caption},
        "shareMediaCategory": "IMAGE" if media_entries else "NONE",
    }
    if media_entries:
        share_content["media"] = media_entries
    ugc_body = {
        "author": person_urn,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": share_content
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    }
    ugc_headers = {
        **headers,
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
    }
    ugc_resp = requests.post(
        "https://api.linkedin.com/v2/ugcPosts",
        headers=ugc_headers,
        json=ugc_body,
        timeout=20,
    )
    if ugc_resp.status_code >= 300:
        return _persist_form_error("LinkedIn UGC post failed", ugc_resp.text, 502)

    post_data = {}
    if ugc_resp.content:
        try:
            post_data = ugc_resp.json()
        except ValueError as exc:
            logger.warning("LinkedIn UGC response was not JSON: %s", exc)
            post_data = {}
    post_urn = post_data.get("id") or ugc_resp.headers.get("x-restli-id")
    if not post_urn:
        return _persist_form_error("LinkedIn response missing post URN", status=502)
    post_url = f"https://www.linkedin.com/feed/update/{quote(post_urn, safe='')}"

    sheet_logged = True
    sheet_error: Optional[str] = None
    try:
        _append_sheet_row(
            event_id,
            post_urn,
            post_url,
            [item["meta"] for item in uploads],
            final_caption_text,
            template_id,
        )
    except sheets.SheetsConfigError as exc:
        sheet_logged = False
        sheet_error = str(exc)
        logger.error("Sheets configuration error: %s", exc)
    except HttpError as exc:
        sheet_logged = False
        sheet_error = getattr(exc, "reason", str(exc))
        logger.error("Sheets append error: %s", exc)
    except Exception as exc:  # pragma: no cover
        sheet_logged = False
        sheet_error = str(exc)
        logger.exception("Unexpected failure appending row to Sheets")

    last_post_payload = {
        "event_id": event_id,
        "file_ids": file_ids,
        "post_urn": post_urn,
        "post_url": post_url,
        "images": [
            {"file_id": item["file_id"], "name": item["meta"].get("name")}
            for item in uploads
        ],
        "caption": final_caption_text,
        "sheet_logged": sheet_logged,
        "sheet_error": sheet_error,
        "text_only": text_only_mode,
    }
    session[LAST_POST_SESSION_KEY] = last_post_payload
    session.modified = True
    session.pop("post_form", None)
    session.pop("post_error", None)

    response_payload = {
        "ok": True,
        "post_urn": post_urn,
        "post_url": post_url,
        "file_ids": file_ids,
        "images": last_post_payload["images"],
        "text_only": text_only_mode,
        "redirect": url_for("post_success"),
        "sheet_logged": sheet_logged,
    }
    if sheet_error:
        response_payload["sheet_error"] = sheet_error
    if expects_json:
        return jsonify(response_payload)
    return redirect(url_for("post_success"))


# --------------------------- Post-success view -----------------------------


@app.route("/post/success")
def post_success():
    _require_linkedin_auth()
    payload = session.pop(LAST_POST_SESSION_KEY, None)
    if not payload:
        return redirect(url_for("upload_form"))
    stored_event = payload.get("event_id")
    if stored_event:
        session["event_id"] = stored_event
        session.modified = True
    event_id = _ensure_event_id()
    attachments = []
    for item in payload.get("images", []):
        fid = item.get("file_id")
        if not fid:
            continue
        attachments.append(
            {
                "file_id": fid,
                "name": item.get("name"),
                "preview": url_for("preview_image", file_id=fid, t=_generate_preview_token(fid)),
            }
        )
    if not attachments:
        legacy_id = payload.get("file_id")
        if legacy_id:
            attachments.append(
                {
                    "file_id": legacy_id,
                    "name": payload.get("image_name"),
                    "preview": url_for(
                        "preview_image", file_id=legacy_id, t=_generate_preview_token(legacy_id)
                    ),
                }
            )
    text_only = bool(payload.get("text_only")) or not attachments
    _set_flow_step(5)
    return render_template(
        "post_success.html",
        event_id=event_id,
        post_url=payload.get("post_url"),
        post_urn=payload.get("post_urn"),
        attachments=attachments,
        caption=payload.get("caption", ""),
        sheet_logged=payload.get("sheet_logged", True),
        sheet_error=payload.get("sheet_error"),
        primary_preview=attachments[0] if attachments else None,
        text_only=text_only,
        current_step=5,
        max_step=session.get("flow_step", 1),
    )


# --------------------------- OAuth routes ----------------------------------

@app.route("/auth/linkedin/login")
def linkedin_login():
    client_id = os.getenv("LINKEDIN_CLIENT_ID")
    redirect_uri = _linkedin_redirect_uri()
    if not client_id or not redirect_uri:
        abort(500, description="LinkedIn OAuth not configured")
    state = secrets.token_urlsafe(16)
    session["oauth_state"] = state
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": "openid profile email w_member_social",
        "state": state,
    }
    auth_url = "https://www.linkedin.com/oauth/v2/authorization"
    return redirect(f"{auth_url}?" + requests.compat.urlencode(params))


@app.route("/auth/linkedin/callback")
def linkedin_callback():
    error = request.args.get("error")
    if error:
        abort(400, description=f"LinkedIn error: {error}")
    state = request.args.get("state")
    if not state or state != session.get("oauth_state"):
        abort(400, description="Invalid OAuth state")
    code = request.args.get("code")
    if not code:
        abort(400, description="Missing authorization code")

    redirect_uri = _linkedin_redirect_uri()
    token_resp = requests.post(
        "https://www.linkedin.com/oauth/v2/accessToken",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": os.getenv("LINKEDIN_CLIENT_ID"),
            "client_secret": os.getenv("LINKEDIN_CLIENT_SECRET"),
        },
        timeout=15,
    )
    if token_resp.status_code >= 300:
        abort(502, description=f"LinkedIn token exchange failed: {token_resp.text}")
    token_data = token_resp.json()
    access_token = token_data.get("access_token")
    expires_in = token_data.get("expires_in", 3600)
    if not access_token:
        abort(502, description="LinkedIn token missing")

    user_info = requests.get(
        "https://api.linkedin.com/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    if user_info.status_code >= 300:
        abort(502, description="Failed to fetch LinkedIn profile")
    profile = user_info.json()
    person_sub = profile.get("sub")
    if not person_sub:
        abort(502, description="LinkedIn profile missing subject")

    session["li_access_token"] = access_token
    session["li_person_urn"] = (
        f"urn:li:person:{person_sub}" if not str(person_sub).startswith("urn:li:") else person_sub
    )
    session["li_expires_at"] = time.time() + int(expires_in)

    redirect_to = session.pop("post_auth_redirect", url_for("upload_form"))
    return redirect(redirect_to)


@app.route("/preview/<file_id>")
def preview_image(file_id: str):
    token = request.args.get("t")
    if not token:
        abort(403)
    _verify_preview_token(file_id, token)
    image = preview_cache.load_cached(file_id)
    if image is None:
        try:
            image_bytes = drive.get_media(file_id)
        except HttpError:
            abort(404)
        try:
            image = preview_cache.cache_from_bytes(file_id, image_bytes)
        except ValueError:
            abort(415, description="Unsupported image format")
    else:
        image = image.copy()

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    watermark = f"{WATERMARK_PREFIX} - {timestamp}"
    font = ImageFont.load_default()
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), watermark, font=font)
        text_w = right - left
        text_h = bottom - top
    else:  # Pillow < 10
        text_w, text_h = draw.textsize(watermark, font=font)
    padding = 8
    rect_height = text_h + padding * 2
    draw.rectangle(
        [(0, image.height - rect_height), (image.width, image.height)],
        fill=(0, 0, 0, 140),
    )
    draw.text(
        (padding, image.height - rect_height + padding),
        watermark,
        font=font,
        fill=(255, 255, 255, 255),
    )
    image = Image.alpha_composite(image, overlay).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    buf.seek(0)

    response = send_file(buf, mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/favicon.ico")
def _favicon():
    return ("", 204)


@app.context_processor
def inject_globals():
    max_step = session.get("flow_step", 1)
    current_step = getattr(g, "current_flow_step", max_step)
    current_step = max(1, min(current_step, 5))
    return {
        "event_id": session.get("event_id", ""),
        "current_step": current_step,
        "max_step": max_step,
    }


@app.route("/")
def root():
    return redirect(url_for("start_flow"))


def _set_flow_step(step: int) -> None:
    """Track highest step reached in the session for navigation."""
    session["flow_step"] = max(step, session.get("flow_step", 1))
    session.modified = True
    g.current_flow_step = step


def _set_caption_origin(origin: Optional[str]) -> None:
    if origin:
        session["caption_from"] = origin.strip().lower()
        session.modified = True


def _build_match_view(match_rows: List[Dict], event_id: str, low_confidence: bool) -> Dict:
    mandatory_ids = get_mandatory_file_ids(event_id)
    present_mandatory: set = set()
    matches: List[Dict] = []
    for row in match_rows:
        fid = row["file_id"]
        is_locked = fid in mandatory_ids
        matches.append(
            {
                "file_id": fid,
                "name": row.get("name"),
                "score": row.get("score"),
                "preview": url_for("preview_image", file_id=fid, t=_generate_preview_token(fid)),
                "is_mandatory": is_locked,
            }
        )
        if is_locked:
            present_mandatory.add(fid)

    required_cards = []
    missing_required = [fid for fid in mandatory_ids if fid not in present_mandatory]
    if missing_required:
        required_meta = _fetch_file_records(missing_required)
        for fid in mandatory_ids:
            if fid in present_mandatory:
                continue
            row = required_meta.get(fid)
            if not row:
                continue
            required_cards.append(
                {
                    "file_id": fid,
                    "name": row["name"],
                    "preview": url_for("preview_image", file_id=fid, t=_generate_preview_token(fid)),
                    "is_mandatory": True,
                }
            )
    default_selected: List[str] = []
    for card in matches:
        if not card.get("is_mandatory"):
            default_selected = [card["file_id"]]
            break

    return {
        "matches": matches,
        "event_id": event_id,
        "low_confidence": low_confidence,
        "match_threshold": MATCH_MIN_SCORE,
        "mandatory_cards": required_cards,
        "default_selected": default_selected,
        "mandatory_ids": mandatory_ids,
        "current_step": 2,
        "max_step": session.get("flow_step", 1),
    }


def _linkedin_redirect_uri() -> str:
    configured = (os.getenv("LINKEDIN_REDIRECT_URI") or "").strip()
    if configured and configured.lower() != "auto":
        return configured
    scheme = (os.getenv("PREFERRED_URL_SCHEME") or request.scheme or "http").strip() or "http"
    return url_for("linkedin_callback", _external=True, _scheme=scheme)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=ALLOW_DEV)
