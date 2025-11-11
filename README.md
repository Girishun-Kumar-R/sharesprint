# ShareSprint Event Poster

ShareSprint turns event photos stored in Google Drive into LinkedIn-ready posts with selfie matching, templated captions, and live engagement tracking.

## Prerequisites
- Windows with Python 3.10 or 3.11 available on PATH
- Google Cloud service account JSON with Drive + Sheets access to the event folder/sheet
- LinkedIn Marketing Developer Program approval for w_member_social scope and redirect URL `http://127.0.0.1:5000/auth/linkedin/callback`

## Setup
1. Clone or download this repository.
2. Create and activate a virtual environment (optional but recommended):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
3. Install dependencies. On Windows, use the pinned PowerShell block below. On macOS/Linux:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt -c constraints.txt
   ```
4. Copy environment defaults and fill in secrets:
   ```powershell
   Copy-Item .env.example .env
   ```
   Edit `.env` to include your `SECRET_KEY`, Google IDs, and LinkedIn OAuth credentials. Keep `.env` and `service_account.json` out of version control.

### Windows install (pinned stack)
```powershell
# From repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip

# Remove conflicting wheels if present
pip uninstall -y tensorflow tensorflow-intel keras tf-keras keras-nightly tf-keras-nightly numpy tensorboard typing_extensions protobuf opencv-python grpcio

# Install pinned stack with constraints
pip install -r requirements.win.txt -c constraints.txt

# Sanity check
python - << 'PY'
import cv2, numpy as np, tensorflow as tf
from tensorflow import keras
print("OpenCV", cv2.__version__)
print("NumPy", np.__version__)
print("TF", tf.__version__)
print("tf.keras", keras.__version__)
PY
```
If PowerShell reports `cv2.pyd` is locked, close any Python processes (`taskkill /IM python.exe /F`) and rerun the install.

### Sanity check (any platform)
Run this anytime to confirm the runtime matches the pinned stack:
```bash
python - <<'PY'
import cv2, numpy as np, tensorflow as tf
from tensorflow import keras
print("OpenCV", cv2.__version__)
print("NumPy", np.__version__)
print("TF", tf.__version__)
print("tf.keras", keras.__version__)
PY
```

## Initial Face Index Build
Before running the web app, ingest the Drive folder so FAISS can answer selfie queries:
```powershell
python indexer.py build
```
This downloads event photos, detects faces (ArcFace embeddings via DeepFace), and fills:
- `face_index.sqlite` – Drive metadata and embeddings
- `faces.index` + `faces.idmap.json` – persisted FAISS cosine index

### Continuous Sync
Keep the index warm during events:
```powershell
python indexer.py watch
```
The watcher listens to the Drive Changes API, ingests new/updated photos, and refreshes FAISS roughly every minute.

## Run the App
Start Flask locally:
```powershell
python app.py
```
Visit `http://127.0.0.1:5000/start?event=DEMO` and follow the QR flow:
1. LinkedIn login (openid profile email w_member_social scopes).
2. Upload a selfie from mobile camera/roll.
3. Review top facial matches or browse all photos.
4. Choose / customise a caption.
5. Post to LinkedIn – original Drive bytes are uploaded without re-encoding.

Successful posts append a row to the configured Google Sheet with audit metadata and LinkedIn post URL.

### Use the app from your phone on the same Wi-Fi
The dev server now listens on `0.0.0.0`, so anything on your LAN can reach it.

1. Start the app as usual: `python app.py`
2. In a new PowerShell window, grab your laptop's LAN IP (look for the Wi-Fi adapter entry):
   ```powershell
   ipconfig | findstr /R "IPv4"
   ```
3. Update your `.env` **and** LinkedIn Developer redirect list:
   - Set `LINKEDIN_REDIRECT_URI=auto` to let the app reuse whatever host the browser used (e.g., your LAN IP).
   - Add `http://<your-ip>:5000/auth/linkedin/callback` (for example `http://192.168.1.42:5000/auth/linkedin/callback`) to the allowed redirect URIs inside LinkedIn’s developer portal.
4. On your phone (connected to the same Wi-Fi), open `http://<your-ip>:5000/start?event=DEMO` — e.g. `http://192.168.1.42:5000/start?event=DEMO`.
5. If Windows prompts about firewall access when you start the app, allow it for private networks so mobile devices can connect.
6. Whenever your laptop’s IP changes, update the redirect entry in LinkedIn’s dashboard (and restart the app) so the mobile browser can reach the callback.

Use Ctrl+C in the terminal when you are done; the server will stop listening on the network.

## Multi-photo posts & mandatory images
- ShareSprint now supports selecting multiple Drive photos per attendee. From the match results or the full gallery, tick 2–4 favorites and continue—LinkedIn receives a single UGC post with all the attachments in the order you picked (required images append automatically).
- Hosts can force sponsor banners or other must-have visuals into every post via environment variables:
  - `MANDATORY_FILE_IDS`: comma-separated Drive file IDs applied to every event (e.g. `MANDATORY_FILE_IDS=1aBannerId,1bSponsorBoard`).
  - `MANDATORY_FILE_IDS_<EVENTID>`: per-event override in uppercase, e.g. `MANDATORY_FILE_IDS_DEMO=123abc,456def`. If set, it replaces the global list for that event.
- Mandatory images render as locked/required chips in the gallery and caption screens and cannot be deselected. Even if an attendee skips their own photos, the locked files still post so sponsors always receive placement.
- When an attendee taps **Skip personal photos**, we send them straight to captions. If mandatory images exist those are posted alone; if not, LinkedIn receives a caption-only (text) update so shy guests can still celebrate the event.
- Google Sheets logging now records every file ID/name that went out with a post (`drive_file_ids`, `drive_file_names` columns) for easier auditing.

## Metrics Poller
Schedule engagement refreshes (e.g., Windows Task Scheduler or cron on a server):
```powershell
$env:LI_ACCESS_TOKEN = '<member-or-page-admin-token>'
python metrics_poller.py
```
The poller reads the sheet, fetches Social Actions for each post URN, and updates reactions/comments/reshares/last_checked columns in-place. For member posts use a valid member token; org pages require a page-admin token. LinkedIn does not expose impression counts for member posts via public APIs.

## Dev Mode Notes
- Set `ALLOW_DEV_DEFAULTS=1` in `.env` to auto-generate a transient Flask secret (never use in prod).
- Selfie uploads are never written to disk; only normalized face embeddings are stored locally. Remove the generated FAISS/SQLite files after the event if desired.
- Gallery previews are short-lived, signed, watermarked JPEGs served with strict CSP and shortcut blocking to deter downloads.

## Why this works
- TensorFlow 2.13.x, Keras 2.13.1, and NumPy 1.24.3 are ABI-compatible with DeepFace/RetinaFace; the constraints enforce the trio so embedding backends load cleanly.
- Protobuf 4.25.3 and grpcio 1.62.2 stay within the range TensorFlow 2.13 expects, eliminating MessageFactory/GetPrototype crashes.
- Platform markers keep Windows on `tensorflow-intel` while non-Windows uses stock `tensorflow`, preventing accidental jumps to TF 2.20 + Keras 3.
- The reinstall block clears any stale wheels so OpenCV and TensorFlow DLLs refresh without lingering cv2.pyd handle locks.

## Troubleshooting
- Re-run `python indexer.py build` if FAISS or SQLite files go missing.
- Delete `faces.index` / `faces.idmap.json` whenever you change `MODEL_NAME` or embeddings drift to force a clean rebuild.
- Ensure the LinkedIn redirect URI in the Developer Console matches `.env` exactly.
