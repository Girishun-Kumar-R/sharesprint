import json, os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from google_drive import Drive
from face_embed import bytes_to_rgb, embed_image_rgb

load_dotenv()
FOLDER_ID = os.getenv("FOLDER_ID")
SERVICE_JSON = os.getenv("SERVICE_ACCOUNT_JSON", "service_account.json")
MODEL_NAME = os.getenv("MODEL_NAME", "Facenet512")
DETECTOR = os.getenv("DETECTOR_BACKEND", "retinaface")
DB_PATH = os.getenv("DB_PATH", "face_index.sqlite")

assert FOLDER_ID, "FOLDER_ID not set"

engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
  file_id TEXT PRIMARY KEY,
  name TEXT,
  mimeType TEXT,
  webViewLink TEXT,
  thumbnailLink TEXT,
  embedding TEXT
);
"""

if __name__ == "__main__":
    from sqlalchemy import text as sqltext

    drive = Drive(SERVICE_JSON)
    with engine.begin() as conn:
        conn.exec_driver_sql(SCHEMA)

    files = drive.list_images_in_folder(FOLDER_ID)
    print(f"Found {len(files)} image files in folder {FOLDER_ID}")

    with engine.begin() as conn:
        existing = set(row[0] for row in conn.execute(sqltext("SELECT file_id FROM images")))
    added = 0
    skipped = 0

    for f in files:
        fid = f["id"]
        if fid in existing:
            skipped += 1
            continue
        try:
            data = drive.download_bytes(fid)
            rgb = bytes_to_rgb(data)
            emb = embed_image_rgb(rgb, MODEL_NAME, DETECTOR)
            rec = {
                "file_id": fid,
                "name": f.get("name"),
                "mimeType": f.get("mimeType"),
                "webViewLink": f.get("webViewLink"),
                "thumbnailLink": f.get("thumbnailLink"),
                "embedding": json.dumps([float(x) for x in emb.tolist()]),
            }
            with engine.begin() as conn:
                conn.execute(
                    sqltext(
                        """INSERT OR REPLACE INTO images(file_id,name,mimeType,webViewLink,thumbnailLink,embedding)
                               VALUES (:file_id,:name,:mimeType,:webViewLink,:thumbnailLink,:embedding)"""
                    ),
                    rec,
                )
            added += 1
            print(f"Indexed: {rec['name']} ({fid})")
        except Exception as e:
            print(f"[WARN] Skipped {f.get('name')} ({fid}) -> {e}")
            continue

    print(f"Done. Added {added}, skipped {skipped}. DB={DB_PATH}")
