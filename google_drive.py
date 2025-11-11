import io
from typing import List, Dict
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class Drive:
    def __init__(self, service_account_json: str):
        creds = service_account.Credentials.from_service_account_file(
            service_account_json, scopes=SCOPES
        )
        self.service = build("drive", "v3", credentials=creds)

    def list_images_in_folder(self, folder_id: str, page_size: int = 1000) -> List[Dict]:
        q = f"'{folder_id}' in parents and trashed = false and mimeType contains 'image/'"
        fields = "nextPageToken, files(id, name, mimeType, thumbnailLink, webViewLink)"
        results = []
        page_token = None
        while True:
            resp = self.service.files().list(
                q=q,
                fields=fields,
                pageSize=page_size,
                pageToken=page_token,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            ).execute()
            results.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return results

    def download_bytes(self, file_id: str) -> bytes:
        request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        return fh.getvalue()
