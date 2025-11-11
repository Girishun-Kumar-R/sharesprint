from __future__ import annotations

import os
import platform
import sys
from importlib import metadata

from dotenv import load_dotenv


EXPECTED_COMMON = {
    "opencv-python": "4.8.1.78",
    "numpy": "1.24.3",
    "keras": "2.13.1",
    "tensorboard": "2.13.0",
    "protobuf": "4.25.3",
    "typing_extensions": "4.5.0",
    "grpcio": "1.62.2",
    "faiss-cpu": "1.7.4",
    "retina-face": "0.0.13",
    "deepface": "0.0.81",
}

EXPECTED_WINDOWS_ONLY = {"tensorflow-intel": "2.13.1"}
EXPECTED_NON_WINDOWS_ONLY = {"tensorflow": "2.13.1"}
OPTIONAL_TENSORFLOW_DISTS = {"tensorflow": "2.13.1", "tensorflow-intel": "2.13.1"}


def _validate_environment() -> None:
    load_dotenv()
    errors: list[str] = []

    service_account_path = os.getenv("SERVICE_ACCOUNT_JSON")
    if not service_account_path:
        errors.append("SERVICE_ACCOUNT_JSON is not set in the environment")
    elif not os.path.exists(service_account_path):
        errors.append(f"Service account file not found: {service_account_path}")

    folder_id = os.getenv("FOLDER_ID")
    if not folder_id:
        errors.append("FOLDER_ID is not set in .env")

    if errors:
        for msg in errors:
            print(f"[ERROR] {msg}")
        sys.exit(1)


def _installed_version(dist: str) -> tuple[str | None, str | None]:
    try:
        installed = metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None, f"{dist} is not installed (expected {dist}=={_expected_for(dist)})"
    return installed, None


def _expected_for(dist: str) -> str:
    combined = EXPECTED_COMMON | EXPECTED_WINDOWS_ONLY | EXPECTED_NON_WINDOWS_ONLY
    return combined.get(dist, "unknown")


def _validate_constraints() -> list[str]:
    failures: list[str] = []
    expected = dict(EXPECTED_COMMON)
    if platform.system() == "Windows":
        expected.update(EXPECTED_WINDOWS_ONLY)
    else:
        expected.update(EXPECTED_NON_WINDOWS_ONLY)

    for dist, expected_version in expected.items():
        installed, error = _installed_version(dist)
        if error:
            failures.append(error)
            continue
        if installed != expected_version:
            failures.append(f"{dist}=={installed} (expected {expected_version})")

    # If the optional TF variant is present ensure it matches the expected pin.
    for dist, expected_version in OPTIONAL_TENSORFLOW_DISTS.items():
        installed, error = _installed_version(dist)
        if error:
            continue
        if installed != expected_version:
            failures.append(f"{dist}=={installed} (expected {expected_version})")

    return failures


def _print_versions() -> list[str]:
    failures: list[str] = []
    reports: list[tuple[str, str]] = []

    try:
        import cv2

        reports.append(("OpenCV", cv2.__version__))
    except Exception as exc:  # pragma: no cover - diagnostics
        failures.append(f"Import cv2 failed: {exc}")

    try:
        import numpy as np

        reports.append(("NumPy", np.__version__))
    except Exception as exc:
        failures.append(f"Import numpy failed: {exc}")

    try:
        import tensorflow as tf

        reports.append(("TensorFlow", tf.__version__))
        reports.append(("tf.keras", tf.keras.__version__))
    except Exception as exc:
        failures.append(f"Import tensorflow failed: {exc}")

    try:
        from google.protobuf import __version__ as protobuf_version

        reports.append(("protobuf", protobuf_version))
    except Exception as exc:
        failures.append(f"Import protobuf failed: {exc}")

    try:
        import grpc

        reports.append(("grpcio", grpc.__version__))
    except Exception as exc:
        failures.append(f"Import grpc failed: {exc}")

    try:
        import faiss

        faiss_version = getattr(faiss, "__version__", metadata.version("faiss-cpu"))
        reports.append(("faiss", faiss_version))
    except Exception as exc:
        failures.append(f"Import faiss failed: {exc}")

    try:
        retina_version = metadata.version("retina-face")
        reports.append(("retina-face", retina_version))
    except metadata.PackageNotFoundError:
        failures.append("retina-face is not installed")
    except Exception as exc:
        failures.append(f"retina-face version lookup failed: {exc}")

    try:
        deepface_version = metadata.version("deepface")
        reports.append(("deepface", deepface_version))
    except metadata.PackageNotFoundError:
        failures.append("deepface is not installed")
    except Exception as exc:
        failures.append(f"deepface version lookup failed: {exc}")

    print("Dependency versions:")
    for label, value in reports:
        print(f"  {label:12s} {value}")

    return failures


def _check_drive_access() -> None:
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except Exception as exc:
        print(f"[WARN] Skipping Drive API smoke test (imports failed: {exc})")
        return

    service_account_path = os.getenv("SERVICE_ACCOUNT_JSON", "")
    folder_id = os.getenv("FOLDER_ID", "")

    try:
        creds = service_account.Credentials.from_service_account_file(
            service_account_path,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        svc = build("drive", "v3", credentials=creds, cache_discovery=False)
        resp = (
            svc.files()
            .list(
                q=f"'{folder_id}' in parents and trashed=false and mimeType contains 'image/'",
                fields="files(id,name)",
                pageSize=3,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            .execute()
        )
        print(f"Drive API access OK. Sample files: {resp.get('files', [])}")
    except Exception as exc:
        print(f"[WARN] Drive API check failed: {exc}")


def main() -> None:
    _validate_environment()
    failures = _validate_constraints()
    failures.extend(_print_versions())

    if failures:
        print("\nDependency issues detected:")
        for problem in failures:
            print(f"  - {problem}")
        print("\nFix suggestions:")
        print("  pip install -r requirements.win.txt -c constraints.txt  # Windows")
        print("  pip install -r requirements.txt -c constraints.txt      # Other platforms")
        sys.exit(1)

    print("\nAll dependency checks passed.")
    _check_drive_access()


if __name__ == "__main__":
    main()
