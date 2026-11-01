"""
Scan Upload Validation & Local File Storage Service
Owner: Shivam (Backend & Storage)

Validates uploaded Chest X-Ray image files by inspecting actual binary headers/pixels
(never trusting client filename or declared MIME type alone), writes files safely
to local disk, and persists Scan metadata in SQLAlchemy.
"""

import os
import re
import uuid
import logging
from pathlib import Path
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.db.models import Scan
from backend.app.schemas.radiology import ScanUploadResponse
from ai.vision.preprocessing import ChestXRayPreprocessor

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """Strip path traversal characters and sanitize filename."""
    basename = os.path.basename(filename or "chest_xray.png")
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", basename)
    return cleaned[:120] or "chest_xray.png"


class StorageService:
    """Handles validation, disk storage, and database persistence for uploaded scans."""

    def __init__(self):
        self.preprocessor = ChestXRayPreprocessor()

    async def save_upload(self, file: UploadFile, db: Session) -> ScanUploadResponse:
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file was provided in the upload request.")

        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

        if len(raw_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File size ({len(raw_bytes) / (1024 * 1024):.2f} MB) exceeds maximum allowed "
                    f"size of {settings.MAX_UPLOAD_SIZE_MB} MB."
                ),
            )

        # Inspect actual image binary content using Pillow (do not trust declared MIME or extension)
        try:
            preprocessed = self.preprocessor.preprocess_bytes(raw_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        detected_format = preprocessed.metadata.format.upper()
        if detected_format == "PNG":
            verified_media_type = "image/png"
            ext = ".png"
        elif detected_format in ("JPEG", "JPG"):
            verified_media_type = "image/jpeg"
            ext = ".jpg"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image format '{detected_format}'. Only PNG and JPEG Chest X-Rays are supported.",
            )

        scan_id = str(uuid.uuid4())
        safe_name = sanitize_filename(file.filename)
        if not safe_name.lower().endswith((".png", ".jpg", ".jpeg")):
            safe_name = f"{safe_name}{ext}"

        upload_dir: Path = settings.get_upload_dir()
        stored_filename = f"{scan_id}_{safe_name}"
        stored_path = upload_dir / stored_filename

        with open(stored_path, "wb") as out_file:
            out_file.write(raw_bytes)

        meta_dict = preprocessed.metadata.model_dump()

        scan_record = Scan(
            id=scan_id,
            filename=safe_name,
            file_path=str(stored_path),
            file_type=verified_media_type,
            file_size_bytes=len(raw_bytes),
            status="uploaded",
            image_metadata_json=meta_dict,
        )
        db.add(scan_record)
        db.commit()
        db.refresh(scan_record)

        logger.info("Persisted Chest X-Ray scan %s (%s, %d bytes)", scan_id, verified_media_type, len(raw_bytes))

        return ScanUploadResponse(
            scan_id=scan_record.id,
            filename=scan_record.filename,
            media_type=verified_media_type,
            status=scan_record.status,
            file_type=verified_media_type,
            file_size_bytes=len(raw_bytes),
            image_metadata=meta_dict,
            message="Scan validated and stored successfully.",
        )

    def get_scan_or_404(self, scan_id: str, db: Session) -> Scan:
        scan = db.query(Scan).filter(Scan.id == scan_id).first()
        if not scan:
            raise HTTPException(
                status_code=404,
                detail=f"Scan with scan_id '{scan_id}' was not found.",
            )
        return scan
