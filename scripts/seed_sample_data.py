"""
Seed Sample Data Script
Owner: Shivam (Backend & DB) / Shreyansh (Data Management)

Generates a synthetic non-patient grayscale Chest X-Ray test phantom (`data/samples/sample_cxr.png`)
and seeds an initial Scan and Report record into the configured database for local development.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def generate_synthetic_cxr_phantom(output_path: Path) -> Path:
    """
    Create a deterministic 512x512 grayscale synthetic thoracic phantom PNG.
    Contains zero real patient data.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 512, 512

    # Base grayscale gradient
    y_grid, x_grid = np.ogrid[:height, :width]
    radial = np.sqrt((x_grid - 256) ** 2 + (y_grid - 256) ** 2)
    bg = np.clip(210 - (radial * 0.45), 35, 220).astype(np.uint8)

    img = Image.fromarray(bg, mode="L")
    draw = ImageDraw.Draw(img)

    # Simulated radiolucent lung fields (darker ovals)
    draw.ellipse((95, 90, 235, 385), fill=55, outline=130, width=3)
    draw.ellipse((275, 90, 415, 385), fill=58, outline=130, width=3)

    # Simulated cardiomediastinal silhouette (central brighter region)
    draw.ellipse((205, 195, 325, 365), fill=165)

    # Simulated right lower zone subtle opacity for test feature variance
    draw.ellipse((125, 285, 210, 355), fill=115)

    img.save(output_path, format="PNG")
    return output_path


def seed_database() -> None:
    from backend.app.db.database import init_db, SessionLocal
    from backend.app.db.models import Scan
    from backend.app.schemas.radiology import AnalysisRequest
    from backend.app.services.analysis_service import AnalysisService
    from ai.vision.preprocessing import ChestXRayPreprocessor

    sample_png = REPO_ROOT / "data" / "samples" / "sample_cxr.png"
    generate_synthetic_cxr_phantom(sample_png)
    print(f"[seed] Generated synthetic Chest X-Ray phantom at: {sample_png}")

    init_db()
    preprocessor = ChestXRayPreprocessor()
    preprocessed = preprocessor.preprocess_file(str(sample_png), scan_id="scan_sample_cxr_001")

    db = SessionLocal()
    try:
        existing = db.query(Scan).filter(Scan.id == "scan_sample_cxr_001").first()
        if not existing:
            scan = Scan(
                id="scan_sample_cxr_001",
                filename="sample_cxr.png",
                file_path=str(sample_png),
                file_type="image/png",
                file_size_bytes=sample_png.stat().st_size,
                status="uploaded",
                image_metadata_json=preprocessed.metadata.model_dump(),
            )
            db.add(scan)
            db.commit()
            db.refresh(scan)
            print(f"[seed] Inserted sample scan record: {scan.id}")

        service = AnalysisService()
        report = service.run_analysis(
            AnalysisRequest(
                scan_id="scan_sample_cxr_001",
                notes="Synthetic Chest X-Ray test phantom for pipeline verification.",
            ),
            db=db,
        )
        print(f"[seed] Generated sample mock report: {report.report_id} (mode={report.mode})")
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
