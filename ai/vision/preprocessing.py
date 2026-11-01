"""
Medical Image Preprocessing & Chest X-Ray Pipeline
Owner: Shanks (Medical Vision & DICOM Preprocessing)

Handles Chest X-Ray image loading, integrity validation, grayscale conversion,
model-compatible resizing, and intensity normalization. Preserves DICOMPreprocessor
interface for future DICOM modality extensions.
"""

import io
import os
from typing import Dict, Any, Tuple, Union
import numpy as np
from PIL import Image, UnidentifiedImageError

from ai.config import AIConfig
from ai.common.schemas import ImageMetadata, PreprocessedImage


class ChestXRayPreprocessor:
    """
    Validates, loads, converts to grayscale, resizes, and normalizes Chest X-Ray images
    (PNG, JPEG/JPG) without requiring PyTorch or GPU dependencies in mock mode.
    """

    SUPPORTED_FORMATS = {"PNG", "JPEG", "JPG", "MPO"}

    def __init__(self, target_size: Tuple[int, int] = AIConfig.IMAGE_SIZE):
        self.target_size = target_size

    def validate_image_bytes(self, raw_bytes: bytes) -> Tuple[str, Tuple[int, int], str]:
        """
        Verify raw image bytes using Pillow (never trusting file extension or MIME header alone).
        Returns (detected_format, (width, height), color_mode).
        Raises ValueError if bytes are not a valid supported image.
        """
        if not raw_bytes:
            raise ValueError("Uploaded image file is empty.")

        try:
            with Image.open(io.BytesIO(raw_bytes)) as img:
                img.verify()
            with Image.open(io.BytesIO(raw_bytes)) as img:
                fmt = (img.format or "").upper()
                if fmt == "MPO":
                    fmt = "JPEG"
                if fmt not in self.SUPPORTED_FORMATS:
                    raise ValueError(
                        f"Unsupported image format '{fmt or 'unknown'}'. Only PNG and JPEG Chest X-Ray images are supported."
                    )
                width, height = img.size
                if width < 16 or height < 16:
                    raise ValueError(
                        f"Image dimensions ({width}x{height}) are too small for Chest X-Ray analysis."
                    )
                return fmt, (width, height), img.mode
        except UnidentifiedImageError as exc:
            raise ValueError("File is not a valid image or is corrupted.") from exc
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to validate image file: {exc}") from exc

    def preprocess_file(
        self,
        file_path: str,
        scan_id: str = "scan_local",
    ) -> PreprocessedImage:
        """
        Load a Chest X-Ray image from disk, extract metadata, convert to grayscale,
        resize to model input dimensions, and normalize intensities to [0.0, 1.0].
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Scan image file not found at '{file_path}'.")

        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        return self.preprocess_bytes(raw_bytes, file_path=file_path, scan_id=scan_id)

    def preprocess_bytes(
        self,
        raw_bytes: bytes,
        file_path: str = "<memory>",
        scan_id: str = "scan_memory",
    ) -> PreprocessedImage:
        """
        Validate and preprocess raw Chest X-Ray image bytes into a normalized grayscale array.
        """
        fmt, (width, height), original_mode = self.validate_image_bytes(raw_bytes)

        with Image.open(io.BytesIO(raw_bytes)) as img:
            channels = len(img.getbands())
            grayscale_img = img.convert("L")
            resized_img = grayscale_img.resize(self.target_size, Image.Resampling.BILINEAR)
            pixel_array = np.asarray(resized_img, dtype=np.float32) / 255.0

        mean_intensity = float(np.mean(pixel_array))
        std_intensity = float(np.std(pixel_array))

        metadata = ImageMetadata(
            width=width,
            height=height,
            channels=channels,
            format=fmt,
            color_mode=original_mode,
            mean_intensity=round(mean_intensity, 4),
            std_intensity=round(std_intensity, 4),
            modality="DX",
        )

        return PreprocessedImage(
            scan_id=scan_id,
            file_path=file_path,
            metadata=metadata,
            target_size=self.target_size,
            pixel_array=pixel_array,
        )


class DICOMPreprocessor:
    """
    Interface for future DICOM header parsing and windowing extensions.
    Note: Initial prototype focuses on PNG/JPEG Chest X-Ray analysis.
    """

    def parse_dicom(self, file_path: str) -> Dict[str, Any]:
        """
        Parse DICOM file if pydicom is installed and file exists.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"DICOM file not found at '{file_path}'. DICOM parsing requires an existing .dcm file."
            )
        try:
            import pydicom  # type: ignore

            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            return {
                "modality": str(getattr(ds, "Modality", "DX")),
                "rows": int(getattr(ds, "Rows", 0)),
                "columns": int(getattr(ds, "Columns", 0)),
                "status": "parsed",
            }
        except ImportError as exc:
            raise RuntimeError(
                "DICOM support is not enabled in the initial prototype (pydicom not installed)."
            ) from exc

    def apply_windowing(
        self, image_data: Union[np.ndarray, Any], window_center: int = 40, window_width: int = 400
    ) -> Any:
        """
        Apply intensity windowing to a numeric pixel array.
        """
        if isinstance(image_data, np.ndarray):
            lower = window_center - (window_width / 2.0)
            upper = window_center + (window_width / 2.0)
            clipped = np.clip(image_data, lower, upper)
            return (clipped - lower) / max(window_width, 1e-6)
        return image_data
