"""
Medical Vision Model Adapter & Wrapper
Owner: Shanks (Medical Vision Model & Feature Extraction)

Provides a clean adapter interface so pretrained DenseNet121, Rad-DINO, or BiomedCLIP
vision backbones can be integrated without rewriting downstream services.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np

from ai.config import AIConfig
from ai.common.schemas import PreprocessedImage


class VisionModelAdapter(ABC):
    """Abstract adapter for Chest X-Ray vision backbone models."""

    @abstractmethod
    def extract_features(self, preprocessed_image: PreprocessedImage) -> np.ndarray:
        """Extract a 1D feature embedding vector from a preprocessed Chest X-Ray."""


class MockVisionModelAdapter(VisionModelAdapter):
    """
    Lightweight deterministic feature extractor for mock mode.
    Computes spatial quadrant intensity statistics from the Chest X-Ray pixel array.
    """

    def __init__(self, model_name: str = "demo-cxr-densenet121-mock"):
        self.model_name = model_name

    def extract_features(self, preprocessed_image: PreprocessedImage) -> np.ndarray:
        arr = preprocessed_image.pixel_array
        if arr is None or not isinstance(arr, np.ndarray):
            return np.zeros(16, dtype=np.float32)

        h, w = arr.shape[:2]
        h_mid, w_mid = max(1, h // 2), max(1, w // 2)
        quadrants = [
            arr[:h_mid, :w_mid],
            arr[:h_mid, w_mid:],
            arr[h_mid:, :w_mid],
            arr[h_mid:, w_mid:],
        ]
        stats: List[float] = []
        for q in quadrants:
            stats.extend(
                [
                    float(np.mean(q)),
                    float(np.std(q)),
                    float(np.percentile(q, 25)),
                    float(np.percentile(q, 75)),
                ]
            )
        return np.array(stats, dtype=np.float32)


class RealVisionModelAdapter(VisionModelAdapter):
    """
    Real PyTorch / MONAI vision backbone adapter.
    Validates that model weights exist before attempting inference.
    """

    def __init__(
        self,
        model_name: str = AIConfig.VISION_MODEL_NAME,
        weights_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.weights_path = weights_path or os.getenv(
            "VISION_WEIGHTS_PATH", AIConfig.VISION_WEIGHTS_PATH
        )

    def extract_features(self, preprocessed_image: PreprocessedImage) -> np.ndarray:
        if not self.weights_path or not os.path.isfile(self.weights_path):
            raise RuntimeError(
                f"Real vision model '{self.model_name}' is unavailable: "
                f"VISION_WEIGHTS_PATH ('{self.weights_path or 'unset'}') does not point to a valid model checkpoint."
            )
        raise NotImplementedError(
            f"Real forward pass for '{self.model_name}' requires loading checkpoint '{self.weights_path}'."
        )


class MedicalVisionModel:
    """
    High-level wrapper for medical vision backbone models.
    Preserves backward compatibility while delegating to mode-specific adapters.
    """

    def __init__(self, model_name: Optional[str] = None, mode: Optional[str] = None):
        self.mode = (mode or AIConfig.get_ai_mode()).lower()
        self.model_name = model_name or (
            "demo-cxr-densenet121-mock" if self.mode == "mock" else AIConfig.VISION_MODEL_NAME
        )
        if self.mode == "mock":
            self.adapter: VisionModelAdapter = MockVisionModelAdapter(self.model_name)
        else:
            self.adapter = RealVisionModelAdapter(self.model_name)

    def extract_features(self, preprocessed_image: PreprocessedImage) -> np.ndarray:
        return self.adapter.extract_features(preprocessed_image)

    def predict_anomalies(self, processed_image: Any) -> List[Dict[str, Any]]:
        """
        Backward-compatible region summary method.
        """
        if self.mode == "real":
            raise RuntimeError(
                "Real ROI anomaly detection requires a configured VISION_WEIGHTS_PATH checkpoint."
            )
        return [
            {
                "region": "Bilateral Lower Lung Zones (Simulated)",
                "finding": "DEMO MODE: Simulated thoracic feature extraction completed",
                "score": 0.78,
                "mode": "mock",
            }
        ]
