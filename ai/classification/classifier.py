"""
Disease Classifier Module
Owner: Vineet (Disease Classification & Model Training)

Implements DemoPredictionProvider (mock mode) and RealPredictionProvider (real mode)
for multi-label Chest X-Ray abnormality classification.
"""

import hashlib
import os
from typing import Dict, List, Optional
import numpy as np

from ai.config import AIConfig, DEFAULT_CHEST_XRAY_LABELS
from ai.common.schemas import (
    PreprocessedImage,
    LabelPrediction,
    PredictionResult,
)
from ai.common.interfaces import BasePredictionProvider
from ai.vision.model import compute_radiographic_features_64d


class DemoPredictionProvider(BasePredictionProvider):
    """
    Simulated multi-label Chest X-Ray classifier for AI_MODE=mock.
    Generates deterministic, reproducible scores derived from image pixel statistics
    so different test images produce distinct yet stable demo predictions.
    """

    MODEL_NAME = "demo-cxr-densenet121-mock"

    def __init__(self, labels: Optional[List[str]] = None):
        self.labels = labels or list(DEFAULT_CHEST_XRAY_LABELS)

    def predict(
        self,
        preprocessed_image: PreprocessedImage,
        labels: Optional[List[str]] = None,
    ) -> PredictionResult:
        target_labels = labels or self.labels

        # Compute deterministic hash from image intensity statistics
        meta = preprocessed_image.metadata
        fingerprint = f"{meta.width}x{meta.height}:{meta.mean_intensity:.4f}:{meta.std_intensity:.4f}"
        digest = hashlib.sha256(fingerprint.encode("utf-8")).digest()

        # Primary demo findings emphasized for Chest X-Ray demonstration
        primary_weights = {
            "Atelectasis": 0.76,
            "Consolidation": 0.64,
            "Effusion": 0.58,
            "Cardiomegaly": 0.42,
            "Pneumonia": 0.38,
            "Infiltration": 0.35,
            "Edema": 0.24,
            "Nodule": 0.19,
            "Pneumothorax": 0.14,
            "Mass": 0.11,
            "Fibrosis": 0.10,
            "Pleural_Thickening": 0.09,
            "Emphysema": 0.08,
            "Hernia": 0.04,
        }

        predictions: List[LabelPrediction] = []
        for idx, label in enumerate(target_labels):
            base = primary_weights.get(label, 0.15)
            # Small deterministic offset in [-0.06, +0.06] based on image digest
            byte_val = digest[idx % len(digest)]
            offset = ((byte_val / 255.0) - 0.5) * 0.12
            score = float(np.clip(base + offset, 0.01, 0.98))
            predictions.append(LabelPrediction(label=label, score=round(score, 2)))

        # Sort descending by score
        predictions.sort(key=lambda item: item.score, reverse=True)

        return PredictionResult(
            predictions=predictions,
            model_name=self.MODEL_NAME,
            mode="mock",
        )


class RealPredictionProvider(BasePredictionProvider):
    """
    Real Chest X-Ray classifier provider for AI_MODE=real.
    Validates model checkpoint configuration, extracts multiscale 64-D radiographic
    features, and runs temperature-scaled multi-label classification across the 14
    NIH ChestX-ray14 pathology labels.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        weights_path: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ):
        self.model_name = model_name or AIConfig.VISION_MODEL_NAME
        if weights_path is not None:
            self.weights_path = weights_path
        else:
            self.weights_path = AIConfig.get_vision_weights_path()
        self.labels = labels or list(DEFAULT_CHEST_XRAY_LABELS)

    def predict(
        self,
        preprocessed_image: PreprocessedImage,
        labels: Optional[List[str]] = None,
    ) -> PredictionResult:
        if not self.weights_path or not os.path.isfile(self.weights_path):
            raise RuntimeError(
                f"Real classification mode is active (AI_MODE=real), but model checkpoint "
                f"VISION_WEIGHTS_PATH ('{self.weights_path or 'unset'}') is missing or unavailable. "
                f"Configure a valid DenseNet121/Rad-DINO weight file or set AI_MODE=mock."
            )

        target_labels = labels or self.labels
        raw_features = compute_radiographic_features_64d(preprocessed_image.pixel_array)

        ckpt = np.load(self.weights_path, allow_pickle=False)
        W = np.asarray(ckpt["W"], dtype=np.float32)
        b = np.asarray(ckpt["b"], dtype=np.float32)
        f_mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
        f_std = np.asarray(ckpt["feature_std"], dtype=np.float32)
        f_std = np.where(f_std > 1e-6, f_std, 1.0)
        temp = float(ckpt["temperature"][0]) if "temperature" in ckpt else 1.0
        temp = max(temp, 0.1)

        ckpt_labels = (
            [str(x) for x in ckpt["labels"].tolist()]
            if "labels" in ckpt
            else list(DEFAULT_CHEST_XRAY_LABELS)
        )
        label_map = {lbl: idx for idx, lbl in enumerate(ckpt_labels)}

        z = (raw_features - f_mean) / f_std
        logits = (W @ z + b) / temp
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))

        predictions: List[LabelPrediction] = []
        for idx, label in enumerate(target_labels):
            row_idx = label_map.get(label, idx % len(probs))
            score = float(np.clip(probs[row_idx], 0.01, 0.98))
            predictions.append(LabelPrediction(label=label, score=round(score, 2)))

        predictions.sort(key=lambda item: item.score, reverse=True)

        return PredictionResult(
            predictions=predictions,
            model_name=self.model_name,
            mode="real",
        )


class DiseaseClassifier:
    """
    Facade for multi-label Chest X-Ray disease classification.
    Supports both typed PredictionResult output and legacy dictionary output.
    """

    def __init__(
        self,
        num_classes: int = AIConfig.NUM_CLASSES,
        mode: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ):
        self.mode = (mode or AIConfig.get_ai_mode()).lower()
        self.labels = (labels or list(DEFAULT_CHEST_XRAY_LABELS))[:num_classes]
        self.num_classes = len(self.labels)

        if self.mode == "mock":
            self.provider: BasePredictionProvider = DemoPredictionProvider(labels=self.labels)
        elif self.mode == "real":
            self.provider = RealPredictionProvider(labels=self.labels)
        else:
            raise ValueError(f"Unsupported AI_MODE '{self.mode}'.")

    def predict(self, preprocessed_image: PreprocessedImage) -> PredictionResult:
        """Run multi-label classification and return structured PredictionResult."""
        return self.provider.predict(preprocessed_image, labels=self.labels)

    def classify(self, feature_vector: List[float]) -> Dict[str, float]:
        """
        Backward-compatible dictionary classification interface.
        """
        if self.mode == "real":
            weights_path = AIConfig.get_vision_weights_path()
            if not weights_path or not os.path.isfile(weights_path):
                raise RuntimeError("Real classification requires a valid VISION_WEIGHTS_PATH checkpoint.")
            ckpt = np.load(weights_path, allow_pickle=False)
            W = np.asarray(ckpt["W"], dtype=np.float32)
            b = np.asarray(ckpt["b"], dtype=np.float32)
            vec = np.zeros(W.shape[1], dtype=np.float32)
            n = min(len(feature_vector), W.shape[1])
            vec[:n] = np.asarray(feature_vector[:n], dtype=np.float32)
            probs = 1.0 / (1.0 + np.exp(-np.clip(W @ vec + b, -15.0, 15.0)))
            return {
                lbl: round(float(probs[i]), 2)
                for i, lbl in enumerate(self.labels[: len(probs)])
            }
        return {
            "Atelectasis": 0.78,
            "Consolidation": 0.65,
            "Effusion": 0.58,
            "Cardiomegaly": 0.41,
            "Pneumonia": 0.36,
        }
