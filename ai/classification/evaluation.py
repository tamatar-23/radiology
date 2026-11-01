"""
Model Evaluation & Grad-CAM Explainability
Owner: Vineet (Explainability & Model Evaluation)

Separates Grad-CAM explainability from core classification.
Never fabricates fake heatmaps using arbitrary image filters in mock mode.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np

from ai.config import AIConfig
from ai.common.schemas import PreprocessedImage, ExplainabilityResult
from ai.common.interfaces import BaseExplainabilityProvider


class DemoExplainabilityProvider(BaseExplainabilityProvider):
    """
    Honest explainability provider for AI_MODE=mock.
    Does NOT generate fake heatmaps or arbitrary color overlays.
    Clearly reports that Grad-CAM requires a trained model's convolutional activation maps.
    """

    def generate_heatmap(
        self,
        preprocessed_image: PreprocessedImage,
        target_class: str,
    ) -> ExplainabilityResult:
        return ExplainabilityResult(
            heatmap_available=False,
            heatmap_url=None,
            target_class=target_class,
            method="gradcam",
            mode="mock",
            message=(
                f"Grad-CAM heatmap is unavailable in DEMO MODE for target class '{target_class}'. "
                "Genuine spatial activation maps require a loaded convolutional/vision backbone in AI_MODE=real."
            ),
        )


class RealGradCAMProvider(BaseExplainabilityProvider):
    """
    Grad-CAM provider for AI_MODE=real.
    Validates model weights availability before computing gradients.
    """

    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = weights_path or os.getenv(
            "VISION_WEIGHTS_PATH", AIConfig.VISION_WEIGHTS_PATH
        )

    def generate_heatmap(
        self,
        preprocessed_image: PreprocessedImage,
        target_class: str,
    ) -> ExplainabilityResult:
        if not self.weights_path or not os.path.isfile(self.weights_path):
            raise RuntimeError(
                f"Grad-CAM generation failed in AI_MODE=real: VISION_WEIGHTS_PATH "
                f"('{self.weights_path or 'unset'}') is not available."
            )
        raise RuntimeError(
            "Real Grad-CAM computation requires a loaded PyTorch model with target convolutional layer hooks."
        )


class ModelEvaluator:
    """
    Evaluation and explainability coordinator for Chest X-Ray classification.
    """

    def __init__(self, mode: Optional[str] = None):
        self.mode = (mode or AIConfig.get_ai_mode()).lower()
        if self.mode == "mock":
            self.explainability_provider: BaseExplainabilityProvider = DemoExplainabilityProvider()
        elif self.mode == "real":
            self.explainability_provider = RealGradCAMProvider()
        else:
            raise ValueError(f"Unsupported AI_MODE '{self.mode}'.")

    def explain(
        self,
        preprocessed_image: PreprocessedImage,
        target_class: str,
    ) -> ExplainabilityResult:
        return self.explainability_provider.generate_heatmap(preprocessed_image, target_class)

    def generate_gradcam(self, image_tensor: Any, target_class: str) -> Dict[str, Any]:
        """
        Backward-compatible dictionary wrapper around ExplainabilityResult.
        Honestly indicates that no heatmap is generated in mock mode.
        """
        if isinstance(image_tensor, PreprocessedImage):
            res = self.explain(image_tensor, target_class)
            return res.model_dump()

        if self.mode == "real":
            raise RuntimeError("Real Grad-CAM requires a valid PreprocessedImage and loaded model weights.")

        return {
            "target_class": target_class,
            "heatmap_available": False,
            "heatmap_url": None,
            "mode": "mock",
            "status": "unavailable_in_mock_mode",
            "message": "Grad-CAM is unavailable in mock mode; no fake overlay is generated.",
        }

    def compute_metrics(
        self,
        y_true: List[int],
        y_pred: List[float],
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute actual binary classification metrics (Accuracy, Precision, Recall, F1)
        from ground-truth binary labels and predicted scores.
        """
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must be non-empty lists of equal length.")

        y_true_arr = np.asarray(y_true, dtype=np.int32)
        y_pred_arr = np.asarray(y_pred, dtype=np.float64)
        y_hat = (y_pred_arr >= threshold).astype(np.int32)

        tp = int(np.sum((y_true_arr == 1) & (y_hat == 1)))
        tn = int(np.sum((y_true_arr == 0) & (y_hat == 0)))
        fp = int(np.sum((y_true_arr == 0) & (y_hat == 1)))
        fn = int(np.sum((y_true_arr == 1) & (y_hat == 0)))

        total = len(y_true_arr)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2.0 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
        }
