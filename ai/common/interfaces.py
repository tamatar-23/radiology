"""
Abstract Interfaces for AI Pipeline Providers
Owner: Shared AI Team (Gourav, Shanks, Vineet)

Ensures mock providers and real model providers adhere to identical contracts.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from ai.common.schemas import (
    PreprocessedImage,
    PredictionResult,
    ExplainabilityResult,
    EvidenceCitation,
    LabelPrediction,
)


class BasePredictionProvider(ABC):
    """Interface for Chest X-Ray multi-label abnormality classification providers."""

    @abstractmethod
    def predict(
        self,
        preprocessed_image: PreprocessedImage,
        labels: Optional[List[str]] = None,
    ) -> PredictionResult:
        """Run multi-label classification on a preprocessed Chest X-Ray."""


class BaseExplainabilityProvider(ABC):
    """Interface for Grad-CAM visual explainability providers."""

    @abstractmethod
    def generate_heatmap(
        self,
        preprocessed_image: PreprocessedImage,
        target_class: str,
    ) -> ExplainabilityResult:
        """Generate or report availability of Grad-CAM heatmap for target_class."""


class BaseRetrievalProvider(ABC):
    """Interface for medical literature retrieval providers."""

    @abstractmethod
    def retrieve(self, queries: List[str], top_k: int = 3) -> List[EvidenceCitation]:
        """Retrieve relevant medical guideline passages for a list of clinical queries."""


class BaseReranker(ABC):
    """Interface for reranking retrieved medical literature passages."""

    @abstractmethod
    def rerank(
        self,
        queries: List[str],
        candidates: List[EvidenceCitation],
        top_k: int = 3,
    ) -> List[EvidenceCitation]:
        """Rerank candidate citations by relevance to planned queries."""


class BaseReportGenerator(ABC):
    """Interface for structured Chest X-Ray report synthesis."""

    @abstractmethod
    def generate_report(
        self,
        predictions: List[LabelPrediction],
        evidence: List[EvidenceCitation],
        evidence_status: str,
        clinical_notes: Optional[str] = None,
    ) -> dict:
        """Synthesize structured findings and impression grounded in predictions and evidence."""
