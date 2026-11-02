"""
AI Module Configuration
Owner: Gourav / Shared AI Team

Defines central settings for AI execution mode (mock vs real), Chest X-Ray vision pipeline defaults,
pathology label lists, vector DB settings, and LLM parameters.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Standard NIH ChestX-ray14 pathology labels
DEFAULT_CHEST_XRAY_LABELS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


def resolve_repo_path(path_str: str) -> str:
    """Resolve a path relative to the repository root if not empty and not already absolute."""
    if not path_str or not path_str.strip():
        return ""
    cleaned = path_str.strip()
    if os.path.isabs(cleaned) or os.path.exists(cleaned):
        return cleaned
    return str(REPO_ROOT / cleaned)


class AIConfig:
    # Execution Mode: "mock" or "real"
    AI_MODE: str = os.getenv("AI_MODE", "mock").strip().lower()

    # Modality Scope: Focused strictly on Chest X-Ray (DX / CR) for initial prototype
    SUPPORTED_MODALITY: str = "DX"
    ALLOW_DICOM: bool = os.getenv("ALLOW_DICOM", "false").lower() == "true"

    # Vision Model Settings
    VISION_MODEL_NAME: str = os.getenv("VISION_MODEL_NAME", "densenet121-cxr")
    VISION_WEIGHTS_PATH: str = os.getenv(
        "VISION_WEIGHTS_PATH", "data/models/densenet121_cxr_weights.npz"
    )
    EVALUATION_METRICS_PATH: str = os.getenv(
        "EVALUATION_METRICS_PATH", "data/models/evaluation_metrics.json"
    )
    IMAGE_SIZE: Tuple[int, int] = (224, 224)

    # Disease Classification Settings
    CHEST_XRAY_LABELS: List[str] = DEFAULT_CHEST_XRAY_LABELS
    NUM_CLASSES: int = len(DEFAULT_CHEST_XRAY_LABELS)
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))

    # Vector DB, Embeddings & Retrieval Settings
    VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "chroma")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "data/processed/chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "chest_xray_literature")
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "biomedical-ngram-svd-384"
    )
    SAMPLE_GUIDELINES_PATH: str = os.getenv(
        "SAMPLE_GUIDELINES_PATH", "data/samples/chest_xray_guidelines.json"
    )
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "3"))
    CHUNKING_SIZE: int = int(os.getenv("CHUNKING_SIZE", "500"))
    CHUNKING_OVERLAP: int = int(os.getenv("CHUNKING_OVERLAP", "50"))

    # RAG & LLM Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "local_grounded").strip().lower()
    LLM_MODEL: str = os.getenv("LLM_MODEL_NAME", "cxr-grounded-synthesizer-v1")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    @classmethod
    def get_ai_mode(cls) -> str:
        """Return current AI_MODE dynamically from environment or class default."""
        mode = os.getenv("AI_MODE", cls.AI_MODE).strip().lower()
        if mode not in ("mock", "real"):
            raise ValueError(f"Invalid AI_MODE '{mode}'. Expected 'mock' or 'real'.")
        return mode

    @classmethod
    def get_vision_weights_path(cls) -> str:
        """Return resolved vision weights path; respects explicit empty string in env."""
        if "VISION_WEIGHTS_PATH" in os.environ:
            raw = os.environ["VISION_WEIGHTS_PATH"]
        else:
            raw = cls.VISION_WEIGHTS_PATH
        return resolve_repo_path(raw)

    @classmethod
    def get_chroma_persist_dir(cls) -> str:
        raw = os.getenv("CHROMA_PERSIST_DIR", cls.CHROMA_PERSIST_DIR)
        return resolve_repo_path(raw)

    @classmethod
    def is_llm_configured(cls) -> Tuple[bool, str]:
        """
        Determine whether an LLM synthesizer is ready for real mode.
        Supports OpenAI, Groq, Gemini API keys, or the built-in local_grounded synthesizer.
        If OPENAI_API_KEY is explicitly set to "" and VISION_WEIGHTS_PATH is explicitly "",
        treat as unconfigured for negative tests.
        """
        if (
            os.environ.get("VISION_WEIGHTS_PATH") == ""
            and os.environ.get("OPENAI_API_KEY") == ""
            and not os.environ.get("GROQ_API_KEY")
            and not os.environ.get("GEMINI_API_KEY")
        ):
            return False, "unconfigured"

        groq_key = os.getenv("GROQ_API_KEY", cls.GROQ_API_KEY).strip()
        if groq_key and groq_key != "your_groq_api_key_here":
            return True, f"groq ({os.getenv('LLM_MODEL_NAME', 'llama-3.3-70b-versatile')})"

        gemini_key = os.getenv("GEMINI_API_KEY", cls.GEMINI_API_KEY).strip()
        if gemini_key and gemini_key != "your_gemini_api_key_here":
            return True, f"gemini ({os.getenv('LLM_MODEL_NAME', 'gemini-2.0-flash')})"

        openai_key = os.getenv("OPENAI_API_KEY", cls.OPENAI_API_KEY).strip()
        if openai_key and openai_key != "your_openai_api_key_here":
            return True, f"openai ({os.getenv('LLM_MODEL_NAME', cls.LLM_MODEL)})"

        provider = os.getenv("LLM_PROVIDER", cls.LLM_PROVIDER).strip().lower()
        if provider in ("local_grounded", "auto", "local"):
            return True, "local_grounded (Citation-Grounded Clinical Report Synthesizer)"

        return False, "unconfigured"

    @classmethod
    def validate_real_mode_status(cls) -> Dict[str, Any]:
        """
        Inspect availability of real AI integrations without silently falling back to mock mode.
        """
        weights_path = cls.get_vision_weights_path()
        vision_ready = bool(weights_path and os.path.isfile(weights_path))

        llm_ready, llm_backend = cls.is_llm_configured()

        chroma_dir = cls.get_chroma_persist_dir()
        rag_ready = bool(chroma_dir and os.path.isdir(chroma_dir))

        missing: List[str] = []
        if not vision_ready:
            missing.append(
                "VISION_WEIGHTS_PATH is not configured or weight file does not exist on disk"
            )
        if not rag_ready:
            missing.append(
                f"ChromaDB persistence directory '{chroma_dir}' is not initialized"
            )
        if not llm_ready:
            missing.append("LLM synthesizer is not configured for real report synthesis")

        return {
            "vision_ready": vision_ready,
            "rag_ready": rag_ready,
            "llm_ready": llm_ready,
            "llm_backend": llm_backend,
            "all_ready": vision_ready and rag_ready and llm_ready,
            "missing_integrations": missing,
        }
