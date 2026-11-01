"""
Radiology Knowledge Retriever
Owner: Gourav (Agentic RAG & Vector DB)

Coordinates retrieval of thoracic radiology literature and clinical guidelines
using either DemoRetrievalProvider (mock mode) or RealRetrievalProvider (real mode).
"""

from typing import List, Dict, Any, Optional, Union
from ai.config import AIConfig
from ai.common.schemas import EvidenceCitation
from ai.common.interfaces import BaseRetrievalProvider
from ai.rag.providers import DemoRetrievalProvider, RealRetrievalProvider


class RadiologyRetriever:
    """
    High-level retriever facade supporting structured EvidenceCitation objects
    as well as legacy dictionary retrieval calls.
    """

    def __init__(
        self,
        top_k: int = AIConfig.TOP_K_RETRIEVAL,
        mode: Optional[str] = None,
        provider: Optional[BaseRetrievalProvider] = None,
    ):
        self.top_k = top_k
        self.mode = (mode or AIConfig.get_ai_mode()).lower()
        if provider is not None:
            self.provider = provider
        elif self.mode == "mock":
            self.provider = DemoRetrievalProvider()
        elif self.mode == "real":
            self.provider = RealRetrievalProvider()
        else:
            raise ValueError(f"Unsupported AI_MODE '{self.mode}'.")

    def retrieve_citations(self, queries: Union[str, List[str]]) -> List[EvidenceCitation]:
        """Return typed EvidenceCitation objects for one or more queries."""
        query_list = [queries] if isinstance(queries, str) else list(queries)
        return self.provider.retrieve(query_list, top_k=self.top_k)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Backward-compatible dictionary retrieval method.
        """
        citations = self.retrieve_citations([query])
        return [c.model_dump() for c in citations]
