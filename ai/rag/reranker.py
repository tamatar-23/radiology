"""
Medical Evidence Reranker
Owner: Gourav (Agentic RAG & Reranking)

Scores and orders retrieved evidence citations against planned clinical queries.
"""

import re
from typing import List
from ai.common.schemas import EvidenceCitation
from ai.common.interfaces import BaseReranker


class EvidenceReranker(BaseReranker):
    """
    Reranks retrieved medical literature passages by combining initial retrieval relevance
    with clinical query term overlap. Designed so a CrossEncoder (e.g. MedCPT-Cross-Encoder)
    can be plugged in for real mode.
    """

    def __init__(self, min_score_threshold: float = 0.20):
        self.min_score_threshold = min_score_threshold

    def rerank(
        self,
        queries: List[str],
        candidates: List[EvidenceCitation],
        top_k: int = 3,
    ) -> List[EvidenceCitation]:
        if not candidates:
            return []

        query_tokens = set()
        for q in queries:
            for token in re.findall(r"[a-zA-Z]{3,}", q.lower()):
                if token not in {"chest", "xray", "ray", "scan", "findings", "radiograph", "thoracic"}:
                    query_tokens.add(token)

        reranked: List[EvidenceCitation] = []
        for item in candidates:
            haystack = f"{item.document_title} {item.passage}".lower()
            if query_tokens:
                matches = sum(1 for tok in query_tokens if tok in haystack)
                lexical_boost = min(0.18, matches * 0.05)
            else:
                lexical_boost = 0.0

            combined_score = round(min(0.99, item.relevance_score + lexical_boost), 2)
            if combined_score >= self.min_score_threshold:
                reranked.append(
                    EvidenceCitation(
                        evidence_id=item.evidence_id,
                        document_title=item.document_title,
                        source_url=item.source_url,
                        passage=item.passage,
                        chunk_id=item.chunk_id,
                        relevance_score=combined_score,
                    )
                )

        reranked.sort(key=lambda c: c.relevance_score, reverse=True)
        return reranked[:top_k]
