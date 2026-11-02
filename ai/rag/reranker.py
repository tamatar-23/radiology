"""
Medical Evidence Reranker
Owner: Gourav (Agentic RAG & Reranking)

Scores and orders retrieved evidence citations against planned clinical queries
using a hybrid scoring model combining dense vector similarity, TF-IDF cosine
similarity, and clinical pathology term overlap.
"""

import re
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ai.common.schemas import EvidenceCitation
from ai.common.interfaces import BaseReranker


class EvidenceReranker(BaseReranker):
    """
    Reranks retrieved medical literature passages by combining initial dense retrieval
    relevance with TF-IDF n-gram cosine similarity and clinical query term overlap.
    """

    def __init__(self, min_score_threshold: float = 0.25):
        self.min_score_threshold = min_score_threshold

    def rerank(
        self,
        queries: List[str],
        candidates: List[EvidenceCitation],
        top_k: int = 3,
    ) -> List[EvidenceCitation]:
        if not candidates:
            return []

        combined_query = " ".join(queries).strip()
        query_tokens = set()
        for q in queries:
            for token in re.findall(r"[a-zA-Z]{3,}", q.lower()):
                if token not in {
                    "chest",
                    "xray",
                    "ray",
                    "scan",
                    "findings",
                    "radiograph",
                    "radiographic",
                    "thoracic",
                    "criteria",
                    "guidelines",
                    "evaluation",
                }:
                    query_tokens.add(token)

        # Compute TF-IDF cosine similarity across candidate passages
        tfidf_scores = np.zeros(len(candidates), dtype=np.float32)
        if combined_query:
            try:
                corpus = [
                    f"{c.document_title} {c.passage}".lower() for c in candidates
                ]
                vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
                mat = vec.fit_transform([combined_query.lower()] + corpus)
                sims = cosine_similarity(mat[0:1], mat[1:])[0]
                tfidf_scores = np.asarray(sims, dtype=np.float32)
            except Exception:
                tfidf_scores = np.zeros(len(candidates), dtype=np.float32)

        reranked: List[EvidenceCitation] = []
        for idx, item in enumerate(candidates):
            haystack = f"{item.document_title} {item.passage}".lower()
            if query_tokens:
                matches = sum(1 for tok in query_tokens if tok in haystack)
                lexical_boost = min(0.22, matches * 0.06)
            else:
                lexical_boost = 0.0

            tfidf_boost = float(tfidf_scores[idx]) * 0.20
            combined_score = round(
                min(0.99, item.relevance_score * 0.78 + lexical_boost + tfidf_boost),
                2,
            )
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
