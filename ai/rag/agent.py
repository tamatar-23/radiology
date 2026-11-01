"""
Agentic Radiology RAG Workflow Agent (LangGraph)
Owner: Gourav (Agentic RAG & LLM Integration)

Implements a 5-stage stateful LangGraph workflow:
  1. Query Planner (`plan_queries_node`)
  2. Medical Evidence Retriever (`retrieve_evidence_node`)
  3. Reranker (`rerank_evidence_node`)
  4. Evidence Checker (`verify_evidence_node`)
  5. Report Generator (`generate_report_node`)

Supports conditional routing when insufficient evidence is retrieved or unsupported claims
are detected, while remaining decoupled from FastAPI and the frontend.
"""

import time
from typing import Dict, Any, List, Optional, TypedDict, Literal

from ai.config import AIConfig
from ai.common.schemas import (
    ExecutionMode,
    LabelPrediction,
    EvidenceCitation,
    AgentStageTrace,
    RAGWorkflowResult,
)
from ai.common.interfaces import (
    BaseRetrievalProvider,
    BaseReranker,
    BaseReportGenerator,
)
from ai.rag.providers import (
    DemoRetrievalProvider,
    RealRetrievalProvider,
    DemoReportGenerator,
    RealReportGenerator,
)
from ai.rag.reranker import EvidenceReranker


class RAGGraphState(TypedDict, total=False):
    """Typed state dictionary passed across LangGraph nodes."""

    mode: ExecutionMode
    predictions: List[LabelPrediction]
    clinical_notes: Optional[str]
    planned_queries: List[str]
    retrieved_candidates: List[EvidenceCitation]
    reranked_evidence: List[EvidenceCitation]
    evidence_status: Literal["sufficient", "insufficient", "unverified_claims"]
    verification_notes: str
    findings: List[str]
    impression: str
    agent_trace: List[AgentStageTrace]


class RadiologyRAGAgent:
    """
    LangGraph-based Agentic RAG orchestrator for Chest X-Ray report synthesis.
    """

    def __init__(
        self,
        mode: Optional[str] = None,
        retrieval_provider: Optional[BaseRetrievalProvider] = None,
        reranker: Optional[BaseReranker] = None,
        report_generator: Optional[BaseReportGenerator] = None,
        top_k: int = AIConfig.TOP_K_RETRIEVAL,
    ):
        raw_mode = (mode or AIConfig.get_ai_mode()).lower()
        if raw_mode not in ("mock", "real"):
            raise ValueError(f"Unsupported AI_MODE '{raw_mode}'.")
        self.mode: ExecutionMode = "mock" if raw_mode == "mock" else "real"
        self.top_k = top_k

        if retrieval_provider is not None:
            self.retrieval_provider = retrieval_provider
        else:
            self.retrieval_provider = (
                DemoRetrievalProvider() if self.mode == "mock" else RealRetrievalProvider()
            )

        self.reranker = reranker or EvidenceReranker()

        if report_generator is not None:
            self.report_generator = report_generator
        else:
            self.report_generator = (
                DemoReportGenerator() if self.mode == "mock" else RealReportGenerator()
            )

        self._compiled_graph = self._build_graph()

    # ------------------------------------------------------------------
    # Node 1: Query Planner
    # ------------------------------------------------------------------
    def plan_queries_node(self, state: RAGGraphState) -> RAGGraphState:
        t0 = time.perf_counter()
        predictions = state.get("predictions", [])
        notes = state.get("clinical_notes")

        elevated = [p for p in predictions if p.score >= AIConfig.CONFIDENCE_THRESHOLD]
        if not elevated and predictions:
            elevated = predictions[:2]

        queries: List[str] = []
        for pred in elevated[:3]:
            label_clean = pred.label.replace("_", " ")
            queries.append(f"Chest X-Ray radiographic criteria and guidelines for {label_clean}")

        if notes and notes.strip():
            queries.append(f"Thoracic imaging evaluation for: {notes.strip()}")

        if not queries:
            queries.append("Standard posteroanterior chest radiograph interpretation guidelines")

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        trace = list(state.get("agent_trace", []))
        trace.append(
            AgentStageTrace(
                stage_id="query_planning",
                stage_name="Query Planning",
                status="completed",
                mode=self.mode,
                duration_ms=duration_ms,
                details=f"Formulated {len(queries)} targeted thoracic literature query/queries from top predictions.",
            )
        )
        return {
            **state,
            "planned_queries": queries,
            "agent_trace": trace,
        }

    # ------------------------------------------------------------------
    # Node 2: Medical Evidence Retriever
    # ------------------------------------------------------------------
    def retrieve_evidence_node(self, state: RAGGraphState) -> RAGGraphState:
        t0 = time.perf_counter()
        queries = state.get("planned_queries", [])
        candidates = self.retrieval_provider.retrieve(queries, top_k=self.top_k + 2)
        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        trace = list(state.get("agent_trace", []))
        trace.append(
            AgentStageTrace(
                stage_id="evidence_retrieval",
                stage_name="Evidence Retrieval",
                status="completed" if candidates else "flagged",
                mode=self.mode,
                duration_ms=duration_ms,
                details=(
                    f"Retrieved {len(candidates)} candidate guideline passage(s)."
                    if candidates
                    else "No candidate passages matched the planned queries."
                ),
            )
        )
        return {
            **state,
            "retrieved_candidates": candidates,
            "agent_trace": trace,
        }

    # ------------------------------------------------------------------
    # Node 3: Reranker
    # ------------------------------------------------------------------
    def rerank_evidence_node(self, state: RAGGraphState) -> RAGGraphState:
        t0 = time.perf_counter()
        queries = state.get("planned_queries", [])
        candidates = state.get("retrieved_candidates", [])
        reranked = self.reranker.rerank(queries, candidates, top_k=self.top_k)
        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        trace = list(state.get("agent_trace", []))
        trace.append(
            AgentStageTrace(
                stage_id="reranking",
                stage_name="Reranking",
                status="completed" if reranked else "skipped",
                mode=self.mode,
                duration_ms=duration_ms,
                details=(
                    f"Reranked and retained top {len(reranked)} passage(s) (top score: {reranked[0].relevance_score:.2f})."
                    if reranked
                    else "Skipped reranking because no candidate passages were retrieved."
                ),
            )
        )
        return {
            **state,
            "reranked_evidence": reranked,
            "agent_trace": trace,
        }

    # ------------------------------------------------------------------
    # Node 4: Evidence Checker
    # ------------------------------------------------------------------
    def verify_evidence_node(self, state: RAGGraphState) -> RAGGraphState:
        t0 = time.perf_counter()
        reranked = state.get("reranked_evidence", [])
        predictions = state.get("predictions", [])

        if not reranked:
            evidence_status: Literal["sufficient", "insufficient", "unverified_claims"] = "insufficient"
            notes = "Insufficient evidence: zero reference passages passed retrieval and reranking thresholds."
            stage_status: Literal["completed", "skipped", "flagged", "failed"] = "flagged"
        else:
            # Verify that at least one top predicted abnormality is discussed in retrieved passages
            combined_passages = " ".join(
                f"{c.document_title} {c.passage}".lower() for c in reranked
            )
            top_labels = [
                p.label.lower().replace("_", " ")
                for p in predictions[:2]
                if p.score >= AIConfig.CONFIDENCE_THRESHOLD
            ]
            unmatched = [lbl for lbl in top_labels if lbl not in combined_passages]
            if top_labels and len(unmatched) == len(top_labels):
                evidence_status = "unverified_claims"
                notes = (
                    f"Flagged for review: retrieved passages do not directly cover predicted label(s) "
                    f"{', '.join(unmatched)}."
                )
                stage_status = "flagged"
            else:
                evidence_status = "sufficient"
                notes = f"Verified {len(reranked)} reference citation(s) supporting the report context."
                stage_status = "completed"

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        trace = list(state.get("agent_trace", []))
        trace.append(
            AgentStageTrace(
                stage_id="evidence_verification",
                stage_name="Evidence Verification",
                status=stage_status,
                mode=self.mode,
                duration_ms=duration_ms,
                details=notes,
            )
        )
        return {
            **state,
            "evidence_status": evidence_status,
            "verification_notes": notes,
            "agent_trace": trace,
        }

    # ------------------------------------------------------------------
    # Conditional Edge Router
    # ------------------------------------------------------------------
    @staticmethod
    def route_after_verification(
        state: RAGGraphState,
    ) -> Literal["generate_verified_report", "generate_flagged_report"]:
        status = state.get("evidence_status", "insufficient")
        if status == "sufficient":
            return "generate_verified_report"
        return "generate_flagged_report"

    # ------------------------------------------------------------------
    # Node 5: Report Generator
    # ------------------------------------------------------------------
    def generate_report_node(self, state: RAGGraphState) -> RAGGraphState:
        t0 = time.perf_counter()
        predictions = state.get("predictions", [])
        evidence = state.get("reranked_evidence", [])
        evidence_status = state.get("evidence_status", "insufficient")
        clinical_notes = state.get("clinical_notes")

        generated = self.report_generator.generate_report(
            predictions=predictions,
            evidence=evidence,
            evidence_status=evidence_status,
            clinical_notes=clinical_notes,
        )

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        trace = list(state.get("agent_trace", []))
        trace.append(
            AgentStageTrace(
                stage_id="report_generation",
                stage_name="Report Generation",
                status="completed" if evidence_status == "sufficient" else "flagged",
                mode=self.mode,
                duration_ms=duration_ms,
                details=(
                    "Synthesized structured Chest X-Ray findings and impression with inline citations."
                    if evidence_status == "sufficient"
                    else f"Generated cautious report flagged with status '{evidence_status}'."
                ),
            )
        )
        return {
            **state,
            "findings": generated.get("findings", []),
            "impression": generated.get("impression", ""),
            "agent_trace": trace,
        }

    # ------------------------------------------------------------------
    # LangGraph Builder (with lightweight sequential fallback)
    # ------------------------------------------------------------------
    def _build_graph(self) -> Any:
        try:
            from langgraph.graph import StateGraph, END  # type: ignore

            workflow = StateGraph(RAGGraphState)
            workflow.add_node("query_planner", self.plan_queries_node)
            workflow.add_node("evidence_retriever", self.retrieve_evidence_node)
            workflow.add_node("reranker", self.rerank_evidence_node)
            workflow.add_node("evidence_checker", self.verify_evidence_node)
            workflow.add_node("report_generator", self.generate_report_node)

            workflow.set_entry_point("query_planner")
            workflow.add_edge("query_planner", "evidence_retriever")
            workflow.add_edge("evidence_retriever", "reranker")
            workflow.add_edge("reranker", "evidence_checker")
            workflow.add_conditional_edges(
                "evidence_checker",
                self.route_after_verification,
                {
                    "generate_verified_report": "report_generator",
                    "generate_flagged_report": "report_generator",
                },
            )
            workflow.add_edge("report_generator", END)
            return workflow.compile()
        except Exception:
            return None

    def execute_workflow(
        self,
        predictions: List[LabelPrediction],
        clinical_notes: Optional[str] = None,
    ) -> RAGWorkflowResult:
        """
        Execute the complete 5-node Agentic RAG workflow and return a typed RAGWorkflowResult.
        """
        initial_state: RAGGraphState = {
            "mode": self.mode,
            "predictions": predictions,
            "clinical_notes": clinical_notes,
            "planned_queries": [],
            "retrieved_candidates": [],
            "reranked_evidence": [],
            "evidence_status": "sufficient",
            "verification_notes": "",
            "findings": [],
            "impression": "",
            "agent_trace": [],
        }

        if self._compiled_graph is not None:
            final_state: RAGGraphState = self._compiled_graph.invoke(initial_state)
        else:
            state = self.plan_queries_node(initial_state)
            state = self.retrieve_evidence_node(state)
            state = self.rerank_evidence_node(state)
            state = self.verify_evidence_node(state)
            final_state = self.generate_report_node(state)

        return RAGWorkflowResult(
            planned_queries=final_state.get("planned_queries", []),
            findings=final_state.get("findings", []),
            impression=final_state.get("impression", ""),
            evidence=final_state.get("reranked_evidence", []),
            evidence_status=final_state.get("evidence_status", "insufficient"),
            verification_notes=final_state.get("verification_notes", ""),
            agent_trace=final_state.get("agent_trace", []),
            mode=self.mode,
        )

    def run_pipeline(
        self,
        vision_findings: Optional[List[str]] = None,
        scan_metadata: Optional[Dict[str, Any]] = None,
        predictions: Optional[List[LabelPrediction]] = None,
        clinical_notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Backward-compatible dictionary wrapper around `execute_workflow`.
        """
        preds = predictions or [
            LabelPrediction(label="Atelectasis", score=0.76),
            LabelPrediction(label="Consolidation", score=0.64),
        ]
        notes = clinical_notes or (
            "; ".join(vision_findings) if vision_findings else None
        )
        result = self.execute_workflow(predictions=preds, clinical_notes=notes)
        return {
            "impression": result.impression,
            "findings": result.findings,
            "evidence": [ev.model_dump() for ev in result.evidence],
            "evidence_status": result.evidence_status,
            "verification_notes": result.verification_notes,
            "agent_trace": [st.model_dump() for st in result.agent_trace],
            "mode": result.mode,
            "confidence": preds[0].score if preds else 0.0,
        }
