"""
Data models for LLM evaluation pipeline using Pydantic.

This module defines the data structures for conversations, context vectors,
and evaluation results.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """Model for a single conversation turn."""

    turn: int
    sender_id: int
    role: str
    message: str
    created_at: str
    evaluation_note: Optional[str] = None


class ChatConversation(BaseModel):
    """Model for complete chat conversation."""

    chat_id: int
    user_id: int
    conversation_turns: List[ConversationTurn]


class VectorData(BaseModel):
    """Model for a single vector data entry."""

    id: int
    source_url: str
    text: str
    tokens: int
    created_at: str


class VectorSources(BaseModel):
    """Model for vector sources metadata."""

    message_id: int
    vector_ids: List[str]
    vectors_used: List[int]
    final_response: List[str]


class ContextVectorsData(BaseModel):
    """Model for context vectors data structure."""

    vector_data: List[VectorData]
    sources: VectorSources


class ContextVectors(BaseModel):
    """Model for complete context vectors response."""

    status: str
    status_code: int
    data: ContextVectorsData


class HallucinationDetection(BaseModel):
    """Model for detected hallucination."""

    text: str
    reason: str
    severity: str
    confidence: float


class RelevanceMetrics(BaseModel):
    """Model for relevance evaluation metrics."""

    score: float
    query_coverage: float
    context_utilization: float
    semantic_similarity: float
    details: Dict[str, Any] = Field(default_factory=dict)


class CompletenessMetrics(BaseModel):
    """Model for completeness evaluation metrics."""

    score: float
    missing_aspects: List[str] = Field(default_factory=list)
    covered_topics: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


class HallucinationMetrics(BaseModel):
    """Model for hallucination detection metrics."""

    score: float
    detected_hallucinations: List[HallucinationDetection] = Field(
        default_factory=list
    )
    grounded_claims_ratio: float
    total_claims: int
    grounded_claims: int


class FactualAccuracyMetrics(BaseModel):
    """Model for factual accuracy metrics."""

    score: float
    verified_claims: List[str] = Field(default_factory=list)
    unverified_claims: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)


class EvaluationMetrics(BaseModel):
    """Model for all evaluation metrics."""

    relevance: RelevanceMetrics
    completeness: CompletenessMetrics
    hallucination: HallucinationMetrics
    factual_accuracy: FactualAccuracyMetrics


class PerformanceMetrics(BaseModel):
    """Model for performance and cost metrics."""

    total_latency_ms: float
    breakdown: Dict[str, float]
    estimated_cost_usd: float
    evaluation_mode: str
    tokens_processed: int


class EvaluationResult(BaseModel):
    """Model for complete evaluation result."""

    evaluation_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    conversation_id: int
    turn_evaluated: int
    metrics: EvaluationMetrics
    performance: PerformanceMetrics
    overall_score: float
    recommendations: List[str] = Field(default_factory=list)
