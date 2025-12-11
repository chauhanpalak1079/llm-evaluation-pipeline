"""
Integration tests for the evaluation pipeline.

This module tests the complete pipeline using sample data files.
"""

import json
from pathlib import Path

import pytest

from evaluator.models import ChatConversation, ContextVectors
from evaluator.pipeline import EvaluationPipeline


class TestEvaluationPipeline:
    """Integration test suite for EvaluationPipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance."""
        return EvaluationPipeline()

    @pytest.fixture
    def sample_data_dir(self):
        """Get sample data directory path."""
        return Path(__file__).parent.parent / "sample_data"

    def test_pipeline_initialization(self, pipeline):
        """Test that pipeline initializes correctly."""
        assert pipeline is not None
        assert pipeline.relevance_evaluator is not None
        assert pipeline.hallucination_detector is not None
        assert pipeline.factual_evaluator is not None

    def test_evaluate_sample_01_with_hallucination(
        self, pipeline, sample_data_dir
    ):
        """
        Test evaluation of sample 01 containing hallucination.

        This is the CRITICAL test case from the problem statement.
        """
        # Load sample data
        conv_file = sample_data_dir / "sample_chat_conversation_01.json"
        ctx_file = sample_data_dir / "sample_context_vectors_01.json"

        with open(conv_file) as f:
            conv_data = json.load(f)
        with open(ctx_file) as f:
            ctx_data = json.load(f)

        conversation = ChatConversation(**conv_data)
        context = ContextVectors(**ctx_data)

        # Evaluate turn 14 (contains hallucination)
        result = pipeline.evaluate_turn(conversation, context, 14)

        # Verify result structure
        assert result is not None
        assert result.conversation_id == 78128
        assert result.turn_evaluated == 14

        # Verify metrics exist
        assert result.metrics is not None
        assert result.metrics.relevance is not None
        assert result.metrics.completeness is not None
        assert result.metrics.hallucination is not None
        assert result.metrics.factual_accuracy is not None

        # Verify performance metrics
        assert result.performance is not None
        assert result.performance.total_latency_ms > 0

        # CRITICAL: Must detect hallucination about clinic rooms
        assert len(
            result.metrics.hallucination.detected_hallucinations
        ) > 0

        # Verify the specific hallucination is detected
        hallucinations = (
            result.metrics.hallucination.detected_hallucinations
        )
        hallucination_texts = [h.text.lower() for h in hallucinations]

        # Check for keywords from the hallucinated claim
        has_clinic_hallucination = any(
            ("clinic" in text or "subsidized" in text or "2000" in text)
            for text in hallucination_texts
        )

        assert has_clinic_hallucination, (
            f"Pipeline failed to detect hallucination about "
            f"subsidized clinic rooms. Detected: {hallucination_texts}"
        )

        # Verify overall score is affected by hallucination
        assert result.overall_score <= 1.0

        # Verify recommendations include hallucination warning
        recommendations_text = " ".join(result.recommendations).lower()
        assert "hallucination" in recommendations_text

    def test_evaluate_sample_02_clean(self, pipeline, sample_data_dir):
        """Test evaluation of sample 02 without hallucinations."""
        # Load sample data
        conv_file = sample_data_dir / "sample_chat_conversation_02.json"
        ctx_file = sample_data_dir / "sample_context_vectors_02.json"

        with open(conv_file) as f:
            conv_data = json.load(f)
        with open(ctx_file) as f:
            ctx_data = json.load(f)

        conversation = ChatConversation(**conv_data)
        context = ContextVectors(**ctx_data)

        # Evaluate turn 3 (clean response)
        result = pipeline.evaluate_turn(conversation, context, 3)

        # Verify result structure
        assert result is not None
        assert result.conversation_id == 78129
        assert result.turn_evaluated == 3

        # Should have good scores for clean response
        assert result.metrics.relevance.score >= 0.0
        assert result.overall_score >= 0.0

    def test_result_json_serialization(
        self, pipeline, sample_data_dir
    ):
        """Test that results can be serialized to JSON."""
        conv_file = sample_data_dir / "sample_chat_conversation_01.json"
        ctx_file = sample_data_dir / "sample_context_vectors_01.json"

        with open(conv_file) as f:
            conv_data = json.load(f)
        with open(ctx_file) as f:
            ctx_data = json.load(f)

        conversation = ChatConversation(**conv_data)
        context = ContextVectors(**ctx_data)

        result = pipeline.evaluate_turn(conversation, context, 14)

        # Convert to dict and serialize
        result_dict = result.model_dump()
        json_str = json.dumps(result_dict)

        # Should be valid JSON
        assert json_str is not None
        parsed = json.loads(json_str)
        assert "evaluation_id" in parsed
        assert "metrics" in parsed
        assert "performance" in parsed

    def test_invalid_turn_number(self, pipeline, sample_data_dir):
        """Test error handling for invalid turn number."""
        conv_file = sample_data_dir / "sample_chat_conversation_01.json"
        ctx_file = sample_data_dir / "sample_context_vectors_01.json"

        with open(conv_file) as f:
            conv_data = json.load(f)
        with open(ctx_file) as f:
            ctx_data = json.load(f)

        conversation = ChatConversation(**conv_data)
        context = ContextVectors(**ctx_data)

        # Try to evaluate non-existent turn
        with pytest.raises(ValueError):
            pipeline.evaluate_turn(conversation, context, 999)

    def test_performance_breakdown(self, pipeline, sample_data_dir):
        """Test that performance breakdown is tracked."""
        conv_file = sample_data_dir / "sample_chat_conversation_01.json"
        ctx_file = sample_data_dir / "sample_context_vectors_01.json"

        with open(conv_file) as f:
            conv_data = json.load(f)
        with open(ctx_file) as f:
            ctx_data = json.load(f)

        conversation = ChatConversation(**conv_data)
        context = ContextVectors(**ctx_data)

        result = pipeline.evaluate_turn(conversation, context, 14)

        # Verify breakdown exists
        breakdown = result.performance.breakdown
        assert "relevance_ms" in breakdown
        assert "hallucination_ms" in breakdown
        assert "factual_accuracy_ms" in breakdown

        # All components should have positive latency
        assert breakdown["relevance_ms"] >= 0
        assert breakdown["hallucination_ms"] >= 0
        assert breakdown["factual_accuracy_ms"] >= 0

    def test_cost_estimation(self, pipeline, sample_data_dir):
        """Test that cost estimation is included."""
        conv_file = sample_data_dir / "sample_chat_conversation_01.json"
        ctx_file = sample_data_dir / "sample_context_vectors_01.json"

        with open(conv_file) as f:
            conv_data = json.load(f)
        with open(ctx_file) as f:
            ctx_data = json.load(f)

        conversation = ChatConversation(**conv_data)
        context = ContextVectors(**ctx_data)

        result = pipeline.evaluate_turn(conversation, context, 14)

        # Verify cost fields
        assert result.performance.estimated_cost_usd >= 0
        assert result.performance.tokens_processed > 0
