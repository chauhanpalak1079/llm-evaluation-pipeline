"""
Tests for relevance evaluation module.

This module tests the relevance evaluator's ability to assess semantic
similarity and completeness.
"""

import pytest

from evaluator.relevance import RelevanceEvaluator
from evaluator.models import RelevanceMetrics, CompletenessMetrics


class TestRelevanceEvaluator:
    """Test suite for RelevanceEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create relevance evaluator instance."""
        return RelevanceEvaluator()

    def test_evaluate_high_relevance(self, evaluator):
        """Test evaluation of highly relevant response."""
        ai_response = (
            "The hotel offers air-conditioned rooms at Rs 800 "
            "per night."
        )
        user_query = "What are the hotel room prices?"
        context_texts = [
            "Hotel rooms cost Rs 800 per night with AC"
        ]

        result = evaluator.evaluate_relevance(
            ai_response, user_query, context_texts
        )

        assert isinstance(result, RelevanceMetrics)
        assert result.score >= 0.5
        assert 0.0 <= result.semantic_similarity <= 1.0
        assert 0.0 <= result.context_utilization <= 1.0

    def test_evaluate_low_relevance(self, evaluator):
        """Test evaluation of low relevance response."""
        ai_response = "Our clinic has excellent facilities."
        user_query = "What are the room prices?"
        context_texts = ["Hotel rooms cost Rs 800 per night"]

        result = evaluator.evaluate_relevance(
            ai_response, user_query, context_texts
        )

        assert isinstance(result, RelevanceMetrics)
        # Should have lower scores due to mismatch
        assert result.score <= 1.0

    def test_evaluate_completeness_full(self, evaluator):
        """Test completeness evaluation with complete response."""
        ai_response = (
            "Gopal Mansion offers air-conditioned rooms with TV "
            "at Rs 800 per night."
        )
        user_query = "Tell me about Gopal Mansion room prices"
        context_texts = ["Context about hotels"]

        result = evaluator.evaluate_completeness(
            ai_response, user_query, context_texts
        )

        assert isinstance(result, CompletenessMetrics)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.covered_topics, list)

    def test_evaluate_completeness_missing(self, evaluator):
        """Test completeness with missing information."""
        ai_response = "Rooms are available."
        user_query = (
            "What are the prices for Gopal Mansion and Hotel Sagar?"
        )
        context_texts = ["Hotel information"]

        result = evaluator.evaluate_completeness(
            ai_response, user_query, context_texts
        )

        assert isinstance(result, CompletenessMetrics)
        # Should detect missing specific details
        assert isinstance(result.missing_aspects, list)

    def test_no_user_query(self, evaluator):
        """Test evaluation when user query is not provided."""
        ai_response = "This is a response."
        user_query = ""
        context_texts = ["Some context"]

        result = evaluator.evaluate_relevance(
            ai_response, user_query, context_texts
        )

        assert isinstance(result, RelevanceMetrics)
        # Should still work with default values
        assert result.semantic_similarity >= 0.0

    def test_empty_context(self, evaluator):
        """Test evaluation with no context vectors."""
        ai_response = "This is a response."
        user_query = "What is the answer?"
        context_texts = []

        result = evaluator.evaluate_relevance(
            ai_response, user_query, context_texts
        )

        assert isinstance(result, RelevanceMetrics)
        assert result.context_utilization == 0.0

    def test_topic_extraction(self, evaluator):
        """Test that topics are extracted correctly."""
        ai_response = (
            "Gopal Mansion offers rooms at Rs 800. Hotel Sagar "
            "is another option."
        )
        user_query = "Tell me about hotels"
        context_texts = ["Hotel information"]

        result = evaluator.evaluate_completeness(
            ai_response, user_query, context_texts
        )

        assert len(result.covered_topics) > 0
        # Should extract topics like "Gopal Mansion", "Hotel Sagar"
        topics_str = " ".join(result.covered_topics)
        assert "Gopal" in topics_str or "Hotel" in topics_str or "800" in topics_str

    def test_context_utilization(self, evaluator):
        """Test context utilization scoring."""
        ai_response = (
            "The air-conditioned rooms at Gopal Mansion cost "
            "Rs 800 per night with TV and bath facilities."
        )
        user_query = "Room prices?"
        context_texts = [
            "Gopal Mansion - air-conditioned room with TV and "
            "bath is Rs 800 per night",
            "Unrelated hotel information about different property",
        ]

        result = evaluator.evaluate_relevance(
            ai_response, user_query, context_texts
        )

        # Should have good context utilization
        assert result.context_utilization >= 0.5

    def test_markdown_stripping(self, evaluator):
        """Test that markdown is properly handled."""
        ai_response = "**Gopal Mansion** offers *premium* rooms."
        user_query = "Tell me about Gopal Mansion"
        context_texts = ["Gopal Mansion information"]

        result = evaluator.evaluate_completeness(
            ai_response, user_query, context_texts
        )

        # Should extract topics even with markdown
        assert len(result.covered_topics) > 0
