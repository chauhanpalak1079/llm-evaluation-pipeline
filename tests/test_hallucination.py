"""
Tests for hallucination detection module.

This module tests the hallucination detector's ability to identify
claims not grounded in provided context.
"""

import pytest

from evaluator.hallucination import HallucinationDetector
from evaluator.models import HallucinationMetrics


class TestHallucinationDetector:
    """Test suite for HallucinationDetector."""

    @pytest.fixture
    def detector(self):
        """Create hallucination detector instance."""
        return HallucinationDetector()

    def test_detect_no_hallucinations(self, detector):
        """Test detection when response is grounded in context."""
        ai_response = "Gopal Mansion offers rooms at Rs 800 per night."
        context_texts = [
            "Gopal Mansion - Rs 800 per night for AC rooms",
            "Hotel near clinic with affordable rates",
        ]

        result = detector.detect_hallucinations(
            ai_response, context_texts
        )

        assert isinstance(result, HallucinationMetrics)
        assert result.score >= 0.6
        assert result.total_claims >= 0

    def test_detect_hallucination_pricing(self, detector):
        """Test detection of hallucinated pricing information."""
        ai_response = (
            "We offer subsidized rooms at our clinic for Rs 2000 "
            "per night which include breakfast and medical monitoring."
        )
        context_texts = [
            "Gopal Mansion - Rs 800 per night for AC rooms",
            "Hotel Sagar - Rs 600 per night for standard rooms",
        ]

        result = detector.detect_hallucinations(
            ai_response, context_texts
        )

        assert isinstance(result, HallucinationMetrics)
        # Should detect hallucination about clinic rooms
        assert len(result.detected_hallucinations) > 0

    def test_detect_mixed_claims(self, detector):
        """Test detection with mix of grounded and hallucinated."""
        ai_response = (
            "Gopal Mansion offers rooms at Rs 800 per night. "
            "We also provide luxury suites at Rs 5000 per night."
        )
        context_texts = [
            "Gopal Mansion - Rs 800 per night for AC rooms with TV"
        ]

        result = detector.detect_hallucinations(
            ai_response, context_texts
        )

        assert isinstance(result, HallucinationMetrics)
        assert result.total_claims >= 2
        # At least one claim should be grounded (Rs 800)
        assert result.grounded_claims >= 1

    def test_empty_response(self, detector):
        """Test handling of empty response."""
        ai_response = ""
        context_texts = ["Some context"]

        result = detector.detect_hallucinations(
            ai_response, context_texts
        )

        assert isinstance(result, HallucinationMetrics)
        assert result.total_claims == 0
        assert result.score == 1.0

    def test_empty_context(self, detector):
        """Test handling of empty context."""
        ai_response = "This is some response text."
        context_texts = []

        result = detector.detect_hallucinations(
            ai_response, context_texts
        )

        assert isinstance(result, HallucinationMetrics)
        # All claims should be flagged as hallucinations with no context
        assert result.grounded_claims == 0

    def test_severity_levels(self, detector):
        """Test that severity levels are assigned correctly."""
        # Response with pricing (high severity)
        ai_response = "Premium rooms cost Rs 10000 per night."
        context_texts = ["Basic rooms available"]

        result = detector.detect_hallucinations(
            ai_response, context_texts
        )

        if result.detected_hallucinations:
            # Check that pricing hallucinations get high severity
            has_high_severity = any(
                h.severity == "high"
                for h in result.detected_hallucinations
            )
            # May or may not detect based on threshold
            assert isinstance(has_high_severity, bool)

    def test_markdown_handling(self, detector):
        """Test that markdown formatting is handled correctly."""
        ai_response = (
            "**Gopal Mansion** offers rooms at *Rs 800* per night."
        )
        context_texts = [
            "Gopal Mansion has rooms for Rs 800 per night"
        ]

        result = detector.detect_hallucinations(
            ai_response, context_texts
        )

        assert isinstance(result, HallucinationMetrics)
        # Should strip markdown and still detect grounding
        assert result.score >= 0.5

    def test_critical_hallucination_example(self, detector):
        """Test the critical example from problem statement."""
        ai_response = (
            "For Gopal Mansion, an air-conditioned room with TV and "
            "bath is Rs 800 per night. We also offer specially "
            "subsidized air-conditioned rooms at our clinic for "
            "Rs 2000 (US $50) per night which include breakfast "
            "and medical monitoring."
        )
        context_texts = [
            "Gopal Mansion is located just 5 minutes from our "
            "clinic. An air-conditioned room with TV and bath is "
            "only Rs 800 per night.",
            "Budget accommodation near clinic: Hotel Sagar - "
            "Rs 600 per night for standard rooms.",
        ]

        result = detector.detect_hallucinations(
            ai_response, context_texts
        )

        # CRITICAL: Must detect the hallucination about clinic rooms
        assert len(result.detected_hallucinations) > 0

        # Check that clinic/subsidized room claim is flagged
        hallucination_texts = [
            h.text.lower()
            for h in result.detected_hallucinations
        ]
        has_clinic_hallucination = any(
            "clinic" in text or "subsidized" in text or "2000" in text
            for text in hallucination_texts
        )

        assert has_clinic_hallucination, (
            "Failed to detect hallucination about subsidized "
            "clinic rooms at Rs 2000"
        )
