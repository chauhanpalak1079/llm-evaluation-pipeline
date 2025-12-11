"""
Main evaluation pipeline orchestrator.

This module coordinates all evaluation components and produces the final
evaluation results.
"""

import hashlib
from typing import Dict, List, Optional

from config.settings import get_settings
from evaluator.cost_estimator import CostEstimator
from evaluator.factual_accuracy import FactualAccuracyEvaluator
from evaluator.hallucination import HallucinationDetector
from evaluator.latency_tracker import LatencyTracker
from evaluator.models import (
    ChatConversation,
    ContextVectors,
    EvaluationMetrics,
    EvaluationResult,
    PerformanceMetrics,
)
from evaluator.relevance import RelevanceEvaluator


class EvaluationPipeline:
    """Main orchestrator for LLM response evaluation."""

    def __init__(self):
        """Initialize evaluation pipeline with all components."""
        self.settings = get_settings()
        self.relevance_evaluator = RelevanceEvaluator()
        self.hallucination_detector = HallucinationDetector()
        self.factual_evaluator = FactualAccuracyEvaluator()
        self.cost_estimator = CostEstimator()
        self.latency_tracker = LatencyTracker()

    def _extract_context_texts(
        self, context_vectors: ContextVectors
    ) -> List[str]:
        """
        Extract text content from context vectors.

        Args:
            context_vectors: Context vectors data

        Returns:
            List[str]: List of context texts
        """
        return [
            vector.text
            for vector in context_vectors.data.vector_data
        ]

    def _get_previous_user_message(
        self, conversation: ChatConversation, current_turn: int
    ) -> str:
        """
        Get the most recent user message before the current turn.

        Args:
            conversation: Chat conversation data
            current_turn: Current turn number

        Returns:
            str: User message text (empty if none found)
        """
        # Look backwards from current turn
        for turn in reversed(conversation.conversation_turns):
            if (
                turn.turn < current_turn
                and turn.role.lower() not in ["ai", "chatbot", "ai/chatbot"]
            ):
                return turn.message

        return ""

    def _compute_overall_score(
        self, metrics: EvaluationMetrics
    ) -> float:
        """
        Compute weighted overall evaluation score.

        Args:
            metrics: All evaluation metrics

        Returns:
            float: Overall weighted score (0-1)
        """
        # Weighted average of all metrics
        weights = {
            "relevance": 0.25,
            "completeness": 0.20,
            "hallucination": 0.35,  # Higher weight (critical)
            "factual_accuracy": 0.20,
        }

        overall = (
            metrics.relevance.score * weights["relevance"]
            + metrics.completeness.score * weights["completeness"]
            + metrics.hallucination.score * weights["hallucination"]
            + metrics.factual_accuracy.score
            * weights["factual_accuracy"]
        )

        return round(overall, 2)

    def _generate_recommendations(
        self, metrics: EvaluationMetrics, performance: PerformanceMetrics
    ) -> List[str]:
        """
        Generate actionable recommendations based on metrics.

        Args:
            metrics: Evaluation metrics
            performance: Performance metrics

        Returns:
            List[str]: List of recommendations
        """
        recommendations = []

        # Hallucination recommendations
        if metrics.hallucination.detected_hallucinations:
            high_severity = [
                h
                for h in metrics.hallucination.detected_hallucinations
                if h.severity == "high"
            ]
            if high_severity:
                recommendations.append(
                    "High-severity hallucination detected - review AI "
                    "response generation"
                )

        # Relevance recommendations
        if metrics.relevance.score < 0.7:
            recommendations.append(
                "Low relevance score - consider improving context "
                "retrieval"
            )

        # Factual accuracy recommendations
        if metrics.factual_accuracy.score < 0.7:
            recommendations.append(
                "Low factual accuracy - verify knowledge base coverage"
            )

        # Performance recommendations
        perf_recs = self.cost_estimator.get_optimization_recommendations(
            performance.tokens_processed, performance.total_latency_ms
        )
        recommendations.extend(perf_recs)

        # Default if all good
        if not recommendations:
            recommendations.append(
                "Evaluation metrics are within acceptable ranges"
            )

        return recommendations

    def evaluate_turn(
        self,
        conversation: ChatConversation,
        context_vectors: ContextVectors,
        turn_number: int,
    ) -> EvaluationResult:
        """
        Evaluate a specific conversation turn.

        Args:
            conversation: Chat conversation data
            context_vectors: Context vectors for the response
            turn_number: Turn number to evaluate

        Returns:
            EvaluationResult: Complete evaluation result
        """
        # Find the turn to evaluate
        target_turn = None
        for turn in conversation.conversation_turns:
            if turn.turn == turn_number:
                target_turn = turn
                break

        if not target_turn:
            raise ValueError(
                f"Turn {turn_number} not found in conversation"
            )

        ai_response = target_turn.message
        context_texts = self._extract_context_texts(context_vectors)
        user_query = self._get_previous_user_message(
            conversation, turn_number
        )

        # Track total latency
        self.latency_tracker.reset()

        # Evaluate relevance
        with self.latency_tracker.track("relevance"):
            relevance_metrics = (
                self.relevance_evaluator.evaluate_relevance(
                    ai_response, user_query, context_texts
                )
            )

        # Evaluate completeness
        with self.latency_tracker.track("completeness"):
            completeness_metrics = (
                self.relevance_evaluator.evaluate_completeness(
                    ai_response, user_query, context_texts
                )
            )

        # Detect hallucinations
        with self.latency_tracker.track("hallucination"):
            hallucination_metrics = (
                self.hallucination_detector.detect_hallucinations(
                    ai_response, context_texts
                )
            )

        # Evaluate factual accuracy
        with self.latency_tracker.track("factual_accuracy"):
            factual_metrics = (
                self.factual_evaluator.evaluate_factual_accuracy(
                    ai_response, context_texts
                )
            )

        # Create evaluation metrics
        metrics = EvaluationMetrics(
            relevance=relevance_metrics,
            completeness=completeness_metrics,
            hallucination=hallucination_metrics,
            factual_accuracy=factual_metrics,
        )

        # Calculate costs
        cost_breakdown = self.cost_estimator.estimate_evaluation_cost(
            ai_response, context_texts, user_query
        )

        # Create performance metrics
        performance = PerformanceMetrics(
            total_latency_ms=round(
                self.latency_tracker.get_total_latency(), 1
            ),
            breakdown=self.latency_tracker.get_breakdown(),
            estimated_cost_usd=cost_breakdown["total_cost_usd"],
            evaluation_mode=self.settings.evaluation_mode.mode,
            tokens_processed=cost_breakdown["total_tokens"],
        )

        # Compute overall score
        overall_score = self._compute_overall_score(metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            metrics, performance
        )

        return EvaluationResult(
            conversation_id=conversation.chat_id,
            turn_evaluated=turn_number,
            metrics=metrics,
            performance=performance,
            overall_score=overall_score,
            recommendations=recommendations,
        )
