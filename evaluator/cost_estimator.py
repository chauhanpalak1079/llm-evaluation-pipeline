"""
Cost estimation utilities for LLM evaluation pipeline.

This module provides functionality to estimate token usage and associated
costs for different LLM providers and models.
"""

import re
from typing import Dict, List

from config.settings import get_settings


class CostEstimator:
    """Estimates token usage and costs for LLM operations."""

    def __init__(self):
        """Initialize cost estimator with settings."""
        self.settings = get_settings()

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text using simple heuristic.

        This uses a simplified approximation: ~4 characters per token
        for English text, which is reasonably accurate for estimation.

        Args:
            text: Input text

        Returns:
            int: Estimated token count
        """
        # Simple approximation: 4 chars per token on average
        return max(1, len(text) // 4)

    def count_tokens_batch(self, texts: List[str]) -> int:
        """
        Count tokens for multiple texts.

        Args:
            texts: List of text strings

        Returns:
            int: Total token count
        """
        return sum(self.count_tokens(text) for text in texts)

    def estimate_embedding_cost(
        self, text_list: List[str]
    ) -> Dict[str, float]:
        """
        Estimate cost for embedding generation.

        Args:
            text_list: List of texts to embed

        Returns:
            Dict[str, float]: Cost breakdown
        """
        total_tokens = self.count_tokens_batch(text_list)
        cost_per_token = self.settings.cost_estimation.token_costs[
            "embedding"
        ]
        total_cost = total_tokens * cost_per_token

        return {
            "total_tokens": total_tokens,
            "cost_per_token": cost_per_token,
            "total_cost_usd": total_cost,
        }

    def estimate_evaluation_cost(
        self,
        ai_response: str,
        context_texts: List[str],
        user_query: str = "",
    ) -> Dict[str, float]:
        """
        Estimate total cost for evaluating a response.

        Args:
            ai_response: AI response text
            context_texts: List of context vector texts
            user_query: User query text (optional)

        Returns:
            Dict[str, float]: Detailed cost breakdown
        """
        # Count tokens for each component
        response_tokens = self.count_tokens(ai_response)
        context_tokens = self.count_tokens_batch(context_texts)
        query_tokens = self.count_tokens(user_query) if user_query else 0

        # Calculate embedding costs (response + contexts + query)
        embedding_tokens = response_tokens + context_tokens + query_tokens
        embedding_cost = (
            embedding_tokens
            * self.settings.cost_estimation.token_costs["embedding"]
        )

        total_tokens = embedding_tokens
        total_cost = embedding_cost

        return {
            "response_tokens": response_tokens,
            "context_tokens": context_tokens,
            "query_tokens": query_tokens,
            "embedding_tokens": embedding_tokens,
            "total_tokens": total_tokens,
            "embedding_cost_usd": embedding_cost,
            "total_cost_usd": total_cost,
        }

    def get_optimization_recommendations(
        self, tokens_processed: int, latency_ms: float
    ) -> List[str]:
        """
        Generate cost optimization recommendations.

        Args:
            tokens_processed: Number of tokens processed
            latency_ms: Processing latency in milliseconds

        Returns:
            List[str]: Optimization recommendations
        """
        recommendations = []

        # Token-based recommendations
        if tokens_processed > 2000:
            recommendations.append(
                "Consider reducing context window size to decrease "
                "token usage"
            )

        # Latency-based recommendations
        if latency_ms > 500:
            recommendations.append(
                "High latency detected - consider enabling caching "
                "for repeated evaluations"
            )
            recommendations.append(
                "Consider using 'fast' evaluation mode for "
                "lower latency"
            )

        # General recommendations
        if len(recommendations) == 0:
            recommendations.append(
                "Performance is optimal for current settings"
            )

        return recommendations
