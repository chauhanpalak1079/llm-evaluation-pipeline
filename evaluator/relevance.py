"""
Relevance and completeness evaluation module.

This module evaluates semantic similarity between AI responses and user
queries, and assesses coverage of relevant context vectors.
"""

import re
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import get_settings
from evaluator.cache import get_embedding_cache
from evaluator.models import (
    CompletenessMetrics,
    RelevanceMetrics,
)


class RelevanceEvaluator:
    """Evaluates relevance and completeness of AI responses."""

    def __init__(self):
        """Initialize relevance evaluator with embedding model."""
        settings = get_settings()
        self.model_name = settings.model.embedding_model
        self.model = SentenceTransformer(self.model_name)
        self.cache = get_embedding_cache()

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text with caching.

        Args:
            text: Input text

        Returns:
            np.ndarray: Text embedding vector
        """
        # Check cache first
        cached = self.cache.get(text, self.model_name)
        if cached is not None:
            return cached

        # Compute and cache embedding
        embedding = self.model.encode(text, show_progress_bar=False)
        self.cache.set(text, self.model_name, embedding)
        return embedding

    def _compute_similarity(
        self, text1: str, text2: str
    ) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            float: Cosine similarity score (0-1)
        """
        emb1 = self._get_embedding(text1).reshape(1, -1)
        emb2 = self._get_embedding(text2).reshape(1, -1)
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)

    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract key topics from text using simple heuristics.

        Args:
            text: Input text

        Returns:
            List[str]: List of extracted topics
        """
        # Simple topic extraction: capitalized words and key phrases
        # Remove markdown formatting
        clean_text = re.sub(r'\*\*|\*|__|_', '', text)

        # Extract capitalized words (potential topics)
        topics = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
                            clean_text)

        # Extract numbers with context (prices, dates, etc.)
        numeric_topics = re.findall(
            r'\b(?:Rs|USD?|\$)?\s*\d+(?:,\d+)*(?:\.\d+)?\b',
            text
        )

        return list(set(topics + numeric_topics))

    def evaluate_relevance(
        self,
        ai_response: str,
        user_query: str,
        context_texts: List[str],
    ) -> RelevanceMetrics:
        """
        Evaluate relevance of AI response to user query.

        Args:
            ai_response: AI-generated response
            user_query: User's original query
            context_texts: List of context vector texts

        Returns:
            RelevanceMetrics: Relevance evaluation metrics
        """
        # Compute semantic similarity between response and query
        semantic_similarity = self._compute_similarity(
            ai_response, user_query
        ) if user_query else 0.5

        # Compute context utilization
        context_utilization = self._compute_context_utilization(
            ai_response, context_texts
        )

        # Query coverage (how well response addresses query)
        query_coverage = min(
            1.0, semantic_similarity + 0.1
        )  # Slight boost

        # Overall relevance score
        relevance_score = (
            semantic_similarity * 0.4
            + context_utilization * 0.4
            + query_coverage * 0.2
        )

        return RelevanceMetrics(
            score=round(relevance_score, 2),
            query_coverage=round(query_coverage, 2),
            context_utilization=round(context_utilization, 2),
            semantic_similarity=round(semantic_similarity, 2),
            details={
                "model_used": self.model_name,
                "num_context_vectors": len(context_texts),
            },
        )

    def _compute_context_utilization(
        self, ai_response: str, context_texts: List[str]
    ) -> float:
        """
        Compute how well response utilizes provided context.

        Args:
            ai_response: AI response text
            context_texts: List of context texts

        Returns:
            float: Context utilization score (0-1)
        """
        if not context_texts:
            return 0.0

        # Compute similarity with each context
        similarities = [
            self._compute_similarity(ai_response, ctx)
            for ctx in context_texts
        ]

        # Return max similarity (best context match)
        return float(max(similarities))

    def evaluate_completeness(
        self,
        ai_response: str,
        user_query: str,
        context_texts: List[str],
    ) -> CompletenessMetrics:
        """
        Evaluate completeness of AI response.

        Args:
            ai_response: AI-generated response
            user_query: User's original query
            context_texts: List of context vector texts

        Returns:
            CompletenessMetrics: Completeness evaluation metrics
        """
        # Extract topics from response and query
        response_topics = self._extract_topics(ai_response)
        query_topics = self._extract_topics(
            user_query
        ) if user_query else []

        # Determine covered vs missing topics
        covered_topics = [
            topic for topic in query_topics
            if any(
                topic.lower() in ai_response.lower()
                for _ in [topic]
            )
        ]

        missing_topics = [
            topic for topic in query_topics
            if topic not in covered_topics
        ]

        # Calculate completeness score
        if query_topics:
            completeness_score = len(covered_topics) / len(query_topics)
        else:
            # If no query topics, base on response length/substance
            completeness_score = min(1.0, len(ai_response) / 500)

        return CompletenessMetrics(
            score=round(completeness_score, 2),
            missing_aspects=missing_topics[:5],  # Limit to top 5
            covered_topics=response_topics[:10],  # Limit to top 10
            details={
                "total_response_topics": len(response_topics),
                "total_query_topics": len(query_topics),
            },
        )
