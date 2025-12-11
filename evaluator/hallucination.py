"""
Hallucination detection module.

This module identifies claims in AI responses that are not grounded in
the provided context vectors, detecting potential hallucinations.
"""

import re
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import get_settings
from evaluator.cache import get_embedding_cache
from evaluator.models import (
    HallucinationDetection,
    HallucinationMetrics,
)


class HallucinationDetector:
    """Detects hallucinations in AI responses."""

    def __init__(self):
        """Initialize hallucination detector."""
        settings = get_settings()
        self.model_name = settings.model.embedding_model
        self.model = SentenceTransformer(self.model_name)
        self.cache = get_embedding_cache()
        self.threshold = settings.thresholds.hallucination_threshold

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text with caching.

        Args:
            text: Input text

        Returns:
            np.ndarray: Text embedding vector
        """
        cached = self.cache.get(text, self.model_name)
        if cached is not None:
            return cached

        embedding = self.model.encode(text, show_progress_bar=False)
        self.cache.set(text, self.model_name, embedding)
        return embedding

    def _split_into_claims(self, text: str) -> List[str]:
        """
        Split text into individual claims/sentences.

        Args:
            text: Input text

        Returns:
            List[str]: List of individual claims
        """
        # Remove markdown formatting
        clean_text = re.sub(r'\*\*|\*|__|_', '', text)

        # Split by sentences
        sentences = re.split(r'[.!?]+', clean_text)

        # Filter out empty and very short sentences
        claims = [
            s.strip()
            for s in sentences
            if len(s.strip()) > 10
        ]

        return claims

    def _is_claim_grounded(
        self, claim: str, context_texts: List[str]
    ) -> Tuple[bool, float]:
        """
        Check if a claim is grounded in context.

        Args:
            claim: Individual claim text
            context_texts: List of context texts

        Returns:
            Tuple[bool, float]: (is_grounded, confidence_score)
        """
        if not context_texts:
            return False, 1.0

        claim_emb = self._get_embedding(claim).reshape(1, -1)

        # Check similarity with each context
        max_similarity = 0.0
        for context in context_texts:
            ctx_emb = self._get_embedding(context).reshape(1, -1)
            similarity = cosine_similarity(claim_emb, ctx_emb)[0][0]
            max_similarity = max(max_similarity, float(similarity))

        # Claim is grounded if similarity exceeds threshold
        is_grounded = max_similarity >= self.threshold
        confidence = 1.0 - max_similarity  # Higher conf = less similar

        return is_grounded, confidence

    def _determine_severity(
        self, claim: str, confidence: float
    ) -> str:
        """
        Determine severity of hallucination.

        Args:
            claim: Hallucinated claim
            confidence: Confidence score

        Returns:
            str: Severity level (low, medium, high)
        """
        # Check for financial/numeric claims (more severe)
        has_numbers = bool(re.search(r'\d+', claim))
        has_price = bool(
            re.search(r'Rs|USD?|\$|price|cost|fee', claim, re.I)
        )

        if (has_numbers and has_price) or confidence > 0.5:
            return "high"
        elif confidence > 0.3:
            return "medium"
        else:
            return "low"

    def detect_hallucinations(
        self, ai_response: str, context_texts: List[str]
    ) -> HallucinationMetrics:
        """
        Detect hallucinations in AI response.

        Args:
            ai_response: AI-generated response
            context_texts: List of context vector texts

        Returns:
            HallucinationMetrics: Hallucination detection metrics
        """
        # Split response into individual claims
        claims = self._split_into_claims(ai_response)

        if not claims:
            # No claims detected, consider it safe
            return HallucinationMetrics(
                score=1.0,
                detected_hallucinations=[],
                grounded_claims_ratio=1.0,
                total_claims=0,
                grounded_claims=0,
            )

        # Check each claim
        hallucinations = []
        grounded_count = 0

        for claim in claims:
            is_grounded, confidence = self._is_claim_grounded(
                claim, context_texts
            )

            if not is_grounded:
                severity = self._determine_severity(claim, confidence)
                hallucinations.append(
                    HallucinationDetection(
                        text=claim,
                        reason=(
                            "Claim not found in any provided "
                            "context vector"
                        ),
                        severity=severity,
                        confidence=round(confidence, 2),
                    )
                )
            else:
                grounded_count += 1

        # Calculate metrics
        total_claims = len(claims)
        grounded_ratio = grounded_count / total_claims
        # Score is the ratio of grounded claims
        hallucination_score = grounded_ratio

        return HallucinationMetrics(
            score=round(hallucination_score, 2),
            detected_hallucinations=hallucinations,
            grounded_claims_ratio=round(grounded_ratio, 2),
            total_claims=total_claims,
            grounded_claims=grounded_count,
        )
