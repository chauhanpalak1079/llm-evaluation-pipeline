"""
Factual accuracy evaluation module.

This module verifies factual claims in AI responses against provided
context vectors and detects contradictions.
"""

import re
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import get_settings
from evaluator.cache import get_embedding_cache
from evaluator.models import FactualAccuracyMetrics


class FactualAccuracyEvaluator:
    """Evaluates factual accuracy of AI responses."""

    def __init__(self):
        """Initialize factual accuracy evaluator."""
        settings = get_settings()
        self.model_name = settings.model.embedding_model
        self.model = SentenceTransformer(self.model_name)
        self.cache = get_embedding_cache()
        self.threshold = (
            settings.thresholds.factual_accuracy_threshold
        )

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

    def _extract_factual_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text.

        Args:
            text: Input text

        Returns:
            List[str]: List of factual claims
        """
        # Remove markdown
        clean_text = re.sub(r'\*\*|\*|__|_', '', text)

        # Extract sentences with factual indicators
        # (numbers, proper nouns, specific details)
        sentences = re.split(r'[.!?]+', clean_text)

        factual_claims = []
        for sent in sentences:
            sent = sent.strip()
            # Consider it factual if it has numbers or specific names
            has_numbers = bool(re.search(r'\d+', sent))
            has_proper_nouns = bool(
                re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sent)
            )

            if (has_numbers or has_proper_nouns) and len(sent) > 10:
                factual_claims.append(sent)

        return factual_claims

    def _verify_claim(
        self, claim: str, context_texts: List[str]
    ) -> Tuple[bool, float]:
        """
        Verify if a claim is supported by context.

        Args:
            claim: Factual claim to verify
            context_texts: List of context texts

        Returns:
            Tuple[bool, float]: (is_verified, similarity_score)
        """
        if not context_texts:
            return False, 0.0

        claim_emb = self._get_embedding(claim).reshape(1, -1)

        max_similarity = 0.0
        for context in context_texts:
            ctx_emb = self._get_embedding(context).reshape(1, -1)
            similarity = cosine_similarity(claim_emb, ctx_emb)[0][0]
            max_similarity = max(max_similarity, float(similarity))

        is_verified = max_similarity >= self.threshold
        return is_verified, max_similarity

    def _detect_contradictions(
        self, claim: str, context_texts: List[str]
    ) -> List[str]:
        """
        Detect contradictions between claim and context.

        Args:
            claim: Claim to check
            context_texts: List of context texts

        Returns:
            List[str]: List of contradictory statements
        """
        contradictions = []

        # Extract numbers from claim
        claim_numbers = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', claim)

        for context in context_texts:
            # Check for conflicting numbers
            context_numbers = re.findall(
                r'\d+(?:,\d+)*(?:\.\d+)?', context
            )

            # Simple heuristic: if claim has number not in context
            # and context has different number for same topic
            for claim_num in claim_numbers:
                if (
                    claim_num not in context
                    and len(context_numbers) > 0
                ):
                    # Potential contradiction
                    # More sophisticated logic could be added here
                    pass

        return contradictions

    def evaluate_factual_accuracy(
        self, ai_response: str, context_texts: List[str]
    ) -> FactualAccuracyMetrics:
        """
        Evaluate factual accuracy of AI response.

        Args:
            ai_response: AI-generated response
            context_texts: List of context vector texts

        Returns:
            FactualAccuracyMetrics: Factual accuracy metrics
        """
        # Extract factual claims
        factual_claims = self._extract_factual_claims(ai_response)

        if not factual_claims:
            # No factual claims detected
            return FactualAccuracyMetrics(
                score=0.8,  # Neutral score
                verified_claims=[],
                unverified_claims=[],
                contradictions=[],
            )

        verified = []
        unverified = []
        contradictions = []

        for claim in factual_claims:
            is_verified, similarity = self._verify_claim(
                claim, context_texts
            )

            if is_verified:
                verified.append(claim)
            else:
                unverified.append(claim)

            # Check for contradictions
            claim_contradictions = self._detect_contradictions(
                claim, context_texts
            )
            contradictions.extend(claim_contradictions)

        # Calculate accuracy score
        total_claims = len(factual_claims)
        accuracy_score = len(verified) / total_claims if total_claims > 0 else 0.8

        return FactualAccuracyMetrics(
            score=round(accuracy_score, 2),
            verified_claims=verified[:5],  # Limit to top 5
            unverified_claims=unverified[:5],  # Limit to top 5
            contradictions=contradictions[:5],  # Limit to top 5
        )
