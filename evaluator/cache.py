"""
Caching utilities for LLM evaluation pipeline.

This module provides LRU caching and embedding caching mechanisms to
optimize performance for repeated evaluations.
"""

import hashlib
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
from cachetools import LRUCache

from config.settings import get_settings


class EmbeddingCache:
    """Cache for storing text embeddings."""

    def __init__(self, max_size: Optional[int] = None):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum cache size. If None, uses config default.
        """
        settings = get_settings()
        size = max_size or settings.cache.embedding_cache_size
        self._cache: LRUCache = LRUCache(maxsize=size)
        self._hits = 0
        self._misses = 0

    def _generate_key(self, text: str, model_name: str) -> str:
        """
        Generate cache key from text and model name.

        Args:
            text: Input text
            model_name: Name of embedding model

        Returns:
            str: Cache key hash
        """
        content = f"{text}_{model_name}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(
        self, text: str, model_name: str
    ) -> Optional[np.ndarray]:
        """
        Retrieve embedding from cache.

        Args:
            text: Input text
            model_name: Name of embedding model

        Returns:
            Optional[np.ndarray]: Cached embedding or None if not found
        """
        key = self._generate_key(text, model_name)
        embedding = self._cache.get(key)
        if embedding is not None:
            self._hits += 1
        else:
            self._misses += 1
        return embedding

    def set(
        self, text: str, model_name: str, embedding: np.ndarray
    ) -> None:
        """
        Store embedding in cache.

        Args:
            text: Input text
            model_name: Name of embedding model
            embedding: Embedding vector to cache
        """
        key = self._generate_key(text, model_name)
        self._cache[key] = embedding

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dict[str, int]: Cache hits, misses, and size
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "max_size": self._cache.maxsize,
        }

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


class ResultCache:
    """Cache for storing evaluation results."""

    def __init__(self, max_size: Optional[int] = None):
        """
        Initialize result cache.

        Args:
            max_size: Maximum cache size. If None, uses config default.
        """
        settings = get_settings()
        size = max_size or settings.cache.max_cache_size
        self._cache: LRUCache = LRUCache(maxsize=size)

    def _generate_key(
        self, conversation_id: int, turn: int, context_hash: str
    ) -> str:
        """
        Generate cache key from conversation and context.

        Args:
            conversation_id: Conversation identifier
            turn: Turn number
            context_hash: Hash of context vectors

        Returns:
            str: Cache key
        """
        return f"{conversation_id}_{turn}_{context_hash}"

    def get(
        self, conversation_id: int, turn: int, context_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve evaluation result from cache.

        Args:
            conversation_id: Conversation identifier
            turn: Turn number
            context_hash: Hash of context vectors

        Returns:
            Optional[Dict[str, Any]]: Cached result or None
        """
        key = self._generate_key(conversation_id, turn, context_hash)
        return self._cache.get(key)

    def set(
        self,
        conversation_id: int,
        turn: int,
        context_hash: str,
        result: Dict[str, Any],
    ) -> None:
        """
        Store evaluation result in cache.

        Args:
            conversation_id: Conversation identifier
            turn: Turn number
            context_hash: Hash of context vectors
            result: Evaluation result to cache
        """
        key = self._generate_key(conversation_id, turn, context_hash)
        self._cache[key] = result

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()


# Global cache instances
_embedding_cache: Optional[EmbeddingCache] = None
_result_cache: Optional[ResultCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """
    Get global embedding cache instance.

    Returns:
        EmbeddingCache: Global embedding cache
    """
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache


def get_result_cache() -> ResultCache:
    """
    Get global result cache instance.

    Returns:
        ResultCache: Global result cache
    """
    global _result_cache
    if _result_cache is None:
        _result_cache = ResultCache()
    return _result_cache


@lru_cache(maxsize=1000)
def compute_hash(text: str) -> str:
    """
    Compute MD5 hash of text with LRU caching.

    Args:
        text: Input text

    Returns:
        str: MD5 hash
    """
    return hashlib.md5(text.encode()).hexdigest()
