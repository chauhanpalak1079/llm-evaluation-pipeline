"""
Configuration settings for LLM evaluation pipeline.

This module manages all configuration parameters including model settings,
thresholds, and evaluation modes.
"""

from typing import Dict, Any
from pydantic import BaseModel


class EvaluationThresholds(BaseModel):
    """Thresholds for evaluation metrics."""

    relevance_threshold: float = 0.7
    hallucination_threshold: float = 0.8
    factual_accuracy_threshold: float = 0.75
    semantic_similarity_threshold: float = 0.6


class ModelConfig(BaseModel):
    """Configuration for embedding and LLM models."""

    embedding_model: str = "all-MiniLM-L6-v2"
    max_sequence_length: int = 512
    batch_size: int = 32


class CacheConfig(BaseModel):
    """Configuration for caching mechanisms."""

    enable_cache: bool = True
    max_cache_size: int = 1000
    embedding_cache_size: int = 10000


class EvaluationModeConfig(BaseModel):
    """Configuration for different evaluation modes."""

    mode: str = "standard"  # fast, standard, comprehensive
    enable_async: bool = True
    parallel_processing: bool = True


class CostEstimationConfig(BaseModel):
    """Configuration for cost estimation."""

    token_costs: Dict[str, float] = {
        "embedding": 0.0000001,  # per token
        "gpt-3.5-turbo": 0.000002,  # per token
        "gpt-4": 0.00003,  # per token
    }


class Settings(BaseModel):
    """Main settings class for the evaluation pipeline."""

    thresholds: EvaluationThresholds = EvaluationThresholds()
    model: ModelConfig = ModelConfig()
    cache: CacheConfig = CacheConfig()
    evaluation_mode: EvaluationModeConfig = EvaluationModeConfig()
    cost_estimation: CostEstimationConfig = CostEstimationConfig()

    # Maximum line length for PEP-8 compliance
    max_line_length: int = 79


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get the global settings instance.

    Returns:
        Settings: Global settings configuration
    """
    return settings


def update_settings(**kwargs: Any) -> None:
    """
    Update global settings with provided keyword arguments.

    Args:
        **kwargs: Settings to update
    """
    global settings
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
