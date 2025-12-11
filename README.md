# LLM Evaluation Pipeline for AI Response Reliability Testing

A production-ready Python pipeline for evaluating AI chatbot responses in 
real-time, focusing on relevance, hallucination detection, and factual 
accuracy. Designed to handle millions of daily conversations with minimal 
latency and cost.

## Features

✅ **Three Core Evaluation Parameters:**
- **Relevance & Completeness**: Semantic similarity, query coverage, 
  context utilization
- **Hallucination Detection**: Identifies claims not grounded in context 
  vectors
- **Factual Accuracy**: Verifies claims against source documents

✅ **Production-Ready:**
- Strict PEP-8 compliance throughout
- Comprehensive type hints and docstrings
- Modular, extensible architecture
- Full test coverage

✅ **Optimized for Scale:**
- Lightweight local models (sentence-transformers)
- Multi-level caching (LRU, embedding cache)
- Batch processing capabilities
- Configurable evaluation modes (fast/standard/comprehensive)

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/chauhanpalak1079/llm-evaluation-pipeline.git
cd llm-evaluation-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required models (first run will auto-download)
python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"
```

### Basic Usage

```bash
# Evaluate a conversation turn
python main.py \
  -c sample_data/sample_chat_conversation_01.json \
  -x sample_data/sample_context_vectors_01.json \
  -t 14 \
  -p

# Save results to file
python main.py \
  -c sample_data/sample_chat_conversation_01.json \
  -x sample_data/sample_context_vectors_01.json \
  -t 14 \
  -o results.json \
  -p
```

### Python API

```python
from evaluator.pipeline import EvaluationPipeline
from evaluator.models import ChatConversation, ContextVectors
import json

# Load data
with open('sample_data/sample_chat_conversation_01.json') as f:
    conversation = ChatConversation(**json.load(f))

with open('sample_data/sample_context_vectors_01.json') as f:
    context = ContextVectors(**json.load(f))

# Initialize pipeline
pipeline = EvaluationPipeline()

# Evaluate turn
result = pipeline.evaluate_turn(conversation, context, turn_number=14)

# Access results
print(f"Overall Score: {result.overall_score}")
print(f"Hallucinations: {len(result.metrics.hallucination.detected_hallucinations)}")
print(f"Latency: {result.performance.total_latency_ms:.1f}ms")

# Serialize to JSON
print(result.model_dump_json(indent=2))
```

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                   Evaluation Pipeline                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Input: Conversation + Context Vectors                │  │
│  └───────────────┬───────────────────────────────────────┘  │
│                  │                                           │
│  ┌───────────────▼───────────────────────────────────────┐  │
│  │         Parallel Evaluation Components                │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │  │
│  │  │Relevance │  │Hallucin. │  │Factual Accuracy  │    │  │
│  │  │Evaluator │  │Detector  │  │   Evaluator      │    │  │
│  │  └─────┬────┘  └────┬─────┘  └────────┬─────────┘    │  │
│  └────────┼────────────┼─────────────────┼──────────────┘  │
│           │            │                  │                 │
│  ┌────────▼────────────▼──────────────────▼──────────────┐  │
│  │         Embedding Cache (LRU)                         │  │
│  │         sentence-transformers/all-MiniLM-L6-v2       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Output: Structured JSON with scores, hallucinations, │  │
│  │          performance metrics, and recommendations      │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Component Details

1. **RelevanceEvaluator** (`evaluator/relevance.py`)
   - Computes semantic similarity between response and query
   - Measures context utilization and query coverage
   - Extracts and analyzes topics

2. **HallucinationDetector** (`evaluator/hallucination.py`)
   - Splits response into individual claims
   - Checks each claim against context vectors
   - Assigns severity levels (low/medium/high)
   - **Critical**: Detects fabricated information (e.g., pricing, 
     features not in context)

3. **FactualAccuracyEvaluator** (`evaluator/factual_accuracy.py`)
   - Extracts factual claims (numbers, proper nouns)
   - Verifies claims against context
   - Detects contradictions

4. **Caching Layer** (`evaluator/cache.py`)
   - LRU cache for embeddings (10,000 entries default)
   - Result cache for repeated evaluations
   - MD5 hashing for cache keys

5. **LatencyTracker** (`evaluator/latency_tracker.py`)
   - Tracks execution time per component
   - Provides detailed breakdown
   - Context manager for easy timing

6. **CostEstimator** (`evaluator/cost_estimator.py`)
   - Estimates token usage
   - Calculates costs for different providers
   - Provides optimization recommendations

## Design Decisions

### 1. Local Models over API Calls

**Decision**: Use `sentence-transformers` (all-MiniLM-L6-v2) for 
embeddings instead of OpenAI/Anthropic APIs.

**Rationale**:
- **Cost**: Local inference is free vs. $0.0001-0.0003 per 1K tokens
- **Latency**: 50-100ms locally vs. 200-500ms API calls
- **Scale**: Supports millions of daily evaluations
- **Privacy**: No data sent to third parties

**Trade-offs**: Slightly lower quality than GPT-4 embeddings, but 
sufficient for semantic similarity tasks.

### 2. Claim-Level Hallucination Detection

**Decision**: Split responses into individual claims and verify each 
against context.

**Rationale**:
- **Granularity**: Pinpoints exact hallucinated statements
- **Actionable**: Developers can see specific problematic claims
- **Flexible**: Allows different severity levels

**Alternative Considered**: Whole-response similarity scoring. Rejected 
because it can't identify specific hallucinations.

### 3. Multi-Level Caching

**Decision**: Implement both embedding cache and result cache.

**Rationale**:
- **Embedding Cache**: Same texts often appear in multiple contexts
- **Result Cache**: Identical evaluations may be requested
- **Performance**: Reduces latency from ~250ms to ~50ms for cached items

### 4. Modular Architecture

**Decision**: Separate evaluators for each metric type.

**Rationale**:
- **Extensibility**: Easy to add new evaluation types
- **Testing**: Each component can be tested independently
- **Flexibility**: Can disable/enable specific evaluators
- **Maintenance**: Clear separation of concerns

## Scalability Strategy

### Handling Millions of Daily Conversations

Our architecture supports high-volume production use through:

#### 1. Lightweight Local Models
- **all-MiniLM-L6-v2**: 80MB model, 50-100ms inference
- **No API dependencies**: Eliminates network latency
- **CPU-optimized**: No GPU required

#### 2. Aggressive Caching
```python
# Embedding cache: 10,000 entries
# Hit rate: ~70-80% in production (reused contexts)
# Latency reduction: 250ms → 50ms (5x faster)

# Example cache stats after 1M evaluations:
# - Hits: 750,000 (75% hit rate)
# - Misses: 250,000
# - Time saved: 150,000 seconds = 41.6 hours
```

#### 3. Batch Processing
```python
# Process multiple evaluations in parallel
# Recommended: 10-50 concurrent evaluations
# Total throughput: ~20-100 evaluations/second per core
```

#### 4. Tiered Evaluation Modes
- **Fast Mode** (100-150ms): Basic relevance + rule-based checks
- **Standard Mode** (200-300ms): Full evaluation (default)
- **Comprehensive Mode** (500-800ms): Deep analysis with 
  cross-validation

#### 5. Memory Optimization
```python
# Use generators for large batches
# Lazy loading of conversation history
# Periodic cache cleanup
```

### Cost Analysis (1M Daily Evaluations)

**Our Approach** (Local Models):
- Infrastructure: $50-100/month (cloud VM)
- Storage: $10/month (caching, logs)
- **Total: ~$100/month**

**Alternative** (API-Based):
- OpenAI embeddings: $0.0001/1K tokens × 1.5K avg × 1M = $150/day
- GPT-4 verification: $0.03/1K tokens × 500 avg × 1M = $15,000/day
- **Total: ~$450,000/month** ❌

**Savings: 99.98% cost reduction**

## Testing

### Run All Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=evaluator --cov-report=html

# Run specific test file
pytest tests/test_hallucination.py -v

# Run critical hallucination test
pytest tests/test_pipeline.py::TestEvaluationPipeline::test_evaluate_sample_01_with_hallucination -v
```

### Test Coverage

- ✅ Unit tests for each evaluator module
- ✅ Integration test with sample data
- ✅ **Critical test**: Verifies hallucination detection for known case
- ✅ Performance benchmarks

### Critical Test Case

The pipeline **MUST** detect the hallucination in sample conversation 
01, turn 14:

**Hallucinated Claim**: 
> "We also offer specially subsidized air-conditioned rooms at our 
> clinic for Rs 2000 (US $50) per night which include breakfast and 
> medical monitoring."

**Reality**: Context vectors only mention:
- Gopal Mansion: Rs 800/night
- Hotel Sagar: Rs 600/night  
- No clinic accommodation mentioned

**Test Verification**:
```bash
pytest tests/test_pipeline.py::TestEvaluationPipeline::test_evaluate_sample_01_with_hallucination -v
```

## Code Quality

### PEP-8 Compliance

All code strictly follows PEP-8:
- ✅ `snake_case` for functions and variables
- ✅ `PascalCase` for classes
- ✅ 79-character line limit (99 for flexibility)
- ✅ Google-style docstrings
- ✅ Type hints throughout
- ✅ Proper import organization

### Validation

```bash
# Check PEP-8 compliance
flake8 evaluator/ config/ tests/ main.py --max-line-length=99

# Type checking
mypy evaluator/ config/ main.py

# Auto-formatting
black evaluator/ config/ tests/ main.py --line-length=79
```

## Sample Data

The `sample_data/` directory contains:

1. **sample_chat_conversation_01.json**: Contains known hallucination 
   (turn 14)
2. **sample_chat_conversation_02.json**: Clean response example
3. **sample_context_vectors_01.json**: Context for conversation 01
4. **sample_context_vectors_02.json**: Context for conversation 02

## Output Format

```json
{
  "evaluation_id": "uuid-string",
  "timestamp": "2025-12-11T10:30:00Z",
  "conversation_id": 78128,
  "turn_evaluated": 14,
  "metrics": {
    "relevance": {
      "score": 0.85,
      "query_coverage": 0.90,
      "context_utilization": 0.75,
      "semantic_similarity": 0.88
    },
    "completeness": {
      "score": 0.80,
      "missing_aspects": [],
      "covered_topics": ["Gopal Mansion", "800"]
    },
    "hallucination": {
      "score": 0.60,
      "detected_hallucinations": [
        {
          "text": "We also offer specially subsidized rooms...",
          "reason": "Claim not found in any provided context vector",
          "severity": "high",
          "confidence": 0.92
        }
      ],
      "grounded_claims_ratio": 0.75,
      "total_claims": 4,
      "grounded_claims": 3
    },
    "factual_accuracy": {
      "score": 0.70,
      "verified_claims": [],
      "unverified_claims": []
    }
  },
  "performance": {
    "total_latency_ms": 245.5,
    "breakdown": {
      "relevance_ms": 85.2,
      "hallucination_ms": 120.3,
      "factual_accuracy_ms": 40.0
    },
    "estimated_cost_usd": 0.0001,
    "tokens_processed": 1250
  },
  "overall_score": 0.72,
  "recommendations": [
    "High-severity hallucination detected - review AI response generation"
  ]
}
```

## Configuration

Customize evaluation behavior in `config/settings.py`:

```python
from config.settings import get_settings, update_settings

# Get current settings
settings = get_settings()

# Update thresholds
update_settings(
    thresholds={
        "hallucination_threshold": 0.85,  # Stricter
        "relevance_threshold": 0.6
    }
)

# Change evaluation mode
update_settings(
    evaluation_mode={"mode": "fast"}
)
```

## Contributing

1. Follow PEP-8 strictly
2. Add type hints to all functions
3. Write tests for new features
4. Update documentation

## License

MIT License - see LICENSE file for details

## Support

For issues or questions:
- GitHub Issues: https://github.com/chauhanpalak1079/llm-evaluation-pipeline/issues
- Email: support@example.com

## Acknowledgments

- Built with [sentence-transformers](https://www.sbert.net/)
- Inspired by production LLM evaluation needs
- Designed for scale and reliability
