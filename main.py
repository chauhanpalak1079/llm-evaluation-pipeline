#!/usr/bin/env python3
"""
Main CLI entry point for LLM evaluation pipeline.

This script provides command-line interface to evaluate AI responses
against conversation history and context vectors.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from evaluator.models import ChatConversation, ContextVectors
from evaluator.pipeline import EvaluationPipeline


@click.command()
@click.option(
    "--conversation",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to conversation JSON file",
)
@click.option(
    "--context",
    "-x",
    type=click.Path(exists=True),
    required=True,
    help="Path to context vectors JSON file",
)
@click.option(
    "--turn",
    "-t",
    type=int,
    required=True,
    help="Turn number to evaluate",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file path (JSON). If not provided, prints to stdout",
)
@click.option(
    "--pretty",
    "-p",
    is_flag=True,
    default=False,
    help="Pretty-print JSON output",
)
def evaluate(
    conversation: str,
    context: str,
    turn: int,
    output: Optional[str],
    pretty: bool,
) -> None:
    """
    Evaluate AI response for a specific conversation turn.

    This command evaluates the AI response at the specified turn using
    the provided conversation history and context vectors. It produces
    a comprehensive evaluation report including relevance, hallucination
    detection, factual accuracy, and performance metrics.

    Example:
        python main.py -c sample_data/sample_chat_conversation_01.json
                       -x sample_data/sample_context_vectors_01.json
                       -t 14 -p
    """
    try:
        # Load conversation data
        click.echo(f"Loading conversation from: {conversation}")
        with open(conversation, "r") as f:
            conv_data = json.load(f)
        conv = ChatConversation(**conv_data)

        # Load context vectors
        click.echo(f"Loading context vectors from: {context}")
        with open(context, "r") as f:
            ctx_data = json.load(f)
        ctx = ContextVectors(**ctx_data)

        # Initialize pipeline
        click.echo("Initializing evaluation pipeline...")
        pipeline = EvaluationPipeline()

        # Run evaluation
        click.echo(f"Evaluating turn {turn}...")
        result = pipeline.evaluate_turn(conv, ctx, turn)

        # Convert to JSON
        result_dict = result.model_dump()

        # Output results
        if pretty:
            result_json = json.dumps(result_dict, indent=2)
        else:
            result_json = json.dumps(result_dict)

        if output:
            click.echo(f"Writing results to: {output}")
            with open(output, "w") as f:
                f.write(result_json)
            click.echo("Evaluation complete!")
        else:
            click.echo("\n" + "=" * 60)
            click.echo("EVALUATION RESULTS")
            click.echo("=" * 60)
            click.echo(result_json)

        # Print summary
        click.echo("\n" + "=" * 60)
        click.echo("SUMMARY")
        click.echo("=" * 60)
        click.echo(f"Overall Score: {result.overall_score}")
        click.echo(
            f"Hallucinations Detected: "
            f"{len(result.metrics.hallucination.detected_hallucinations)}"
        )
        click.echo(
            f"Total Latency: {result.performance.total_latency_ms:.1f}ms"
        )
        click.echo(
            f"Estimated Cost: ${result.performance.estimated_cost_usd:.6f}"
        )

        if result.metrics.hallucination.detected_hallucinations:
            click.echo("\nWARNING: Hallucinations detected:")
            for h in result.metrics.hallucination.detected_hallucinations:
                click.echo(f"  - [{h.severity.upper()}] {h.text[:80]}...")

    except FileNotFoundError as e:
        click.echo(f"Error: File not found - {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: Unexpected error - {e}", err=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    evaluate()
