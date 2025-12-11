"""
Latency tracking utilities for LLM evaluation pipeline.

This module provides functionality to track and measure execution time
for different components of the evaluation pipeline.
"""

import time
from contextlib import contextmanager
from typing import Dict, Generator, Optional


class LatencyTracker:
    """Tracks execution time for evaluation components."""

    def __init__(self):
        """Initialize latency tracker."""
        self._timings: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}

    @contextmanager
    def track(self, component_name: str) -> Generator[None, None, None]:
        """
        Context manager to track execution time for a component.

        Args:
            component_name: Name of component being tracked

        Yields:
            None

        Example:
            >>> tracker = LatencyTracker()
            >>> with tracker.track("embedding"):
            ...     # code to track
            ...     pass
        """
        start_time = time.time()
        self._start_times[component_name] = start_time
        try:
            yield
        finally:
            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000
            self._timings[component_name] = elapsed_ms
            if component_name in self._start_times:
                del self._start_times[component_name]

    def start(self, component_name: str) -> None:
        """
        Start tracking time for a component.

        Args:
            component_name: Name of component to track
        """
        self._start_times[component_name] = time.time()

    def stop(self, component_name: str) -> float:
        """
        Stop tracking and record elapsed time for a component.

        Args:
            component_name: Name of component to stop tracking

        Returns:
            float: Elapsed time in milliseconds

        Raises:
            KeyError: If component was not started
        """
        if component_name not in self._start_times:
            raise KeyError(
                f"Component '{component_name}' was not started"
            )

        start_time = self._start_times.pop(component_name)
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        self._timings[component_name] = elapsed_ms
        return elapsed_ms

    def get_timing(self, component_name: str) -> Optional[float]:
        """
        Get recorded timing for a component.

        Args:
            component_name: Name of component

        Returns:
            Optional[float]: Elapsed time in ms, or None if not recorded
        """
        return self._timings.get(component_name)

    def get_all_timings(self) -> Dict[str, float]:
        """
        Get all recorded timings.

        Returns:
            Dict[str, float]: Dictionary of component timings in ms
        """
        return self._timings.copy()

    def get_total_latency(self) -> float:
        """
        Get total latency across all tracked components.

        Returns:
            float: Total latency in milliseconds
        """
        return sum(self._timings.values())

    def get_breakdown(self) -> Dict[str, float]:
        """
        Get latency breakdown with component-specific keys.

        Returns:
            Dict[str, float]: Breakdown with '_ms' suffix keys
        """
        return {
            f"{name}_ms": timing
            for name, timing in self._timings.items()
        }

    def reset(self) -> None:
        """Reset all tracked timings."""
        self._timings.clear()
        self._start_times.clear()

    def format_summary(self) -> str:
        """
        Format timing summary as human-readable string.

        Returns:
            str: Formatted timing summary
        """
        if not self._timings:
            return "No timings recorded"

        lines = ["Latency Breakdown:"]
        for component, timing_ms in sorted(self._timings.items()):
            lines.append(f"  {component}: {timing_ms:.2f}ms")
        lines.append(f"Total: {self.get_total_latency():.2f}ms")

        return "\n".join(lines)
