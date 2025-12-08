"""
Lightweight optimizer integration layer for Phoenix playground.

This package wires existing `opik_optimizer` algorithms to Phoenix data/models.
It does not expose any API surface yet; see `job_runner.py` for the main entry
point that backend handlers can call.
"""

from .job_runner import run_optimization_job
from .models import (
    AlgorithmName,
    MetricName,
    MetricConfig,
    OptimizationRequest,
    OptimizationResult,
)
from .optimization_service import OptimizationJob, OptimizationService

__all__ = [
    "AlgorithmName",
    "MetricName",
    "MetricConfig",
    "OptimizationRequest",
    "OptimizationResult",
    "OptimizationJob",
    "OptimizationService",
    "run_optimization_job",
]
