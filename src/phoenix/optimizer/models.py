from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


# Supported algorithms for the first integration slice.
AlgorithmName = Literal[
    "metaprompt",
    "evolutionary",
    "hierarchical",
    "gepa",
]

# Built-in metric choices; LLM judge requires an external prompt/model.
MetricName = Literal["exact_match", "contains", "llm_judge"]


@dataclass
class MetricConfig:
    """User-selected metric configuration."""

    name: MetricName
    contains_keywords: list[str] | None = None
    llm_judge_prompt: str | None = None
    llm_judge_model: str | None = None


@dataclass
class DatasetExample:
    """Normalized dataset item pulled from Playground dataset service."""

    id: str
    input: dict[str, Any]
    reference_output: Any | None = None


@dataclass
class PlaygroundDataset:
    """Minimal dataset wrapper needed by opik_optimizer."""

    id: str
    name: str
    examples: list[DatasetExample]


@dataclass
class OptimizationRequest:
    """
    Payload the backend can build from the Playground UI.

    Attributes:
        algorithm: Which optimizer to run.
        prompt_messages: The current Playground prompt (seed) as a list of
            OpenAI-style message dicts.
        tools: Optional tool definitions in OpenAI/LiteLLM format.
        model: Model name to use inside the optimizer for generation.
        model_kwargs: Extra parameters for the optimizer's internal LLM calls.
        dataset: Dataset items used for evaluation.
        metric: Metric configuration.
        max_trials: Trial budget for the optimizer (meaning depends on algo).
        n_samples: Optional subsampling of dataset items for faster runs.
    """

    algorithm: AlgorithmName
    prompt_messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    model: str
    model_kwargs: dict[str, Any] | None
    dataset: PlaygroundDataset
    metric: MetricConfig
    max_trials: int | None = None
    n_samples: int | None = None
    llm_judge_fn: Callable[[dict[str, Any], Any], float] | None = None
    parameter_space: Any | None = None
    # Few-shot specific
    min_examples: int | None = None
    max_examples_fs: int | None = None


@dataclass
class OptimizationResult:
    """Surface a minimal, Phoenix-friendly optimization result."""

    best_prompt_messages: list[dict[str, Any]]
    best_score: float
    trials_used: int
    history: list[dict[str, Any]] = field(default_factory=list)
