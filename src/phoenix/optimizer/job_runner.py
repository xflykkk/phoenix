from __future__ import annotations

from typing import Any, Callable
import importlib.util

# Ensure local opik_optimizer (vendored in repo) is importable even if not installed.
import sys
from pathlib import Path
import importlib

spec = importlib.util.find_spec("opik_optimizer")
if spec is None or spec.origin is None:  # pragma: no cover - path setup
    repo_root = Path(__file__).resolve().parents[3]
    opik_src = repo_root / "opik_optimizer" / "src"
    sys.path.insert(0, str(opik_src))

from opik_optimizer import (
    ChatPrompt,
    EvolutionaryOptimizer,
    FewShotBayesianOptimizer,
    GepaOptimizer,
    HierarchicalReflectiveOptimizer,
    MetaPromptOptimizer,
    OptimizationResult as OpikOptimizationResult,
    ParameterOptimizer,
    ParameterSearchSpace,
)

from .dataset_adapter import OptimizerDataset
from .metrics import build_metric, llm_judge_metric_factory
from .models import (
    AlgorithmName,
    MetricConfig,
    OptimizationRequest,
    OptimizationResult,
)


_ALGORITHM_REGISTRY: dict[AlgorithmName, type] = {
    "metaprompt": MetaPromptOptimizer,
    "evolutionary": EvolutionaryOptimizer,
    "hierarchical": HierarchicalReflectiveOptimizer,
    "gepa": GepaOptimizer,
    "fewshot": FewShotBayesianOptimizer,
    "parameter": ParameterOptimizer,
}

_DEFAULT_TRIALS: dict[AlgorithmName, int] = {
    "metaprompt": 8,
    "evolutionary": 10,
    "hierarchical": 4,
    "gepa": 10,
    "fewshot": 10,
    "parameter": 20,
}

_REQUIRED_MODULES: dict[AlgorithmName, list[str]] = {
    "metaprompt": [],
    "hierarchical": [],
    "evolutionary": ["deap"],
    "gepa": ["gepa"],
    "fewshot": ["optuna"],
    "parameter": ["optuna"],
}


def _ensure_dependencies(algorithm: AlgorithmName) -> None:
    missing: list[str] = []
    for module_name in _REQUIRED_MODULES.get(algorithm, []):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    if missing:
        raise ImportError(
            f"Algorithm '{algorithm}' requires missing packages: {', '.join(missing)}"
        )


def _build_metric_callable(
    metric_cfg: MetricConfig, judge_fn: Callable[[dict[str, Any], Any], float] | None
) -> Callable[[dict[str, Any], Any], float]:
    if metric_cfg.name == "llm_judge":
        if judge_fn is None:
            raise ValueError("llm_judge metric requires llm_judge_fn")
        return llm_judge_metric_factory(judge_fn)
    return build_metric(metric_cfg)


def _to_chat_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    model: str,
    model_kwargs: dict[str, Any] | None,
) -> ChatPrompt:
    return ChatPrompt(
        messages=messages,
        tools=tools,
        model=model,
        model_parameters=model_kwargs or {},
    )


def _extract_result(opik_result: OpikOptimizationResult) -> OptimizationResult:
    return OptimizationResult(
        best_prompt_messages=opik_result.prompt,
        best_score=opik_result.score,
        trials_used=len(opik_result.history) if opik_result.history else 0,
        history=opik_result.history,
    )


def run_optimization_job(request: OptimizationRequest) -> OptimizationResult:
    """
    Run an optimization job using opik_optimizer algorithms.

    The caller is responsible for supplying a normalized dataset and metric
    configuration. Any LLM judge logic must be provided via `llm_judge_fn`.
    """
    if request.algorithm not in _ALGORITHM_REGISTRY:
        raise ValueError(f"Unsupported algorithm: {request.algorithm}")

    _ensure_dependencies(request.algorithm)
    optimizer_cls = _ALGORITHM_REGISTRY[request.algorithm]

    if request.algorithm == "parameter":
        optimizer = optimizer_cls(
            model=request.model,
            model_parameters=request.model_kwargs or {},
        )
    else:
        optimizer = optimizer_cls(
            model=request.model,
            model_parameters=request.model_kwargs or {},
        )

    dataset = OptimizerDataset(request.dataset)
    metric_callable = _build_metric_callable(request.metric, request.llm_judge_fn)
    chat_prompt = _to_chat_prompt(
        messages=request.prompt_messages,
        tools=request.tools,
        model=request.model,
        model_kwargs=request.model_kwargs,
    )

    max_trials = request.max_trials or _DEFAULT_TRIALS[request.algorithm]

    # Some optimizers accept max_trials; others ignore extra kwargs harmlessly.
    if request.algorithm == "fewshot":
        if request.min_examples is not None:
            optimizer.min_examples = request.min_examples  # type: ignore[attr-defined]
        if request.max_examples_fs is not None:
            optimizer.max_examples = request.max_examples_fs  # type: ignore[attr-defined]

    if request.algorithm == "parameter":
        if not request.parameter_space:
            raise ValueError("parameter_space is required for ParameterOptimizer")
        space = (
            request.parameter_space
            if isinstance(request.parameter_space, ParameterSearchSpace)
            else ParameterSearchSpace.model_validate(request.parameter_space)
        )
        opik_result = optimizer.optimize_parameter(  # type: ignore[assignment]
            prompt=chat_prompt,
            dataset=dataset,
            metric=metric_callable,
            parameter_space=space,
            n_samples=request.n_samples,
            max_trials=max_trials,
        )
    else:
        opik_result = optimizer.optimize_prompt(
            prompt=chat_prompt,
            dataset=dataset,
            metric=metric_callable,
            n_samples=request.n_samples,
            max_trials=max_trials,
        )

    return _extract_result(opik_result)
