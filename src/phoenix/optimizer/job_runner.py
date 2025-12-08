from __future__ import annotations

from typing import Any, Callable
import importlib.util

# Ensure local opik_optimizer (vendored in repo) is importable even if not installed.
import sys
from pathlib import Path
import importlib
import asyncio

from phoenix.server.api.types.ChatCompletionMessageRole import ChatCompletionMessageRole

class OptimizationCancelled(Exception):
    """Raised when an optimization was cancelled."""
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
    # Optionally patch LiteLLM to route through the Playground client chain so that
    # custom provider/base_url/credentials from the Playground UI are reused.
    litellm_completion_restore: Any = None
    litellm_acompletion_restore: Any = None
    if request.llm_client is not None:
        import litellm

        def _simple_response(text: str) -> Any:
            class _Message:
                def __init__(self, content: str) -> None:
                    self.content = content

            class _Choice:
                def __init__(self, content: str) -> None:
                    self.message = _Message(content)
                    self.finish_reason = None

            class _Response:
                def __init__(self, content: str) -> None:
                    self.choices = [_Choice(content)]
                    self.model = request.model

            return _Response(text)

        def _collect_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
            allowed = {
                "temperature",
                "max_tokens",
                "max_completion_tokens",
                "top_p",
                "presence_penalty",
                "frequency_penalty",
                "metadata",
            }
            return {k: v for k, v in kwargs.items() if k in allowed and v is not None}

        async def _run_client(
            messages: list[dict[str, Any]],
            extra_params: dict[str, Any],
        ) -> Any:
            if request.cancel_event is not None and request.cancel_event.is_set():
                raise OptimizationCancelled()
            merged_params = {**(request.invocation_parameters or {}), **extra_params}
            formatted_messages = []
            for m in messages:
                role_value = m.get("role") or "user"
                try:
                    role = ChatCompletionMessageRole(role_value)
                except Exception:
                    role = ChatCompletionMessageRole.user
                formatted_messages.append(
                    (
                        role,
                        m.get("content"),
                        m.get("tool_call_id"),
                        m.get("tool_calls"),
                    )
            )
            text_parts: list[str] = []
            async for chunk in request.llm_client.chat_completion_create(
                messages=formatted_messages,
                tools=request.tools or [],
                **merged_params,
            ):
                if request.cancel_event is not None and request.cancel_event.is_set():
                    raise OptimizationCancelled()
                content = getattr(chunk, "content", None)
                if content:
                    text_parts.append(str(content))
            return _simple_response("".join(text_parts))

        def _completion_proxy(*args: Any, **kwargs: Any) -> Any:
            messages = kwargs.get("messages") or (args[1] if len(args) > 1 else [])
            extra_params = _collect_kwargs(kwargs)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(_run_client(messages, extra_params))
            else:  # pragma: no cover - unexpected loop presence
                return loop.run_until_complete(_run_client(messages, extra_params))

        async def _acompletion_proxy(*args: Any, **kwargs: Any) -> Any:
            messages = kwargs.get("messages") or (args[1] if len(args) > 1 else [])
            extra_params = _collect_kwargs(kwargs)
            return await _run_client(messages, extra_params)

        litellm_completion_restore = getattr(litellm, "completion", None)
        litellm_acompletion_restore = getattr(litellm, "acompletion", None)
        litellm.completion = _completion_proxy  # type: ignore[assignment]
        litellm.acompletion = _acompletion_proxy  # type: ignore[assignment]

    try:
        if request.algorithm not in _ALGORITHM_REGISTRY:
            raise ValueError(f"Unsupported algorithm: {request.algorithm}")

        _ensure_dependencies(request.algorithm)
        if request.cancel_event is not None and request.cancel_event.is_set():
            raise OptimizationCancelled()
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
    finally:
        if request.llm_client is not None and litellm_completion_restore is not None:
            import litellm

            litellm.completion = litellm_completion_restore  # type: ignore[assignment]
            if litellm_acompletion_restore is not None:
                litellm.acompletion = litellm_acompletion_restore  # type: ignore[assignment]
