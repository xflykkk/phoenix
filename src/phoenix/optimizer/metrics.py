from __future__ import annotations

from typing import Any, Callable

from .models import MetricConfig, MetricName


def exact_match_metric(dataset_item: dict[str, Any], llm_output: Any) -> float:
    ref = dataset_item.get("_reference_output")
    if ref is None:
        return 0.0
    return 1.0 if str(llm_output).strip() == str(ref).strip() else 0.0


def contains_metric(
    dataset_item: dict[str, Any], llm_output: Any, keywords: list[str]
) -> float:
    text = str(llm_output).lower()
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw.lower() in text)
    return hits / len(keywords)


def llm_judge_metric_factory(
    judge_fn: Callable[[dict[str, Any], Any], float],
) -> Callable[[dict[str, Any], Any], float]:
    """
    Wrap a custom judge function (likely LLM-based) that returns a score in [0, 1].
    """

    def _metric(dataset_item: dict[str, Any], llm_output: Any) -> float:
        return judge_fn(dataset_item, llm_output)

    return _metric


def build_metric(metric: MetricConfig) -> Callable[[dict[str, Any], Any], float]:
    if metric.name == "exact_match":
        return exact_match_metric
    if metric.name == "contains":
        return lambda item, output: contains_metric(
            item, output, metric.contains_keywords or []
        )
    if metric.name == "llm_judge":
        raise ValueError(
            "llm_judge metric requires an injected judge_fn; "
            "use llm_judge_metric_factory and pass that to the optimizer."
        )
    raise ValueError(f"Unsupported metric: {metric.name}")
