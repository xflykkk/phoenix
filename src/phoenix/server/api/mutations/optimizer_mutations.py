from __future__ import annotations

from typing import Any, Optional

import strawberry
from strawberry.scalars import JSON as JSONScalarType
from strawberry.relay import GlobalID
from strawberry.types import Info

from phoenix.server.api.auth import IsNotReadOnly, IsNotViewer, IsLocked
from phoenix.server.api.context import Context
from phoenix.server.api.exceptions import BadRequest, NotFound
from phoenix.server.api.types.node import from_global_id_with_expected_type
from phoenix.db import models
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from phoenix.optimizer import MetricConfig, OptimizationRequest
from phoenix.optimizer.models import DatasetExample, PlaygroundDataset
from phoenix.optimizer.optimization_service import service as optimizer_service
import litellm


async def _resolve_dataset_examples(session: AsyncSession, dataset_db_id: int) -> PlaygroundDataset:
    """Load latest dataset version examples for the given dataset id."""

    # Resolve dataset name
    dataset_row = await session.execute(
        select(models.Dataset).where(models.Dataset.id == dataset_db_id)
    )
    dataset_obj = dataset_row.scalar_one_or_none()
    if dataset_obj is None:
        raise NotFound(f"Dataset {dataset_db_id} not found")

    latest_version_id = await session.scalar(
        select(func.max(models.DatasetVersion.id)).where(
            models.DatasetVersion.dataset_id == dataset_db_id
        )
    )
    if latest_version_id is None:
        raise NotFound("Dataset has no versions")

    # Subquery to get latest revision per example up to latest_version_id
    subq = (
        select(func.max(models.DatasetExampleRevision.id).label("max_id"))
        .where(models.DatasetExampleRevision.dataset_version_id <= latest_version_id)
        .group_by(models.DatasetExampleRevision.dataset_example_id)
        .subquery()
    )

    query = (
        select(models.DatasetExample, models.DatasetExampleRevision)
        .join(
            models.DatasetExampleRevision,
            models.DatasetExample.id == models.DatasetExampleRevision.dataset_example_id,
        )
        .join(subq, subq.c.max_id == models.DatasetExampleRevision.id)
        .where(models.DatasetExample.dataset_id == dataset_db_id)
        .where(models.DatasetExampleRevision.revision_kind != "DELETE")
        .order_by(models.DatasetExample.id.asc())
    )

    examples: list[DatasetExample] = []
    result = await session.execute(query)
    for example, revision in result.all():
        examples.append(
            DatasetExample(
                id=str(example.id),
                input=revision.input,
                reference_output=revision.output,
            )
        )

    return PlaygroundDataset(
        id=str(dataset_db_id),
        name=dataset_obj.name,
        examples=examples,
    )


@strawberry.input
class OptimizePlaygroundPromptInput:
    dataset_id: str
    algorithm: str
    prompt_messages: JSONScalarType
    tools: Optional[JSONScalarType] = None
    model: str
    model_kwargs: Optional[JSONScalarType] = None
    metric_name: str = "exact_match"
    contains_keywords: Optional[list[str]] = None
    llm_judge_prompt: Optional[str] = None
    llm_judge_model: Optional[str] = None
    max_trials: Optional[int] = None
    n_samples: Optional[int] = None
    min_examples: Optional[int] = None  # few-shot only
    max_examples_fs: Optional[int] = None  # few-shot only
    parameter_space: Optional[JSONScalarType] = None  # parameter optimizer


@strawberry.type
class OptimizePlaygroundPromptPayload:
    job_id: str
    status: str


@strawberry.type
class OptimizerMutationMixin:
    @strawberry.mutation(permission_classes=[IsNotReadOnly, IsNotViewer, IsLocked])  # type: ignore[misc]
    async def optimize_playground_prompt(
        self, info: Info[Context, None], input: OptimizePlaygroundPromptInput
    ) -> OptimizePlaygroundPromptPayload:
        # Validate algorithm
        algorithm = input.algorithm.lower()
        if algorithm not in {
            "metaprompt",
            "evolutionary",
            "hierarchical",
            "gepa",
            "fewshot",
            "parameter",
        }:
            raise BadRequest("Unsupported algorithm")

        metric_name = input.metric_name.lower()
        if metric_name not in {"exact_match", "contains", "llm_judge"}:
            raise BadRequest(
                "Only exact_match, contains, and llm_judge metrics are supported in this endpoint"
            )

        # Resolve dataset
        try:
            dataset_gid = GlobalID.from_id(input.dataset_id)
            dataset_db_id = from_global_id_with_expected_type(
                global_id=dataset_gid, expected_type_name="Dataset"
            )
        except Exception as exc:
            raise BadRequest(f"Invalid dataset id: {input.dataset_id}") from exc

        # Build dataset (sync session is available on context.db as async factory)
        async with info.context.db() as session:
            dataset = await _resolve_dataset_examples(session, dataset_db_id)

        metric_cfg = MetricConfig(
            name=metric_name,
            contains_keywords=input.contains_keywords,
            llm_judge_prompt=input.llm_judge_prompt,
            llm_judge_model=input.llm_judge_model,
        )

        judge_fn = None
        if metric_name == "llm_judge":
            if not input.llm_judge_prompt:
                raise BadRequest("llm_judge_prompt is required when metric_name is llm_judge")
            judge_model = input.llm_judge_model or input.model

            def _judge_fn(dataset_item: dict[str, Any], llm_output: Any) -> float:
                messages = [
                    {"role": "system", "content": input.llm_judge_prompt},
                    {
                        "role": "user",
                        "content": (
                            "Dataset item:\n"
                            f"{dataset_item}\n\nModel output:\n{llm_output}\n\n"
                            "Return a numeric score between 0 and 1."
                        ),
                    },
                ]
                try:
                    resp = litellm.completion(
                        model=judge_model,
                        messages=messages,  # type: ignore[arg-type]
                        temperature=0,
                    )
                    text = (
                        resp.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )
                    return float(text)
                except Exception:
                    return 0.0
            judge_fn = _judge_fn

        request = OptimizationRequest(
            algorithm=algorithm,  # type: ignore[arg-type]
            prompt_messages=input.prompt_messages,
            tools=input.tools or [],
            model=input.model,
            model_kwargs=input.model_kwargs or {},
            dataset=dataset,
            metric=metric_cfg,
            max_trials=input.max_trials,
            n_samples=input.n_samples,
            llm_judge_fn=judge_fn,
            parameter_space=input.parameter_space,
            min_examples=input.min_examples,
            max_examples_fs=input.max_examples_fs,
        )

        job = optimizer_service.submit_job(request, run_async=True)
        return OptimizePlaygroundPromptPayload(job_id=job.id, status=job.status)
