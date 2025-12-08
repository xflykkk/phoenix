from __future__ import annotations

from typing import Any, Optional

import asyncio
import litellm
import strawberry
from strawberry.scalars import JSON as JSONScalarType
from strawberry.relay import GlobalID
from strawberry.types import Info

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from phoenix.db import models
from phoenix.optimizer import MetricConfig, OptimizationRequest
from phoenix.optimizer.models import DatasetExample, PlaygroundDataset
from phoenix.optimizer.optimization_service import service as optimizer_service
from phoenix.server.api.auth import IsLocked, IsNotReadOnly, IsNotViewer
from phoenix.server.api.context import Context
from phoenix.server.api.exceptions import BadRequest, NotFound
from phoenix.server.api.helpers.playground_clients import PlaygroundClientCredential
from phoenix.server.api.helpers.playground_registry import PLAYGROUND_CLIENT_REGISTRY
from phoenix.server.api.input_types.GenerativeCredentialInput import (
    GenerativeCredentialInput,
)
from phoenix.server.api.input_types.GenerativeModelInput import GenerativeModelInput
from phoenix.server.api.input_types.InvocationParameters import (
    InvocationParameterInput,
)
from phoenix.server.api.types.node import from_global_id_with_expected_type


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
    model: GenerativeModelInput
    invocation_parameters: list[InvocationParameterInput] = strawberry.field(
        default_factory=list
    )
    metric_name: str = "exact_match"
    contains_keywords: Optional[list[str]] = None
    llm_judge_prompt: Optional[str] = None
    llm_judge_model: Optional[str] = None
    max_trials: Optional[int] = None
    n_samples: Optional[int] = None
    min_examples: Optional[int] = None  # few-shot only
    max_examples_fs: Optional[int] = None  # few-shot only
    parameter_space: Optional[JSONScalarType] = None  # parameter optimizer
    credentials: Optional[list[GenerativeCredentialInput]] = strawberry.field(
        default_factory=list
    )


@strawberry.type
class OptimizePlaygroundPromptPayload:
    job_id: str
    status: str


@strawberry.type
class CancelPlaygroundPromptPayload:
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

        # Build Playground LLM client (reuse Playground credentials/params)
        llm_client_class = PLAYGROUND_CLIENT_REGISTRY.get_client(
            input.model.provider_key, input.model.name
        )
        if llm_client_class is None:
            raise BadRequest(f"Unknown LLM provider: '{input.model.provider_key.value}'")
        credentials = [
            PlaygroundClientCredential(env_var_name=cred.env_var_name, value=cred.value)
            for cred in (input.credentials or [])
        ]
        try:
            llm_client = llm_client_class(model=input.model, credentials=credentials)
            invocation_parameters = llm_client.construct_invocation_parameters(
                input.invocation_parameters or []
            )
        except Exception as exc:  # pragma: no cover - guarded by UI/client validation
            raise BadRequest(
                f"Failed to initialize LLM client {input.model.provider_key.value} "
                f"{input.model.name}: {exc}"
            ) from exc

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

            def _judge_fn(dataset_item: dict[str, Any], llm_output: Any) -> float:
                messages = [
                    (
                        "system",
                        input.llm_judge_prompt or "",
                        None,
                        None,
                    ),
                    (
                        "user",
                        (
                            "Dataset item:\n"
                            f"{dataset_item}\n\nModel output:\n{llm_output}\n\n"
                            "Return a numeric score between 0 and 1."
                        ),
                        None,
                        None,
                    ),
                ]
                try:
                    text_parts: list[str] = []
                    async def _run() -> None:
                        async for chunk in llm_client.chat_completion_create(
                            messages=messages,
                            tools=[],
                            **invocation_parameters,
                        ):
                            content = getattr(chunk, "content", None)
                            if content:
                                text_parts.append(str(content))

                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        asyncio.run(_run())
                    else:
                        loop.run_until_complete(_run())
                    if not text_parts:
                        return 0.0
                    return float(text_parts[-1].strip())
                except Exception:
                    return 0.0

            judge_fn = _judge_fn

        # Normalize prompt roles to lowercase for opik_optimizer
        normalized_messages = []
        for m in input.prompt_messages:
            if isinstance(m, dict):
                m = {**m, "role": str(m.get("role", "")).lower()}
            normalized_messages.append(m)

        request = OptimizationRequest(
            algorithm=algorithm,  # type: ignore[arg-type]
            prompt_messages=normalized_messages,
            tools=input.tools or [],
            model=input.model.name,
            model_kwargs={},
            dataset=dataset,
            metric=metric_cfg,
            max_trials=input.max_trials,
            n_samples=input.n_samples,
            llm_judge_fn=judge_fn,
            parameter_space=input.parameter_space,
            min_examples=input.min_examples,
            max_examples_fs=input.max_examples_fs,
            invocation_parameters=invocation_parameters,
            llm_client=llm_client,
        )

        job = optimizer_service.submit_job(request, run_async=True)
        return OptimizePlaygroundPromptPayload(job_id=job.id, status=job.status)

    @strawberry.mutation(permission_classes=[IsNotReadOnly, IsNotViewer, IsLocked])  # type: ignore[misc]
    async def cancel_playground_optimization_job(
        self, info: Info[Context, None], id: strawberry.ID
    ) -> CancelPlaygroundPromptPayload:
        job_id = str(id)
        ok = optimizer_service.cancel_job(job_id)
        if not ok:
            raise NotFound(f"Job {job_id} not found")
        job = optimizer_service.get_job(job_id)
        status = job.status if job else "canceled"
        return CancelPlaygroundPromptPayload(job_id=job_id, status=status)
