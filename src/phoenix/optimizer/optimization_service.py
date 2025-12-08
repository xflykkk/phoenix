from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from .job_runner import run_optimization_job
from .models import OptimizationRequest, OptimizationResult


JobStatus = str


@dataclass
class OptimizationJob:
    id: str
    status: JobStatus
    request: OptimizationRequest
    result: OptimizationResult | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class OptimizationService:
    """
    Lightweight in-memory job service.

    This is meant as a placeholder until a persistent queue/worker is wired in.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, OptimizationJob] = {}
        self._lock = threading.Lock()

    def submit_job(
        self,
        request: OptimizationRequest,
        *,
        run_async: bool = True,
        on_progress: Callable[[OptimizationJob], None] | None = None,
    ) -> OptimizationJob:
        job_id = str(uuid.uuid4())
        job = OptimizationJob(id=job_id, status="pending", request=request)
        with self._lock:
            self._jobs[job_id] = job

        def _run() -> None:
            self._update_status(job_id, "running")
            try:
                result = run_optimization_job(request)
                self._update_result(job_id, result)
                self._update_status(job_id, "completed")
            except Exception as exc:  # pragma: no cover - defensive
                self._update_error(job_id, str(exc))
                self._update_status(job_id, "failed")
            if on_progress:
                on_progress(self._jobs[job_id])

        if run_async:
            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
        else:
            _run()

        return job

    def get_job(self, job_id: str) -> Optional[OptimizationJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def _update_status(self, job_id: str, status: JobStatus) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = status

    def _update_result(self, job_id: str, result: OptimizationResult) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.result = result
                job.metadata["trials_used"] = result.trials_used
                job.metadata["best_score"] = result.best_score

    def _update_error(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.error = error


# Global singleton service used by API handlers.
service = OptimizationService()
