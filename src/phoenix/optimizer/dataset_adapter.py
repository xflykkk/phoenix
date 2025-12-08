from __future__ import annotations

import random
from typing import Any

from .models import PlaygroundDataset


class OptimizerDataset:
    """
    Minimal dataset interface to satisfy opik_optimizer expectations.

    opik_optimizer requires:
    - name / id attributes
    - get_items(nb_samples: int | None = None) -> list[dict[str, Any]]
    """

    def __init__(self, dataset: PlaygroundDataset) -> None:
        self._dataset = dataset
        self.name = dataset.name
        self.id = dataset.id

    def get_items(self, nb_samples: int | None = None) -> list[dict[str, Any]]:
        items = [
            {
                "id": example.id,
                **example.input,
                "_reference_output": example.reference_output,
            }
            for example in self._dataset.examples
        ]
        if nb_samples is None or nb_samples >= len(items):
            return items
        return random.sample(items, nb_samples)
