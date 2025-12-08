import copy
import logging
from typing import Any, Callable
from dataclasses import dataclass

import litellm
import gepa
from gepa.core.adapter import EvaluationBatch, GEPAAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StandaloneGEPA")

# -----------------------------------------------------------------------------
# 1. Standalone ChatPrompt (No Opik dependencies)
# -----------------------------------------------------------------------------

class StandaloneChatPrompt:
    def __init__(
        self,
        messages: list[dict[str, Any]],
        model: str = "gpt-4o-mini",
        model_parameters: dict[str, Any] | None = None,
    ) -> None:
        self.messages = messages
        self.model = model
        self.model_kwargs = model_parameters or {}

    def get_messages(self, dataset_item: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        messages = copy.deepcopy(self.messages)
        if dataset_item:
            for key, value in dataset_item.items():
                label = "{" + key + "}"
                for message in messages:
                    if isinstance(message.get("content"), str):
                        message["content"] = message["content"].replace(label, str(value))
        return messages

    def copy(self) -> "StandaloneChatPrompt":
        return StandaloneChatPrompt(
            messages=copy.deepcopy(self.messages),
            model=self.model,
            model_parameters=copy.deepcopy(self.model_kwargs),
        )

# -----------------------------------------------------------------------------
# 2. Standalone Agent (No Opik tracking, just LiteLLM)
# -----------------------------------------------------------------------------

class StandaloneAgent:
    def __init__(self, prompt: StandaloneChatPrompt) -> None:
        self.prompt = prompt
        self.model = prompt.model
        self.model_kwargs = prompt.model_kwargs

    def invoke(self, messages: list[dict[str, str]]) -> str:
        response = litellm.completion(
            model=self.model,
            messages=messages,
            **self.model_kwargs,
        )
        return response.choices[0].message.content

# -----------------------------------------------------------------------------
# 3. Standalone Adapter (Local Evaluation only)
# -----------------------------------------------------------------------------

@dataclass
class StandaloneDataInst:
    input_text: str
    expected: str
    item: dict[str, Any]

class StandaloneGEPAAdapter(GEPAAdapter[StandaloneDataInst, dict[str, Any], dict[str, Any]]):
    def __init__(
        self,
        base_prompt: StandaloneChatPrompt,
        metric: Callable[[dict[str, Any], str], float],
    ) -> None:
        self._base_prompt = base_prompt
        self._metric = metric

    def evaluate(
        self,
        batch: list[StandaloneDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], dict[str, Any]]:
        
        # Apply system prompt from candidate
        system_text = candidate.get("system_prompt", candidate.get("system", ""))
        prompt_variant = self._base_prompt.copy()
        
        # Simple logic to inject system prompt (assuming first message is system or inserting it)
        if prompt_variant.messages and prompt_variant.messages[0].get("role") == "system":
            prompt_variant.messages[0]["content"] = system_text
        else:
            prompt_variant.messages.insert(0, {"role": "system", "content": system_text})

        agent = StandaloneAgent(prompt_variant)
        
        outputs = []
        scores = []
        trajectories = [] if capture_traces else None

        for inst in batch:
            messages = prompt_variant.get_messages(inst.item)
            try:
                raw_output = agent.invoke(messages).strip()
            except Exception as e:
                logger.error(f"Error invoking agent: {e}")
                raw_output = ""
            
            score = float(self._metric(inst.item, raw_output))
            
            outputs.append({"output": raw_output})
            scores.append(score)
            
            if trajectories is not None:
                trajectories.append({
                    "input": inst.item,
                    "output": raw_output,
                    "score": score
                })

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict[str, Any], dict[str, Any]],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        # Minimal reflection dataset generation
        records = []
        if eval_batch.trajectories:
            for traj in eval_batch.trajectories:
                item = traj.get("input", {})
                records.append({
                    "Inputs": {"text": str(item)}, # Simplify input representation
                    "Generated Outputs": traj.get("output", ""),
                    "Feedback": f"Score: {traj.get('score', 0)}",
                })
        
        return {comp: records for comp in (components_to_update or ["system_prompt"])}

# -----------------------------------------------------------------------------
# 4. Standalone Optimizer
# -----------------------------------------------------------------------------

class StandaloneGepaOptimizer:
    def __init__(
        self,
        model: str = "gpt-4o",
        verbose: int = 1,
    ) -> None:
        self.model = model
        self.verbose = verbose

    def optimize(
        self,
        prompt_messages: list[dict[str, Any]],
        train_dataset: list[dict[str, Any]],
        metric: Callable[[dict[str, Any], str], float],
        max_trials: int = 10,
        val_dataset: list[dict[str, Any]] | None = None,
        seed_system_prompt: str = "You are a helpful assistant.",
    ) -> dict[str, Any]:
        
        # Setup Prompts
        # Ensure there is a system message placeholder if not present
        initial_messages = copy.deepcopy(prompt_messages)
        has_system = any(m.get("role") == "system" for m in initial_messages)
        if not has_system:
             initial_messages.insert(0, {"role": "system", "content": seed_system_prompt})

        sc_prompt = StandaloneChatPrompt(
            messages=initial_messages,
            model=self.model
        )

        # Build Adapter
        adapter = StandaloneGEPAAdapter(base_prompt=sc_prompt, metric=metric)

        # Wrap Dataset
        # Assuming dataset items are dicts. We convert them to StandaloneDataInst
        # We need to know which keys correspond to input/output for generic GEPA logic, 
        # but here we pass the whole item to the metric.
        train_insts = [
            StandaloneDataInst(input_text=str(item), expected="", item=item) 
            for item in train_dataset
        ]
        
        val_insts = []
        if val_dataset:
            val_insts = [
                StandaloneDataInst(input_text=str(item), expected="", item=item) 
                for item in val_dataset
            ]
        else:
            val_insts = train_insts

        # Run GEPA
        print(f"Starting Standalone GEPA Optimization (max_trials={max_trials})...")
        
        gepa_result = gepa.optimize(
            seed_candidate={"system_prompt": seed_system_prompt},
            trainset=train_insts,
            valset=val_insts,
            adapter=adapter,
            task_lm=None, # Not used since we use local agent in adapter
            reflection_lm=self.model, # Model string for reflection
            max_metric_calls=max_trials * len(train_insts) * 2, # Rough budget
            display_progress_bar=bool(self.verbose),
            logger=None, # Use default or None
        )
        
        best_candidate = getattr(gepa_result, "best_candidate", {})
        print(f"Optimization Finished. Best Score: {gepa_result.best_score}")
        print(f"Best Candidate: {best_candidate}")

        return {
            "best_prompt": best_candidate.get("system_prompt"),
            "best_score": gepa_result.best_score,
            "candidates": gepa_result.candidates,
            "scores": gepa_result.val_aggregate_scores
        }
