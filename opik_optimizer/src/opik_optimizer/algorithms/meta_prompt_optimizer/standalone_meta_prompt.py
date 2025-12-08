import copy
import json
import logging
import random
import textwrap
from typing import Any, Callable, Dict, List, Optional
import litellm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StandaloneMetaPrompt")

# -----------------------------------------------------------------------------
# 1. Standalone ChatPrompt (No Opik dependencies)
# -----------------------------------------------------------------------------


class StandaloneChatPrompt:
    def __init__(
        self,
        messages: List[Dict[str, Any]],
        model: str = "gpt-4o",
        model_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.messages = messages
        self.model = model
        self.model_kwargs = model_parameters or {}

    def get_messages(
        self, dataset_item: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        messages = copy.deepcopy(self.messages)
        if dataset_item:
            for key, value in dataset_item.items():
                label = "{" + key + "}"
                for message in messages:
                    if isinstance(message.get("content"), str):
                        message["content"] = message["content"].replace(
                            label, str(value)
                        )
        return messages

    def copy(self) -> "StandaloneChatPrompt":
        return StandaloneChatPrompt(
            messages=copy.deepcopy(self.messages),
            model=self.model,
            model_parameters=copy.deepcopy(self.model_kwargs),
        )


# -----------------------------------------------------------------------------
# 2. Standalone Meta-Prompt Optimizer
# -----------------------------------------------------------------------------


class StandaloneMetaPromptOptimizer:
    def __init__(
        self,
        model: str = "gpt-4o",
        prompts_per_round: int = 4,
        max_context_examples: int = 3,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.prompts_per_round = prompts_per_round
        self.max_context_examples = max_context_examples
        self.verbose = verbose

    def optimize(
        self,
        prompt_messages: List[Dict[str, Any]],
        train_dataset: List[Dict[str, Any]],
        metric: Callable[[Dict[str, Any], str], float],
        max_trials: int = 10,
        val_dataset: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize a prompt using LLM-based meta-reasoning (Standalone version).
        """
        logger.info(f"Starting Meta-Prompt Optimization (max_trials={max_trials})")

        initial_prompt = StandaloneChatPrompt(prompt_messages, model=self.model)
        val_dataset = val_dataset or train_dataset

        # Initial Evaluation
        initial_score = self._evaluate_prompt(initial_prompt, val_dataset, metric)
        logger.info(f"Initial Score: {initial_score:.4f}")

        best_prompt = initial_prompt
        best_score = initial_score
        history = []
        trials_used = 0
        round_num = 0

        while trials_used < max_trials:
            logger.info(f"--- Round {round_num + 1} ---")

            # 1. Generate Candidates
            candidates = self._generate_candidates(
                current_prompt=best_prompt,
                best_score=best_score,
                history=history,
                task_examples=train_dataset[: self.max_context_examples],
                metric_name=getattr(metric, "__name__", "custom_metric"),
            )

            # 2. Evaluate Candidates
            round_results = []
            for cand in candidates:
                if trials_used >= max_trials:
                    break

                score = self._evaluate_prompt(cand, val_dataset, metric)
                round_results.append((cand, score))
                trials_used += 1
                logger.debug(f"Candidate score: {score:.4f}")

            if not round_results:
                break

            # 3. Update Best and History
            round_results.sort(key=lambda x: x[1], reverse=True)
            round_best_cand, round_best_score = round_results[0]

            history.append(
                {
                    "round": round_num,
                    "best_score": round_best_score,
                    "best_messages": round_best_cand.messages,
                    "candidates": [
                        {"messages": c[0].messages, "score": c[1]}
                        for c in round_results
                    ],
                }
            )

            if round_best_score > best_score:
                logger.info(
                    f"New best score found: {round_best_score:.4f} (prev: {best_score:.4f})"
                )
                best_score = round_best_score
                best_prompt = round_best_cand
            else:
                logger.info(f"No improvement this round (best: {round_best_score:.4f})")

            round_num += 1

        return {
            "best_prompt": best_prompt.messages,
            "best_score": best_score,
            "history": history,
        }

    def _evaluate_prompt(
        self,
        prompt: StandaloneChatPrompt,
        dataset: List[Dict[str, Any]],
        metric: Callable[[Dict[str, Any], str], float],
    ) -> float:
        total_score = 0.0
        count = 0
        for item in dataset:
            try:
                # Basic agent invocation
                messages = prompt.get_messages(item)
                response = litellm.completion(
                    model=prompt.model, messages=messages, **prompt.model_kwargs
                )
                output = response.choices[0].message.content
                score = metric(item, output)
                total_score += score
                count += 1
            except Exception as e:
                logger.warning(f"Error evaluating item: {e}")

        return total_score / count if count > 0 else 0.0

    def _generate_candidates(
        self,
        current_prompt: StandaloneChatPrompt,
        best_score: float,
        history: List[Dict[str, Any]],
        task_examples: List[Dict[str, Any]],
        metric_name: str,
    ) -> List[StandaloneChatPrompt]:
        # Construct Meta-Prompt
        context_str = self._build_context_str(task_examples, metric_name)
        history_str = self._build_history_str(history)
        current_prompt_str = json.dumps(current_prompt.messages, indent=2)

        system_prompt = textwrap.dedent(
            """
            You are an expert prompt engineer. Your task is to generate valid JSON containing improved prompt variations.
            Focus on clarity, specificity, and constraints.
            Return a JSON object with a single key "prompts" containing a list of objects.
            Each object must have:
            - "prompt": a list of message objects [{"role": "...", "content": "..."}]
            - "reasoning": string explaining the change
        """
        )

        user_prompt = textwrap.dedent(
            f"""
            Optimize this prompt to maximize the '{metric_name}' metric.

            CURRENT PROMPT:
            {current_prompt_str}

            CURRENT BEST SCORE: {best_score:.4f}

            TASK CONTEXT:
            {context_str}

            OPTIMIZATION HISTORY:
            {history_str}

            Generate {self.prompts_per_round} improved prompt variations.
            Vary the tone, structure, and specificity.
            IMPORTANT: Ensure placeholders (e.g., {{input}}) are preserved if they exist in the original prompt.
        """
        )

        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            data = json.loads(content)

            candidates = []
            for p_data in data.get("prompts", []):
                if "prompt" in p_data:
                    candidates.append(
                        StandaloneChatPrompt(
                            messages=p_data["prompt"],
                            model=current_prompt.model,
                            model_parameters=current_prompt.model_kwargs,
                        )
                    )
            return candidates

        except Exception as e:
            logger.error(f"Failed to generate candidates: {e}")
            return []

    def _build_context_str(
        self, examples: List[Dict[str, Any]], metric_name: str
    ) -> str:
        ctx = f"Metric: {metric_name}\n\nExamples:\n"
        for i, ex in enumerate(examples):
            ctx += f"--- Example {i+1} ---\n{json.dumps(ex)}\n"
        return ctx

    def _build_history_str(self, history: List[Dict[str, Any]]) -> str:
        if not history:
            return "No previous rounds."

        # Show last 2 rounds
        recent = history[-2:]
        hist_str = ""
        for r in recent:
            hist_str += f"Round {r['round']}: Best Score {r['best_score']:.4f}\n"
        return hist_str
