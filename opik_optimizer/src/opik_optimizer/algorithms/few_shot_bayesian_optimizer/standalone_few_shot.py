import copy
import json
import logging
import textwrap
from typing import Any, List, Dict, Optional, Callable, Union

import optuna
import optuna.samplers
import litellm
from pydantic import BaseModel, Field

# Placeholder string used to mark where few-shot examples should be inserted
FEW_SHOT_EXAMPLE_PLACEHOLDER = "FEW_SHOT_EXAMPLE_PLACEHOLDER"

# System prompt for generating few-shot prompt templates
SYSTEM_PROMPT_TEMPLATE = (
    textwrap.dedent(
        """
    You are a prompt editor that modifies a message list to support few-shot learning. Your job is to insert a placeholder where few-shot examples can be inserted and generate a reusable string template for formatting those examples.

    You will receive a JSON object with the following fields:

    - "message_list": a list of messages, each with a role (system, user, or assistant) and a content field.
    - "examples": a list of example pairs, each with input and output fields.

    Your task:

    - Insert the string "{FEW_SHOT_EXAMPLE_PLACEHOLDER}" into one of the messages in the list. Make sure to:
        - Insert it at the most logical point for including few-shot examples — typically as part of the system message
        - Add a section title in XML or markdown format. The examples will be provided as `example_1\nexample_2\n...` with each example following the example template.
    - Analyze the examples to infer a consistent structure, and create a single string few_shot_example_template using the Python .format() style. Make sure to follow the following instructions:
        - Unless absolutely relevant, do not return an object but instead a string that can be inserted as part of {FEW_SHOT_EXAMPLE_PLACEHOLDER}
        - Make sure to include the variables as part of this string so we can before string formatting with actual examples. Only variables available in the examples can be used.
        - Do not apply any transformations to the variables either, only the variable name should be included in the format `{{<variable_name>}}`
        - The few shot examples should include the expected response as the goal is to provide examples of the response.
        - Ensure the format of the few shot examples are consistent with how the model will be called

    Return your output as a JSON object with:

    - message_list_with_placeholder: the updated list with "FEW_SHOT_EXAMPLE_PLACEHOLDER" inserted.
    - example_template: a string template using the fields provided in the examples (you don't need to use all of them)

    Respond only with the JSON object. Do not include any explanation or extra text.
    """
    )
    .strip()
    .format(FEW_SHOT_EXAMPLE_PLACEHOLDER=FEW_SHOT_EXAMPLE_PLACEHOLDER)
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StandaloneFewShot")

# -----------------------------------------------------------------------------
# 1. Standalone Helpers
# -----------------------------------------------------------------------------


class StandaloneChatPrompt:
    def __init__(
        self,
        messages: List[Dict[str, Any]],
        model: str = "gpt-4o-mini",
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
# 2. Standalone Few-Shot Bayesian Optimizer
# -----------------------------------------------------------------------------


class StandaloneFewShotBayesianOptimizer:
    """
    Standalone Few-Shot Bayesian Optimizer using Optuna.
    Selects the best combination of few-shot examples from the dataset.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        model_parameters: Optional[Dict[str, Any]] = None,
        min_examples: int = 2,
        max_examples: int = 5,
        n_trials: int = 20,
        verbose: bool = True,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.model_parameters = model_parameters or {}
        self.min_examples = min_examples
        self.max_examples = max_examples
        self.n_trials = n_trials
        self.verbose = verbose
        self.seed = seed

    def optimize(
        self,
        prompt_messages: List[Dict[str, Any]],
        train_dataset: List[Dict[str, Any]],
        metric: Callable[[Dict[str, Any], str], float],
        val_dataset: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run the optimization.
        """
        logger.info(
            f"Starting Few-Shot Bayesian Optimization (n_trials={self.n_trials})"
        )

        # 1. Create Few-Shot Template
        # We need to know WHERE to insert examples.
        # For simplicity, we assume we insert examples into the system prompt or first user message.
        # Ideally, we used the LLM to generate a template (like in the original), but here let's do a simplified approach:
        # We will append examples to the system prompt or create one.

        template_messages = self._create_template(prompt_messages)

        # 2. Define Objective
        val_dataset = (
            val_dataset or train_dataset
        )  # If no split, evaluate on train (careful with overfitting)
        if val_dataset is train_dataset:
            logger.warning(
                "Using training dataset for evaluation. Risk of overfitting."
            )

        # Pruner and Sampler
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial):
            # 1. Select number of examples
            n_examples = trial.suggest_int(
                "n_examples", self.min_examples, self.max_examples
            )

            # 2. Select specific indices (without replacement ideally, but Optuna suggests independent params)
            # To simulate selection without replacement, we can just suggest n integers from 0 to len-1
            # But standard TPE treats them independently.
            # A simple workaround: suggest n indices, if duplicates, ignore or take unique.

            selected_indices = []
            for i in range(n_examples):
                # Suggest categorical index
                idx = trial.suggest_categorical(
                    f"example_{i}", list(range(len(train_dataset)))
                )
                selected_indices.append(idx)

            # Remove duplicates for the actual prompt
            unique_indices = sorted(list(set(selected_indices)))

            # Build prompt with these examples
            examples_str = self._format_examples(train_dataset, unique_indices)
            candidate_messages = self._inject_examples(template_messages, examples_str)

            # Evaluate
            score = self._evaluate_prompt(candidate_messages, val_dataset, metric)
            return score

        # 3. Optimize
        study.optimize(objective, n_trials=self.n_trials)

        logger.info(f"Best trial: {study.best_trial.value}")
        logger.info(f"Best params: {study.best_trial.params}")

        # Reconstruct best prompt
        best_n = study.best_trial.params["n_examples"]
        best_indices = []
        for i in range(best_n):
            best_indices.append(study.best_trial.params[f"example_{i}"])
        unique_best = sorted(list(set(best_indices)))

        best_examples_str = self._format_examples(train_dataset, unique_best)
        best_messages = self._inject_examples(template_messages, best_examples_str)

        return {
            "best_prompt": best_messages,
            "best_score": study.best_trial.value,
            "best_examples_indices": unique_best,
        }

    def _create_template(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Check if we already have a placeholder
        # If not, let's look for system prompt, or insert one
        template = copy.deepcopy(messages)

        has_system = False
        for msg in template:
            if msg["role"] == "system":
                has_system = True
                if "{examples}" not in msg["content"]:
                    msg["content"] += "\n\nExamples:\n{examples}"
                break

        if not has_system:
            template.insert(
                0,
                {
                    "role": "system",
                    "content": "You are a helpful assistant.\n\nExamples:\n{examples}",
                },
            )

        return template

    def _format_examples(
        self, dataset: List[Dict[str, Any]], indices: List[int]
    ) -> str:
        out = ""
        for idx in indices:
            item = dataset[idx]
            # Naive formatting: JSON dump of item
            # Realistically we might want "Input: ... Output: ..." based on keys
            out += f"Example:\n{json.dumps(item)}\n\n"
        return out

    def _inject_examples(
        self, template: List[Dict[str, Any]], examples_str: str
    ) -> List[Dict[str, Any]]:
        msgs = copy.deepcopy(template)
        for msg in msgs:
            if "{examples}" in msg.get("content", ""):
                msg["content"] = msg["content"].replace("{examples}", examples_str)
        return msgs

    def _evaluate_prompt(
        self,
        messages: List[Dict[str, Any]],
        dataset: List[Dict[str, Any]],
        metric: Callable[[Dict[str, Any], str], float],
    ) -> float:
        prompt = StandaloneChatPrompt(
            messages, model=self.model, model_parameters=self.model_parameters
        )
        total_score = 0.0

        for item in dataset:
            try:
                msgs = prompt.get_messages(item)
                response = litellm.completion(
                    model=self.model, messages=msgs, **self.model_parameters
                )
                output = response.choices[0].message.content
                total_score += metric(item, output)
            except Exception as e:
                pass

        return total_score / len(dataset) if dataset else 0.0
