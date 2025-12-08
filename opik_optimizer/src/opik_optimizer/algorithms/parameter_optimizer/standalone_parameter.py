import logging
import copy
from typing import Any, List, Dict, Optional, Callable, Union

import optuna
import optuna.samplers
import litellm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StandaloneParameter")

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
        messages = [msg.copy() for msg in self.messages]
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
            messages=[msg.copy() for msg in self.messages],
            model=self.model,
            model_parameters=self.model_kwargs.copy(),
        )


# -----------------------------------------------------------------------------
# 2. Standalone Parameter Optimizer
# -----------------------------------------------------------------------------


class StandaloneParameterOptimizer:
    """
    Standalone Parameter Optimizer using Optuna to tune model parameters (temperature, etc.)
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        n_trials: int = 20,
        verbose: bool = True,
        seed: int = 42,
    ):
        self.model = model
        self.n_trials = n_trials
        self.verbose = verbose
        self.seed = seed

    def optimize(
        self,
        prompt_messages: List[Dict[str, Any]],
        train_dataset: List[Dict[str, Any]],
        metric: Callable[[Dict[str, Any], str], float],
        parameter_space: Dict[str, Any],  # Dict defining the search space
        val_dataset: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize model parameters.
        parameter_space format example:
        {
            "temperature": {"type": "float", "low": 0.0, "high": 1.0},
            "top_p": {"type": "float", "low": 0.1, "high": 1.0},
        }
        """
        logger.info(f"Starting Parameter Optimization (n_trials={self.n_trials})")

        dataset = val_dataset or train_dataset

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial):
            # 1. Suggest parameters
            suggested_params = {}
            for param_name, config in parameter_space.items():
                p_type = config.get("type")
                if p_type == "float":
                    suggested_params[param_name] = trial.suggest_float(
                        param_name, config["low"], config["high"]
                    )
                elif p_type == "int":
                    suggested_params[param_name] = trial.suggest_int(
                        param_name, config["low"], config["high"]
                    )
                elif p_type == "categorical":
                    suggested_params[param_name] = trial.suggest_categorical(
                        param_name, config["choices"]
                    )

            # 2. Evaluate with these parameters
            score = self._evaluate_prompt(
                prompt_messages, dataset, metric, model_params=suggested_params
            )
            return score

        study.optimize(objective, n_trials=self.n_trials)

        best_params = study.best_trial.params
        best_score = study.best_trial.value

        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {best_score}")

        return {
            "best_parameters": best_params,
            "best_score": best_score,
            "best_prompt": prompt_messages,  # Prompt doesn't change
        }

    def _evaluate_prompt(
        self,
        messages: List[Dict[str, Any]],
        dataset: List[Dict[str, Any]],
        metric: Callable[[Dict[str, Any], str], float],
        model_params: Dict[str, Any],
    ) -> float:
        prompt = StandaloneChatPrompt(
            messages, model=self.model, model_parameters=model_params
        )
        total_score = 0.0

        count = 0
        for item in dataset:
            try:
                msgs = prompt.get_messages(item)
                response = litellm.completion(
                    model=self.model, messages=msgs, **model_params
                )
                output = response.choices[0].message.content
                total_score += metric(item, output)
                count += 1
            except Exception as e:
                # logger.warning(f"Eval error: {e}")
                pass

        return total_score / count if count > 0 else 0.0
