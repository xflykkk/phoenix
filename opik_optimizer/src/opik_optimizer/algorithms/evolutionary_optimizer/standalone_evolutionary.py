import copy
import json
import logging
import random
import sys
import warnings
from typing import Any, List, Dict, Tuple, Optional, Callable
import numpy as np

# DEAP imports
from deap import base, tools, creator

import litellm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StandaloneEvolutionary")

# -----------------------------------------------------------------------------
# 1. Standalone ChatPrompt & Helper Classes
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
# 2. Standalone Evolutionary Optimizer
# -----------------------------------------------------------------------------


class StandaloneEvolutionaryOptimizer:
    """
    Standalone Evolutionary Optimizer that uses genetic algorithms to evolve prompts.
    Does not require opik backend.
    """

    DEFAULT_POPULATION_SIZE = 10
    DEFAULT_NUM_GENERATIONS = 5
    DEFAULT_MUTATION_RATE = 0.2
    DEFAULT_CROSSOVER_RATE = 0.5
    DEFAULT_TOURNAMENT_SIZE = 3
    DEFAULT_ELITISM_SIZE = 1

    def __init__(
        self,
        model: str = "gpt-4o",
        model_parameters: Optional[Dict[str, Any]] = None,
        population_size: int = DEFAULT_POPULATION_SIZE,
        num_generations: int = DEFAULT_NUM_GENERATIONS,
        mutation_rate: float = DEFAULT_MUTATION_RATE,
        crossover_rate: float = DEFAULT_CROSSOVER_RATE,
        tournament_size: int = DEFAULT_TOURNAMENT_SIZE,
        elitism_size: int = DEFAULT_ELITISM_SIZE,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.model_parameters = model_parameters or {}
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size
        self.verbose = verbose

        # Initialize DEAP
        self._setup_deap()

    def _setup_deap(self):
        # Create Fitness and Individual classes if they don't exist
        # Minimizing negative score or maximizing positive score? Assuming maximizing score.
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        # Not registering attribute generators here as we initialize population manually

    def optimize(
        self,
        prompt_messages: List[Dict[str, Any]],
        train_dataset: List[Dict[str, Any]],
        metric: Callable[[Dict[str, Any], str], float],
        max_trials: int = 50,
        val_dataset: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run the evolutionary optimization loop.
        """
        logger.info(f"Starting Evolutionary Optimization (max_trials={max_trials})")

        # 1. Initialize Population
        # For simplicity, we create a population of identical individuals (clones of initial prompt)
        # In a real scenario, you might want to start with variations.
        initial_messages = copy.deepcopy(prompt_messages)
        population = [
            creator.Individual(copy.deepcopy(initial_messages))
            for _ in range(self.population_size)
        ]

        # We need a way to track total evaluations to respect max_trials
        self.eval_count = 0
        self.max_trials = max_trials

        # Statistics
        best_score_overall = -float("inf")
        best_individual_overall = None
        history = []

        # 2. Evolution Loop
        for gen in range(self.num_generations):
            if self.eval_count >= self.max_trials:
                logger.info("Max trials reached.")
                break

            logger.info(f"--- Generation {gen + 1} ---")

            # Evaluate offspring/population
            # We only evaluate individuals with invalid fitness (new ones)
            invalid_ind = [ind for ind in population if not ind.fitness.valid]

            for ind in invalid_ind:
                if self.eval_count >= self.max_trials:
                    break

                score = self._evaluate_individual(ind, train_dataset, metric)
                ind.fitness.values = (score,)
                self.eval_count += 1

                if score > best_score_overall:
                    best_score_overall = score
                    best_individual_overall = copy.deepcopy(ind)
                    logger.info(f"New best score: {score:.4f}")

            # Select next generation
            offspring = tools.selTournament(
                population, len(population) - self.elitism_size, self.tournament_size
            )
            offspring = [copy.deepcopy(ind) for ind in offspring]

            # Apply Crossover
            for i in range(0, len(offspring) - 1, 2):
                if random.random() < self.crossover_rate:
                    self._crossover(offspring[i], offspring[i + 1])
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values

            # Apply Mutation
            for ind in offspring:
                if random.random() < self.mutation_rate:
                    self._mutate(ind)
                    del ind.fitness.values

            # Elitism: keep best from previous population
            elites = tools.selBest(population, self.elitism_size)
            population[:] = elites + offspring

            # Record stats
            valid_fits = [
                ind.fitness.values[0] for ind in population if ind.fitness.valid
            ]
            if valid_fits:
                avg_fit = sum(valid_fits) / len(valid_fits)
                max_fit = max(valid_fits)
                history.append({"gen": gen, "avg": avg_fit, "max": max_fit})
                logger.debug(f"Gen {gen+1}: Avg={avg_fit:.4f}, Max={max_fit:.4f}")

        return {
            "best_prompt": list(best_individual_overall)
            if best_individual_overall
            else initial_messages,
            "best_score": best_score_overall,
            "history": history,
        }

    def _evaluate_individual(
        self,
        individual: List[Dict[str, str]],
        dataset: List[Dict[str, Any]],
        metric: Callable[[Dict[str, Any], str], float],
    ) -> float:
        prompt = StandaloneChatPrompt(
            list(individual), model=self.model, model_parameters=self.model_parameters
        )
        total_score = 0.0

        # Simple evaluation loop
        # For speed, maybe sample dataset? using full dataset for now.
        for item in dataset:
            try:
                messages = prompt.get_messages(item)
                response = litellm.completion(
                    model=self.model, messages=messages, **self.model_parameters
                )
                output = response.choices[0].message.content
                total_score += metric(item, output)
            except Exception as e:
                logger.error(f"Evaluation error: {e}")

        return total_score / len(dataset) if dataset else 0.0

    # ... (previous code)

    def _extract_json_arrays(self, text: str) -> List[str]:
        """Extract top-level JSON array substrings from arbitrary text."""
        arrays: List[str] = []
        depth = 0
        start: Optional[int] = None
        in_str = False
        escape = False
        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "[":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "]" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    arrays.append(text[start : i + 1])
                    start = None
        return arrays

    def _crossover(self, ind1: List[Dict[str, str]], ind2: List[Dict[str, str]]):
        """
        LLM-based crossover: ask LLM to blend two parent prompts.
        """
        # Fallback to DEAP crossover if LLM fails or for simple cases (not implemented here fully)
        # But we aim for LLM crossover as primary.

        output_style_guidance = "Produce clear, effective, and high-quality responses suitable for the task."

        # Exact System Prompt
        system_prompt = f"""You are an expert prompt engineer specializing in creating novel prompts by intelligently blending existing ones.
Given two parent prompts, your task is to generate one or two new child prompts that effectively combine the strengths, styles, or core ideas of both parents.
The children should be coherent and aim to explore a potentially more effective region of the prompt design space, with a key goal of eliciting responses from the target language model in the following style: '{output_style_guidance}'.

Consider the following when generating children:
- Identify the key instructions, constraints, and desired output formats in each parent, paying attention to any hints about desired output style.
- Explore ways to merge these elements such that the resulting prompt strongly guides the target LLM towards the desired output style.
- You can create a child that is a direct blend, or one that takes a primary structure from one parent and incorporates specific elements from the other, always optimizing for clear instruction towards the desired output style.
- If generating two children, try to make them distinct from each other and from the parents, perhaps by emphasizing different aspects of the parental combination that could lead to the desired output style.

All generated prompts must aim for eliciting answers in the style: '{output_style_guidance}'.

Return a JSON object that is a list of both child prompts. Each child prompt is a list of LLM messages. Example:
[
    {{"role": "<role>", "content": "<content>"}},
    {{"role": "<role>", "content": "<content>"}}
]
"""

        # Exact User Prompt
        user_prompt = f"""Parent Prompt 1:
'''{ind1}'''

Parent Prompt 2:
'''{ind2}'''

Desired output style from target LLM for children prompts: '{output_style_guidance}'

Please generate TWO child prompts by intelligently blending the ideas, styles, or structures from these two parents, ensuring the children aim to elicit the desired output style.
Follow the instructions provided in the system prompt regarding the JSON output format:
[
    {{"role": "<role>", "content": "<content>"}}, {{"role": "<role>", "content": "<content>"}}
]
"""

        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **self.model_parameters,
            )
            content = response.choices[0].message.content

            # Simple parsing attempt
            try:
                children = json.loads(content)
            except json.JSONDecodeError:
                # Fallback extraction
                extracted = self._extract_json_arrays(content)
                children = []
                for arr in extracted:
                    try:
                        parsed = json.loads(arr)
                        if isinstance(parsed, list):
                            children.append(parsed)
                    except:
                        pass

            # Normalize structure: we expect a list of prompts (where each prompt is a list of dicts)
            # The prompt says "Return a JSON object that is a list of both child prompts." -> [[msg, msg], [msg, msg]]
            # But the example shows `[{"role":...}, ...]` which is one prompt. The text says "list of BOTH child prompts".
            # The return format is ambiguous in prompt text vs example.
            # "Example: [{"role":...}, ...]" implies one prompt.
            # But the logic expects 2 children.
            # Let's assume the LLM returns a list of lists if successful, or we handle list of dicts as one child.

            final_children = []
            if isinstance(children, list) and len(children) > 0:
                if isinstance(children[0], list):  # List of lists
                    final_children = children
                elif isinstance(children[0], dict):  # List of dicts (single prompt)
                    final_children = [children]

            if len(final_children) >= 1:
                # Update parents in place to new children (standard DEAP way)
                ind1[:] = final_children[0]
                if len(final_children) >= 2:
                    ind2[:] = final_children[1]
                else:
                    # If only one child, ind2 stays same or becomes clone of child 1?
                    # DEAP crossover usually updates both. Let's leave ind2 if only 1 child found.
                    pass

        except Exception as e:
            logger.warning(f"LLM Crossover failed: {e}")
            # Fallback to simple swap
            self._simple_crossover(ind1, ind2)

        return ind1, ind2

    def _simple_crossover(self, ind1, ind2):
        # Fallback simple 1-point crossover
        if len(ind1) != len(ind2):
            return ind1, ind2
        idx = random.randint(0, len(ind1) - 1)
        if ind1[idx]["role"] == ind2[idx]["role"]:
            ind1[idx]["content"], ind2[idx]["content"] = (
                ind2[idx]["content"],
                ind1[idx]["content"],
            )
        return ind1, ind2

    def _mutate(self, ind: List[Dict[str, str]]):
        """
        LLM-based mutation using strategies.
        """
        if not ind:
            return ind

        strategies = {
            "rephrase": "Create a different way to express the same instruction, possibly with a different length or structure.",
            "simplify": "Simplify the instruction while maintaining its core meaning, potentially making it more concise.",
            "elaborate": "Add more relevant detail and specificity to the instruction, potentially increasing its length.",
            "restructure": "Change the structure of the instruction (e.g., reorder sentences, combine/split ideas) while keeping its intent.",
            "focus": "Emphasize the key aspects of the instruction, perhaps by rephrasing or adding clarifying statements.",
        }

        strategy_name, strategy_desc = random.choice(list(strategies.items()))
        output_style_guidance = "Produce clear, effective, and high-quality responses suitable for the task."

        # Select random message to mutate
        # In a full impl we might not mutate 'user' messages if fixed, or 'system' content specifically.
        # Assuming all are mutable prompt parts.
        idx = random.randint(0, len(ind) - 1)

        # System Prompt
        system_prompt = (
            "You are a prompt engineering expert. Your goal is to modify prompts to improve their "
            f"effectiveness in eliciting specific types of answers, particularly matching the style: '{output_style_guidance}'. "
            "Follow the specific modification instruction provided."
        )

        # User Prompt
        user_prompt = f"""Given this prompt: '{json.dumps(ind)}'
Task context: None provided
Desired output style from target LLM: '{output_style_guidance}'
Instruction for this modification: {strategy_desc} ensuring it still aims for an answer from the target LLM in the style of: '{output_style_guidance}'.

Return only the modified prompt message list, nothing else. Make sure to return a valid JSON object.
"""
        # Note: The original `semantic_mutation_user_prompt` takes the WHOLE prompt list?
        # Yes: `Given this prompt: '{prompt_messages}'`.
        # So it mutates the entire conversation structure?
        # "Return only the modified prompt message list" suggests it returns the whole list.
        # So we should pass `ind` entirely, not just one message.

        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **self.model_parameters,
            )
            content = response.choices[0].message.content.strip()

            # Extract JSON
            try:
                new_ind = json.loads(content)
            except:
                extracted = self._extract_json_arrays(content)
                if extracted:
                    new_ind = json.loads(extracted[0])
                else:
                    new_ind = None

            if new_ind and isinstance(new_ind, list) and len(new_ind) > 0:
                ind[:] = new_ind

        except Exception as e:
            logger.warning(f"Mutation failed: {e}")

        return ind
