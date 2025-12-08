import logging
import sys
import os

# Add paths to sys.path to import directly without package initialization side-effects
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "algorithms", "evolutionary_optimizer"))
sys.path.append(os.path.join(current_dir, "algorithms", "few_shot_bayesian_optimizer"))
sys.path.append(
    os.path.join(current_dir, "algorithms", "hierarchical_reflective_optimizer")
)
sys.path.append(os.path.join(current_dir, "algorithms", "parameter_optimizer"))

from standalone_evolutionary import StandaloneEvolutionaryOptimizer
from standalone_few_shot import StandaloneFewShotBayesianOptimizer
from standalone_hierarchical import StandaloneHierarchicalReflectiveOptimizer
from standalone_parameter import StandaloneParameterOptimizer


# Basic Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestStandalone")

# Dummy Dataset and Metric
DATASET = [
    {"input": "What is 2+2?", "expected": "4"},
    {"input": "Capital of France?", "expected": "Paris"},
    {"input": "Color of the sky?", "expected": "blue"},
]


def dummy_metric(item, output):
    # Simple keyword match
    if item["expected"].lower() in output.lower():
        return 1.0
    return 0.0


PROMPT_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Question: {input}\nAnswer:"},
]


def test_evolutionary():
    logger.info("Testing StandaloneEvolutionaryOptimizer...")
    opt = StandaloneEvolutionaryOptimizer(
        num_generations=2, population_size=4, model="gpt-4o-mini", verbose=True
    )
    result = opt.optimize(PROMPT_MESSAGES, DATASET, dummy_metric, max_trials=10)
    logger.info(f"Evolutionary Result: {result['best_score']}")
    assert "best_prompt" in result
    assert "best_score" in result


def test_few_shot():
    logger.info("Testing StandaloneFewShotBayesianOptimizer...")
    opt = StandaloneFewShotBayesianOptimizer(
        n_trials=5, model="gpt-4o-mini", min_examples=1, max_examples=2, verbose=True
    )
    result = opt.optimize(PROMPT_MESSAGES, DATASET, dummy_metric)
    logger.info(f"Few-Shot Result: {result['best_score']}")
    assert "best_prompt" in result
    assert "best_examples_indices" in result


def test_hierarchical():
    logger.info("Testing StandaloneHierarchicalReflectiveOptimizer...")
    opt = StandaloneHierarchicalReflectiveOptimizer(
        max_iterations=2, batch_size=2, model="gpt-4o-mini", verbose=True
    )
    result = opt.optimize(PROMPT_MESSAGES, DATASET, dummy_metric)
    logger.info(f"Hierarchical Result: {result['best_score']}")
    assert "best_prompt" in result
    assert "history" in result


def test_parameter():
    logger.info("Testing StandaloneParameterOptimizer...")
    opt = StandaloneParameterOptimizer(n_trials=5, model="gpt-4o-mini", verbose=True)
    param_space = {
        "temperature": {"type": "float", "low": 0.1, "high": 1.0},
        "max_tokens": {"type": "int", "low": 50, "high": 100},
    }
    result = opt.optimize(PROMPT_MESSAGES, DATASET, dummy_metric, param_space)
    logger.info(f"Parameter Result: {result['best_score']}")
    assert "best_parameters" in result


if __name__ == "__main__":
    try:
        test_evolutionary()
        print("-" * 30)
        test_few_shot()
        print("-" * 30)
        test_hierarchical()
        print("-" * 30)
        test_parameter()
        print("All standalone tests passed!")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)
