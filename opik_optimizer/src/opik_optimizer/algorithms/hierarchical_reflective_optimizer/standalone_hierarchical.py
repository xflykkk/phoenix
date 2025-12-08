import logging
import asyncio
import json
import copy
from typing import Any, List, Dict, Optional, Callable, Union

from pydantic import BaseModel, Field
import litellm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StandaloneHierarchical")

# -----------------------------------------------------------------------------
# 1. Standalone API Objects & Types
# -----------------------------------------------------------------------------


class FailureMode(BaseModel):
    name: str = Field(description="Name of the failure mode")
    description: str = Field(description="Detailed description of the failure mode")
    root_cause: str = Field(description="Hypothesized root cause of the failure mode")


class RootCauseAnalysis(BaseModel):
    failure_modes: List[FailureMode] = Field(
        description="List of identified failure modes"
    )


class ImprovedPrompt(BaseModel):
    reasoning: str = Field(description="Reasoning behind the improvements")
    messages: List[Dict[str, Any]] = Field(description="The improved prompt messages")


# -----------------------------------------------------------------------------
# 2. Standalone Helpers
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
# 3. Standalone Root Cause Analyzer
# -----------------------------------------------------------------------------


class StandaloneHierarchicalRootCauseAnalyzer:
    def __init__(
        self,
        model: str,
        model_parameters: Dict[str, Any],
        batch_size: int = 5,
        verbose: bool = True,
    ):
        self.model = model
        self.model_parameters = model_parameters
        self.batch_size = batch_size
        self.verbose = verbose

    async def analyze(self, failures: List[Dict[str, Any]]) -> List[FailureMode]:
        """
        Analyze failures in batches and synthesize.
        failures: list of dicts with 'input', 'output', 'score', 'expected_output' etc.
        """
        if not failures:
            return []

        # 1. Analyze batches
        batch_analyses = []
        for i in range(0, len(failures), self.batch_size):
            batch = failures[i : i + self.batch_size]
            analysis = await self._analyze_batch(batch, i // self.batch_size)
            batch_analyses.extend(analysis.failure_modes)

        # 2. Synthesize (Simplified: just deduplicate or re-summarize)
        # For true hierarchical, we'd feed batch_analyses back to LLM.
        # Here we just take unique ones or re-ask LLM to synthesize.
        synthesized = await self._synthesize(batch_analyses)
        return synthesized

    async def _analyze_batch(
        self, batch: List[Dict[str, Any]], batch_idx: int
    ) -> RootCauseAnalysis:
        # Format batch like in original: "Test Case #X (ID: ...)\nScores: ..."
        formatted_results = []
        for idx, item in enumerate(batch):
            # Simplified formatting compared to original which supports detailed score breakdown
            # Here we just show input, output and main score
            result_text = f"""Test Case #{idx + 1} (Score: {item.get('score')})
Input: {item.get('input')}
Output: {item.get('output')}
"""
            formatted_results.append(result_text)

        formatted_batch = "\n\n" + ("=" * 80 + "\n\n").join(formatted_results)

        # Using exact prompt from original (inlined here for standalone)
        BATCH_ANALYSIS_PROMPT = """You are analyzing evaluation results to identify failure patterns.

TEST RESULTS:
```
{formatted_batch}
```

Important constraint: Base your analysis exclusively on the TEST RESULTS shown above. Do not infer, speculate, or hypothesize failure modes that are not directly evidenced in the provided results.

Think through the failures systematically:

1. IDENTIFY: List all distinct types of failures you observe in the test results
2. GROUP: Which failures share similar characteristics or root causes?
3. FREQUENCY: Which patterns appear multiple times across different test cases?
4. PRIORITIZE: Which failures are most critical to address?

Then, for each distinct failure pattern provide:
1. A clear, descriptive name that captures the essence of the failure
2. A comprehensive description of what is failing
3. The underlying root cause explaining why this failure occurs

Focus on patterns that appear multiple times. Be specific about what is failing and why.
Provide a list of failure modes, each with a name, description, and root cause."""

        prompt = BATCH_ANALYSIS_PROMPT.format(formatted_batch=formatted_batch)

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=RootCauseAnalysis,
                **self.model_parameters,
            )
            return RootCauseAnalysis.model_validate_json(
                response.choices[0].message.content
            )
        except Exception as e:
            logger.warning(f"Batch analysis failed: {e}")
            return RootCauseAnalysis(failure_modes=[])

    async def _synthesize(self, failure_modes: List[FailureMode]) -> List[FailureMode]:
        if not failure_modes:
            return []

        # Format batch summaries like original
        # Original takes batch_analyses list, here we just have a flat list of modes.
        # We'll adapt slightly to match the "BATCH ANALYSES" format expected by the prompt.

        failure_list = []
        for fm in failure_modes:
            failure_list.append(
                f"  - {fm.name}\n"
                f"    Description: {fm.description}\n"
                f"    Root Cause: {fm.root_cause}"
            )

        # We treat all failure modes as coming from "Batch 1" for simplicity as we don't track batches in this simplified synthesis
        # Or we can just list them.
        batch_summaries = f"""Batch 1 (Synthesis):
{chr(10).join(failure_list)}"""

        # Exact prompt
        SYNTHESIS_PROMPT = """You are synthesizing root cause analyses from multiple batches of evaluation results.

BATCH ANALYSES:
```
{batch_summaries}
```

Your task is to synthesize these batch-level analyses into a unified root cause analysis.

1. MERGE similar failure modes across batches:
   - If multiple batches identify the same or very similar failure pattern, combine them into one unified failure mode
   - Create a comprehensive description that captures the pattern across all relevant batches
   - Identify the core root cause

2. PRIORITIZE the most critical failure modes:
   - Focus on patterns that appear in multiple batches
   - Consider the severity and frequency of each failure
   - Eliminate one-off or minor issues unless they're particularly impactful

3. PROVIDE SYNTHESIS NOTES:
   - Briefly explain which batch-level patterns were merged and why
   - Note any cross-batch trends or patterns
   - Highlight the most critical areas for improvement

Provide:
1. A unified list of failure modes (name, description, root cause)
2. Synthesis notes explaining your analysis process and key findings"""

        prompt = SYNTHESIS_PROMPT.format(batch_summaries=batch_summaries)

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=RootCauseAnalysis,
                **self.model_parameters,
            )
            return RootCauseAnalysis.model_validate_json(
                response.choices[0].message.content
            ).failure_modes
        except Exception as e:
            logger.warning(f"Synthesis failed: {e}")
            return failure_modes[:5]  # Fallback

    # ... in StandaloneHierarchicalReflectiveOptimizer ...
    def _improve_prompt(
        self, messages: List[Dict[str, Any]], failure_mode: FailureMode
    ) -> List[Dict[str, Any]]:
        current_prompt_str = json.dumps(messages, indent=2)

        # Exact prompt from original
        IMPROVE_PROMPT_TEMPLATE = """You are an expert prompt engineer. You are given a prompt and a failure mode identified during evaluation.
Your task is to improve the prompt to address this failure mode.

CURRENT PROMPT:
```
{current_prompt}
```
FAILURE MODE TO ADDRESS:
 - Name: {failure_mode_name}
 - Description: {failure_mode_description}
 - Root Cause: {failure_mode_root_cause}

INSTRUCTIONS FOR IMPROVING THE PROMPT:

1. **Analyze First**: Carefully review the current prompt to understand what instructions already exist.

2. **Choose the Right Approach**:
   - If relevant instructions already exist but are unclear or incomplete, UPDATE and CLARIFY them in place
   - If the prompt is missing critical instructions needed to address this failure mode, ADD new targeted instructions
   - If existing instructions contradict what's needed, REPLACE them with corrected versions

3. **Be Surgical**: Make targeted changes that directly address the root cause. Don't add unnecessary instructions or rewrite the entire prompt.

4. **Maintain Structure**: Keep the same message structure (role and content format). Only modify the content where necessary.

5. **Do NOT Add Messages**: Do not add new messages to the prompt. Only modify existing messages. The number of messages in the prompt must remain exactly the same.

6. **Be Specific**: Ensure your changes provide concrete, actionable guidance that directly addresses the identified failure mode.

Do not remove any variables or placeholders from any prompt message. You can reposition them within the same message content if needed but never remove them.

Provide your reasoning for the changes you made, explaining WHY each change addresses the failure mode, and then provide the improved prompt."""

        prompt = IMPROVE_PROMPT_TEMPLATE.format(
            current_prompt=current_prompt_str,
            failure_mode_name=failure_mode.name,
            failure_mode_description=failure_mode.description,
            failure_mode_root_cause=failure_mode.root_cause,
        )

        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=ImprovedPrompt,
                **self.model_parameters,
            )
            improved = ImprovedPrompt.model_validate_json(
                response.choices[0].message.content
            )
            return improved.messages
        except Exception as e:
            logger.warning(f"Improvement failed: {e}")
            return messages


# -----------------------------------------------------------------------------
# 4. Standalone Hierarchical Reflective Optimizer
# -----------------------------------------------------------------------------


class StandaloneHierarchicalReflectiveOptimizer:
    def __init__(
        self,
        model: str = "gpt-4o",
        model_parameters: Optional[Dict[str, Any]] = None,
        batch_size: int = 5,
        max_iterations: int = 3,
        verbose: bool = True,
    ):
        self.model = model
        self.model_parameters = model_parameters or {}
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.verbose = verbose

        self.analyzer = StandaloneHierarchicalRootCauseAnalyzer(
            model=model,
            model_parameters=self.model_parameters,
            batch_size=batch_size,
            verbose=verbose,
        )

    def optimize(
        self,
        prompt_messages: List[Dict[str, Any]],
        train_dataset: List[Dict[str, Any]],
        metric: Callable[[Dict[str, Any], str], float],
        val_dataset: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize prompt by iteratively analyzing failures and improving.
        """
        current_messages = copy.deepcopy(prompt_messages)
        best_messages = copy.deepcopy(prompt_messages)
        best_score = -float("inf")
        history = []

        dataset = (
            val_dataset or train_dataset
        )  # evaluations on validation set if available

        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")

            # 1. Evaluate current prompt
            failures = []
            total_score = 0.0

            prompt = StandaloneChatPrompt(
                current_messages, self.model, self.model_parameters
            )

            for item in dataset:
                try:
                    msgs = prompt.get_messages(item)
                    response = litellm.completion(
                        model=self.model, messages=msgs, **self.model_parameters
                    )
                    output = response.choices[0].message.content
                    score = metric(item, output)
                    total_score += score

                    if (
                        score < 1.0
                    ):  # Assuming 1.0 is perfect, collect "failures" or low scores
                        failures.append(
                            {"input": item, "output": output, "score": score}
                        )
                except Exception as e:
                    logger.warning(f"Eval error: {e}")

            avg_score = total_score / len(dataset) if dataset else 0.0
            logger.info(f"Current score: {avg_score}")

            if avg_score > best_score:
                best_score = avg_score
                best_messages = copy.deepcopy(current_messages)

            history.append({"iteration": iteration, "score": avg_score})

            if not failures:
                logger.info("No failures found (perfect score?), stopping.")
                break

            # 2. Analyze Root Causes
            try:
                # Synchronous wrapper for async call
                failure_modes = asyncio.run(self.analyzer.analyze(failures))
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                break

            if not failure_modes:
                logger.info("No failure modes identified.")
                break

            # 3. Improve Prompt based on Top Failure Mode
            top_failure = failure_modes[0]  # Pick the first one
            logger.info(f"Addressing failure mode: {top_failure.name}")

            new_messages = self._improve_prompt(current_messages, top_failure)
            current_messages = new_messages

        return {
            "best_prompt": best_messages,
            "best_score": best_score,
            "history": history,
        }

    def _improve_prompt(
        self, messages: List[Dict[str, Any]], failure_mode: FailureMode
    ) -> List[Dict[str, Any]]:
        prompt = f"""
        You are an expert prompt engineer. The following prompt has a failure mode.
        
        Current Prompt:
        {json.dumps(messages, indent=2)}
        
        Failure Mode: {failure_mode.name}
        Description: {failure_mode.description}
        Root Cause: {failure_mode.root_cause}
        
        Please generate an improved version of the prompt that fixes this issue while maintaining performance on other aspects.
        Return the response as an ImprovedPrompt JSON object.
        """

        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=ImprovedPrompt,
                **self.model_parameters,
            )
            improved = ImprovedPrompt.model_validate_json(
                response.choices[0].message.content
            )
            return improved.messages
        except Exception as e:
            logger.warning(f"Improvement failed: {e}")
            return messages
