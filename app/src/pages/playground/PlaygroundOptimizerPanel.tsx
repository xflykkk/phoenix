import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router";
import { css } from "@emotion/react";

import {
  Button,
  Flex,
  Heading,
  Label,
  Radio,
  RadioGroup,
  TextArea,
  TextField,
  View,
} from "@phoenix/components";
import { usePlaygroundContext } from "@phoenix/contexts/PlaygroundContext";
import {
  selectPlaygroundInstance,
  selectPlaygroundInstanceMessages,
} from "@phoenix/store/playground/selectors";

type JobStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "unknown";

type PlaygroundOptimizationJob = {
  id: string;
  status: JobStatus;
  bestScore?: number | null;
  bestPrompt?: any;
  error?: string | null;
};

const panelCSS = css`
  border: 1px solid var(--ac-global-color-gray-200);
  border-radius: 8px;
  padding: 12px;
  background: var(--ac-global-color-gray-50);
`;

const preCSS = css`
  padding: 8px;
  border-radius: 6px;
  background: var(--ac-global-color-gray-100);
  max-height: 260px;
  overflow: auto;
  font-size: 12px;
`;

async function fetchGraphQL<T>(query: string, variables: Record<string, any>) {
  const resp = await fetch("/graphql", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, variables }),
    credentials: "include",
  });
  const json = await resp.json();
  if (json.errors) {
    throw new Error(json.errors.map((e: any) => e.message).join("; "));
  }
  return json.data as T;
}

export function PlaygroundOptimizerPanel() {
  const [searchParams] = useSearchParams();
  const datasetId = searchParams.get("datasetId");
  const [algorithm, setAlgorithm] = useState("metaprompt");
  const [metric, setMetric] = useState("exact_match");
  const [containsKeywords, setContainsKeywords] = useState("");
  const [judgePrompt, setJudgePrompt] = useState("");
  const [judgeModel, setJudgeModel] = useState("");
  const [minExamples, setMinExamples] = useState<number | undefined>(undefined);
  const [maxExamplesFs, setMaxExamplesFs] = useState<number | undefined>(undefined);
  const [parameterSpace, setParameterSpace] = useState("");
  const [maxTrials, setMaxTrials] = useState<number | undefined>(undefined);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<PlaygroundOptimizationJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const firstInstanceId = usePlaygroundContext(
    (state) => state.instances[0]?.id
  );
  const firstInstance = usePlaygroundContext(
    useMemo(
      () => (state) =>
        firstInstanceId != null
          ? selectPlaygroundInstance(firstInstanceId)(state)
          : undefined,
      [firstInstanceId]
    )
  );
  const promptMessages = usePlaygroundContext(
    useMemo(
      () => (state) =>
        firstInstanceId != null
          ? selectPlaygroundInstanceMessages(firstInstanceId)(state)
          : [],
      [firstInstanceId]
    )
  );

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    const interval = setInterval(async () => {
      try {
        const data = await fetchGraphQL<{
          playgroundOptimizationJob: PlaygroundOptimizationJob;
        }>(
          `
          query PlaygroundOptimizationJob($id: ID!) {
            playgroundOptimizationJob(id: $id) {
              id
              status
              bestScore
              bestPrompt
              error
            }
          }
        `,
          { id: jobId }
        );
        if (!cancelled) {
          setJob(data.playgroundOptimizationJob);
          if (
            data.playgroundOptimizationJob?.status === "completed" ||
            data.playgroundOptimizationJob?.status === "failed"
          ) {
            clearInterval(interval);
          }
        }
      } catch (err: any) {
        if (!cancelled) setError(err.message);
      }
    }, 2000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [jobId]);

  const handleRun = async () => {
    setError(null);
    if (!datasetId) {
      setError("请选择 dataset 后再运行优化器");
      return;
    }
    if (!promptMessages || promptMessages.length === 0) {
      setError("当前 Prompt 为空，无法优化");
      return;
    }
    setIsSubmitting(true);
    try {
      const data = await fetchGraphQL<{
        optimizePlaygroundPrompt: { jobId: string; status: string };
      }>(
        `
        mutation OptimizePlaygroundPrompt($input: OptimizePlaygroundPromptInput!) {
          optimizePlaygroundPrompt(input: $input) {
            jobId
            status
          }
        }
      `,
        {
          input: {
            datasetId,
            algorithm,
            promptMessages,
            tools: firstInstance?.tools ?? [],
            model: firstInstance?.model?.modelName ?? "gpt-4o-mini",
            modelKwargs: firstInstance?.model?.invocationParameters ?? {},
            metricName: metric,
            containsKeywords:
              metric === "contains" && containsKeywords
                ? containsKeywords.split(",").map((s) => s.trim())
                : undefined,
            llmJudgePrompt: metric === "llm_judge" ? judgePrompt : undefined,
            llmJudgeModel: metric === "llm_judge" ? judgeModel : undefined,
            maxTrials: maxTrials,
            minExamples: algorithm === "fewshot" ? minExamples : undefined,
            maxExamplesFs: algorithm === "fewshot" ? maxExamplesFs : undefined,
            parameterSpace:
              algorithm === "parameter" && parameterSpace
                ? JSON.parse(parameterSpace)
                : undefined,
          },
        }
      );
      setJobId(data.optimizePlaygroundPrompt.jobId);
      setJob({
        id: data.optimizePlaygroundPrompt.jobId,
        status: data.optimizePlaygroundPrompt.status as JobStatus,
      });
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <View marginTop="size-200" padding="size-200" backgroundColor="gray-50">
      <div css={panelCSS}>
        <Flex direction="column" gap="size-200">
          <Heading level={3}>Prompt Optimizer</Heading>
          <RadioGroup
            label="算法"
            value={algorithm}
            onChange={setAlgorithm}
            orientation="horizontal"
          >
            <Radio value="metaprompt">MetaPrompt</Radio>
            <Radio value="evolutionary">Evolutionary</Radio>
            <Radio value="hierarchical">Hierarchical</Radio>
            <Radio value="gepa">GEPA</Radio>
            <Radio value="fewshot">FewShotBayesian</Radio>
            <Radio value="parameter">ParameterOptimizer</Radio>
          </RadioGroup>
          <RadioGroup
            label="评估指标"
            value={metric}
            onChange={setMetric}
            orientation="horizontal"
          >
            <Radio value="exact_match">Exact Match</Radio>
            <Radio value="contains">Contains</Radio>
            <Radio value="llm_judge">LLM Judge</Radio>
          </RadioGroup>
          {metric === "contains" ? (
            <TextField
              label="关键词（逗号分隔）"
              value={containsKeywords}
              onChange={setContainsKeywords}
            />
          ) : null}
          {metric === "llm_judge" ? (
            <Flex direction="column" gap="size-100">
              <TextArea
                label="LLM 裁判提示词"
                value={judgePrompt}
                onChange={setJudgePrompt}
              />
              <TextField
                label="裁判模型（可选，默认沿用上方模型）"
                value={judgeModel}
                onChange={setJudgeModel}
              />
            </Flex>
          ) : null}
          {algorithm === "fewshot" ? (
            <Flex direction="row" gap="size-100">
              <TextField
                label="最少示例数"
                type="number"
                value={minExamples?.toString() ?? ""}
                onChange={(v) =>
                  setMinExamples(v ? parseInt(v, 10) : undefined)
                }
              />
              <TextField
                label="最多示例数"
                type="number"
                value={maxExamplesFs?.toString() ?? ""}
                onChange={(v) =>
                  setMaxExamplesFs(v ? parseInt(v, 10) : undefined)
                }
              />
            </Flex>
          ) : null}
          {algorithm === "parameter" ? (
            <TextArea
              label="参数空间 JSON（参见 ParameterSearchSpace schema）"
              value={parameterSpace}
              onChange={setParameterSpace}
              placeholder='{"temperature": {"type": "float", "min": 0, "max": 1}}'
            />
          ) : null}
          <TextField
            label="最大试验数（可选）"
            type="number"
            value={maxTrials?.toString() ?? ""}
            onChange={(val) =>
              setMaxTrials(val ? parseInt(val, 10) : undefined)
            }
          />
          <Button
            variant="primary"
            isDisabled={isSubmitting}
            onPress={handleRun}
          >
            {isSubmitting ? "提交中..." : "运行优化"}
          </Button>
          {error ? <Label color="negative">{error}</Label> : null}
          {job ? (
            <div>
              <Label>Job</Label>
              <div>ID: {job.id}</div>
              <div>状态: {job.status}</div>
              {job.bestScore != null ? <div>最佳得分: {job.bestScore}</div> : null}
              {job.error ? (
                <Label color="negative">{job.error}</Label>
              ) : null}
              {job.bestPrompt ? (
                <div>
                  <Label>最佳 Prompt</Label>
                  <pre css={preCSS}>
                    {JSON.stringify(job.bestPrompt, null, 2)}
                  </pre>
                </div>
              ) : null}
            </div>
          ) : null}
        </Flex>
      </div>
    </View>
  );
}
