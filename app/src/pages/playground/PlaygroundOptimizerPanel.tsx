import { useEffect, useState } from "react";
import { useSearchParams } from "react-router";
import { css } from "@emotion/react";

import {
  Button,
  Flex,
  Heading,
  Label,
  Radio,
  RadioGroup,
  Text,
  TextArea,
  TextField,
  Tooltip,
  TooltipTrigger,
  View,
} from "@phoenix/components";
import {
  usePlaygroundContext,
  usePlaygroundStore,
} from "@phoenix/contexts/PlaygroundContext";
import { useCredentialsContext } from "@phoenix/contexts/CredentialsContext";
import { getBaseChatCompletionInputForInstance } from "./playgroundUtils";

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
  const [nSamples, setNSamples] = useState<number | undefined>(undefined);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<PlaygroundOptimizationJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isCanceling, setIsCanceling] = useState(false);

  const algoHelp: Record<string, string> = {
    metaprompt: "LLM 自反思改写，多轮生成候选，适合通用提示快速迭代。",
    evolutionary: "遗传算法多代演化，耗时高但能探索大空间。",
    hierarchical: "层级反思+根因分析，聚焦失败模式迭代。",
    gepa: "遗传+帕累托搜索，可兼顾多目标（当前主要以单指标得分）。",
    fewshot: "自动挑选/排序 few-shot 示例，贝叶斯优化示例组合。",
    parameter: "仅调模型参数（温度/Top-p 等），不改提示文本。",
  };
  const algoTips: Record<string, string[]> = {
    metaprompt: [
      "max_trials 8-12 常用；适合通用提示改写。",
      "LLM Judge 能给出更平滑的质量信号。",
    ],
    evolutionary: [
      "max_trials 10-20；数据大时设置 n_samples 先粗筛。",
      "Contains/LLM Judge 作为“适应度”，关键词要明确。",
    ],
    hierarchical: [
      "max_trials 4-8；适合定位失败模式。",
      "优先用 Exact/Contains，问题复杂时用 LLM Judge。",
    ],
    gepa: [
      "max_trials 8-15；兼顾长度/得分场景可用。",
      "大数据集时建议设置 n_samples。",
    ],
    fewshot: [
      "最少/最多示例数建议 1-4；数据多时调低 n_samples。",
      "指标最好选 Contains/LLM Judge 以容忍表述差异。",
    ],
    parameter: [
      "参数空间先小范围，如温度 0-0.8、top_p 0.7-1.0。",
      "适合已较优提示，仅需微调生成稳定性。",
    ],
  };

  const playgroundStore = usePlaygroundStore();
  const credentialsState = useCredentialsContext((state) => state);
  const firstInstanceId = usePlaygroundContext(
    (state) => state.instances[0]?.id
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
            data.playgroundOptimizationJob?.status === "failed" ||
            data.playgroundOptimizationJob?.status === "canceled"
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
    const instanceId = firstInstanceId;
    setIsSubmitting(true);
    try {
      const baseInput =
        instanceId != null
          ? getBaseChatCompletionInputForInstance({
              playgroundStore,
              instanceId,
              credentials: credentialsState,
            })
          : null;
      if (!baseInput || !baseInput.messages || baseInput.messages.length === 0) {
        throw new Error("当前 Prompt 为空，无法优化");
      }
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
            promptMessages: baseInput.messages,
            tools: baseInput.tools ?? [],
            model: baseInput.model,
            invocationParameters: baseInput.invocationParameters ?? [],
            credentials: baseInput.credentials ?? [],
            metricName: metric,
            containsKeywords:
              metric === "contains" && containsKeywords
                ? containsKeywords.split(",").map((s) => s.trim())
                : undefined,
            llmJudgePrompt: metric === "llm_judge" ? judgePrompt : undefined,
            llmJudgeModel: metric === "llm_judge" ? judgeModel : undefined,
            maxTrials: maxTrials,
            nSamples: nSamples,
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

  const handleCancel = async () => {
    if (!jobId) return;
    setIsCanceling(true);
    setError(null);
    try {
      const data = await fetchGraphQL<{
        cancelPlaygroundOptimizationJob: { jobId: string; status: string };
      }>(
        `
        mutation CancelPlaygroundOptimizationJob($id: ID!) {
          cancelPlaygroundOptimizationJob(id: $id) {
            jobId
            status
          }
        }
      `,
        { id: jobId }
      );
      setJob((prev) =>
        prev
          ? { ...prev, status: data.cancelPlaygroundOptimizationJob.status as JobStatus }
          : prev
      );
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsCanceling(false);
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
            <TooltipTrigger delay={0} closeDelay={0}>
              <span>
                <Radio value="metaprompt">MetaPrompt</Radio>
              </span>
              <Tooltip>LLM 反思+生成改写，适合通用提示迭代。</Tooltip>
            </TooltipTrigger>
            <TooltipTrigger delay={0} closeDelay={0}>
              <span>
                <Radio value="evolutionary">Evolutionary</Radio>
              </span>
              <Tooltip>遗传算法多代迭代，尝试大量变体，耗时/成本高。</Tooltip>
            </TooltipTrigger>
            <TooltipTrigger delay={0} closeDelay={0}>
              <span>
                <Radio value="hierarchical">Hierarchical</Radio>
              </span>
              <Tooltip>层级反思+根因分析，适合找失败模式并迭代。</Tooltip>
            </TooltipTrigger>
            <TooltipTrigger delay={0} closeDelay={0}>
              <span>
                <Radio value="gepa">GEPA</Radio>
              </span>
              <Tooltip>遗传+帕累托搜索，兼顾多目标（如长度/得分）。</Tooltip>
            </TooltipTrigger>
            <TooltipTrigger delay={0} closeDelay={0}>
              <span>
                <Radio value="fewshot">FewShotBayesian</Radio>
              </span>
              <Tooltip>为提示自动选取 few-shot 示例，贝叶斯优化示例组合。</Tooltip>
            </TooltipTrigger>
            <TooltipTrigger delay={0} closeDelay={0}>
              <span>
                <Radio value="parameter">ParameterOptimizer</Radio>
              </span>
              <Tooltip>只调模型参数（温度/Top-p 等），不改提示文本。</Tooltip>
            </TooltipTrigger>
          </RadioGroup>
          <Text size="S" color="gray-700">
            {algoHelp[algorithm] ?? ""}
          </Text>
          <View paddingTop="size-50">
            <Text size="S" color="gray-700">
              最佳实践：
            </Text>
            {(algoTips[algorithm] ?? []).map((tip, idx) => (
              <Text key={idx} size="S" color="gray-700">
                • {tip}
              </Text>
            ))}
          </View>
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
            <View paddingTop="size-50">
              <TextField
                label="关键词（逗号分隔）"
                placeholder="例如：correct,pass,success"
                value={containsKeywords}
                onChange={setContainsKeywords}
              />
            </View>
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
            <View paddingTop="size-50">
              <Flex direction="row" gap="size-100" wrap>
                <TextField
                  label="最少示例数"
                  type="number"
                  value={minExamples?.toString() ?? ""}
                  onChange={(v) =>
                    setMinExamples(v ? parseInt(v, 10) : undefined)
                  }
                  placeholder="如 1"
                />
                <TextField
                  label="最多示例数"
                  type="number"
                  value={maxExamplesFs?.toString() ?? ""}
                  onChange={(v) =>
                    setMaxExamplesFs(v ? parseInt(v, 10) : undefined)
                  }
                  placeholder="如 4"
                />
              </Flex>
            </View>
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
            label="最大试验数/迭代（可选）"
            type="number"
            value={maxTrials?.toString() ?? ""}
            onChange={(val) =>
              setMaxTrials(val ? parseInt(val, 10) : undefined)
            }
          />
          <TextField
            label="抽样样本数 n_samples（可选，用于加速大数据集）"
            type="number"
            value={nSamples?.toString() ?? ""}
            onChange={(val) =>
              setNSamples(val ? parseInt(val, 10) : undefined)
            }
            placeholder="为空则使用全部样本"
          />
          <Flex gap="size-100">
            <Button
              variant="primary"
              isDisabled={isSubmitting}
              onPress={handleRun}
            >
              {isSubmitting ? "提交中..." : "运行优化"}
            </Button>
            {job && ["running", "pending"].includes(job.status) ? (
              <Button
                variant="secondary"
                isDisabled={isCanceling}
                onPress={handleCancel}
              >
                {isCanceling ? "停止中..." : "停止"}
              </Button>
            ) : null}
          </Flex>
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
