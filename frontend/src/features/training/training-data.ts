function genCurve(len, start, end, noise = 0.05, shape = "exp") {
  const d = [];
  for (let i = 0; i < len; i++) {
    const t = i / (len - 1);
    let v;
    if (shape === "exp") v = start * Math.pow(end / start, t);
    else if (shape === "log") v = start + (end - start) * (1 - Math.exp(-3 * t));
    else v = start + (end - start) * t;
    v += (Math.random() - 0.5) * noise * Math.abs(v);
    d.push(Math.round(v * 1000) / 1000);
  }
  return d;
}

/* ═══════════════════════════════════════════
   SFT EXPERIMENTS (SFT + DPO)
   ═══════════════════════════════════════════ */

export const SFT_EXPERIMENTS = [
  {
    id: "exp-001",
    name: "CX Quality Improvement",
    hypothesis: "DPO on curated preference pairs will improve CX response quality without degrading factual accuracy",
    status: "active",
    createdBy: "A. Kovács",
    createdAt: "2026-03-20",
    tags: ["cx", "quality", "dpo", "sft"],
    baseModel: "meta-llama/Llama-3.1-8B-Instruct",
    runs: [
      {
        id: "ft-0040", name: "CX SFT Baseline (v3)", method: "SFT", status: "completed",
        dataset: "cx-convos-v4", datasetSamples: 2500,
        model: "llama-3.1-8b-cx-v3", baseModel: "meta-llama/Llama-3.1-8B-Instruct",
        compute: "local", gpu: "4× A100 80GB", gpuUtil: 88,
        startedAt: "2026-03-20 09:00", completedAt: "2026-03-20 14:30", duration: "5h 30m",
        epochs: { current: 3, total: 3 }, steps: { current: 4200, total: 4200 },
        lr: 2e-5, batchSize: 8, warmupSteps: 200, weightDecay: 0.01, gradAccum: 4,
        scheduler: "cosine", optimizer: "AdamW", lora: { rank: 64, alpha: 128, target: "q_proj,v_proj,k_proj,o_proj" },
        bestLoss: 0.445, finalLoss: 0.445,
        lossCurve: genCurve(80, 2.1, 0.445, 0.08, "exp"),
        lrCurve: genCurve(80, 0, 2e-5, 0.01, "log").map((v, i, a) => i > 60 ? a[60] * Math.cos((i - 60) / 20 * Math.PI / 2) : v),
        gradNormCurve: genCurve(80, 4.2, 0.8, 0.15, "exp"),
        checkpoints: [
          { step: 1400, loss: 0.82, path: "ckpt/ft-0040/step-1400" },
          { step: 2800, loss: 0.52, path: "ckpt/ft-0040/step-2800" },
          { step: 4200, loss: 0.445, path: "ckpt/ft-0040/step-4200", best: true },
        ],
        evalResults: { gsm8k: 78.1, mmlu: 67.8, "cx-quality": 4.1, mtbench: 7.4 },
        hubRef: "models/llama-3.1-8b-cx/v3",
        color: "#6B7585",
      },
      {
        id: "ft-0042", name: "CX SFT Round 4", method: "SFT", status: "running",
        dataset: "cx-convos-v5", datasetSamples: 2840,
        model: "llama-3.1-8b-cx-v4", baseModel: "models/llama-3.1-8b-cx/v3",
        compute: "local", gpu: "4× H100 80GB", gpuUtil: 92,
        startedAt: "2026-03-28 12:00", completedAt: null, duration: null,
        epochs: { current: 2, total: 3 }, steps: { current: 2814, total: 4260 },
        progress: 66,
        lr: 2e-5, batchSize: 8, warmupSteps: 200, weightDecay: 0.01, gradAccum: 4,
        scheduler: "cosine", optimizer: "AdamW", lora: { rank: 64, alpha: 128, target: "q_proj,v_proj,k_proj,o_proj" },
        bestLoss: 0.398, finalLoss: null,
        lossCurve: genCurve(53, 1.8, 0.398, 0.06, "exp"),
        lrCurve: genCurve(53, 0, 2e-5, 0.01, "log"),
        gradNormCurve: genCurve(53, 3.8, 0.6, 0.12, "exp"),
        checkpoints: [
          { step: 1420, loss: 0.72, path: "ckpt/ft-0042/step-1420" },
          { step: 2814, loss: 0.398, path: "ckpt/ft-0042/step-2814", best: true },
        ],
        evalResults: { gsm8k: 82.4, mmlu: 68.2, "cx-quality": 4.6, mtbench: 7.8 },
        hubRef: null,
        color: "#F59E0B",
      },
      {
        id: "ft-0043", name: "CX DPO Phase 1", method: "DPO", status: "queued",
        dataset: "cx-dpo-pairs-v1", datasetSamples: 1800,
        model: "llama-3.1-8b-cx-v4-dpo", baseModel: "models/llama-3.1-8b-cx/v4",
        compute: "local", gpu: "4× H100 80GB", gpuUtil: 0,
        startedAt: null, completedAt: null, duration: null,
        epochs: { current: 0, total: 2 }, steps: { current: 0, total: 2250 },
        lr: 5e-7, batchSize: 4, warmupSteps: 100, weightDecay: 0.01, gradAccum: 8,
        scheduler: "cosine", optimizer: "AdamW", lora: { rank: 32, alpha: 64, target: "q_proj,v_proj" },
        beta: 0.1, labelSmoothing: 0.0, refModel: "models/llama-3.1-8b-cx/v4",
        bestLoss: null, finalLoss: null,
        lossCurve: [], lrCurve: [], gradNormCurve: [],
        rewardMarginCurve: [],
        checkpoints: [],
        evalResults: {},
        hubRef: null,
        color: "#8B5CF6",
      },
    ],
  },
  {
    id: "exp-003",
    name: "Safety Classifier",
    hypothesis: "Fine-tuned LlamaGuard on expanded demographic labels will improve toxicity detection recall",
    status: "completed",
    createdBy: "A. Kovács",
    createdAt: "2026-03-18",
    tags: ["safety", "classification", "sft"],
    baseModel: "meta-llama/Llama-Guard-3-1B",
    runs: [
      {
        id: "ft-0039", name: "Guard SFT v1", method: "SFT", status: "completed",
        dataset: "safety-labels-v1", datasetSamples: 4800,
        model: "guard-3b-v1", baseModel: "meta-llama/Llama-Guard-3-1B",
        compute: "local", gpu: "1× A100 80GB", gpuUtil: 0,
        startedAt: "2026-03-18 10:00", completedAt: "2026-03-18 11:20", duration: "1h 20m",
        epochs: { current: 3, total: 3 }, steps: { current: 1800, total: 1800 },
        lr: 5e-5, batchSize: 16, warmupSteps: 100, weightDecay: 0.01, gradAccum: 2,
        scheduler: "cosine", optimizer: "AdamW", lora: { rank: 16, alpha: 32, target: "q_proj,v_proj" },
        bestLoss: 0.489, finalLoss: 0.489,
        lossCurve: genCurve(60, 1.4, 0.489, 0.06, "exp"),
        lrCurve: genCurve(60, 0, 5e-5, 0.01, "log"),
        gradNormCurve: genCurve(60, 2.8, 0.5, 0.1, "exp"),
        checkpoints: [{ step: 1800, loss: 0.489, path: "ckpt/ft-0039/step-1800", best: true }],
        evalResults: { toxigen: 94.2, truthfulqa: 55.8 },
        hubRef: "models/guard-3b/v1",
        color: "#6B7585",
      },
      {
        id: "ft-0040b", name: "Guard SFT v2", method: "SFT", status: "completed",
        dataset: "safety-labels-v2", datasetSamples: 6200,
        model: "guard-3b-v2", baseModel: "meta-llama/Llama-Guard-3-1B",
        compute: "local", gpu: "1× A100 80GB", gpuUtil: 0,
        startedAt: "2026-03-26 09:00", completedAt: "2026-03-26 10:40", duration: "1h 40m",
        epochs: { current: 3, total: 3 }, steps: { current: 2325, total: 2325 },
        lr: 5e-5, batchSize: 16, warmupSteps: 100, weightDecay: 0.01, gradAccum: 2,
        scheduler: "cosine", optimizer: "AdamW", lora: { rank: 16, alpha: 32, target: "q_proj,v_proj" },
        bestLoss: 0.312, finalLoss: 0.312,
        lossCurve: genCurve(70, 1.2, 0.312, 0.05, "exp"),
        lrCurve: genCurve(70, 0, 5e-5, 0.01, "log"),
        gradNormCurve: genCurve(70, 2.5, 0.4, 0.1, "exp"),
        checkpoints: [{ step: 2325, loss: 0.312, path: "ckpt/ft-0040b/step-2325", best: true }],
        evalResults: { toxigen: 97.8, truthfulqa: 61.2 },
        hubRef: "models/guard-3b/v2",
        color: "#22C55E",
      },
    ],
  },
];


/* ═══════════════════════════════════════════
   RL: AGENT FLOWS
   Callable that takes a Task → Episode.
   Defines agent logic, tools, multi-agent patterns.
   ═══════════════════════════════════════════ */

export const AGENT_FLOWS = [
  {
    id: "af-code-solver",
    name: "Code Solver",
    icon: "▷",
    description: "Single-agent flow. Reads a coding problem, generates a Python solution, optionally self-debugs once if first attempt fails. Produces one Trajectory with 1-2 Steps.",
    type: "single-agent",
    agents: ["solver"],
    stepsPerEpisode: "1–2",
    maxRolloutTokens: 4096,
    status: "active",
    usedByRuns: 2,
    source: "flows/code_solver.py",
    config: {
      maxTurns: 2,
      selfDebug: true,
      systemPrompt: "You are a Python coding assistant. Solve the problem step by step.",
      tools: ["code-executor"],
    },
  },
  {
    id: "af-cx-support",
    name: "CX Support Agent",
    icon: "⬡",
    description: "Single-agent flow simulating a customer support conversation. Agent receives a customer query as Task, uses tools (order-lookup, subscription-manager) and follows SKILLS.md guidelines.",
    type: "single-agent",
    agents: ["cx-agent"],
    stepsPerEpisode: "3–8",
    maxRolloutTokens: 2048,
    status: "active",
    usedByRuns: 1,
    source: "flows/cx_support.py",
    config: {
      maxTurns: 8,
      systemPrompt: "skills/cx-support-skills/v3.2.0",
      tools: ["order-lookup", "subscription-manager", "kb-search", "escalation-router"],
      mcpServers: ["internal-crm", "stripe-billing"],
    },
  },
  {
    id: "af-solver-judge",
    name: "Solver-Judge",
    icon: "◈",
    description: "Multi-agent flow. A solver agent generates an answer, then a judge agent critiques it. Produces two Trajectories per Episode (solver + judge). Used for debate-style RL.",
    type: "multi-agent",
    agents: ["solver", "judge"],
    stepsPerEpisode: "2–4",
    maxRolloutTokens: 4096,
    status: "active",
    usedByRuns: 0,
    source: "flows/solver_judge.py",
    config: {
      solverSystemPrompt: "Solve the problem carefully. Show your reasoning.",
      judgeSystemPrompt: "Evaluate the solution. Is it correct? Score 1-5.",
      judgeModel: "same",
      maxSolverTurns: 2,
      maxJudgeTurns: 1,
    },
  },
  {
    id: "af-tool-agent",
    name: "Tool Use Agent",
    icon: "⚙",
    description: "Single-agent flow for tool-use training. Agent receives a task requiring tool calls, selects tools, provides parameters, and produces a final answer from tool outputs.",
    type: "single-agent",
    agents: ["agent"],
    stepsPerEpisode: "2–6",
    maxRolloutTokens: 8192,
    status: "active",
    usedByRuns: 1,
    source: "flows/tool_agent.py",
    config: {
      maxTurns: 6,
      tools: ["sql-generator", "chart-renderer", "code-executor", "kb-search"],
      toolCallFormat: "openai",
      parallelToolCalls: false,
    },
  },
];


/* ═══════════════════════════════════════════
   RL: EVALUATORS
   Scores an Episode → EvalOutput (reward, is_correct, signals).
   ═══════════════════════════════════════════ */

export const EVALUATORS = [
  {
    id: "eval-exact-match",
    name: "Exact Match",
    icon: "✓",
    description: "String comparison between agent output and ground truth. Extracts answer from \\boxed{} if present. Binary reward: 1.0 if match, 0.0 otherwise.",
    rewardType: "binary",
    signals: ["accuracy"],
    builtIn: true,
    status: "active",
    usedByRuns: 1,
    config: {
      extractPattern: "\\boxed{}",
      normalize: "strip + lowercase",
      tolerance: null,
    },
  },
  {
    id: "eval-code-test",
    name: "Code Test Runner",
    icon: "▷",
    description: "Executes generated code against unit test suites in a sandboxed environment. Reward = pass@1 rate. Supports Python, JavaScript, Rust.",
    rewardType: "binary",
    signals: ["pass_rate", "execution_time", "syntax_valid"],
    builtIn: false,
    status: "active",
    usedByRuns: 2,
    config: {
      sandboxImage: "registry.internal/sandbox:latest",
      cpuLimit: "2000m",
      memLimit: "1Gi",
      netPolicy: "deny-all",
      testFramework: "pytest / jest / cargo test",
      timeout: "30s",
    },
  },
  {
    id: "eval-llm-judge-cx",
    name: "LLM Judge — CX Quality",
    icon: "◈",
    description: "Uses a judge LLM to score customer support responses. Returns composite reward from multiple signals: empathy, accuracy, resolution, tone. Score range 0-5.",
    rewardType: "continuous",
    signals: ["empathy", "accuracy", "resolution", "tone"],
    builtIn: false,
    status: "active",
    usedByRuns: 1,
    config: {
      judgeModel: "gpt-4o-2024-08-06",
      judgeEndpoint: "https://api.openai.com/v1",
      rubric: "cx-quality-rubric-v3",
      temperature: 0.0,
      aggregation: "weighted_mean (0.3, 0.3, 0.25, 0.15)",
    },
  },
  {
    id: "eval-tool-match",
    name: "Tool Call Matcher (BFCL)",
    icon: "⚙",
    description: "Function call matching evaluator. Validates tool selection, parameter accuracy, and call ordering against ground truth trajectories. Continuous reward 0-1.",
    rewardType: "continuous",
    signals: ["tool_selection", "param_accuracy", "call_order"],
    builtIn: true,
    status: "active",
    usedByRuns: 1,
    config: {
      matchMode: "fuzzy",
      orderWeight: 0.3,
      toolSelectionWeight: 0.4,
      paramAccuracyWeight: 0.3,
      partialCredit: true,
    },
  },
  {
    id: "eval-safety-gate",
    name: "Safety Classifier Gate",
    icon: "◈",
    description: "Binary reward from safety classifier. Penalizes responses that trigger toxicity, PII leakage, or policy violations. Used as a constraint signal during RL.",
    rewardType: "binary",
    signals: ["safe", "toxicity_score"],
    builtIn: false,
    status: "error",
    usedByRuns: 0,
    config: {
      classifierModel: "guard-3b",
      classifierEndpoint: "http://guard-3b.staging-da:8000/v1",
      threshold: 0.9,
      penaltyWeight: -1.0,
    },
  },
];


/* ═══════════════════════════════════════════
   RL EXPERIMENTS (GRPO / PPO)
   Each run references an AgentFlow + Evaluator.
   Training loop: AgentFlow.run(task) → Episode → Evaluator.evaluate() → reward → advantage → update
   ═══════════════════════════════════════════ */

export const RL_EXPERIMENTS = [
  {
    id: "exp-002",
    name: "Code RL Training",
    hypothesis: "GRPO with test-outcome rewards will improve code generation pass rate beyond SFT ceiling",
    status: "active",
    createdBy: "M. Chen",
    createdAt: "2026-03-22",
    tags: ["code", "grpo", "rl", "tool-use"],
    baseModel: "deepseek-ai/DeepSeek-R1",
    runs: [
      {
        id: "ft-0041", name: "Code RL Phase 2", method: "GRPO", status: "running",
        agentFlowId: "af-code-solver",
        evaluatorId: "eval-code-test",
        dataset: "code-trajectories-v2", datasetSamples: 1420,
        model: "deepseek-r1-code-rl", baseModel: "deepseek-ai/DeepSeek-R1",
        compute: "aws", gpu: "8× A100 80GB", gpuUtil: 78,
        startedAt: "2026-03-28 08:00", completedAt: null, duration: null,
        epochs: { current: 1, total: 5 }, steps: { current: 680, total: 3550 },
        progress: 19,
        lr: 1e-6, batchSize: 32, warmupSteps: 50, weightDecay: 0.01, gradAccum: 16,
        scheduler: "cosine", optimizer: "AdamW", lora: null,
        algorithm: { name: "grpo", groupSize: 8, klCoeff: 0.05, clipRange: 0.2 },
        workflow: { nParallelTasks: 64, retryLimit: 3 },
        bestLoss: 1.203, finalLoss: null,
        lossCurve: genCurve(34, 2.4, 1.203, 0.1, "exp"),
        lrCurve: genCurve(34, 0, 1e-6, 0.02, "log"),
        gradNormCurve: genCurve(34, 6.2, 1.8, 0.2, "exp"),
        rewardCurve: genCurve(34, 0.12, 0.68, 0.08, "log"),
        klDivCurve: genCurve(34, 0, 0.042, 0.15, "log"),
        policyLossCurve: genCurve(34, -0.02, -0.18, 0.1, "linear"),
        episodes: { total: 5440, completed: 1088 },
        avgStepsPerEpisode: 1.4,
        checkpoints: [
          { step: 350, loss: 1.65, reward: 0.38, path: "ckpt/ft-0041/step-350" },
          { step: 680, loss: 1.203, reward: 0.68, path: "ckpt/ft-0041/step-680", best: true },
        ],
        evalResults: {},
        hubRef: null,
        color: "#3B82F6",
      },
    ],
  },
  {
    id: "exp-004",
    name: "CX Agent RL",
    hypothesis: "GRPO with LLM judge rewards will improve CX agent quality beyond SFT+DPO, especially on empathy and resolution scores",
    status: "active",
    createdBy: "A. Kovács",
    createdAt: "2026-03-26",
    tags: ["cx", "grpo", "rl", "judge"],
    baseModel: "models/llama-3.1-8b-cx/v4",
    runs: [
      {
        id: "ft-0044", name: "CX GRPO Judge", method: "GRPO", status: "completed",
        agentFlowId: "af-cx-support",
        evaluatorId: "eval-llm-judge-cx",
        dataset: "cx-convos-v5", datasetSamples: 2840,
        model: "llama-3.1-8b-cx-v4-rl", baseModel: "models/llama-3.1-8b-cx/v4",
        compute: "local", gpu: "4× H100 80GB", gpuUtil: 0,
        startedAt: "2026-03-26 14:00", completedAt: "2026-03-27 02:00", duration: "12h",
        epochs: { current: 3, total: 3 }, steps: { current: 4260, total: 4260 },
        lr: 5e-7, batchSize: 16, warmupSteps: 100, weightDecay: 0.01, gradAccum: 16,
        scheduler: "cosine", optimizer: "AdamW", lora: { rank: 64, alpha: 128, target: "q_proj,v_proj,k_proj,o_proj" },
        algorithm: { name: "grpo", groupSize: 4, klCoeff: 0.04, clipRange: 0.15 },
        workflow: { nParallelTasks: 32, retryLimit: 3 },
        bestLoss: 0.82, finalLoss: 0.82,
        lossCurve: genCurve(80, 1.6, 0.82, 0.06, "exp"),
        lrCurve: genCurve(80, 0, 5e-7, 0.01, "log"),
        gradNormCurve: genCurve(80, 5.0, 1.2, 0.15, "exp"),
        rewardCurve: genCurve(80, 2.8, 4.3, 0.05, "log"),
        klDivCurve: genCurve(80, 0, 0.038, 0.12, "log"),
        policyLossCurve: genCurve(80, -0.01, -0.14, 0.08, "linear"),
        episodes: { total: 11360, completed: 11360 },
        avgStepsPerEpisode: 5.2,
        checkpoints: [
          { step: 1420, loss: 1.1, reward: 3.4, path: "ckpt/ft-0044/step-1420" },
          { step: 2840, loss: 0.92, reward: 3.9, path: "ckpt/ft-0044/step-2840" },
          { step: 4260, loss: 0.82, reward: 4.3, path: "ckpt/ft-0044/step-4260", best: true },
        ],
        evalResults: { "cx-quality": 4.8, mtbench: 8.1, toxigen: 97.0 },
        hubRef: null,
        color: "#F59E0B",
      },
      {
        id: "ft-0045", name: "CX Tool Use RL", method: "GRPO", status: "queued",
        agentFlowId: "af-tool-agent",
        evaluatorId: "eval-tool-match",
        dataset: "cx-convos-v5", datasetSamples: 2840,
        model: "llama-3.1-8b-cx-v4-rl2", baseModel: "models/llama-3.1-8b-cx/v4",
        compute: "aws", gpu: "8× A100 80GB", gpuUtil: 0,
        startedAt: null, completedAt: null, duration: null,
        epochs: { current: 0, total: 3 }, steps: { current: 0, total: 4260 },
        lr: 3e-7, batchSize: 16, warmupSteps: 100, weightDecay: 0.01, gradAccum: 16,
        scheduler: "cosine", optimizer: "AdamW", lora: { rank: 64, alpha: 128, target: "q_proj,v_proj,k_proj,o_proj" },
        algorithm: { name: "grpo", groupSize: 4, klCoeff: 0.03, clipRange: 0.15 },
        workflow: { nParallelTasks: 32, retryLimit: 3 },
        bestLoss: null, finalLoss: null,
        lossCurve: [], lrCurve: [], gradNormCurve: [],
        rewardCurve: [], klDivCurve: [], policyLossCurve: [],
        episodes: { total: 0, completed: 0 },
        avgStepsPerEpisode: 0,
        checkpoints: [],
        evalResults: {},
        hubRef: null,
        color: "#8B5CF6",
      },
    ],
  },
];

// Legacy combined export
export const EXPERIMENTS = [...SFT_EXPERIMENTS, ...RL_EXPERIMENTS];
