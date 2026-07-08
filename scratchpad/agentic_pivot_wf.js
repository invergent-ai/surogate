export const meta = {
  name: 'agentic-pivot-decision',
  description: 'Decide whether to pivot the Conductor run from single-step to paper-faithful mixed agentic training',
  phases: [
    { title: 'Ground', detail: 'establish hard constraints: cost, integration, plateau, paper recipe' },
    { title: 'Design', detail: '3 independent training-plan proposals from different priors' },
    { title: 'Judge', detail: '3 scorers rate the proposals adversarially' },
    { title: 'Synthesize', detail: 'graft the winner + best-of into one plan' },
  ],
}

const SC = `CONTEXT — Fugu-Ultra replication (repo /home/densemax/work/flavius/surogate), 2026-07-08.
We train a Conductor orchestrator: raw Qwen3-8B + LoRA, GRPO, paper-exact (output = three lists model_id/subtasks/access_list; reward 0 unparseable / 0.5 valid-wrong / 1.0 correct; no KL; <=5 workflow steps; temp 1.0). It designs agentic workflows over a worker pool GLM-5.2 + Gemini-3.1-Pro + Opus-4.8 + GPT-5.5(via Yunwu). In TRAINING, workers are HANDICAPPED to 4096 tokens / minimal reasoning effort (keeps tasks hard + cheap; matches paper section 4.3 constrained setting).

LIVE RUN NOW: trainer at step ~143 of a 200-step ceiling. Diet = 4 SINGLE-CALL lanes (math / code / reason-RLPR / repair). Held-out eval every 10 steps on a FIXED 60-task single-call set. Bar = best solo worker 0.883; oracle 0.983. Recent held-out overall: step110=0.850, 120=0.900 (first time OVER the bar, +0.017), 130=0.867, 140=0.850. Code per-cap stuck ~0.767-0.800. => STRADDLING the bar, arguably plateaued ~0.86.

RE-READING THE FUGU REPORT (Table 1): Fugu-Ultra's BIG wins (+4 to +5 pts over best worker) are ALL CODING: LiveCodeBench +4.7 (single-call), SWE-Bench-Pro +4.5 (agentic), Terminal-Bench +3.9 (agentic). Ties/losses are niche single-answer reasoning + long-context. The paper trains a MIXTURE of single-step public data AND expert-designed END-TO-END multi-turn agentic environments (Claude Code / Codex / OpenCode trajectories) in ONE GRPO run, no KL. Paper evals workers at FULL reasoning effort (not handicapped).

USER'S CHALLENGE (the decision): "Why do another 50 steps of single-step training? Shouldn't we adhere to the paper and mix in ACTUAL multi-turn agentic tasks now?" The user thinks pure single-step is low-value and we should pivot the remaining budget to the paper-faithful mixed diet with real agentic coding.

ASSETS (all built/proven):
- Repair lane: DEPLOYED + training NOW. Data-only single-turn proxy of paper 'build-and-debug' (a failed solo-builder attempt embedded in prompt; conductor must orchestrate a fix; grader unchanged). 40 tasks.
- MultiTurnEnv scratchpad/fugu_ultra_multiturn.py: 2-turn conductor env (plan -> execute -> feedback -> replan), single-call workers, both paper 3.2.2 memory scopes. BUILT + 5/5 offline logic tests + seq-len retired. NOT wired to orchestrator, NOT live-smoke-tested on a failing task.
- Real agentic harnesses ultra/ultra/harness/opencode.py (opencode + claude_code + codex): tool-using workers inside SWE-smith Docker containers. PROVEN live end-to-end (2 bugs x both harnesses: real edits + tests + official grade). NOT wired as a GRPO training lane.
- Cached-execution Stage-2c design: pay per unique (task, canonical-plan) not per rollout; ~10-20 distinct plans per 16-rollout group at temp 1.0, cache hit-rate rises as policy converges; 16x16 geometry; 10-40 steps; continue from best checkpoint. Designed, never run.
- Cost data: director/manifests/fugu_clean_v1/derisk_swesmith_routability.jsonl (200 real agentic runs, fields include cost, elapsed_s, arm, reward). NOTE Yunwu premium workers report $0 locally (blind spot) — estimate their true cost via tokens.

CONSTRAINTS: GPT MUST stay on Yunwu (never OpenRouter). Spend + restarts are user-gated. Do NOT corrupt the 0/0.5/1.0 reward. Kill processes by PID only. Trainer is turn-agnostic (per-token loss mask) so a 2-turn rollout is 1 RolloutOutput => NO trainer change needed for multi-turn.`

const GROUND_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['headline', 'key_numbers', 'findings', 'confidence'],
  properties: {
    headline: { type: 'string', description: 'one-line verdict answering the question' },
    key_numbers: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['label', 'value'], properties: { label: { type: 'string' }, value: { type: 'string' } } } },
    findings: { type: 'array', items: { type: 'string' } },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    caveats: { type: 'array', items: { type: 'string' } },
  },
}

phase('Ground')
const [g_cost, g_integ, g_plateau, g_paper] = await parallel([
  () => agent(`${SC}

YOUR JOB (Ground-1, COST/FEASIBILITY — the gating number): Read director/manifests/fugu_clean_v1/derisk_swesmith_routability.jsonl. Compute the real per-agentic-run distribution of cost(USD) and elapsed_s, broken out by arm. Then, using the cached-execution design (pay per unique (task,plan); assume ~12 distinct plans explored per 16-rollout group early, dropping as policy converges), ESTIMATE per-GRPO-step and total dollars + wall-clock for an agentic coding lane at (a) 8 tasks x 16 rollouts and (b) 16 x 16, for a 10-step and a 30-step run. State every assumption. Give a crisp verdict: is agentic-in-the-loop affordable within a few hundred to ~1-2k USD and days-not-weeks of wall-clock? What is the BINDING constraint — wall-clock or dollars? Estimate true premium-worker cost via tokens given the Yunwu 0-dollar blind spot.`,
    { label: 'ground:cost', phase: 'Ground', schema: GROUND_SCHEMA }),
  () => agent(`${SC}

YOUR JOB (Ground-2, INTEGRATION AUDIT): Read ultra/ultra/harness/opencode.py, director/manifests/fugu_clean_v1/grpo_pilot_train/orch_paper.yaml, and scratchpad file /tmp/claude-1000/-home-densemax-work-flavius-surogate/1636be7a-c882-47c0-8ed5-6ece7392008f/scratchpad/fugu_ultra_multiturn.py. Determine EXACTLY what integration work is required for: PATH A = add a REAL agentic coding lane (opencode tool-using workers in containers) as a 5th lane in the running orchestrator; PATH B = wire the conductor-level 2-turn MultiTurnEnv as a lane. For EACH path list concrete code/config gaps, the risk surface, and a rough eng-effort estimate (hours). Which path is closer to shippable? Is the worker_harnesses override mechanism sufficient for Path A, or is deeper plumbing needed?`,
    { label: 'ground:integration', phase: 'Ground', schema: GROUND_SCHEMA }),
  () => agent(`${SC}

YOUR JOB (Ground-3, PLATEAU CHECK): Read output/fugu_ultra_paper/heldout_trend.log and skim MISSION.md for the same-task conversion / eviction dynamics. Assess whether the single-call held-out series (rows ~90-140) is genuinely PLATEAUED vs still improving, with attention to the code lane specifically. Then answer: what would 50 more PURE single-step steps plausibly buy (any evidence of ongoing conversions, or is the frontier exhausted)? Crisp verdict: is continuing pure single-step to step 200 LOW / MEDIUM / HIGH value? On an n=60 held-out set, quantify the noise band so 'plateau' is defensible not eyeballed.`,
    { label: 'ground:plateau', phase: 'Ground', schema: GROUND_SCHEMA }),
  () => agent(`${SC}

YOUR JOB (Ground-4, PAPER FIDELITY): Using the report details in MISSION.md plus the context above, specify precisely: (a) what END-TO-END tasks the paper mixed in and in what rough proportion (as best determinable — separate STATED from INFERRED); (b) does the paper CONTINUE Ultra from a single-step checkpoint or train the mixture from scratch; (c) the paper's worker EFFORT setting during training (handicap vs full) and whether our handicap is a faithful choice or a deviation; (d) any recipe detail bearing on 'mix agentic in NOW vs finish single-step first'. Be explicit about what is stated vs inferred.`,
    { label: 'ground:paper', phase: 'Ground', schema: GROUND_SCHEMA }),
])

const groundBlob = JSON.stringify({ cost: g_cost, integration: g_integ, plateau: g_plateau, paper: g_paper }, null, 1)

const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['name', 'one_line', 'steps', 'current_run_disposition', 'budget_usd', 'time_to_first_signal', 'biggest_risk'],
  properties: {
    name: { type: 'string' },
    one_line: { type: 'string' },
    steps: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['n', 'action', 'detail'], properties: {
        n: { type: 'integer' }, action: { type: 'string' }, detail: { type: 'string' } } } },
    current_run_disposition: { type: 'string', description: 'kill / checkpoint-and-repurpose / continue-in-parallel, with why' },
    budget_usd: { type: 'string' },
    time_to_first_signal: { type: 'string' },
    biggest_risk: { type: 'string' },
    paper_adherence_note: { type: 'string' },
  },
}

phase('Design')
const designs = await parallel([
  () => agent(`${SC}

GROUNDING FACTS (established by investigation — treat as authoritative):
${groundBlob}

YOUR JOB (Design-1, PAPER-MAXIMALIST): Propose the plan that adheres MOST closely to the Fugu paper — mix REAL agentic end-to-end coding tasks into the training diet, continuing from the current best checkpoint, per the paper one-mixed-run recipe. Give exact lane structure, env_ratios, geometry, step budget, and the smoke-test gate BEFORE going live. Respect the grounding cost/integration facts. Be concrete about what happens to the running step-143 process.`,
    { label: 'design:paper-max', phase: 'Design', schema: DESIGN_SCHEMA }),
  () => agent(`${SC}

GROUNDING FACTS (authoritative):
${groundBlob}

YOUR JOB (Design-2, COST/RISK-MINIMALIST): Propose the plan that captures the paper BENEFIT (training the build-and-debug / agentic-coding skill and the handicap-lift question) at MINIMUM cost and risk. Prefer the cheap proxies already available (repair lane live + conductor 2-turn MultiTurnEnv, single-call workers) and defer expensive container-agentic until a measured probe justifies it. Give the exact sequence, gates, and the smallest spend that still moves us toward the coding win.`,
    { label: 'design:cost-min', phase: 'Design', schema: DESIGN_SCHEMA }),
  () => agent(`${SC}

GROUNDING FACTS (authoritative):
${groundBlob}

YOUR JOB (Design-3, DISCRIMINATING-EXPERIMENT / VERDICT-FIRST): Propose the sequence that, with the LEAST wasted motion, answers the real question 'does mixing agentic tasks in actually help OUR setup, and is our headroom being hidden by the worker handicap?'. Design the SMALLEST experiment that discriminates between hypotheses (single-step-is-done vs agentic-adds-real-headroom vs handicap-hides-headroom). Must include the full-strength worker eval as a first-class experiment. State exactly what result would trigger a full agentic training commit vs abandon.`,
    { label: 'design:discriminating', phase: 'Design', schema: DESIGN_SCHEMA }),
])

const designBlob = JSON.stringify(designs.filter(Boolean), null, 1)

const JUDGE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['scores', 'winner', 'rationale'],
  properties: {
    scores: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['proposal', 'paper_adherence', 'expected_headroom', 'cost_efficiency', 'risk_low', 'time_to_signal', 'total'],
      properties: {
        proposal: { type: 'string' },
        paper_adherence: { type: 'integer' }, expected_headroom: { type: 'integer' },
        cost_efficiency: { type: 'integer' }, risk_low: { type: 'integer' },
        time_to_signal: { type: 'integer' }, total: { type: 'integer' } } } },
    winner: { type: 'string' },
    rationale: { type: 'string' },
    best_ideas_from_losers: { type: 'array', items: { type: 'string' } },
    fatal_flaws: { type: 'array', items: { type: 'string' } },
  },
}

phase('Judge')
const judges = await parallel(['pragmatist (cares most about time-to-real-signal and not burning money)',
  'paper-purist (cares most about faithful replication of the Fugu recipe)',
  'risk-officer (cares most about not damaging the live run or corrupting the reward, and about defensible verdicts)']
  .map((lens, i) => () => agent(`${SC}

The three candidate plans:
${designBlob}

YOU ARE JUDGE ${i + 1}, lens = ${lens}. Score each proposal 1-10 on each dimension (paper_adherence, expected_headroom, cost_efficiency, risk_low where 10=lowest-risk, time_to_signal where 10=fastest useful signal); total = sum. Pick a winner. List the best ideas worth grafting from the non-winners, and any FATAL flaws in any proposal. Be adversarial and specific — do not be diplomatic.`,
    { label: `judge:${i + 1}`, phase: 'Judge', schema: JUDGE_SCHEMA, effort: 'high' })))

const judgeBlob = JSON.stringify(judges.filter(Boolean), null, 1)

const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['recommendation_one_line', 'plan', 'current_run_disposition', 'total_budget_usd', 'first_action', 'open_decisions_for_user'],
  properties: {
    recommendation_one_line: { type: 'string' },
    plan: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['n', 'action', 'detail', 'gate'], properties: {
        n: { type: 'integer' }, action: { type: 'string' }, detail: { type: 'string' },
        gate: { type: 'string', description: 'what must be true to proceed / what result advances vs kills' } } } },
    current_run_disposition: { type: 'string' },
    total_budget_usd: { type: 'string' },
    first_action: { type: 'string', description: 'the single concrete next step' },
    honest_tradeoff: { type: 'string' },
    open_decisions_for_user: { type: 'array', items: { type: 'string' } },
  },
}

phase('Synthesize')
const plan = await agent(`${SC}

GROUNDING FACTS:
${groundBlob}

THE THREE CANDIDATE PLANS:
${designBlob}

THREE ADVERSARIAL JUDGE VERDICTS:
${judgeBlob}

YOUR JOB: Synthesize ONE recommended plan. Take the winning proposal as the spine but GRAFT the best ideas the judges flagged from the others, and AVOID every fatal flaw the judges identified. The plan must: (1) give a decisive answer to the user challenge (is 50 more single-step steps worth it, and should we mix agentic in now); (2) be concrete about the live step-143 run disposition; (3) sequence experiments cheapest-discriminating-first; (4) include the full-strength eval; (5) name the ONE first action and total budget. Be honest about the central trade-off. List only genuine user decisions (spend/restart gates).`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA, effort: 'high' })

return { ground: { cost: g_cost, integration: g_integ, plateau: g_plateau, paper: g_paper }, designs: designs.filter(Boolean), judges: judges.filter(Boolean), plan }
