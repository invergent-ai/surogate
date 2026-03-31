// ── Types ─────────────────────────────────────────────────────

export interface DatasetSampleMetadata {
  turns: number;
  tokens: number;
  sentiment: string;
  tags: string[];
  quality_score: number;
  reward?: number;
}

export interface DatasetSample {
  id: string;
  instruction: string;
  response: string;
  metadata: DatasetSampleMetadata;
}

export interface DatasetPipelineStep {
  id: string;
  step: string;
  type: "source" | "transform" | "output";
  detail: string;
  status: "completed" | "running" | "pending";
  icon: string;
}

export interface DatasetVersion {
  version: string;
  date: string;
  samples: number;
  change: string;
  author: string;
}

export interface DatasetUsedBy {
  name: string;
  type: string;
  status: string;
}

export interface Dataset {
  id: string;
  name: string;
  displayName: string;
  description: string;
  format: string;
  source: string;
  status: string;
  samples: number;
  tokens: string;
  size: string;
  createdAt: string;
  updatedAt: string;
  createdBy: string;
  tags: string[];
  color: string;
  hubRef: string | null;
  published: boolean;
  usedBy: DatasetUsedBy[];
  versions: DatasetVersion[];
  stats: {
    avgTurns: number;
    avgTokensPerSample: number;
    maxTokens: number;
    minTokens: number;
    sentimentDist: { positive: number; neutral: number; negative: number };
    topTags: { tag: string; count: number }[];
    tokenHistogram: number[];
  };
  pipeline: DatasetPipelineStep[];
  sampleData: DatasetSample[];
}

// ── Styling constants ─────────────────────────────────────────

export const FORMAT_STYLES: Record<string, { bg: string; fg: string; border: string }> = {
  SFT:  { bg: "bg-green-500/10", fg: "text-green-500", border: "border-green-500/20" },
  DPO:  { bg: "bg-violet-500/10", fg: "text-violet-500", border: "border-violet-500/20" },
  GRPO: { bg: "bg-blue-500/10", fg: "text-blue-500", border: "border-blue-500/20" },
  RLHF: { bg: "bg-amber-500/10", fg: "text-amber-500", border: "border-amber-500/20" },
};

export const SOURCE_LABELS: Record<string, string> = {
  conversations: "From Conversations",
  manual: "Manual / Uploaded",
  synthetic: "Synthetic / Generated",
};

export const STATUS_MAP: Record<string, "running" | "serving" | "completed" | "deploying" | "error" | "stopped"> = {
  ready: "serving",
  building: "deploying",
  error: "error",
  running: "running",
  completed: "completed",
  pending: "stopped",
};

export function toStatus(raw: string) {
  return STATUS_MAP[raw] ?? "stopped";
}

export const PIPELINE_TYPE_STYLES: Record<string, { bg: string; fg: string }> = {
  source:    { bg: "bg-blue-500/10", fg: "text-blue-500" },
  transform: { bg: "bg-violet-500/10", fg: "text-violet-500" },
  output:    { bg: "bg-green-500/10", fg: "text-green-500" },
};

// ── Demo data ─────────────────────────────────────────────────

export const DATASETS: Dataset[] = [
  {
    id: "cx-convos-v5",
    name: "cx-convos-v5",
    displayName: "CX Conversations v5",
    description: "Curated customer support conversations from cx-support-v3 production traffic. Includes tool call traces, annotations, and sentiment labels. Filtered for quality and de-identified.",
    format: "SFT",
    source: "conversations",
    status: "ready",
    samples: 2840,
    tokens: "4.2M",
    size: "48MB",
    createdAt: "2026-03-25",
    updatedAt: "2h ago",
    createdBy: "A. Kovács",
    tags: ["cx", "support", "sft", "production", "annotated"],
    color: "#F59E0B",
    hubRef: "datasets/cx-convos-v5",
    published: true,
    usedBy: [
      { name: "CX SFT Round 4", type: "training", status: "running" },
    ],
    versions: [
      { version: "v5", date: "2h ago", samples: 2840, change: "Added 340 new annotated conversations", author: "A. Kovács" },
      { version: "v4", date: "1w ago", samples: 2500, change: "De-identification pipeline applied", author: "A. Kovács" },
      { version: "v3", date: "3w ago", samples: 2200, change: "Added sentiment labels", author: "R. Silva" },
    ],
    stats: {
      avgTurns: 7.2, avgTokensPerSample: 1480, maxTokens: 8420, minTokens: 120,
      sentimentDist: { positive: 58, neutral: 28, negative: 14 },
      topTags: [{ tag: "subscription", count: 420 }, { tag: "billing", count: 380 }, { tag: "refund", count: 310 }, { tag: "upgrade", count: 280 }, { tag: "escalation", count: 190 }],
      tokenHistogram: [12, 45, 120, 280, 520, 680, 440, 320, 180, 110, 72, 38, 15, 8],
    },
    pipeline: [
      { id: "p1", step: "Source", type: "source", detail: "cx-support-v3 conversations (flagged + starred)", status: "completed", icon: "⊡" },
      { id: "p2", step: "Filter", type: "transform", detail: "Min 4 turns, exclude escalated-only", status: "completed", icon: "▽" },
      { id: "p3", step: "De-identify", type: "transform", detail: "NeMo PII redaction pipeline", status: "completed", icon: "◈" },
      { id: "p4", step: "Format", type: "transform", detail: "Convert to SFT instruction/response", status: "completed", icon: "⊞" },
      { id: "p5", step: "Quality Score", type: "transform", detail: "NeMo quality classifier (threshold > 0.7)", status: "completed", icon: "★" },
      { id: "p6", step: "Publish", type: "output", detail: "Push to Hub: datasets/cx-convos-v5", status: "completed", icon: "⊕" },
    ],
    sampleData: [
      {
        id: "s-0001",
        instruction: "A customer (premium member) wants to return a product purchased 45 days ago. The item arrived damaged. Handle the return request following the correct policy for premium members.",
        response: "I'd be happy to help with your return! As a Premium member, you have a 60-day return window, so your purchase from 45 days ago is fully eligible for a return and refund.\n\nSince the item arrived damaged:\n• Full refund to your original payment method\n• No return shipping required\n• Processing time: 24-48 hours (expedited for Premium)\n• $10 courtesy credit applied to your account\n\nI've initiated the return process. You'll receive a confirmation email shortly. Is there anything else I can help with?",
        metadata: { turns: 6, tokens: 1240, sentiment: "positive", tags: ["refund", "premium", "damaged"], quality_score: 0.92 },
      },
      {
        id: "s-0002",
        instruction: "Customer is asking why they were charged twice for order #ORD-7823. They are frustrated and want immediate resolution.",
        response: "I'm sorry about the double charge — I can see that's very frustrating. Let me investigate right away.\n\nI found the issue: there are two charges of $149.99:\n1. The actual order charge (completed March 25)\n2. A pending authorization hold from March 24 that wasn't properly released\n\nThis is a billing system error on our end. I'm escalating this to our billing specialist team who will:\n• Release the hold within 1-2 hours\n• Provide written confirmation\n• Apply a courtesy credit for the inconvenience\n\nEscalation Reference: ESC-2026-3109-BIL\n\nYou'll hear from the billing team within 1 hour.",
        metadata: { turns: 14, tokens: 4120, sentiment: "negative", tags: ["billing", "double-charge", "escalation"], quality_score: 0.88 },
      },
      {
        id: "s-0003",
        instruction: "Help a customer upgrade their subscription from Starter ($29/mo) to Pro ($79/mo). Show the prorated billing calculation.",
        response: "I'd be happy to help you upgrade to Pro! Here's the breakdown:\n\n• Current plan: Starter ($29/month)\n• New plan: Pro ($79/month)\n• Remaining days in cycle: 18 days\n• Prorated charge today: $30.00\n• Next full billing: $79.00 on April 14th\n\nPro includes: Unlimited API calls, priority support, advanced analytics, and custom integrations.\n\nYour upgrade is confirmed! Confirmation #: SUB-2026-8472-PRO. New features are available immediately.",
        metadata: { turns: 8, tokens: 2340, sentiment: "positive", tags: ["subscription", "upgrade", "proration"], quality_score: 0.95 },
      },
      {
        id: "s-0004",
        instruction: "Customer asks about the refund policy for standard (non-premium) members. They purchased an item 35 days ago.",
        response: "Our standard return policy allows returns within 30 days of purchase. Since your purchase was 35 days ago, it unfortunately falls outside the standard return window.\n\nHowever, I have some options for you:\n• Store credit: I can issue a full store credit that never expires\n• Exchange: We can exchange the item for a different product\n• Premium upgrade: If you upgrade to Premium membership, you'd get a 60-day return window for future purchases\n\nWould any of these options work for you?",
        metadata: { turns: 5, tokens: 890, sentiment: "neutral", tags: ["refund", "policy", "standard-member"], quality_score: 0.84 },
      },
    ],
  },
  {
    id: "code-trajectories-v2",
    name: "code-trajectories-v2",
    displayName: "Code Trajectories v2",
    description: "Multi-turn code generation trajectories from code-assist-v2. Includes full tool call traces (repo-indexer, code-executor, LSP), file diffs, and test outcomes.",
    format: "GRPO",
    source: "conversations",
    status: "ready",
    samples: 1420,
    tokens: "12.8M",
    size: "142MB",
    createdAt: "2026-03-20",
    updatedAt: "1d ago",
    createdBy: "M. Chen",
    tags: ["code", "trajectories", "grpo", "tool-use", "multi-turn"],
    color: "#3B82F6",
    hubRef: "datasets/code-trajectories-v2",
    published: true,
    usedBy: [
      { name: "Code RL Phase 2", type: "training", status: "running" },
    ],
    versions: [
      { version: "v2", date: "1d ago", samples: 1420, change: "Added reward signals from test outcomes", author: "M. Chen" },
      { version: "v1", date: "2w ago", samples: 980, change: "Initial extraction from production", author: "M. Chen" },
    ],
    stats: {
      avgTurns: 18.4, avgTokensPerSample: 9014, maxTokens: 32000, minTokens: 1200,
      sentimentDist: { positive: 72, neutral: 22, negative: 6 },
      topTags: [{ tag: "refactoring", count: 380 }, { tag: "debugging", count: 320 }, { tag: "generation", count: 290 }, { tag: "testing", count: 210 }, { tag: "documentation", count: 140 }],
      tokenHistogram: [5, 12, 28, 65, 140, 280, 320, 240, 160, 85, 42, 25, 12, 6],
    },
    pipeline: [
      { id: "p1", step: "Source", type: "source", detail: "code-assist-v2 conversations (starred)", status: "completed", icon: "⊡" },
      { id: "p2", step: "Extract Trajectories", type: "transform", detail: "NeMo trajectory extraction", status: "completed", icon: "⬡" },
      { id: "p3", step: "Compute Rewards", type: "transform", detail: "Test pass rate + code quality scores", status: "completed", icon: "★" },
      { id: "p4", step: "Format", type: "transform", detail: "Convert to GRPO format with rewards", status: "completed", icon: "⊞" },
      { id: "p5", step: "Publish", type: "output", detail: "Push to Hub: datasets/code-trajectories-v2", status: "completed", icon: "⊕" },
    ],
    sampleData: [
      {
        id: "s-0001",
        instruction: "Refactor this React class component to use hooks. The component manages a complex form with validation, async submission, and error handling.",
        response: "I'll break this into custom hooks for clean separation:\n\n```jsx\nfunction useFormState(initial) {\n  const [fields, setFields] = useState(initial);\n  const [errors, setErrors] = useState({});\n  // ...\n}\n```\n\n[Full refactored implementation with useFormState, useFormSubmission, useFieldValidation hooks]",
        metadata: { turns: 22, tokens: 8900, sentiment: "positive", tags: ["react", "refactoring", "hooks"], quality_score: 0.94, reward: 0.89 },
      },
    ],
  },
  {
    id: "safety-labels-v2",
    name: "safety-labels-v2",
    displayName: "Safety Labels v2",
    description: "Human-labeled safety classification dataset. Covers toxicity, PII, bias, and harmful content categories across 13 demographic groups.",
    format: "SFT",
    source: "manual",
    status: "ready",
    samples: 6200,
    tokens: "1.8M",
    size: "22MB",
    createdAt: "2026-03-18",
    updatedAt: "2d ago",
    createdBy: "A. Kovács",
    tags: ["safety", "classification", "toxicity", "pii", "human-labeled"],
    color: "#EF4444",
    hubRef: "datasets/safety-labels-v2",
    published: true,
    usedBy: [
      { name: "Guard classifier v2", type: "training", status: "completed" },
    ],
    versions: [
      { version: "v2", date: "2d ago", samples: 6200, change: "Extended to 13 demographic groups", author: "A. Kovács" },
      { version: "v1", date: "1w ago", samples: 4800, change: "Initial labeled dataset", author: "A. Kovács" },
    ],
    stats: {
      avgTurns: 1, avgTokensPerSample: 290, maxTokens: 1200, minTokens: 20,
      sentimentDist: { positive: 15, neutral: 35, negative: 50 },
      topTags: [{ tag: "toxicity", count: 2400 }, { tag: "safe", count: 2100 }, { tag: "bias", count: 820 }, { tag: "pii", count: 540 }, { tag: "harmful", count: 340 }],
      tokenHistogram: [180, 420, 880, 1400, 1200, 920, 540, 340, 180, 80, 40, 15, 4, 1],
    },
    pipeline: [],
    sampleData: [],
  },
  {
    id: "cx-dpo-pairs-v1",
    name: "cx-dpo-pairs-v1",
    displayName: "CX DPO Pairs v1",
    description: "Preference pairs for DPO training. Each sample contains a chosen (good) and rejected (bad) response, annotated from cx-convos-v4 using human and model judgments.",
    format: "DPO",
    source: "synthetic",
    status: "ready",
    samples: 1800,
    tokens: "5.6M",
    size: "64MB",
    createdAt: "2026-03-22",
    updatedAt: "5d ago",
    createdBy: "A. Kovács",
    tags: ["cx", "dpo", "preference", "pairs", "synthetic"],
    color: "#8B5CF6",
    hubRef: "datasets/cx-dpo-pairs-v1",
    published: true,
    usedBy: [],
    versions: [
      { version: "v1", date: "5d ago", samples: 1800, change: "Initial DPO pair generation", author: "A. Kovács" },
    ],
    stats: {
      avgTurns: 2, avgTokensPerSample: 3111, maxTokens: 9200, minTokens: 400,
      sentimentDist: { positive: 50, neutral: 30, negative: 20 },
      topTags: [{ tag: "chosen", count: 1800 }, { tag: "rejected", count: 1800 }, { tag: "quality-gap", count: 1200 }],
      tokenHistogram: [8, 24, 80, 240, 380, 420, 310, 180, 90, 42, 18, 6, 2, 0],
    },
    pipeline: [
      { id: "p1", step: "Source", type: "source", detail: "cx-convos-v4 dataset", status: "completed", icon: "▤" },
      { id: "p2", step: "Generate Rejections", type: "transform", detail: "NeMo synthetic rejection generation", status: "completed", icon: "◬" },
      { id: "p3", step: "Judge Pairs", type: "transform", detail: "GPT-4 + human judge scoring", status: "completed", icon: "★" },
      { id: "p4", step: "Format", type: "transform", detail: "Convert to DPO chosen/rejected format", status: "completed", icon: "⊞" },
      { id: "p5", step: "Publish", type: "output", detail: "Push to Hub: datasets/cx-dpo-pairs-v1", status: "completed", icon: "⊕" },
    ],
    sampleData: [],
  },
  {
    id: "sql-gold-v1",
    name: "sql-gold-v1",
    displayName: "SQL Gold Queries v1",
    description: "Natural language to SQL gold standard dataset. Contains verified question-query-result triples from the company data warehouse.",
    format: "SFT",
    source: "manual",
    status: "ready",
    samples: 420,
    tokens: "380K",
    size: "4.2MB",
    createdAt: "2026-03-15",
    updatedAt: "1w ago",
    createdBy: "R. Silva",
    tags: ["sql", "nl2sql", "gold-standard", "data-warehouse"],
    color: "#22C55E",
    hubRef: "datasets/sql-gold-v1",
    published: false,
    usedBy: [],
    versions: [
      { version: "v1", date: "1w ago", samples: 420, change: "Initial gold standard collection", author: "R. Silva" },
    ],
    stats: {
      avgTurns: 1, avgTokensPerSample: 905, maxTokens: 3200, minTokens: 80,
      sentimentDist: { positive: 40, neutral: 55, negative: 5 },
      topTags: [{ tag: "select", count: 380 }, { tag: "join", count: 240 }, { tag: "aggregate", count: 190 }, { tag: "window", count: 65 }, { tag: "cte", count: 42 }],
      tokenHistogram: [15, 42, 85, 110, 72, 48, 25, 12, 6, 3, 1, 1, 0, 0],
    },
    pipeline: [],
    sampleData: [],
  },
  {
    id: "onboarding-qa-v1",
    name: "onboarding-qa-v1",
    displayName: "Onboarding Q&A v1",
    description: "Question-answer pairs from onboarding-bot conversations. Covers IT setup, benefits enrollment, policy FAQs. Used for RAG evaluation.",
    format: "SFT",
    source: "conversations",
    status: "building",
    samples: 180,
    tokens: "142K",
    size: "1.6MB",
    createdAt: "2026-03-27",
    updatedAt: "4h ago",
    createdBy: "L. Park",
    tags: ["onboarding", "qa", "hr", "it", "benefits"],
    color: "#06B6D4",
    hubRef: null,
    published: false,
    usedBy: [],
    versions: [
      { version: "v1-draft", date: "4h ago", samples: 180, change: "Initial extraction in progress", author: "L. Park" },
    ],
    stats: {
      avgTurns: 4.8, avgTokensPerSample: 789, maxTokens: 2400, minTokens: 120,
      sentimentDist: { positive: 75, neutral: 20, negative: 5 },
      topTags: [{ tag: "it-setup", count: 62 }, { tag: "benefits", count: 48 }, { tag: "vpn", count: 32 }, { tag: "email", count: 28 }, { tag: "policy", count: 10 }],
      tokenHistogram: [8, 20, 35, 48, 32, 18, 10, 5, 3, 1, 0, 0, 0, 0],
    },
    pipeline: [
      { id: "p1", step: "Source", type: "source", detail: "onboarding-bot conversations", status: "completed", icon: "⊡" },
      { id: "p2", step: "Extract Q&A", type: "transform", detail: "NeMo Q&A extraction pipeline", status: "completed", icon: "▽" },
      { id: "p3", step: "Quality Filter", type: "transform", detail: "NeMo quality scorer (threshold > 0.65)", status: "running", icon: "★" },
      { id: "p4", step: "Human Review", type: "transform", detail: "Manual review queue", status: "pending", icon: "◈" },
      { id: "p5", step: "Publish", type: "output", detail: "Push to Hub", status: "pending", icon: "⊕" },
    ],
    sampleData: [],
  },
];
