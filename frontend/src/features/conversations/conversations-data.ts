// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//

// ── Types ───────────────────────────────────────────────────────

export interface ToolCall {
  name: string;
  action: string;
  status: "success" | "error";
  latency: number;
}

export interface Annotation {
  type: "skill_gap" | "trajectory_correction" | "quality_issue";
  note: string;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  tokens: number;
  latency?: number;
  tools?: ToolCall[];
  annotation?: Annotation;
}

export interface Conversation {
  id: string;
  agent: string;
  agentColor: string;
  user: string;
  preview: string;
  status: "completed" | "active" | "failed";
  sentiment: "positive" | "negative" | "neutral";
  sentimentScore: number;
  flagged: boolean;
  starred: boolean;
  tags: string[];
  tokens: { in: number; out: number };
  turns: number;
  duration: string;
  latencyAvg: number;
  time: string;
  timestamp: string;
  resolved: boolean;
  escalated: boolean;
  toolCalls: number;
  dataset: string | null;
  messages: Message[];
}

// ── Constants ──────────────────────────────────────────────────

export const SENTIMENT_COLORS: Record<string, string> = {
  positive: "#22C55E",
  neutral: "#6B7585",
  negative: "#EF4444",
};

export const ANNOTATION_STYLES: Record<
  string,
  { bg: string; fg: string; border: string; label: string }
> = {
  skill_gap: {
    bg: "#F59E0B12",
    fg: "#F59E0B",
    border: "#F59E0B30",
    label: "SKILL GAP",
  },
  trajectory_correction: {
    bg: "#3B82F612",
    fg: "#3B82F6",
    border: "#3B82F630",
    label: "TRAJECTORY",
  },
  quality_issue: {
    bg: "#EF444412",
    fg: "#EF4444",
    border: "#EF444430",
    label: "QUALITY",
  },
};

export const CONVERSATIONS: Conversation[] = [
  {
    id: "c-9821",
    agent: "cx-support-v3",
    agentColor: "#F59E0B",
    user: "user_8472",
    preview: "I need to change my subscription plan to the pro tier",
    status: "completed",
    sentiment: "positive",
    sentimentScore: 0.87,
    flagged: false,
    starred: false,
    tags: ["subscription", "upgrade"],
    tokens: { in: 1420, out: 920 },
    turns: 8,
    duration: "2m 14s",
    latencyAvg: 245,
    time: "2m ago",
    timestamp: "2026-03-28T14:58:00Z",
    resolved: true,
    escalated: false,
    toolCalls: 3,
    dataset: null,
    messages: [
      { role: "user", content: "Hi, I need to change my subscription plan to the pro tier. Can you help me with that?", timestamp: "14:55:12", tokens: 28 },
      { role: "assistant", content: "Of course! I'd be happy to help you upgrade to the Pro tier. Let me look up your current subscription details first.", timestamp: "14:55:13", tokens: 32, latency: 198, tools: [{ name: "subscription-manager", action: "lookup", status: "success", latency: 45 }] },
      { role: "assistant", content: "I can see you're currently on the **Starter** plan at $29/month. The Pro plan is $79/month and includes:\n\n• Unlimited API calls\n• Priority support\n• Advanced analytics\n• Custom integrations\n\nWould you like me to proceed with the upgrade? I can show you the prorated billing for this cycle.", timestamp: "14:55:14", tokens: 68, latency: 210 },
      { role: "user", content: "Yes please, go ahead with the upgrade. What will I be charged?", timestamp: "14:55:42", tokens: 18 },
      { role: "assistant", content: "Let me calculate the prorated amount for the remainder of your billing cycle.", timestamp: "14:55:43", tokens: 18, latency: 180, tools: [{ name: "subscription-manager", action: "calculate_proration", status: "success", latency: 120 }] },
      { role: "assistant", content: "Here's the breakdown:\n\n• Remaining days in cycle: 18 days\n• Prorated charge: **$30.00** (difference between plans)\n• Next full billing: **$79.00** on April 14th\n\nShall I confirm the upgrade?", timestamp: "14:55:44", tokens: 52, latency: 195 },
      { role: "user", content: "That looks good. Please confirm it.", timestamp: "14:56:08", tokens: 12 },
      { role: "assistant", content: "Your subscription has been upgraded to **Pro**! 🎉\n\nHere's your confirmation:\n• Plan: Pro ($79/month)\n• Prorated charge: $30.00 (charged today)\n• Next billing: April 14, 2026\n• Confirmation #: SUB-2026-8472-PRO\n\nYour new features are available immediately. Is there anything else I can help with?", timestamp: "14:56:09", tokens: 72, latency: 310, tools: [{ name: "subscription-manager", action: "upgrade", status: "success", latency: 180 }] },
    ],
  },
  {
    id: "c-9820",
    agent: "cx-support-v3",
    agentColor: "#F59E0B",
    user: "user_3109",
    preview: "Why was I charged twice for the same order?",
    status: "completed",
    sentiment: "negative",
    sentimentScore: 0.28,
    flagged: true,
    starred: true,
    tags: ["billing", "double-charge", "escalation-candidate"],
    tokens: { in: 2480, out: 1640 },
    turns: 14,
    duration: "5m 38s",
    latencyAvg: 285,
    time: "5m ago",
    timestamp: "2026-03-28T14:53:00Z",
    resolved: false,
    escalated: true,
    toolCalls: 5,
    dataset: "cx-convos-v5",
    messages: [
      { role: "user", content: "Why was I charged twice for the same order? I see two charges of $149.99 on my credit card statement. Order #ORD-7823.", timestamp: "14:48:22", tokens: 38 },
      { role: "assistant", content: "I'm sorry to hear about the double charge. Let me look into order #ORD-7823 right away.", timestamp: "14:48:23", tokens: 24, latency: 210, tools: [{ name: "order-lookup", action: "get_order", status: "success", latency: 45 }] },
      { role: "assistant", content: "I found your order. I can see one charge of $149.99 for Order #ORD-7823 (Wireless Headphones). Let me check the billing system for any duplicate transactions.", timestamp: "14:48:24", tokens: 42, latency: 240, tools: [{ name: "subscription-manager", action: "check_billing", status: "success", latency: 120 }] },
      { role: "user", content: "I definitely see two charges. Check my account again. This is really frustrating.", timestamp: "14:49:10", tokens: 22 },
      { role: "assistant", content: "I completely understand your frustration, and I apologize for the inconvenience. Let me do a deeper investigation into your billing history.", timestamp: "14:49:11", tokens: 28, latency: 190, tools: [{ name: "order-lookup", action: "billing_history", status: "success", latency: 85 }] },
      { role: "assistant", content: "I found the issue. There are indeed two charges:\n\n1. **$149.99** — ORD-7823 (completed, March 25)\n2. **$149.99** — A pending authorization from March 24 that should have been released\n\nThe second charge appears to be a pre-authorization hold that wasn't properly released. This is a billing system issue on our end.", timestamp: "14:49:12", tokens: 68, latency: 280 },
      { role: "user", content: "So when will I get my money back? I need this resolved today.", timestamp: "14:49:48", tokens: 18 },
      { role: "assistant", content: "I want to make sure this is resolved properly for you. I can initiate a release of the pending authorization right now, but since this involves a billing system error, I'd like to escalate this to our billing specialist team who can:\n\n1. Release the hold immediately (within 1-2 hours)\n2. Provide written confirmation\n3. Apply a courtesy credit for the inconvenience\n\nWould you like me to escalate this?", timestamp: "14:49:49", tokens: 82, latency: 320, tools: [{ name: "escalation-router", action: "evaluate", status: "success", latency: 25 }], annotation: { type: "trajectory_correction", note: "Agent correctly identified need for escalation instead of attempting billing mutation directly" } },
      { role: "user", content: "Yes, escalate it. And I want a credit for the trouble.", timestamp: "14:50:35", tokens: 16 },
      { role: "assistant", content: "Absolutely. I'm escalating this now with full context. You'll hear from our billing team within 1 hour. I've noted the request for a courtesy credit as well.\n\nEscalation Reference: ESC-2026-3109-BIL\n\nIs there anything else I can help with while we get this resolved?", timestamp: "14:50:36", tokens: 54, latency: 290, tools: [{ name: "escalation-router", action: "escalate", status: "success", latency: 180 }] },
      { role: "user", content: "No, just get this fixed.", timestamp: "14:51:02", tokens: 8 },
      { role: "assistant", content: "I understand. The escalation is in progress and the billing team will reach out shortly. I'm sorry again for the inconvenience. Have a good day.", timestamp: "14:51:03", tokens: 32, latency: 195 },
    ],
  },
  {
    id: "c-9819",
    agent: "code-assist-v2",
    agentColor: "#3B82F6",
    user: "dev_riley",
    preview: "Help me refactor this React component to use hooks",
    status: "completed",
    sentiment: "positive",
    sentimentScore: 0.92,
    flagged: false,
    starred: false,
    tags: ["react", "refactoring", "hooks"],
    tokens: { in: 5200, out: 3700 },
    turns: 22,
    duration: "12m 45s",
    latencyAvg: 820,
    time: "8m ago",
    timestamp: "2026-03-28T14:50:00Z",
    resolved: true,
    escalated: false,
    toolCalls: 8,
    dataset: null,
    messages: [
      { role: "user", content: "I have a class component that manages a complex form with validation. Can you help me refactor it to use React hooks? Here's the component:\n\n```jsx\nclass UserForm extends React.Component {\n  constructor(props) {\n    super(props);\n    this.state = {\n      name: '',\n      email: '',\n      errors: {},\n      submitting: false\n    };\n  }\n  // ... 200 more lines\n}\n```", timestamp: "14:37:15", tokens: 120 },
      { role: "assistant", content: "I'll help you refactor this to hooks. Let me first analyze the component structure to understand the state management pattern and lifecycle methods being used.", timestamp: "14:37:17", tokens: 36, latency: 780, tools: [{ name: "repo-indexer", action: "analyze_component", status: "success", latency: 120 }] },
      { role: "assistant", content: "Based on the analysis, I'll break this into several custom hooks for clean separation of concerns:\n\n1. `useFormState` — manages form fields and validation\n2. `useFormSubmission` — handles async submission logic\n3. `useFieldValidation` — real-time field validation\n\nHere's the refactored version:\n\n```jsx\nfunction UserForm() {\n  const { fields, setField, errors, validate } = useFormState({\n    name: '',\n    email: '',\n  });\n  // ...\n}\n```\n\nShall I generate the complete implementation with all custom hooks?", timestamp: "14:37:18", tokens: 180, latency: 920 },
      { role: "user", content: "Yes, give me the full implementation.", timestamp: "14:38:02", tokens: 10 },
    ],
  },
  {
    id: "c-9818",
    agent: "data-analyst-v1",
    agentColor: "#8B5CF6",
    user: "analyst_jen",
    preview: "Generate a quarterly revenue breakdown by region",
    status: "completed",
    sentiment: "neutral",
    sentimentScore: 0.55,
    flagged: false,
    starred: false,
    tags: ["sql", "revenue", "quarterly"],
    tokens: { in: 1800, out: 1400 },
    turns: 6,
    duration: "1m 52s",
    latencyAvg: 1180,
    time: "12m ago",
    timestamp: "2026-03-28T14:46:00Z",
    resolved: true,
    escalated: false,
    toolCalls: 3,
    dataset: null,
    messages: [
      { role: "user", content: "Can you generate a quarterly revenue breakdown by region for 2025? I need it for the board deck.", timestamp: "14:44:08", tokens: 24 },
      { role: "assistant", content: "I'll query the data warehouse for Q1-Q4 2025 revenue broken down by region. Let me first check the schema.", timestamp: "14:44:10", tokens: 28, latency: 980, tools: [{ name: "schema-introspect", action: "introspect", status: "success", latency: 450 }, { name: "sql-generator", action: "generate", status: "success", latency: 220 }] },
      { role: "assistant", content: "Here are the results:\n\n| Region | Q1 | Q2 | Q3 | Q4 | Total |\n|--------|-------|-------|-------|-------|--------|\n| North America | $4.2M | $4.8M | $5.1M | $5.6M | $19.7M |\n| Europe | $2.1M | $2.4M | $2.6M | $2.9M | $10.0M |\n| APAC | $1.8M | $2.0M | $2.3M | $2.5M | $8.6M |\n| LATAM | $0.6M | $0.7M | $0.8M | $0.9M | $3.0M |\n\n**Total: $41.3M** (+18% YoY)\n\nWould you like me to generate a chart for the board deck?", timestamp: "14:44:12", tokens: 160, latency: 1200, tools: [{ name: "chart-renderer", action: "render", status: "success", latency: 340 }] },
    ],
  },
  {
    id: "c-9817",
    agent: "cx-support-v3",
    agentColor: "#F59E0B",
    user: "user_0091",
    preview: "The agent couldn't understand my request about refund policy",
    status: "completed",
    sentiment: "negative",
    sentimentScore: 0.22,
    flagged: true,
    starred: false,
    tags: ["refund", "misunderstanding", "skill-gap"],
    tokens: { in: 1200, out: 690 },
    turns: 11,
    duration: "4m 10s",
    latencyAvg: 260,
    time: "15m ago",
    timestamp: "2026-03-28T14:43:00Z",
    resolved: false,
    escalated: false,
    toolCalls: 2,
    dataset: "cx-convos-v5",
    messages: [
      { role: "user", content: "I want to return a product I bought 45 days ago. What's your refund policy for items past the 30-day window?", timestamp: "14:38:50", tokens: 28 },
      { role: "assistant", content: "I'd be happy to help with your return! Let me look up the details.", timestamp: "14:38:51", tokens: 16, latency: 180, tools: [{ name: "kb-search", action: "search", status: "success", latency: 85 }] },
      { role: "assistant", content: "Our standard return policy allows returns within 30 days of purchase. Since your purchase was 45 days ago, it falls outside the standard window. However, you may be eligible for store credit. Would you like me to check?", timestamp: "14:38:52", tokens: 48, latency: 220, annotation: { type: "skill_gap", note: "Agent didn't search for exception policies. KB has a 60-day extended window for premium members that wasn't retrieved." } },
      { role: "user", content: "I'm a premium member. Don't I get extended returns?", timestamp: "14:39:30", tokens: 14 },
      { role: "assistant", content: "Let me check the premium member benefits for you.", timestamp: "14:39:31", tokens: 12, latency: 195, tools: [{ name: "kb-search", action: "search", status: "success", latency: 90 }] },
      { role: "assistant", content: "You're right, I apologize for the oversight! As a Premium member, you have a 60-day return window. Since your purchase was 45 days ago, you're fully eligible for a return and refund. Would you like me to initiate the return process?", timestamp: "14:39:32", tokens: 48, latency: 240, annotation: { type: "trajectory_correction", note: "Self-corrected after user prompt, but should have checked membership tier proactively on first query" } },
      { role: "user", content: "Yes, but I'm not happy that I had to tell you about my membership. You should have checked that first.", timestamp: "14:40:05", tokens: 24 },
      { role: "assistant", content: "You're absolutely right, and I sincerely apologize. I should have checked your membership tier before quoting the standard policy. I'll make sure to verify membership status for all return inquiries going forward. Let me process your return right away.", timestamp: "14:40:06", tokens: 44, latency: 280 },
    ],
  },
  {
    id: "c-9816",
    agent: "onboarding-bot",
    agentColor: "#22C55E",
    user: "new_hire_88",
    preview: "How do I set up my company email and VPN access?",
    status: "completed",
    sentiment: "positive",
    sentimentScore: 0.81,
    flagged: false,
    starred: false,
    tags: ["onboarding", "it-setup", "vpn"],
    tokens: { in: 620, out: 480 },
    turns: 6,
    duration: "1m 20s",
    latencyAvg: 185,
    time: "22m ago",
    timestamp: "2026-03-28T14:36:00Z",
    resolved: true,
    escalated: false,
    toolCalls: 2,
    dataset: null,
    messages: [
      { role: "user", content: "Hi! I just started today. How do I set up my company email and VPN?", timestamp: "14:34:40", tokens: 18 },
      { role: "assistant", content: "Welcome aboard! 🎉 I'll walk you through both setups. Let me pull up the IT onboarding checklist for you.", timestamp: "14:34:41", tokens: 24, latency: 165, tools: [{ name: "policy-search", action: "search", status: "success", latency: 95 }] },
    ],
  },
  {
    id: "c-9815",
    agent: "code-assist-v2",
    agentColor: "#3B82F6",
    user: "dev_alex",
    preview: "Debug this async race condition in our worker pool",
    status: "active",
    sentiment: "neutral",
    sentimentScore: 0.52,
    flagged: false,
    starred: true,
    tags: ["debugging", "async", "race-condition", "concurrency"],
    tokens: { in: 6800, out: 4200 },
    turns: 28,
    duration: "18m 32s",
    latencyAvg: 890,
    time: "now",
    timestamp: "2026-03-28T14:40:00Z",
    resolved: false,
    escalated: false,
    toolCalls: 12,
    dataset: "code-trajectories-v2",
    messages: [
      { role: "user", content: "I'm seeing intermittent failures in our worker pool implementation. Tasks occasionally get processed twice or dropped. Can you help debug?", timestamp: "14:40:00", tokens: 32 },
      { role: "assistant", content: "This sounds like a classic race condition in concurrent task processing. Let me analyze your worker pool code to identify the issue.", timestamp: "14:40:02", tokens: 28, latency: 850, tools: [{ name: "repo-indexer", action: "search", status: "success", latency: 120 }, { name: "lsp-bridge", action: "diagnostics", status: "success", latency: 35 }] },
    ],
  },
];
