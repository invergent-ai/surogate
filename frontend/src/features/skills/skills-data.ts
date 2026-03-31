// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { Status } from "@/components/ui/status-dot";

// ── Status mapping ──────────────────────────────────────────────
// The mockup uses "active"/"connected"/"disconnected" which don't
// exist on StatusDot, so we map them to the nearest semantic match.
const STATUS_MAP: Record<string, Status> = {
  active: "running",
  connected: "serving",
  disconnected: "stopped",
  running: "running",
  error: "error",
  deploying: "deploying",
};

export function toStatus(raw: string): Status {
  return STATUS_MAP[raw] ?? "stopped";
}

// ── Types ───────────────────────────────────────────────────────

export interface SkillVersion {
  version: string;
  date: string;
  author: string;
  change: string;
  status: string;
}

export interface Skill {
  id: string;
  name: string;
  displayName: string;
  description: string;
  agent: string;
  status: string;
  author: string;
  updatedAt: string;
  version: string;
  tags: string[];
  content: string;
  versions: SkillVersion[];
}

export interface SchemaField {
  name: string;
  type: string;
  required?: boolean;
  description: string;
}

export interface ToolConfig {
  key: string;
  value: string;
  secret: boolean;
}

export interface AgentRef {
  name: string;
  status: string;
}

export interface Tool {
  id: string;
  name: string;
  displayName: string;
  description: string;
  category: string;
  version: string;
  status: string;
  author: string;
  tags: string[];
  usedByAgents: AgentRef[];
  inputSchema: SchemaField[];
  outputSchema: SchemaField[];
  config: ToolConfig[];
  metrics: { calls24h: number; avgLatency: string; errorRate: string; p99: string };
}

export interface ToolCategory {
  id: string;
  label: string;
  icon: string;
}

export interface McpServer {
  id: string;
  name: string;
  url: string;
  transport: string;
  status: string;
  latency: string;
  description: string;
  toolCount: number;
  connectedAgents: string[];
  tools: string[];
  auth: string;
  updatedAt: string;
}

// ── Category colours (Tailwind-compatible) ──────────────────────

export const CAT_STYLES: Record<string, { bg: string; fg: string; border: string }> = {
  tool: { bg: "bg-blue-500/10", fg: "text-blue-500", border: "border-blue-500/15" },
  rag: { bg: "bg-green-500/10", fg: "text-green-500", border: "border-green-500/15" },
  workflow: { bg: "bg-amber-500/10", fg: "text-amber-500", border: "border-amber-500/15" },
  guardrail: { bg: "bg-red-500/10", fg: "text-red-500", border: "border-red-500/15" },
  output: { bg: "bg-violet-500/10", fg: "text-violet-500", border: "border-violet-500/15" },
};

// ── Demo data ───────────────────────────────────────────────────

export const TOOL_CATEGORIES: ToolCategory[] = [
  { id: "all", label: "All", icon: "✦" },
  { id: "tool", label: "Tools", icon: "⚙" },
  { id: "rag", label: "RAG / Retrieval", icon: "▤" },
  { id: "workflow", label: "Workflows", icon: "⬡" },
  { id: "guardrail", label: "Guardrails", icon: "◈" },
  { id: "output", label: "Output Parsers", icon: "⊡" },
];

export const SKILLS: Skill[] = [
  {
    id: "cx-support-skills",
    name: "cx-support-skills",
    displayName: "CX Support Skills",
    description: "Core customer support capabilities: greeting protocols, empathy-first response patterns, refund & cancellation workflows, escalation criteria, and SLA-aware prioritization.",
    agent: "cx-support-v3",
    status: "active",
    author: "A. Kovács",
    updatedAt: "2d ago",
    version: "3.2.0",
    tags: ["customer-support", "empathy", "refunds", "escalation"],
    content: `# CX Support Agent Skills

## Greeting & Tone
- Always greet the customer by name when available
- Use empathy-first language: acknowledge frustration before solving
- Match formality to customer's tone

## Refund Workflow
1. Verify order via order-lookup tool
2. Check refund eligibility (< 30 days, unused)
3. Process via subscription-manager tool
4. Confirm with customer and provide timeline

## Escalation Criteria
- Customer mentions legal action
- Sentiment score drops below 0.3
- 3+ failed resolution attempts
- VIP customer flag detected

## SLA Priorities
- P0: Service outage affecting billing → 15 min response
- P1: Failed transaction → 1 hour response
- P2: General inquiry → 4 hour response`,
    versions: [
      { version: "3.2.0", date: "2d ago", author: "A. Kovács", change: "Added SLA priority matrix", status: "active" },
      { version: "3.1.0", date: "2w ago", author: "A. Kovács", change: "Revised escalation criteria", status: "previous" },
      { version: "3.0.0", date: "1mo ago", author: "M. Chen", change: "Full rewrite for v3 agent", status: "archived" },
    ],
  },
  {
    id: "code-assist-skills",
    name: "code-assist-skills",
    displayName: "Code Assistant Skills",
    description: "Developer productivity capabilities: code generation patterns, refactoring strategies, debugging workflows, PR review guidelines, and documentation standards.",
    agent: "code-assist-v2",
    status: "active",
    author: "M. Chen",
    updatedAt: "1w ago",
    version: "2.7.0",
    tags: ["coding", "refactoring", "debugging", "review"],
    content: `# Code Assistant Skills

## Code Generation
- Always generate type-safe code with explicit return types
- Include error handling for external calls
- Follow project conventions from .editorconfig

## Refactoring
- Identify code smells: long methods, deep nesting, duplicated logic
- Suggest extract-method for functions > 30 lines
- Prefer composition over inheritance

## Debugging Workflow
1. Reproduce the issue with a minimal test case
2. Use LSP diagnostics to check types
3. Trace data flow from input to failure point
4. Suggest fix with explanation

## PR Review Guidelines
- Check for security issues (injection, XSS, auth)
- Verify test coverage for new code paths
- Flag breaking API changes`,
    versions: [
      { version: "2.7.0", date: "1w ago", author: "M. Chen", change: "Added PR review guidelines", status: "active" },
      { version: "2.6.0", date: "1mo ago", author: "M. Chen", change: "Debugging workflow section", status: "previous" },
    ],
  },
  {
    id: "data-analyst-skills",
    name: "data-analyst-skills",
    displayName: "Data Analyst Skills",
    description: "Data analysis capabilities: natural language to SQL translation, chart selection heuristics, statistical summary patterns, and data quality checks.",
    agent: "data-analyst-v1",
    status: "active",
    author: "R. Silva",
    updatedAt: "4d ago",
    version: "1.4.0",
    tags: ["sql", "analytics", "charts", "data-quality"],
    content: `# Data Analyst Skills

## SQL Generation
- Always use parameterized queries
- Prefer CTEs over subqueries for readability
- Add LIMIT to exploratory queries
- Validate generated SQL before execution

## Chart Selection
- Time series → line chart
- Comparison across categories → bar chart
- Part-of-whole → pie chart (max 6 slices)
- Correlation → scatter plot

## Data Quality Checks
- Flag NULL rates > 5%
- Detect outliers using IQR method
- Warn on date gaps in time series`,
    versions: [
      { version: "1.4.0", date: "4d ago", author: "R. Silva", change: "Data quality checks section", status: "active" },
      { version: "1.3.0", date: "2w ago", author: "R. Silva", change: "Chart selection heuristics", status: "previous" },
    ],
  },
  {
    id: "onboarding-skills",
    name: "onboarding-skills",
    displayName: "Onboarding Skills",
    description: "Employee onboarding guidance: first-day checklists, IT setup procedures, benefits enrollment walkthrough, and HR policy Q&A patterns.",
    agent: "onboarding-bot",
    status: "active",
    author: "L. Park",
    updatedAt: "3d ago",
    version: "1.0.3",
    tags: ["hr", "onboarding", "it-setup", "benefits"],
    content: `# Onboarding Bot Skills

## First Day Checklist
1. Welcome message with team info
2. IT equipment setup guide
3. Account provisioning status check
4. Benefits enrollment reminder

## IT Setup
- Guide laptop configuration step by step
- Create IT ticket if provisioning is delayed
- Verify VPN and SSO access

## HR Policy Q&A
- Use policy-search tool for answers
- Always cite the source document and section
- Escalate to HR for ambiguous cases`,
    versions: [
      { version: "1.0.3", date: "3d ago", author: "L. Park", change: "Updated IT setup for 2026 laptops", status: "active" },
    ],
  },
  {
    id: "safety-reviewer-skills",
    name: "safety-reviewer-skills",
    displayName: "Safety Reviewer Skills",
    description: "Content moderation capabilities: toxicity classification rules, PII detection patterns, compliance check procedures, and incident escalation protocols.",
    agent: "safety-reviewer",
    status: "error",
    author: "A. Kovács",
    updatedAt: "1d ago",
    version: "0.9.1",
    tags: ["safety", "moderation", "pii", "compliance"],
    content: `# Safety Reviewer Skills

## Toxicity Classification
- Run all user-facing text through toxicity-classifier
- Block responses scoring > 0.9 on any category
- Log borderline cases (0.7-0.9) for human review

## PII Detection
- Scan all inputs and outputs with pii-detector
- Auto-redact SSN, credit card, phone patterns
- Flag but don't redact email addresses

## Compliance Checks
- GDPR: ensure right-to-deletion requests are honored
- SOC2: log all data access decisions
- HIPAA: block health information in non-medical contexts`,
    versions: [
      { version: "0.9.1", date: "1d ago", author: "A. Kovács", change: "HIPAA compliance rules", status: "error" },
    ],
  },
];

export const TOOLS: Tool[] = [
  {
    id: "order-lookup", name: "order-lookup", displayName: "Order Lookup",
    description: "Retrieves order details, status, and tracking information from the CRM and logistics systems. Supports bulk lookups and fuzzy matching on order IDs.",
    category: "tool", version: "2.1.0", status: "active", author: "A. Kovács",
    tags: ["crm", "orders", "lookup", "customer-support"],
    usedByAgents: [{ name: "cx-support-v3", status: "running" }],
    inputSchema: [
      { name: "order_id", type: "string", required: true, description: "Order ID or tracking number" },
      { name: "customer_email", type: "string", required: false, description: "Customer email for fuzzy match" },
      { name: "include_history", type: "boolean", required: false, description: "Include full order history" },
    ],
    outputSchema: [
      { name: "order", type: "object", description: "Order details including status, items, dates" },
      { name: "tracking", type: "object", description: "Shipping/tracking information" },
    ],
    config: [
      { key: "CRM_ENDPOINT", value: "https://crm.internal/api/v2", secret: false },
      { key: "CRM_API_KEY", value: "••••••••", secret: true },
      { key: "TIMEOUT_MS", value: "5000", secret: false },
    ],
    metrics: { calls24h: 4820, avgLatency: "45ms", errorRate: "0.1%", p99: "120ms" },
  },
  {
    id: "subscription-manager", name: "subscription-manager", displayName: "Subscription Manager",
    description: "Manages customer subscription lifecycle: upgrades, downgrades, cancellations, and plan comparisons. Integrates with Stripe for billing mutations.",
    category: "tool", version: "1.4.2", status: "active", author: "M. Chen",
    tags: ["billing", "stripe", "subscriptions"],
    usedByAgents: [{ name: "cx-support-v3", status: "running" }],
    inputSchema: [
      { name: "customer_id", type: "string", required: true, description: "Customer identifier" },
      { name: "action", type: "enum", required: true, description: "upgrade | downgrade | cancel | compare" },
      { name: "target_plan", type: "string", required: false, description: "Target plan ID" },
    ],
    outputSchema: [
      { name: "result", type: "object", description: "Subscription mutation result" },
      { name: "billing_preview", type: "object", description: "Prorated billing preview" },
    ],
    config: [
      { key: "STRIPE_SECRET_KEY", value: "••••••••", secret: true },
      { key: "WEBHOOK_SECRET", value: "••••••••", secret: true },
    ],
    metrics: { calls24h: 620, avgLatency: "180ms", errorRate: "0.3%", p99: "450ms" },
  },
  {
    id: "kb-search", name: "kb-search", displayName: "Knowledge Base Search",
    description: "Semantic search over company knowledge base using embeddings. Supports hybrid search (dense + sparse), re-ranking, and citation extraction.",
    category: "rag", version: "3.0.1", status: "active", author: "A. Kovács",
    tags: ["search", "embeddings", "rag", "hybrid-search"],
    usedByAgents: [{ name: "cx-support-v3", status: "running" }],
    inputSchema: [
      { name: "query", type: "string", required: true, description: "Search query" },
      { name: "top_k", type: "integer", required: false, description: "Number of results (default: 5)" },
      { name: "rerank", type: "boolean", required: false, description: "Apply cross-encoder reranking" },
    ],
    outputSchema: [
      { name: "results", type: "array", description: "Ranked search results with scores" },
      { name: "citations", type: "array", description: "Extracted citations from source docs" },
    ],
    config: [
      { key: "VECTOR_DB_URL", value: "http://qdrant.internal:6333", secret: false },
      { key: "EMBEDDING_MODEL", value: "bge-large-en-v1.5", secret: false },
    ],
    metrics: { calls24h: 8940, avgLatency: "85ms", errorRate: "0.05%", p99: "220ms" },
  },
  {
    id: "code-executor", name: "code-executor", displayName: "Code Executor",
    description: "Sandboxed code execution environment. Supports Python, JavaScript, and bash. Runs in an isolated container with resource limits.",
    category: "tool", version: "1.2.1", status: "active", author: "M. Chen",
    tags: ["code", "execution", "sandbox"],
    usedByAgents: [{ name: "code-assist-v2", status: "running" }],
    inputSchema: [
      { name: "code", type: "string", required: true, description: "Code to execute" },
      { name: "language", type: "enum", required: true, description: "python | javascript | bash" },
      { name: "timeout_s", type: "integer", required: false, description: "Execution timeout (default: 30s)" },
    ],
    outputSchema: [
      { name: "stdout", type: "string", description: "Standard output" },
      { name: "stderr", type: "string", description: "Standard error" },
      { name: "exit_code", type: "integer", description: "Process exit code" },
    ],
    config: [
      { key: "SANDBOX_IMAGE", value: "registry.internal/sandbox:latest", secret: false },
      { key: "CPU_LIMIT", value: "1000m", secret: false },
      { key: "MEM_LIMIT", value: "512Mi", secret: false },
    ],
    metrics: { calls24h: 1420, avgLatency: "2.1s", errorRate: "1.2%", p99: "8.5s" },
  },
  {
    id: "repo-indexer", name: "repo-indexer", displayName: "Repository Indexer",
    description: "Indexes code repositories for semantic search. Builds AST-aware index with function/class-level chunking for precise code retrieval.",
    category: "rag", version: "2.0.0", status: "active", author: "M. Chen",
    tags: ["code", "indexing", "rag", "tree-sitter"],
    usedByAgents: [{ name: "code-assist-v2", status: "running" }],
    inputSchema: [
      { name: "repo_url", type: "string", required: true, description: "Git repository URL or local path" },
      { name: "query", type: "string", required: true, description: "Code search query" },
    ],
    outputSchema: [
      { name: "chunks", type: "array", description: "Matched code chunks with file paths and line numbers" },
    ],
    config: [
      { key: "INDEX_BACKEND", value: "qdrant", secret: false },
      { key: "CHUNK_STRATEGY", value: "ast-aware", secret: false },
    ],
    metrics: { calls24h: 2340, avgLatency: "120ms", errorRate: "0.2%", p99: "380ms" },
  },
  {
    id: "lsp-bridge", name: "lsp-bridge", displayName: "LSP Bridge",
    description: "Language Server Protocol bridge for real-time type checking, diagnostics, and code intelligence. Supports TypeScript, Python, Rust.",
    category: "tool", version: "0.8.0", status: "active", author: "L. Park",
    tags: ["lsp", "type-checking", "diagnostics"],
    usedByAgents: [{ name: "code-assist-v2", status: "running" }],
    inputSchema: [
      { name: "file_uri", type: "string", required: true, description: "File URI to analyze" },
      { name: "action", type: "enum", required: true, description: "diagnostics | hover | completion | references" },
    ],
    outputSchema: [
      { name: "diagnostics", type: "array", description: "Type errors, warnings, hints" },
      { name: "result", type: "object", description: "Action-specific result" },
    ],
    config: [],
    metrics: { calls24h: 3800, avgLatency: "35ms", errorRate: "0.4%", p99: "90ms" },
  },
  {
    id: "escalation-router", name: "escalation-router", displayName: "Escalation Router",
    description: "Multi-step workflow that evaluates conversation sentiment, detects agent uncertainty, and routes to human support with full context handoff.",
    category: "workflow", version: "1.0.0", status: "active", author: "A. Kovács",
    tags: ["escalation", "routing", "human-handoff"],
    usedByAgents: [{ name: "cx-support-v3", status: "running" }],
    inputSchema: [
      { name: "conversation", type: "array", required: true, description: "Full conversation history" },
      { name: "sentiment_score", type: "float", required: false, description: "Pre-computed sentiment (0-1)" },
    ],
    outputSchema: [
      { name: "should_escalate", type: "boolean", description: "Whether to escalate" },
      { name: "reason", type: "string", description: "Escalation reason" },
      { name: "priority", type: "enum", description: "low | medium | high | urgent" },
    ],
    config: [
      { key: "THRESHOLD", value: "0.85", secret: false },
      { key: "QUEUE_URL", value: "https://support.internal/queue", secret: false },
    ],
    metrics: { calls24h: 2200, avgLatency: "25ms", errorRate: "0.0%", p99: "60ms" },
  },
  {
    id: "sentiment-guard", name: "sentiment-guard", displayName: "Sentiment Guard",
    description: "Real-time sentiment analysis guardrail. Monitors agent responses for empathy, tone, and appropriateness. Blocks or rewrites responses that fail checks.",
    category: "guardrail", version: "0.9.3", status: "active", author: "A. Kovács",
    tags: ["sentiment", "safety", "guardrail"],
    usedByAgents: [{ name: "cx-support-v3", status: "running" }],
    inputSchema: [
      { name: "response", type: "string", required: true, description: "Agent response to evaluate" },
    ],
    outputSchema: [
      { name: "pass", type: "boolean", description: "Whether response passes checks" },
      { name: "score", type: "float", description: "Sentiment score (0-1)" },
      { name: "rewrite", type: "string", description: "Suggested rewrite if failed" },
    ],
    config: [
      { key: "MIN_EMPATHY_SCORE", value: "0.6", secret: false },
      { key: "BLOCK_ON_FAIL", value: "true", secret: false },
    ],
    metrics: { calls24h: 4820, avgLatency: "18ms", errorRate: "0.0%", p99: "45ms" },
  },
  {
    id: "toxicity-classifier", name: "toxicity-classifier", displayName: "Toxicity Classifier",
    description: "Classifies text for toxicity, hate speech, and harmful content. Uses LlamaGuard model for multi-label classification with configurable thresholds.",
    category: "guardrail", version: "1.0.0", status: "error", author: "A. Kovács",
    tags: ["toxicity", "safety", "moderation"],
    usedByAgents: [{ name: "safety-reviewer", status: "error" }],
    inputSchema: [
      { name: "text", type: "string", required: true, description: "Text to classify" },
    ],
    outputSchema: [
      { name: "safe", type: "boolean", description: "Whether content is safe" },
      { name: "categories", type: "object", description: "Per-category scores" },
    ],
    config: [
      { key: "MODEL_ENDPOINT", value: "http://guard-3b.staging-da:8000/v1", secret: false },
      { key: "THRESHOLD", value: "0.9", secret: false },
    ],
    metrics: { calls24h: 0, avgLatency: "\u2014", errorRate: "\u2014", p99: "\u2014" },
  },
  {
    id: "pii-detector", name: "pii-detector", displayName: "PII Detector",
    description: "Detects and redacts personally identifiable information (emails, phones, SSNs, credit cards) using regex + NER hybrid approach.",
    category: "guardrail", version: "0.8.0", status: "active", author: "R. Silva",
    tags: ["pii", "privacy", "redaction", "compliance"],
    usedByAgents: [{ name: "safety-reviewer", status: "error" }],
    inputSchema: [
      { name: "text", type: "string", required: true, description: "Text to scan for PII" },
      { name: "redact", type: "boolean", required: false, description: "Redact detected PII" },
    ],
    outputSchema: [
      { name: "entities", type: "array", description: "Detected PII entities with spans" },
      { name: "redacted_text", type: "string", description: "Text with PII redacted" },
    ],
    config: [
      { key: "NER_MODEL", value: "dslim/bert-base-NER", secret: false },
    ],
    metrics: { calls24h: 3100, avgLatency: "32ms", errorRate: "0.0%", p99: "80ms" },
  },
  {
    id: "sql-generator", name: "sql-generator", displayName: "SQL Generator",
    description: "Converts natural language questions to SQL queries. Introspects database schema, validates generated SQL, handles complex joins and aggregations.",
    category: "tool", version: "1.3.0", status: "active", author: "R. Silva",
    tags: ["sql", "database", "natural-language"],
    usedByAgents: [{ name: "data-analyst-v1", status: "deploying" }],
    inputSchema: [
      { name: "question", type: "string", required: true, description: "Natural language question" },
      { name: "dialect", type: "enum", required: false, description: "postgresql | mysql | sqlite" },
    ],
    outputSchema: [
      { name: "sql", type: "string", description: "Generated SQL query" },
      { name: "explanation", type: "string", description: "Query explanation" },
    ],
    config: [
      { key: "VALIDATE_BEFORE_EXEC", value: "true", secret: false },
    ],
    metrics: { calls24h: 340, avgLatency: "220ms", errorRate: "1.8%", p99: "680ms" },
  },
  {
    id: "chart-renderer", name: "chart-renderer", displayName: "Chart Renderer",
    description: "Generates interactive charts from data. Supports bar, line, scatter, pie, and heatmap. Outputs PNG, SVG, or interactive HTML.",
    category: "output", version: "0.9.1", status: "active", author: "R. Silva",
    tags: ["charts", "visualization", "data"],
    usedByAgents: [{ name: "data-analyst-v1", status: "deploying" }],
    inputSchema: [
      { name: "data", type: "array", required: true, description: "Data rows" },
      { name: "chart_type", type: "enum", required: true, description: "bar | line | scatter | pie | heatmap" },
    ],
    outputSchema: [
      { name: "image", type: "string", description: "Base64-encoded chart image" },
      { name: "html", type: "string", description: "Interactive HTML chart" },
    ],
    config: [
      { key: "DEFAULT_FORMAT", value: "svg", secret: false },
    ],
    metrics: { calls24h: 180, avgLatency: "340ms", errorRate: "0.5%", p99: "1.2s" },
  },
];

export const MCP_SERVERS: McpServer[] = [
  {
    id: "internal-crm",
    name: "Internal CRM",
    url: "https://crm.internal/mcp",
    transport: "SSE",
    status: "connected",
    latency: "45ms",
    description: "Customer relationship management system. Exposes order lookup, customer profiles, and subscription data.",
    toolCount: 8,
    connectedAgents: ["cx-support-v3", "onboarding-bot"],
    tools: ["get_order", "search_orders", "get_customer", "update_customer", "list_subscriptions", "get_invoice", "create_note", "get_interactions"],
    auth: "OAuth2",
    updatedAt: "2h ago",
  },
  {
    id: "stripe-billing",
    name: "Stripe Billing",
    url: "https://billing.internal/mcp",
    transport: "SSE",
    status: "connected",
    latency: "120ms",
    description: "Billing and payment processing. Provides subscription management, invoice generation, and payment method operations.",
    toolCount: 6,
    connectedAgents: ["cx-support-v3"],
    tools: ["create_subscription", "cancel_subscription", "update_plan", "get_invoice", "preview_proration", "list_payment_methods"],
    auth: "API Key",
    updatedAt: "1h ago",
  },
  {
    id: "jira-service",
    name: "Jira Service Desk",
    url: "https://jira.internal/mcp",
    transport: "Stdio",
    status: "connected",
    latency: "180ms",
    description: "IT service management. Creates and manages support tickets, queries knowledge base, and tracks SLA compliance.",
    toolCount: 5,
    connectedAgents: ["onboarding-bot"],
    tools: ["create_ticket", "update_ticket", "search_tickets", "add_comment", "get_sla_status"],
    auth: "Bearer Token",
    updatedAt: "4h ago",
  },
  {
    id: "github-server",
    name: "GitHub",
    url: "npx @modelcontextprotocol/server-github",
    transport: "Stdio",
    status: "connected",
    latency: "95ms",
    description: "GitHub repository operations. Provides code search, PR management, issue tracking, and file operations.",
    toolCount: 12,
    connectedAgents: ["code-assist-v2"],
    tools: ["search_code", "get_file_contents", "create_pull_request", "list_issues", "create_issue", "get_diff", "list_branches", "get_commit", "create_review", "merge_pr", "list_workflows", "get_workflow_run"],
    auth: "GitHub PAT",
    updatedAt: "30m ago",
  },
  {
    id: "qdrant-vector",
    name: "Qdrant Vector DB",
    url: "http://qdrant.internal:6333/mcp",
    transport: "SSE",
    status: "connected",
    latency: "12ms",
    description: "Vector database for semantic search. Manages collections, performs similarity search, and handles embeddings storage.",
    toolCount: 4,
    connectedAgents: ["cx-support-v3", "code-assist-v2", "data-analyst-v1"],
    tools: ["search", "upsert", "delete", "get_collection_info"],
    auth: "API Key",
    updatedAt: "5m ago",
  },
  {
    id: "slack-server",
    name: "Slack",
    url: "npx @modelcontextprotocol/server-slack",
    transport: "Stdio",
    status: "disconnected",
    latency: "\u2014",
    description: "Slack workspace integration. Send messages, read channels, manage threads, and post notifications.",
    toolCount: 7,
    connectedAgents: [],
    tools: ["send_message", "read_channel", "list_channels", "create_thread", "add_reaction", "upload_file", "get_user_info"],
    auth: "OAuth2",
    updatedAt: "3d ago",
  },
];
