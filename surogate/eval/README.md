## 1. Core Infrastructure

### 1.1 Configuration Management
- **JSON config parser** - Custom (simple file reading)
- **YAML config support** - DeepTeam (use existing library)
- **Config validation** - Custom (JSON schema validation)

### 1.2 Execution Backends
- **Local execution backend** - EvalScope
- **Sandbox environments** - EvalScope

### 1.3 Parallel Execution
- **Multi-worker support** - EvalScope
- **Result aggregation** - Custom (simple dict merging)

## 2. Model & Application Support

### 2.1 Model Types
- **Large Language Models (LLMs)** - EvalScope, DeepEval
- **Multimodal Large Models** - EvalScope, DeepEval
- **Embedding Models** - EvalScope
- **CLIP Models** - EvalScope

### 2.2 Application Types
- **AI Agents** - DeepEval
- **Chatbots** - DeepEval
- **RAG Systems** - DeepEval
- **MCP (Model Context Protocol)** - DeepEval

### 2.3 Target Configuration
- **Model endpoint configuration** - Custom (simple HTTP client wrapper)
- **Application endpoint configuration** - Custom (simple HTTP client wrapper)

## 3. Dataset Management

### 3.1 Dataset Handling
- **Custom dataset support** - EvalScope, DeepEval
- **JSONL format support** - DeepEval
- **CSV format support** - Custom (use pandas)
- **Dataset validation** - Custom (basic schema check)
- **Mixed data evaluation** - EvalScope

### 3.2 Test Cases
- **Test case framework** - DeepEval, DeepTeam
- **Test case loading from files** - Custom (file I/O)

### 3.3 Prompts
- **Prompt management** - DeepEval
- **Prompt templates** - Custom (string templating)

## 4. Functional Evaluation

### 4.1 Core Metrics
- **G-Eval** - DeepEval
- **Conversational G-Eval** - DeepEval
- **Multimodal G-Eval** - DeepEval
- **Arena G-Eval** - DeepEval
- **DAG (Directed Acyclic Graph)** - DeepEval
- **Conversational DAG** - DeepEval

### 4.2 RAG-Specific Metrics
- **Context relevance** - DeepEval
- **Answer relevance** - DeepEval
- **Faithfulness** - DeepEval
- **Answer correctness** - DeepEval

### 4.3 Agentic Metrics
- **Tool usage correctness** - DeepEval
- **Action sequence validity** - DeepEval
- **Goal achievement** - DeepEval

### 4.4 Multi-Turn Metrics
- **Conversation coherence** - DeepEval
- **Context retention** - DeepEval
- **Turn-level analysis** - DeepEval

### 4.5 MCP Metrics
- **Protocol compliance** - DeepEval
- **Context management** - DeepEval

### 4.6 Safety Metrics
- **Toxicity detection** - DeepEval
- **Bias detection** - DeepEval
- **Harm assessment** - DeepEval

### 4.7 Non-LLM Metrics
- **Embedding similarity** - DeepEval
- **Classification metrics** - DeepEval

## 5. Performance Evaluation

### 5.1 Speed Benchmarking
- **Latency measurement** - EvalScope
- **Throughput measurement** - EvalScope
- **Token generation speed** - EvalScope

### 5.2 Stress Testing
- **Model inference stress testing** - EvalScope
- **Concurrent request handling** - EvalScope

## 6. Standard Benchmarks

### 6.1 Academic Benchmarks
- **MMLU** - EvalScope, DeepEval
- **HellaSwag** - DeepEval
- **BIG-Bench Hard** - DeepEval
- **TruthfulQA** - DeepEval
- **ARC** - DeepEval
- **Winogrande** - DeepEval
- **LAMBADA** - DeepEval

### 6.2 Reading Comprehension
- **SQuAD** - DeepEval
- **DROP** - DeepEval
- **BoolQ** - DeepEval

### 6.3 Reasoning Benchmarks
- **GSM8K** - DeepEval
- **MathQA** - DeepEval
- **LogiQA** - DeepEval

### 6.4 Coding Benchmarks
- **HumanEval** - DeepEval
- **IFEval** - DeepEval

### 6.5 Specialized Benchmarks
- **BBQ (Bias Benchmark)** - DeepEval

## 7. Third-Party Benchmarks

### 7.1 External Tools
- **τ-bench** - EvalScope
- **τ²-bench** - EvalScope
- **BFCL-v3** - EvalScope
- **BFCL-v4** - EvalScope
- **Needle in a Haystack** - EvalScope
- **ToolBench** - EvalScope
- **LongBench-Write** - EvalScope

## 8. Red-Teaming & Security

### 8.1 Adversarial Attacks
- **Single-turn attacks** - DeepTeam
- **Multi-turn attacks** - DeepTeam
- **Jailbreak attempts** - DeepTeam, DeepEval
- **Prompt injection** - DeepTeam, DeepEval
- **Role-play attacks** - DeepTeam
- **Gradual escalation** - DeepTeam
- **Context manipulation** - DeepTeam

### 8.2 Vulnerability Scanning
- **Data Privacy vulnerabilities** - DeepTeam, DeepEval
- **PII leakage detection** - DeepTeam, DeepEval
- **Prompt leakage** - DeepTeam, DeepEval
- **Bias detection** - DeepTeam, DeepEval
- **Toxicity detection** - DeepTeam, DeepEval
- **Misinformation susceptibility** - DeepTeam, DeepEval
- **Illegal activity prompting** - DeepTeam, DeepEval
- **Personal safety threats** - DeepTeam, DeepEval
- **Unauthorized access** - DeepTeam, DeepEval
- **Intellectual property leakage** - DeepTeam, DeepEval
- **Excessive agency** - DeepTeam, DeepEval
- **Graphic content generation** - DeepTeam, DeepEval
- **Competitive information leakage** - DeepTeam, DeepEval
- **Robustness testing** - DeepTeam
- **Custom vulnerability definition** - DeepTeam

### 8.3 Risk Assessment
- **Risk scoring** - DeepTeam
- **Severity classification** - DeepTeam
- **Attack success rate tracking** - DeepTeam
- **Vulnerability reporting** - DeepTeam

## 9. Guardrails

### 9.1 Input Guardrails
- **Input validation** - DeepTeam
- **Content filtering** - DeepTeam
- **PII detection and redaction** - DeepTeam
- **Prompt injection detection** - DeepTeam

### 9.2 Output Guardrails
- **Output filtering** - DeepTeam
- **Toxicity filtering** - DeepTeam
- **PII filtering** - DeepTeam
- **Hallucination detection** - DeepTeam
- **Factuality checking** - DeepTeam

### 9.3 Runtime Guardrails
- **Rate limiting** - DeepTeam
- **Content moderation** - DeepTeam

### 9.4 Custom Guards
- **Custom guard framework** - DeepTeam

## 10. Compliance & Frameworks

### 10.1 Security Frameworks
- **OWASP Top 10 for LLMs** - DeepTeam
- **NIST AI RMF** - DeepTeam
- **MITRE ATLAS** - DeepTeam
- **BeaverTails** - DeepTeam
- **Aegis** - DeepTeam

### 10.2 Compliance Reporting
- **Framework compliance scoring** - DeepTeam
- **Custom policy support** - DeepTeam

## 11. Testing & Development

### 11.1 Component Testing
- **Component-level evaluation** - DeepEval
- **Retriever testing** - DeepEval
- **LLM component testing** - DeepEval
- **Isolated component metrics** - DeepEval

### 11.2 Integration Testing
- **End-to-end evaluation** - DeepEval
- **Integration test cases** - DeepEval
- **Multi-component workflows** - DeepEval

### 11.3 Unit Testing
- **Unit test framework** - DeepEval

### 11.4 CI/CD Integration
- **GitHub Actions integration** - DeepEval
- **Quality gates** - DeepEval

## 12. Monitoring & Observability

### 12.1 Tracing
- **Execution tracing** - DeepEval

### 12.2 Logging
- **Structured logging** - Custom (use Python logging library)

## 13. Comparative Evaluation

### 13.1 Arena Mode
- **Model comparison** - EvalScope, DeepEval
- **Head-to-head evaluation** - EvalScope
- **Multi-model benchmarking** - EvalScope

## 14. Reporting

### 14.1 Report Generation
- **JSON reports** - Custom (json.dump)
- **HTML reports** - Custom (simple Jinja2 template)
- **Markdown reports** - Custom (string formatting)

### 14.2 Report Content
- **Executive summary** - Custom (simple aggregation)
- **Detailed metrics** - Custom (dict to table)
- **Vulnerability reports** - DeepTeam
- **Compliance reports** - DeepTeam

## 15. AIGC Evaluation

### 15.1 AI-Generated Content
- **AIGC evaluation** - EvalScope