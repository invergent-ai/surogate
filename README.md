<div align="center">
<a href="https://surogate.ai/">
<img width="50" alt="surogate llmops framework" src="./docs/static/img/logo-white.svg" />
</a>

<div align="center">
<h1>Surogate</h1>
<h3>The Enterprise LLMOps Framework</h3>
</div>

<div align="center">
    <a href="https://surogate.ai">Home Page</a> |
    <a href="https://docs.surogate.ai">Documentation</a> |
    <a href="https://github.com/invergent-ai/surogate/tree/master/examples">Examples</a> 
</div>

<br/>

<div align="center">
    
[![GitHub stars](https://img.shields.io/github/stars/invergent-ai/surogate?style=social)](https://github.com/invergent-ai/surogate)
[![GitHub issues](https://img.shields.io/github/issues/invergent-ai/surogate)](https://github.com/invergent-ai/surogate/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/invergent-ai/surogate)](https://github.com/invergent-ai/surogate/pulls)
[![Twitter Follow](https://img.shields.io/twitter/follow/invergentai?style=social)](https://twitter.com/invergentai)

</div>

<div align="center">
Do you like what we're doing? Give us a star ️⬆⭐
</div>
<br/>
</div>

Surogate is an end-to-end Enterprise LLMOps framework that simplifies the development, deployment, and maintenance of organization-specific Large Language Models (LLMs). It offers a complete toolkit and proven workflows for data processing, model training and fine-tuning, evaluation, quantization, and deployment — enabling efficient, reliable, and scalable LLM operations tailored to enterprise needs.

Surogate is built for enterprises that need fast experimentation scalability and predictable outcomes — whether running on-premise, in private clouds, or inside turnkey systems such as the DenseMAX Appliance.


## All-in-One LLMOps Platform
Everything required to build, adapt, deploy, and monitor generative AI systems:

* **Enterprise Knowledge Integration**: Tools for ingesting and processing proprietary documents, internal wikis, codebases, and domain-specific knowledge bases.
* **Custom Model Training**: Pre-training and continued pre-training on your organization's data, terminology, and business processes.
* **Fine-tuning & Alignment**: LoRA techniques to adapt models to internal language, workflows, and compliance requirements.
* **Reinforcement Learning** from Human Feedback: DPO, PPO, and GRPO workflows for aligning models with organizational values and safety standards.
* **Enterprise-Grade Security**: Role-based access control, audit logging, data encryption, and support for air-gapped deployments.
* **On-Premise & Private Cloud Deployment**: Full control over infrastructure with support for on-premise, private cloud, and hybrid environments.
* **Model Serving & Optimization**: High-throughput inference with KV-cache routing, GPU sharding, and quantization (4-bit, 8-bit, GPTQ, AWQ).
* **Comprehensive Evaluation**: Built-in benchmarks (MMLU, ARC, GSM8k, TruthfulQA) and Red Teaming evaluations.
* **Data Governance & Compliance**: Data management tools with versioning, lineage tracking, and compliance reporting for regulated industries.
* **Experiment Tracking & Versioning**: Built-in Data Hub for reproducible experiments, model versioning, and lifecycle management.
* **Synthetic Data Generation**: Create domain-specific training data and reward models tailored to enterprise use cases.
* **Model Distillation**: Compress larger models into efficient variants optimized for cost and latency.
* **Modular & Extensible**: Open architecture for integrating with existing enterprise systems and custom tooling.

## Why Surogate?
**Why build a LLMOps framework when you can piece together open-source tools?**: because building and maintaining reliable, scalable, and efficient LLM systems is **hard** — and the current open-source landscape is highly fragmented. 

Compared to generic, public LLMs, with **Enterprise LLM systems introduces additional challenges around security, compliance, scalability, and reliability that many community tools are not designed to address**:

1. **Data & Knowledge Requirements:**
    - ingest, understand, and reason over private, domain-specific data.
    - integration with proprietary knowledge bases, wikis, documents, and enterprise systems.
    - support for real-time or near-real-time updates to reflect organizational changes.
    - manage data silos, access control, and varying data quality.
    - contextual understanding of organizational structures, roles, products, processes, and policies.
2. **Security, Privacy & Compliance:**
    - strict access control, data governance, and identity integration (SSO, OAuth, LDAP, Azure AD).
    - operate within enterprise-grade security environments (VPC, on-prem, isolated compute).
    - comply with industry regulations (HIPAA, FINRA, GDPR, SOC 2, ISO 27001).
    - strong capabilities for auditing, traceability, and data lineage.
    - zero-trust architectural compatibility.
3. **Customization & Fine-Tuning:**
    - fine-tuning or domain adaptation on internal data. 
    - retraining pipelines for ongoing knowledge updates. 
    - multi-tenant customization for different teams or departments. 
    - LoRA, adapters, and structured grounding. 
    - policy-tuned behavior: response style, safety rules, escalation logic, and brand tone.
4. **Deployment & Infrastructure**:
    - deployment on-prem, private cloud, air-gapped, or hybrid environments.
    - scalable inference optimized for internal workloads and SLAs.
    - integrate with enterprise LLMOps pipelines for versioning, model lifecycle management, monitoring and observability
    - cost observability and quota management per department or user.
5. **Evaluation & Testing Requirements**:
    - evaluation against domain-specific benchmarks, not just general NLP benchmarks.
    - performance scoring on Internal correctness,  Policy compliance, PII handling, Procedural accuracy, Auditability & explainability
    - evaluate models in both QA and workflow/agent scenarios.
    - alignment with enterprise risk management frameworks.
6. **Operational Governance:**
    - well-defined processes for model approval and change management, human review workflows, output governance (redaction, PII blocking) and SLA definitions for accuracy and uptime
    - full audit trails of prompts, responses, model versions and user interactions
7. **Multi-Agent / Workflow Integration:**
    - integrate with internal APIs, Enterprise systems (ERP, CRM, ITSM, HRIS), Workflow engines and RPA systems
    - safe “action-taking” with role-based permissions.
    - guardrails to prevent unintended actions.
8. **Reliability & SLAs:**
    - meet strict availability, latency, and consistency requirements.
    - operate predictably under enterprise load patterns.
9. **Cost & Resource Management:**
    - detailed cost controls
    - Quantization, distillation, or custom inference optimizations to reduce cost.

## 


## Getting Started
Coming soon: installation guides, deployment configuration, developer docs, and API references.


# Installation

```bash
uv venv --python 3.12
sh requirements/raw-deps.sh
uv pip install -r requirements/torch29.txt
uv pip install -r requirements/build.txt
uv pip install -r requirements/common.txt
MAX_JOBS=8 uv pip install -r requirements/cuda.txt

# build sgl-kernel
# build lmcache

uv pip install "numpy==2.2.6"
rm -rf .venv/lib/python3.12/site-packages/triton_kernels
```

## Contributing
Contributions are welcome! Please open a PR or issue, and follow the contributing guidelines (to be published).

## License
Surogate is released under the Apache 2.0 License. See [LICENSE](./LICENSE) for details.

