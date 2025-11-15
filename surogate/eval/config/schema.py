# surogate/eval/config/schema.py
"""JSON Schema for configuration validation."""

# Metric schema (reusable)
METRIC_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "type": {"type": "string"},
        "criteria": {"type": "string"},
        "evaluation_params": {"type": "array"},
        "judge_model": {
            "type": "object",
            "properties": {
                "target": {"type": "string"}  # Reference to target name
            },
            "required": ["target"]
        },
        "window_size": {"type": "integer"},
        "key_info_threshold": {"type": "number"},
        "analyze_all_turns": {"type": "boolean"},
        "threshold": {"type": "number"},
        "threshold_ms": {"type": "number"},
        "min_tokens_per_sec": {"type": "number"},
        "min_rps": {"type": "number"},
        "backend": {"type": "string"},
        "bias_types": {"type": "array"},
        "harm_categories": {"type": "array"},
        "similarity_function": {"type": "string"},
        "metric_type": {"type": "string"}
    },
    "required": ["name", "type"]
}



BENCHMARK_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "path": {"type": "string"},  # NEW: Custom dataset path
        "num_fewshot": {"type": "integer", "minimum": 0},
        "limit": {
            "oneOf": [
                {"type": "integer", "minimum": 1},
                {"type": "number", "minimum": 0.0, "maximum": 1.0}
            ]
        },
        "tasks": {"type": "array", "items": {"type": "string"}},
        "subset": {  # NEW: Support subsets
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}}
            ]
        },
        "use_cache": {"type": "boolean"},
        "cache_dir": {"type": "string"},
        "backend_params": {"type": "object"},
        "judge_model": {
            "type": "object",
            "properties": {
                "target": {"type": "string"}
            },
            "required": ["target"]
        }
    },
    "required": ["name"]
}

# Update EVALUATION_SCHEMA - remove backend field
EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "dataset": {"type": "string"},
        "metrics": {
            "type": "array",
            "items": METRIC_SCHEMA
        },
        "benchmarks": {
            "type": "array",
            "items": BENCHMARK_SCHEMA
        }
    },
    "required": [],  # Nothing required at top level
    "anyOf": [
        {"required": ["dataset"]},      # Has dataset for custom metrics
        {"required": ["benchmarks"]},   # Has benchmarks for standard evaluation
        {
            "required": ["dataset", "benchmarks"]  # Has both
        }
    ]
}

# Infrastructure schema (reusable)
INFRASTRUCTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "backend": {"type": "string", "enum": ["local", "cloud", "kubernetes"]},
        "workers": {"type": "integer", "minimum": 1},
        "sandbox": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "timeout_seconds": {"type": "number"}
            }
        },
        "parallel_execution": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "max_workers": {"type": "integer", "minimum": 1}
            }
        }
    }
}

# Red teaming schema (reusable)
RED_TEAMING_SCHEMA = {
    "type": "object",
    "properties": {
        "enabled": {"type": "boolean"},
        "attacks": {"type": "array"},
        "attack_types": {"type": "object"},
        "vulnerabilities": {"type": "object"},
        "risk_assessment": {"type": "object"}
    }
}




# Guardrails schema (reusable)
GUARDRAILS_SCHEMA = {
    "type": "object",
    "properties": {
        "enabled": {"type": "boolean"},
        "input_guards": {"type": "array"},
        "output_guards": {"type": "array"},
        "pre_processing": {"type": "array"},
        "post_processing": {"type": "array"},
        "runtime": {"type": "object"},
        "custom_guards": {"type": "array"}
    }
}

# Stress testing schema (NEW)
STRESS_TESTING_SCHEMA = {
    "type": "object",
    "properties": {
        "enabled": {"type": "boolean"},
        "dataset": {"type": "string"},
        "num_concurrent": {"type": "integer", "minimum": 1},
        "num_requests": {"type": "integer", "minimum": 1},
        "duration_seconds": {"type": "integer", "minimum": 1},
        "progressive": {"type": "boolean"},
        "start_concurrent": {"type": "integer", "minimum": 1},
        "step_concurrent": {"type": "integer", "minimum": 1},
        "step_duration_seconds": {"type": "integer", "minimum": 1},
        "monitor_resources": {"type": "boolean"},
        "monitoring_interval": {"type": "number"},
        "warmup_requests": {"type": "integer", "minimum": 0},
        "max_failures": {"type": "integer", "minimum": 1},
        "retry_on_failure": {"type": "boolean"}
    }
}

# Main config schema
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "project": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["name"]
        },
        "targets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # Target identification
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["llm", "multimodal", "embedding", "reranker", "clip", "custom"]
                    },

                    # Model configuration
                    "provider": {
                        "type": "string",
                        "enum": ["openai", "anthropic", "azure", "cohere", "huggingface", "vllm", "ollama", "local",
                                 "custom"]
                    },
                    "model": {"type": "string"},
                    "model_path": {"type": "string"},
                    "base_url": {"type": "string"},
                    "api_key": {"type": "string"},
                    "endpoint": {"type": "string"},
                    "backend": {"type": "string", "enum": ["transformers", "vllm"]},
                    "device": {"type": "string"},
                    "timeout": {"type": "number"},
                    "headers": {"type": "object"},
                    "health_endpoint": {"type": "string"},
                    "load_in_8bit": {"type": "boolean"},
                    "load_in_4bit": {"type": "boolean"},
                    "tensor_parallel_size": {"type": "integer"},

                    # Target-specific configuration
                    "infrastructure": INFRASTRUCTURE_SCHEMA,
                    "evaluations": {
                        "type": "array",
                        "items": EVALUATION_SCHEMA
                    },
                    "stress_testing": STRESS_TESTING_SCHEMA,  # NEW
                    "red_teaming": RED_TEAMING_SCHEMA,
                    "guardrails": GUARDRAILS_SCHEMA,

                    "comment": {"type": "string"}
                },
                "required": ["name", "type"],
                "allOf": [
                    {
                        "if": {
                            "properties": {"type": {"enum": ["llm", "multimodal"]}}
                        },
                        "then": {
                            "required": ["provider", "model"]
                        }
                    },
                    {
                        "if": {
                            "properties": {"type": {"enum": ["embedding", "reranker"]}}
                        },
                        "then": {
                            "required": ["provider", "model"]
                        }
                    },
                    {
                        "if": {
                            "properties": {"provider": {"const": "local"}}
                        },
                        "then": {
                            "required": ["model"]
                        }
                    },
                    # MODIFIED: Only require api_key for specific providers that need it
                    {
                        "if": {
                            "properties": {"provider": {"enum": ["anthropic", "cohere", "azure"]}}
                        },
                        "then": {
                            "required": ["api_key"]
                        }
                    },
                    # OpenAI provider needs api_key UNLESS it's a local deployment (localhost)
                    {
                        "if": {
                            "allOf": [
                                {"properties": {"provider": {"const": "openai"}}},
                                {
                                    "not": {
                                        "properties": {
                                            "base_url": {
                                                "type": "string",
                                                "pattern": "^https?://localhost|^https?://127\\.0\\.0\\.1|^https?://0\\.0\\.0\\.0"
                                            }
                                        }
                                    }
                                }
                            ]
                        },
                        "then": {
                            "required": ["api_key"]
                        }
                    }
                ]
            },
            "minItems": 1
        }
    },
    "required": ["project", "targets"]
}