# surogate/eval/config/schema.py
"""JSON Schema for configuration validation."""

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
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["llm", "multimodal", "embedding", "reranker", "clip", "rag", "agent", "chatbot", "mcp", "custom"]
                    },
                    "provider": {
                        "type": "string",
                        "enum": ["openai", "anthropic", "azure", "cohere", "huggingface", "vllm", "ollama", "local", "custom"]
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
                    "comment": {"type": "string"}  # Allow comments in config
                },
                "required": ["name", "type"],
                "allOf": [
                    {
                        "if": {
                            "properties": {"type": {"const": "llm"}}
                        },
                        "then": {
                            "required": ["provider", "model"]
                        }
                    },
                    {
                        "if": {
                            "properties": {"type": {"const": "embedding"}}
                        },
                        "then": {
                            "required": ["provider", "model"]
                        }
                    },
                    {
                        "if": {
                            "properties": {"type": {"const": "reranker"}}
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
                    {
                        "if": {
                            "properties": {"provider": {"enum": ["openai", "anthropic", "cohere"]}}
                        },
                        "then": {
                            "required": ["api_key"]
                        }
                    }
                ]
            },
            "minItems": 1
        },
        "evaluation": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "datasets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "path": {"type": "string"},
                            "format": {"type": "string", "enum": ["json", "jsonl", "csv", "parquet"]}
                        },
                        "required": ["path"]
                    }
                },
                "metrics": {
                    "type": "object",
                    "properties": {
                        "functional": {"type": "array"},
                        "performance": {"type": "array"}
                    }
                },
                "benchmarks": {
                    "type": "object",
                    "properties": {
                        "standard": {"type": "array"},
                        "third_party": {"type": "array"}
                    }
                }
            }
        },
        "red_teaming": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "attack_types": {"type": "object"},
                "vulnerabilities": {"type": "object"},
                "risk_assessment": {"type": "object"}
            }
        },
        "guardrails": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "pre_processing": {"type": "array"},
                "post_processing": {"type": "array"},
                "runtime": {"type": "object"},
                "custom_guards": {"type": "array"}
            }
        },
        "testing": {
            "type": "object",
            "properties": {
                "unit_tests": {"type": "object"},
                "integration_tests": {"type": "object"},
                "component_tests": {"type": "array"}
            }
        },
        "infrastructure": {
            "type": "object",
            "properties": {
                "backend": {"type": "string", "enum": ["local", "cloud", "kubernetes"]},
                "workers": {"type": "integer"},
                "sandbox": {"type": "object"},
                "parallel_execution": {"type": "object"}
            }
        },
        "monitoring": {
            "type": "object",
            "properties": {
                "tracing": {"type": "object"},
                "logging": {"type": "object"}
            }
        },
        "reporting": {
            "type": "object",
            "properties": {
                "format": {"type": "array"},
                "output_dir": {"type": "string"},
                "include": {"type": "object"},
                "compare_with": {"type": "object"}
            }
        },
        "ci_cd": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "blocking": {"type": "object"}
            }
        },
        "compliance": {
            "type": "object",
            "properties": {
                "frameworks": {"type": "array"}
            }
        },
        "arena": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "models": {"type": "array"},
                "comparison_metrics": {"type": "array"},
                "dataset": {"type": "string"}
            }
        }
    },
    "required": ["project", "targets"]
}