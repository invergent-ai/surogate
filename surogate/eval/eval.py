from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import re

import evalscope
from evalscope.constants import EvalType
from evalscope.run import run_task
from swift import get_logger

from surogate.utils.config import load_config

logger = get_logger()


class SurogateEval:
    def __init__(self, **kwargs):
        self.args = kwargs
        self.config = load_config(self.args['config'])

    def _resolve_env_vars(self, value: Any) -> Any:
        """
        Recursively resolve environment variables in config values.
        Supports ${VAR_NAME} and $VAR_NAME syntax.
        """
        if isinstance(value, str):
            def replace_env(match):
                var_name = match.group(1) or match.group(2)
                env_value = os.getenv(var_name)
                if env_value is None:
                    logger.warning(f"Environment variable {var_name} not found, keeping placeholder")
                    return match.group(0)
                return env_value

            # Match ${VAR} or $VAR patterns
            value = re.sub(r'\$\{([^}]+)\}|\$([A-Z_][A-Z0-9_]*)', replace_env, value)
            return value

        elif isinstance(value, dict):
            return {k: self._resolve_env_vars(v) for k, v in value.items()}

        elif isinstance(value, list):
            return [self._resolve_env_vars(item) for item in value]

        return value

    def _apply_env_mappings(self, config: Dict[str, Any], mappings: Dict[str, str]) -> Dict[str, Any]:
        """Apply environment variable mappings to config"""
        for config_key, env_var in mappings.items():
            if config_key not in config or config[config_key] is None:
                env_value = os.getenv(env_var)
                if env_value:
                    config[config_key] = env_value
                    logger.debug(f"Mapped {env_var} -> {config_key}")
        return config

    def _process_target_config(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process target configuration generically.
        Passes through ALL parameters, resolves env vars, handles multiple naming conventions.
        """
        # First, resolve all environment variables
        target = self._resolve_env_vars(target.copy())

        # Apply environment variable mappings if specified
        if target.get('env_mappings'):
            target = self._apply_env_mappings(target, target['env_mappings'])

        processed = {}

        # Model name (required for most evals)
        if target.get('model'):
            processed['model'] = target['model']

        # API URL - support multiple naming conventions
        api_url = (target.get('api_url') or
                   target.get('api_base') or
                   target.get('base_url') or
                   target.get('endpoint'))
        if api_url:
            processed['api_url'] = api_url

        # API Key - try multiple naming conventions and common env vars
        api_key = (target.get('api_key') or
                   target.get('auth_token') or
                   target.get('bearer_token') or
                   target.get('access_token'))

        if api_key is None:
            # Try common environment variable names
            for env_name in ['API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY',
                             'OPENROUTER_API_KEY', 'DASHSCOPE_API_KEY', 'HF_TOKEN',
                             'HUGGINGFACE_TOKEN', 'MISTRAL_API_KEY', 'COHERE_API_KEY']:
                api_key = os.getenv(env_name)
                if api_key:
                    logger.debug(f"Using API key from {env_name}")
                    break

        if api_key:
            processed['api_key'] = api_key

        # Initialize model_args dict
        if 'model_args' not in processed:
            processed['model_args'] = {}

        # Headers - add to model_args as default_headers (OpenAI client parameter)
        headers = target.get('headers', {})
        if headers:
            processed['model_args']['default_headers'] = headers

        # Request parameters - merge into model_args
        if target.get('request_params'):
            processed['model_args'].update(target['request_params'])

        # Model args from config - merge without overwriting
        model_args = target.get('model_args', {})
        if model_args:
            for key, value in model_args.items():
                if key not in processed['model_args']:
                    processed['model_args'][key] = value

        # Generation config - pass through everything
        if target.get('generation_config'):
            processed['generation_config'] = target['generation_config']

        # Pass through any other fields that might be needed
        passthrough_fields = [
            'model_type', 'pooling_mode', 'max_seq_length', 'prompt',
            'encode_kwargs', 'is_cross_encoder', 'model_name_or_path',
            'config_kwargs', 'tokenizer_kwargs', 'device', 'batch_size',
            'trust_remote_code', 'revision', 'torch_dtype'
        ]

        for field in passthrough_fields:
            if field in target and target[field] is not None:
                processed[field] = target[field]

        return processed

    def _build_task_config(self) -> Dict[str, Any]:
        """Build evalscope TaskConfig from YAML configuration"""
        # Check for multi-model configuration
        if self.config.get('models'):
            return self._build_multi_model_config()

        # Determine eval backend
        eval_config = self.config.get('eval', {})
        backend = eval_config.get('backend')

        # Start with empty config
        task_cfg = {}

        if backend:
            task_cfg['eval_backend'] = backend
            if eval_config.get('eval_config'):
                # Pass through backend-specific config as-is
                task_cfg['eval_config'] = self._resolve_env_vars(eval_config['eval_config'])

        # Route to specific builder based on enabled features
        if self._is_enabled('embedding_reranker_eval'):
            return self._build_embedding_reranker_config()
        elif self._is_enabled('rag_eval'):
            return self._build_rag_eval_config()
        elif self._is_enabled('vlm_eval'):
            return self._build_vlm_eval_config()
        elif self._is_enabled('aigc'):
            return self._build_aigc_config()
        elif self._is_enabled('stress_test'):
            return self._build_stress_test_config()
        elif self._is_enabled('arena'):
            return self._build_arena_config()
        elif self._is_enabled('third_party'):
            return self._build_third_party_config()
        elif self._is_enabled('sandbox'):
            return self._build_sandbox_config()
        elif self._is_enabled('clip_eval'):
            return self._build_clip_eval_config()
        else:
            return self._build_standard_config()

    def _build_multi_model_config(self) -> Dict[str, Any]:
        """Build configuration for multiple models"""
        task_cfg = {}

        models = self.config.get('models', [])
        processed_models = []

        for model_cfg in models:
            processed = self._process_target_config(model_cfg)
            processed_models.append(processed)

        task_cfg['models'] = processed_models

        # Add eval type
        eval_config = self.config.get('eval', {})
        eval_type = eval_config.get('type', 'service')
        eval_type_map = {
            'service': EvalType.SERVICE,
            'local': EvalType.LOCAL,
            'custom': EvalType.CUSTOM
        }
        task_cfg['eval_type'] = eval_type_map.get(eval_type, EvalType.SERVICE)

        # Add benchmarks if specified
        if self.config.get('benchmarks'):
            task_cfg['datasets'] = self._process_benchmarks(self.config['benchmarks'])

        # Pass through all other parameters
        self._add_common_params(task_cfg)

        return task_cfg

    def _is_enabled(self, section: str) -> bool:
        """Check if a config section is enabled"""
        section_config = self.config.get(section, {})
        return section_config.get('enabled', False)

    def _build_standard_config(self) -> Dict[str, Any]:
        """Build config for standard LLM/VLM evaluation"""
        task_cfg = {}

        # Process target configuration generically
        target = self.config.get('target', {})
        processed_target = self._process_target_config(target)
        task_cfg.update(processed_target)

        eval_config = self.config.get('eval', {})
        eval_type_map = {
            'service': EvalType.SERVICE,
            'local': EvalType.CHECKPOINT,
            'checkpoint': EvalType.CHECKPOINT,
            'custom': EvalType.CUSTOM
        }
        eval_type = eval_config.get('type', 'service')
        task_cfg['eval_type'] = eval_type_map.get(eval_type, EvalType.SERVICE)

        # Benchmarks/datasets
        benchmarks = self.config.get('benchmarks')
        if benchmarks:
            task_cfg['datasets'] = self._process_benchmarks(benchmarks)

            # Add per-dataset args if any were collected
            if hasattr(self, '_dataset_args_per_benchmark'):
                if 'dataset_args' not in task_cfg:
                    task_cfg['dataset_args'] = {}
                task_cfg['dataset_args'].update(self._dataset_args_per_benchmark)

                # ALSO check if there's a global limit to apply
                # Try adding 'limit' at top level as well
                if any('num_samples' in args for args in self._dataset_args_per_benchmark.values()):
                    # Check if all datasets have the same limit
                    limits = [args.get('num_samples') for args in self._dataset_args_per_benchmark.values() if
                              'num_samples' in args]
                    if limits and all(l == limits[0] for l in limits):
                        task_cfg['limit'] = limits[0]

        # Custom datasets
        custom_datasets = self.config.get('custom_datasets')
        if custom_datasets:
            if 'datasets' not in task_cfg:
                task_cfg['datasets'] = []
            task_cfg['datasets'].extend(self._process_generic_items(custom_datasets))

        # Add common parameters
        self._add_common_params(task_cfg)

        return task_cfg

    def _add_common_params(self, task_cfg: Dict[str, Any]):
        """Add common parameters to task config - passed through as-is"""
        # General parameters
        params = self.config.get('parameters', {})
        for key, value in params.items():
            if value is not None and key not in task_cfg:
                task_cfg[key] = value

        # Output configuration - FIX: use 'work_dir' instead of 'output_dir'
        output = self.config.get('output', {})
        if output.get('dir'):
            # Create directory if it doesn't exist
            output_path = Path(output['dir'])
            output_path.mkdir(parents=True, exist_ok=True)

            # Try both possible parameter names
            task_cfg['work_dir'] = str(output_path)  # Common in evalscope

        # Pass through other output configs without 'output_' prefix
        for key, value in output.items():
            if key != 'dir' and value is not None:
                task_cfg[key] = value

    def _build_embedding_reranker_config(self) -> Dict[str, Any]:
        """Build config for embedding/reranker evaluation (MTEB/CMTEB)"""
        task_cfg = {
            'eval_backend': 'RAGEval',
            'eval_config': {}
        }

        embed_config = self.config['embedding_reranker_eval']
        target = self.config.get('target', {})

        # Tool selection
        task_cfg['eval_config']['tool'] = embed_config.get('tool', 'MTEB')

        # Model configuration
        models = []
        evaluation_mode = embed_config.get('evaluation_mode', 'one_stage')

        if evaluation_mode == 'two_stage':
            retrieval_model = embed_config.get('retrieval_model') or target
            reranker_model = embed_config.get('reranker_model')

            models.append(self._build_model_config(retrieval_model, is_reranker=False))
            if reranker_model:
                models.append(self._build_model_config(reranker_model, is_reranker=True))
        else:
            models.append(self._build_model_config(target))

        task_cfg['eval_config']['model'] = models

        # Pass through all eval settings
        eval_settings = {}
        for key, value in embed_config.items():
            if key not in ['enabled', 'evaluation_mode', 'retrieval_model', 'reranker_model'] and value is not None:
                eval_settings[key] = value

        task_cfg['eval_config']['eval'] = eval_settings

        return task_cfg

    def _build_model_config(self, model_cfg: Dict[str, Any], is_reranker: bool = False) -> Dict[str, Any]:
        """Build model configuration generically for embedding/reranker"""
        # Resolve env vars
        model_cfg = self._resolve_env_vars(model_cfg.copy())

        config = {
            'model_name_or_path': model_cfg.get('model'),
        }

        # Pass through all fields
        for key, value in model_cfg.items():
            if key == 'model':
                continue  # Already handled as model_name_or_path
            elif key == 'model_args':
                config['model_kwargs'] = value
            elif key == 'api_url' or key == 'api_key' or key == 'headers':
                continue  # Not needed for local embedding models
            else:
                config[key] = value

        # Set defaults only if not specified
        if 'model_kwargs' not in config:
            config['model_kwargs'] = {'torch_dtype': 'auto'}
        if 'encode_kwargs' not in config:
            config['encode_kwargs'] = {'batch_size': 128}
        if 'max_seq_length' not in config:
            config['max_seq_length'] = 512
        if 'prompt' not in config:
            config['prompt'] = ''
        if 'is_cross_encoder' not in config:
            config['is_cross_encoder'] = is_reranker

        return config

    def _build_rag_eval_config(self) -> Dict[str, Any]:
        """Build config for RAG evaluation - pass through all fields"""
        task_cfg = {
            'eval_backend': 'RAGEval',
            'eval_config': {'tool': 'RAGAS'}
        }

        rag_config = self.config['rag_eval']
        target = self.config.get('target', {})

        # Pass through all rag_config fields
        for key, value in rag_config.items():
            if key != 'enabled' and value is not None:
                task_cfg['eval_config'][key] = value

        # Add target info
        if target.get('model'):
            task_cfg['eval_config']['model'] = target['model']
        if target.get('api_url'):
            task_cfg['eval_config']['api_url'] = target['api_url']
        if target.get('api_key'):
            task_cfg['eval_config']['api_key'] = target['api_key']

        return task_cfg

    def _build_clip_eval_config(self) -> Dict[str, Any]:
        """Build config for CLIP evaluation - pass through all fields"""
        return self._build_generic_backend_config('clip_eval', 'RAGEval', 'CLIP_Benchmark')

    def _build_vlm_eval_config(self) -> Dict[str, Any]:
        """Build config for VLM evaluation - pass through all fields"""
        return self._build_generic_backend_config('vlm_eval', 'VLMEvalKit')

    def _build_aigc_config(self) -> Dict[str, Any]:
        """Build config for AIGC evaluation - pass through all fields"""
        return self._build_generic_feature_config('aigc')

    def _build_stress_test_config(self) -> Dict[str, Any]:
        """Build config for stress testing - pass through all fields"""
        return self._build_generic_feature_config('stress_test')

    def _build_arena_config(self) -> Dict[str, Any]:
        """Build config for arena mode - pass through all fields"""
        task_cfg = {}
        arena_config = self.config['arena']

        # Process models
        models = arena_config.get('models', [])
        if models:
            task_cfg['models'] = [self._process_target_config(m) for m in models]

        # Pass through all other fields
        for key, value in arena_config.items():
            if key not in ['enabled', 'models'] and value is not None:
                task_cfg[key] = value

        # Add benchmarks
        if self.config.get('benchmarks'):
            task_cfg['datasets'] = self._process_benchmarks(self.config['benchmarks'])

        return task_cfg

    def _build_third_party_config(self) -> Dict[str, Any]:
        """Build config for third-party tools - pass through all fields"""
        return self._build_generic_backend_config('third_party', 'ThirdParty')

    def _build_sandbox_config(self) -> Dict[str, Any]:
        """Build config for sandbox - pass through all fields"""
        return self._build_generic_feature_config('sandbox')

    def _build_generic_backend_config(self, section: str, backend: str, tool: str = None) -> Dict[str, Any]:
        """Generic builder for backend-based configs"""
        task_cfg = {
            'eval_backend': backend,
            'eval_config': {}
        }

        section_config = self.config[section]
        target = self.config.get('target', {})

        if tool:
            task_cfg['eval_config']['tool'] = tool

        # Pass through all fields from section
        for key, value in section_config.items():
            if key != 'enabled' and value is not None:
                task_cfg['eval_config'][key] = value

        # Add target info
        processed_target = self._process_target_config(target)
        for key, value in processed_target.items():
            if key not in task_cfg['eval_config']:
                task_cfg['eval_config'][key] = value

        return task_cfg

    def _build_generic_feature_config(self, section: str) -> Dict[str, Any]:
        """Generic builder for feature-based configs"""
        task_cfg = {}

        section_config = self.config[section]
        target = self.config.get('target', {})

        # Add processed target
        task_cfg.update(self._process_target_config(target))

        # Pass through all section fields
        for key, value in section_config.items():
            if key != 'enabled' and value is not None:
                task_cfg[key] = value

        return task_cfg

    def _process_benchmarks(self, benchmarks: List[Any]) -> List[Any]:
        """Process benchmark configurations into evalscope format"""
        processed = []
        for item in benchmarks:
            if isinstance(item, str):
                processed.append(item)
            elif isinstance(item, dict):
                item = self._resolve_env_vars(item)
                bench_name = item.get('name')
                if not bench_name:
                    logger.warning(f"Benchmark config missing 'name' field: {item}")
                    continue

                # Always just append the benchmark name
                processed.append(bench_name)

                # Store ALL config (subset, limit, etc.) in dataset_args
                if not hasattr(self, '_dataset_args_per_benchmark'):
                    self._dataset_args_per_benchmark = {}

                dataset_arg = {}

                # Handle subsets
                if item.get('subset') or item.get('subsets'):
                    subsets = item.get('subset') or item.get('subsets')
                    if isinstance(subsets, str):
                        subsets = [subsets]
                    dataset_arg['subset_list'] = subsets

                # Handle limit
                if item.get('limit'):
                    dataset_arg['num_samples'] = item['limit']

                if dataset_arg:
                    self._dataset_args_per_benchmark[bench_name] = dataset_arg
            else:
                processed.append(item)

        return processed

    def _process_generic_items(self, items: List[Any]) -> List[Any]:
        """Generic processor - just resolves env vars and passes through"""
        processed = []
        for item in items:
            if isinstance(item, (dict, list, str)):
                processed.append(self._resolve_env_vars(item))
            else:
                processed.append(item)
        return processed

    def debug_config(self):
        """Debug: print the built configuration without running"""
        import json

        task_cfg = self._build_task_config()
        print("\n" + "=" * 60)
        print("YAML Config Input:")
        print("=" * 60)
        print(json.dumps(self.config, indent=2, default=str))

        print("\n" + "=" * 60)
        print("Generated EvalScope TaskConfig:")
        print("=" * 60)
        print(json.dumps(task_cfg, indent=2, default=str))
        print("=" * 60 + "\n")

        return task_cfg

    def run(self):
        """Run evaluation based on configuration"""
        batch_config = self.config.get('batch', {})

        if batch_config.get('enabled'):
            return self.run_batch()
        else:
            task_cfg = self._build_task_config()
            logger.info(f"Running evaluation with config: {task_cfg}")
            return run_task(task_cfg=task_cfg)

    def run_batch(self):
        """Run batch evaluation"""
        batch_config = self.config.get('batch', {})
        tasks = batch_config.get('tasks', [])

        if not tasks:
            logger.warning("Batch mode enabled but no tasks specified")
            return

        results = []
        for task in tasks:
            logger.info(f"Running batch task: {task}")
            # Resolve env vars in task config
            task = self._resolve_env_vars(task)
            result = run_task(task_cfg=task)
            results.append(result)

        return results