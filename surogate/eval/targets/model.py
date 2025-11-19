# surogate/eval/targets/model.py
import time

import httpx
from typing import Dict, Any, Optional, List
from surogate.utils.logger import get_logger

from .base import BaseTarget, TargetRequest, TargetResponse, ModelProvider, TargetType

logger = get_logger()


class APIModelTarget(BaseTarget):
    """Target for API-based models (OpenAI-compatible)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider = ModelProvider(config.get('provider', 'openai'))
        self.base_url = config.get('base_url', self._get_default_base_url())
        self.api_key = config.get('api_key')
        self.model = config.get('model')
        self.timeout = config.get('timeout', 60)

        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self._build_headers()
        )

    def _validate_config(self):
        """Validate API model configuration."""
        required = ['model']
        for field in required:
            if field not in self.config:
                raise ValueError(f"Missing required field: {field}")

    def _get_default_base_url(self) -> str:
        """Get default base URL for provider."""
        defaults = {
            ModelProvider.OPENAI: "https://api.openai.com/v1",
            ModelProvider.ANTHROPIC: "https://api.anthropic.com/v1",
            ModelProvider.AZURE: None,  # Must be provided
            ModelProvider.COHERE: "https://api.cohere.ai/v1",
        }
        return defaults.get(self.provider, "http://localhost:8000")

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}

        if self.api_key:
            if self.provider == ModelProvider.ANTHROPIC:
                headers["x-api-key"] = self.api_key
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"

        # Add custom headers from config
        custom_headers = self.config.get('headers', {})
        headers.update(custom_headers)

        return headers

    def send_request(self, request: TargetRequest) -> TargetResponse:
        """Send request to API model."""
        import time

        start_time = time.time()

        try:
            # Build request payload
            payload = self._build_payload(request)

            # Send request
            response = self.client.post("/chat/completions", json=payload)
            response.raise_for_status()

            # Parse response
            data = response.json()

            end_time = time.time()
            total_time = end_time - start_time

            result = self._parse_response(data)

            # Add timing information
            result.timing = {
                'total_time': total_time,
                'start_time': start_time,
                'end_time': end_time
            }

            return result

        except Exception as e:
            logger.error(f"Error calling model API: {e}")
            end_time = time.time()
            return TargetResponse(
                content="",
                raw_response={},
                metadata={},
                timing={'total_time': end_time - start_time},
                error=str(e)
            )

    def _build_payload(self, request: TargetRequest) -> Dict[str, Any]:
        """Build API request payload."""
        payload = {
            "model": self.model,
        }

        # Handle messages or prompt
        if request.messages:
            payload["messages"] = request.messages
        elif request.prompt:
            payload["messages"] = [{"role": "user", "content": request.prompt}]
        else:
            raise ValueError("Either messages or prompt must be provided")

        # Add parameters
        if request.parameters:
            payload.update(request.parameters)

        return payload

    def _parse_response(self, data: Dict[str, Any]) -> TargetResponse:
        """Parse API response to standard format."""
        content = ""

        # Extract content (OpenAI format)
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice:
                content = choice["message"].get("content", "")
            elif "text" in choice:
                content = choice["text"]

        metadata = {
            "model": data.get("model"),
            "usage": data.get("usage", {}),
            "finish_reason": data.get("choices", [{}])[0].get("finish_reason")
        }

        return TargetResponse(
            content=content,
            raw_response=data,
            metadata=metadata
        )

    def health_check(self) -> bool:
        """Check if API is accessible."""
        try:
            # For localhost/local endpoints - try various health endpoints
            # Check this FIRST before provider-specific logic
            if any(x in self.base_url for x in ['localhost', '127.0.0.1', '0.0.0.0']):
                # Try /v1/models first (most common for vLLM/OpenAI-compatible)
                try:
                    response = self.client.get("/v1/models", timeout=5)
                    if response.status_code == 200:
                        logger.debug(f"{self.name}: /v1/models endpoint healthy")
                        return True
                except:
                    pass

                # Try /models
                try:
                    response = self.client.get("/models", timeout=5)
                    if response.status_code == 200:
                        logger.debug(f"{self.name}: /models endpoint healthy")
                        return True
                except:
                    pass

                # Try /health
                try:
                    response = self.client.get("/health", timeout=5)
                    if response.status_code == 200:
                        logger.debug(f"{self.name}: /health endpoint healthy")
                        return True
                except:
                    pass

                logger.warning(f"Health check failed for {self.name}. Ensure server is running at {self.base_url}")
                return False

            # Special handling for OpenAI and Anthropic APIs (non-localhost)
            # Only check API key if it's actually needed (remote API)
            if self.provider in [ModelProvider.OPENAI, ModelProvider.ANTHROPIC]:
                has_key = bool(self.api_key)
                if not has_key:
                    logger.error(f"No API key provided for {self.name}")
                else:
                    logger.debug(f"{self.name}: API key present, assuming healthy")
                return has_key

            # For other remote APIs (OpenRouter, Cohere, etc.)
            try:
                response = self.client.get("/models", timeout=10)
                return response.status_code == 200
            except:
                logger.warning(f"Could not verify health for {self.name}")
                # Be optimistic if we have credentials
                return bool(self.api_key)

        except Exception as e:
            logger.error(f"Health check error for {self.name}: {e}")
            return False

    def cleanup(self):
        """Close HTTP client."""
        self.client.close()


class LocalModelTarget(BaseTarget):
    """Target for locally loaded models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_path = config.get('model_path')
        self.device = config.get('device', 'auto')
        self.load_in_8bit = config.get('load_in_8bit', False)
        self.load_in_4bit = config.get('load_in_4bit', False)

        self.model = None
        self.tokenizer = None
        self._load_model()

    def _validate_config(self):
        """Validate local model configuration."""
        required = ['model_path']
        for field in required:
            if field not in self.config:
                raise ValueError(f"Missing required field: {field}")

    def _load_model(self):
        """Load model using transformers or vLLM."""
        backend = self.config.get('backend', 'transformers')

        if backend == 'transformers':
            self._load_transformers_model()
        elif backend == 'vllm':
            self._load_vllm_model()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _load_transformers_model(self):
        """Load model using transformers."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading model from {self.model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device,
                load_in_8bit=self.load_in_8bit,
                load_in_4bit=self.load_in_4bit
            )

            logger.info("Model loaded successfully")

        except ImportError:
            raise ImportError("transformers not installed. Install with: pip install transformers")

    def _load_vllm_model(self):
        """Load model using vLLM."""
        try:
            from vllm import LLM

            logger.info(f"Loading model with vLLM from {self.model_path}")

            self.model = LLM(
                model=self.model_path,
                tensor_parallel_size=self.config.get('tensor_parallel_size', 1)
            )

            logger.info("Model loaded successfully with vLLM")

        except ImportError:
            raise ImportError("vllm not installed. Install with: pip install vllm")

    def send_request(self, request: TargetRequest) -> TargetResponse:
        """Generate response using local model."""
        backend = self.config.get('backend', 'transformers')

        if backend == 'transformers':
            return self._generate_transformers(request)
        elif backend == 'vllm':
            return self._generate_vllm(request)

    def _generate_transformers(self, request: TargetRequest) -> TargetResponse:
        """Generate using transformers."""
        try:
            start_time = time.time()
            # Get prompt
            if request.messages:
                prompt = self.tokenizer.apply_chat_template(
                    request.messages,
                    tokenize=False
                )
            elif request.prompt:
                prompt = request.prompt
            else:
                raise ValueError("Either messages or prompt required")

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            # Generate
            gen_params = request.parameters or {}
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=gen_params.get('max_tokens', 512),
                temperature=gen_params.get('temperature', 1.0),
                top_p=gen_params.get('top_p', 1.0),
            )

            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            end_time = time.time()

            return TargetResponse(
                content=generated_text,
                raw_response={'output': generated_text},
                metadata={'backend': 'transformers'},
                timing={
                    'total_time': end_time - start_time,
                    'start_time': start_time,
                    'end_time': end_time
                }
            )

        except Exception as e:
            logger.error(f"Generation error: {e}")
            end_time = time.time()
            return TargetResponse(
                content="",
                raw_response={},
                metadata={},
                timing={'total_time': end_time - start_time},
                error=str(e)
            )

    def _generate_vllm(self, request: TargetRequest) -> TargetResponse:
        """Generate using vLLM."""
        try:
            from vllm import SamplingParams
            start_time = time.time()
            # Get prompt
            if request.messages:
                # vLLM expects string prompts
                prompt = str(request.messages)
            elif request.prompt:
                prompt = request.prompt
            else:
                raise ValueError("Either messages or prompt required")

            # Sampling params
            gen_params = request.parameters or {}
            sampling_params = SamplingParams(
                temperature=gen_params.get('temperature', 1.0),
                top_p=gen_params.get('top_p', 1.0),
                max_tokens=gen_params.get('max_tokens', 512)
            )

            # Generate
            outputs = self.model.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text

            end_time = time.time()

            return TargetResponse(
                content=generated_text,
                raw_response={'output': generated_text},
                metadata={'backend': 'vllm'},
                timing={
                    'total_time': end_time - start_time,
                    'start_time': start_time,
                    'end_time': end_time
                }
            )

        except Exception as e:
            logger.error(f"Generation error: {e}")
            end_time = time.time()
            return TargetResponse(
                content="",
                raw_response={},
                metadata={},
                timing={'total_time': end_time - start_time},
                error=str(e)
            )


    def health_check(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def cleanup(self):
        """Cleanup model from memory."""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None



class EmbeddingTarget(BaseTarget):
    """Target for embedding models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider = config.get('provider', 'openai')
        self.model = config.get('model')
        self.base_url = config.get('base_url', 'https://api.openai.com/v1')

        if self.provider == 'openai':
            self.client = self._init_openai_client()
        elif self.provider == 'local':
            self._load_local_embeddings()

    def _validate_config(self):
        """Validate embedding configuration."""
        required = ['model']
        for field in required:
            if field not in self.config:
                raise ValueError(f"Missing required field: {field}")

    def _init_openai_client(self):
        """Initialize OpenAI client."""
        import httpx
        return httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.config.get('api_key')}",
                "Content-Type": "application/json"
            },
            timeout=self.config.get('timeout', 60)
        )

    def _load_local_embeddings(self):
        """Load local embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model}")
            self.model_obj = SentenceTransformer(self.model)
            logger.info("Embedding model loaded successfully")
        except ImportError:
            raise ImportError("sentence-transformers not installed")

    def send_request(self, request: TargetRequest) -> TargetResponse:
        """Get embeddings."""
        if self.provider == 'openai':
            return self._embed_openai(request)
        elif self.provider == 'local':
            return self._embed_local(request)

    # surogate/eval/targets/model.py - Update _embed_openai method

    def _embed_openai(self, request: TargetRequest) -> TargetResponse:
        """Get embeddings from OpenAI."""
        try:
            start_time = time.time()
            text = request.prompt or request.inputs.get('text')

            response = self.client.post(
                "/embeddings",
                json={
                    "model": self.model,
                    "input": text
                }
            )
            response.raise_for_status()
            data = response.json()

            end_time = time.time()

            embedding = data['data'][0]['embedding']

            return TargetResponse(
                content=str(embedding),
                raw_response=data,
                metadata={
                    'embedding': embedding,  # ← ADD THIS!
                    'dimension': len(embedding),
                    'usage': data.get('usage', {})
                },
                timing={
                    'total_time': end_time - start_time,
                    'start_time': start_time,
                    'end_time': end_time
                }
            )
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            end_time = time.time()
            return TargetResponse(
                content="",
                raw_response={},
                metadata={},
                timing={'total_time': end_time - start_time},
                error=str(e)
            )

    def _embed_local(self, request: TargetRequest) -> TargetResponse:
        """Get embeddings from local model."""
        try:
            start_time = time.time()
            text = request.prompt or request.inputs.get('text')
            embedding = self.model_obj.encode(text)

            end_time = time.time()

            return TargetResponse(
                content=str(embedding.tolist()),
                raw_response={'embedding': embedding.tolist()},
                metadata={
                    'embedding': embedding.tolist(),  # ← ADD THIS!
                    'dimension': len(embedding)
                },
                timing={
                    'total_time': end_time - start_time,
                    'start_time': start_time,
                    'end_time': end_time
                }
            )
        except Exception as e:
            logger.error(f"Local embedding error: {e}")
            end_time = time.time()
            return TargetResponse(
                content="",
                raw_response={},
                metadata={},
                timing={'total_time': end_time - start_time},
                error=str(e)
            )

    def health_check(self) -> bool:
        """Check if embedding service is available."""
        try:
            if self.provider == 'openai':
                return (self.client is not None and
                        self.config.get('api_key') is not None)
            return self.model_obj is not None
        except:
            return False

    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'client'):
            self.client.close()
        if hasattr(self, 'model_obj'):
            del self.model_obj


# surogate/eval/targets/model.py - Updated RerankerTarget

class RerankerTarget(BaseTarget):
    """Target for reranking models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider = config.get('provider', 'local')
        self.model = config.get('model')
        self.base_url = config.get('base_url')

        if self.provider == 'cohere':
            self.client = self._init_cohere_client()
        elif self.provider in ['openai', 'custom']:
            # OpenAI-compatible APIs (Gemini, Qwen, etc. via OpenRouter/proxies)
            self.client = self._init_openai_compatible_client()
        elif self.provider == 'local':
            self._load_local_reranker()

    def _validate_config(self):
        """Validate reranker configuration."""
        required = ['model']
        for field in required:
            if field not in self.config:
                raise ValueError(f"Missing required field: {field}")

    def _init_cohere_client(self):
        """Initialize Cohere client."""
        import httpx
        return httpx.Client(
            base_url=self.config.get('base_url', 'https://api.cohere.ai/v1'),
            headers={
                "Authorization": f"Bearer {self.config.get('api_key')}",
                "Content-Type": "application/json"
            },
            timeout=self.config.get('timeout', 60)
        )

    def _init_openai_compatible_client(self):
        """Initialize OpenAI-compatible client (for Gemini, Qwen, etc.)."""
        import httpx

        if not self.base_url:
            raise ValueError("base_url required for OpenAI-compatible reranker")

        headers = {
            "Content-Type": "application/json"
        }

        # Add API key if provided
        if self.config.get('api_key'):
            headers["Authorization"] = f"Bearer {self.config.get('api_key')}"

        # Add custom headers
        headers.update(self.config.get('headers', {}))

        return httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.config.get('timeout', 60)
        )

    def _load_local_reranker(self):
        """Load local reranker model."""
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker model: {self.model}")
            self.model_obj = CrossEncoder(self.model)
            logger.info("Reranker model loaded successfully")
        except ImportError:
            raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")

    def send_request(self, request: TargetRequest) -> TargetResponse:
        """Rerank documents."""
        if self.provider == 'cohere':
            return self._rerank_cohere(request)
        elif self.provider in ['openai', 'custom']:
            return self._rerank_openai_compatible(request)
        elif self.provider == 'local':
            return self._rerank_local(request)

    def _rerank_cohere(self, request: TargetRequest) -> TargetResponse:
        """Rerank using Cohere API."""
        import time
        start_time = time.time()

        try:
            query = request.inputs.get('query')
            documents = request.inputs.get('documents')
            top_n = request.parameters.get('top_n', len(documents)) if request.parameters else len(documents)

            response = self.client.post(
                "/rerank",
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n
                }
            )
            response.raise_for_status()
            data = response.json()

            end_time = time.time()

            # Extract reranked results
            results = data.get('results', [])
            reranked_docs = [
                {
                    'index': r['index'],
                    'document': documents[r['index']],
                    'relevance_score': r['relevance_score']
                }
                for r in results
            ]

            return TargetResponse(
                content=str(reranked_docs),
                raw_response=data,
                metadata={
                    'num_results': len(results),
                    'top_n': top_n,
                    'provider': 'cohere'
                },
                timing={
                    'total_time': end_time - start_time,
                    'start_time': start_time,
                    'end_time': end_time
                }
            )
        except Exception as e:
            logger.error(f"Cohere reranking error: {e}")
            end_time = time.time()
            return TargetResponse(
                content="",
                raw_response={},
                metadata={},
                timing={'total_time': end_time - start_time},
                error=str(e)
            )

    def _rerank_openai_compatible(self, request: TargetRequest) -> TargetResponse:
        """Rerank using OpenAI-compatible API (Gemini, Qwen, etc.)."""
        import time
        start_time = time.time()

        try:
            query = request.inputs.get('query')
            documents = request.inputs.get('documents')
            top_n = request.parameters.get('top_n', len(documents)) if request.parameters else len(documents)

            # OpenAI-compatible rerank format (similar to Cohere)
            response = self.client.post(
                "/rerank",
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n
                }
            )
            response.raise_for_status()
            data = response.json()

            end_time = time.time()

            # Parse response (try both Cohere and custom formats)
            results = data.get('results', data.get('data', []))

            reranked_docs = []
            for r in results:
                # Handle different response formats
                if 'index' in r:
                    reranked_docs.append({
                        'index': r['index'],
                        'document': documents[r['index']],
                        'relevance_score': r.get('relevance_score', r.get('score', 0.0))
                    })
                elif 'document' in r and 'score' in r:
                    # Some APIs return document directly
                    reranked_docs.append({
                        'index': r.get('index', len(reranked_docs)),
                        'document': r['document'],
                        'relevance_score': r['score']
                    })

            return TargetResponse(
                content=str(reranked_docs),
                raw_response=data,
                metadata={
                    'num_results': len(reranked_docs),
                    'top_n': top_n,
                    'provider': self.provider
                },
                timing={
                    'total_time': end_time - start_time,
                    'start_time': start_time,
                    'end_time': end_time
                }
            )
        except Exception as e:
            logger.error(f"OpenAI-compatible reranking error: {e}")
            end_time = time.time()
            return TargetResponse(
                content="",
                raw_response={},
                metadata={},
                timing={'total_time': end_time - start_time},
                error=str(e)
            )

    def _rerank_local(self, request: TargetRequest) -> TargetResponse:
        """Rerank using local model."""
        import time
        start_time = time.time()

        try:
            query = request.inputs.get('query')
            documents = request.inputs.get('documents')
            top_n = request.parameters.get('top_n', len(documents)) if request.parameters else len(documents)

            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]

            # Get scores
            scores = self.model_obj.predict(pairs)

            # Sort by score (descending)
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

            # Get top_n results
            reranked_docs = [
                {
                    'index': idx,
                    'document': documents[idx],
                    'relevance_score': float(scores[idx])
                }
                for idx in ranked_indices[:top_n]
            ]

            end_time = time.time()

            return TargetResponse(
                content=str(reranked_docs),
                raw_response={'results': reranked_docs},
                metadata={
                    'num_results': len(reranked_docs),
                    'top_n': top_n,
                    'provider': 'local'
                },
                timing={
                    'total_time': end_time - start_time,
                    'start_time': start_time,
                    'end_time': end_time
                }
            )
        except Exception as e:
            logger.error(f"Local reranking error: {e}")
            end_time = time.time()
            return TargetResponse(
                content="",
                raw_response={},
                metadata={},
                timing={'total_time': end_time - start_time},
                error=str(e)
            )

    def health_check(self) -> bool:
        """Check if reranker is available."""
        try:
            if self.provider in ['cohere', 'openai', 'custom']:
                return (self.client is not None and
                       self.config.get('api_key') is not None)
            return self.model_obj is not None
        except:
            return False

    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'client'):
            self.client.close()
        if hasattr(self, 'model_obj'):
            del self.model_obj

class CLIPTarget(BaseTarget):
    """Target for CLIP (Contrastive Language-Image Pre-training) models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider = config.get('provider', 'local')
        self.model = config.get('model', 'openai/clip-vit-base-patch32')

        if self.provider == 'openai':
            self.client = self._init_openai_client()
        elif self.provider == 'local':
            self._load_local_clip()

    def _validate_config(self):
        """Validate CLIP configuration."""
        # Model is optional as it has a default
        pass

    def _init_openai_client(self):
        """Initialize OpenAI client for CLIP."""
        import httpx
        return httpx.Client(
            base_url=self.config.get('base_url', 'https://api.openai.com/v1'),
            headers={
                "Authorization": f"Bearer {self.config.get('api_key')}",
                "Content-Type": "application/json"
            },
            timeout=self.config.get('timeout', 60)
        )

    def _load_local_clip(self):
        """Load local CLIP model."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            from PIL import Image

            logger.info(f"Loading CLIP model: {self.model}")
            self.model_obj = CLIPModel.from_pretrained(self.model)
            self.processor = CLIPProcessor.from_pretrained(self.model)
            logger.info("CLIP model loaded successfully")
        except ImportError:
            raise ImportError("transformers and PIL required. Install with: pip install transformers pillow")

    def send_request(self, request: TargetRequest) -> TargetResponse:
        """Process image-text pairs with CLIP."""
        if self.provider == 'openai':
            return self._process_openai(request)
        elif self.provider == 'local':
            return self._process_local(request)

    def _process_openai(self, request: TargetRequest) -> TargetResponse:
        """Process using OpenAI CLIP API."""
        import time
        start_time = time.time()

        try:
            # OpenAI doesn't have a direct CLIP API, use embeddings endpoint
            text = request.inputs.get('text')

            response = self.client.post(
                "/embeddings",
                json={
                    "model": self.model,
                    "input": text
                }
            )
            response.raise_for_status()
            data = response.json()

            end_time = time.time()

            embedding = data['data'][0]['embedding']

            return TargetResponse(
                content=str(embedding),
                raw_response=data,
                metadata={'dimension': len(embedding)},
                timing={
                    'total_time': end_time - start_time,
                    'start_time': start_time,
                    'end_time': end_time
                }
            )
        except Exception as e:
            logger.error(f"OpenAI CLIP error: {e}")
            end_time = time.time()
            return TargetResponse(
                content="",
                raw_response={},
                metadata={},
                timing={'total_time': end_time - start_time},
                error=str(e)
            )

    def _process_local(self, request: TargetRequest) -> TargetResponse:
        """Process using local CLIP model."""
        import time
        start_time = time.time()

        try:
            from PIL import Image
            import torch

            task = request.inputs.get('task', 'similarity')  # similarity, classify, embed

            if task == 'similarity':
                # Calculate similarity between image and text
                image = request.inputs.get('image')  # Can be path or PIL Image
                text = request.inputs.get('text')  # Single text or list

                if isinstance(image, str):
                    image = Image.open(image)

                inputs = self.processor(
                    text=text if isinstance(text, list) else [text],
                    images=image,
                    return_tensors="pt",
                    padding=True
                )

                outputs = self.model_obj(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

                end_time = time.time()

                return TargetResponse(
                    content=str(probs.tolist()),
                    raw_response={'probabilities': probs.tolist()},
                    metadata={'task': 'similarity'},
                    timing={
                        'total_time': end_time - start_time,
                        'start_time': start_time,
                        'end_time': end_time
                    }
                )

            elif task == 'embed_text':
                # Get text embeddings
                text = request.inputs.get('text')
                inputs = self.processor(text=[text], return_tensors="pt", padding=True)
                text_features = self.model_obj.get_text_features(**inputs)

                end_time = time.time()

                return TargetResponse(
                    content=str(text_features.tolist()),
                    raw_response={'embedding': text_features.tolist()},
                    metadata={'task': 'embed_text', 'dimension': text_features.shape[-1]},
                    timing={
                        'total_time': end_time - start_time,
                        'start_time': start_time,
                        'end_time': end_time
                    }
                )

            elif task == 'embed_image':
                # Get image embeddings
                image = request.inputs.get('image')
                if isinstance(image, str):
                    image = Image.open(image)

                inputs = self.processor(images=image, return_tensors="pt")
                image_features = self.model_obj.get_image_features(**inputs)

                end_time = time.time()

                return TargetResponse(
                    content=str(image_features.tolist()),
                    raw_response={'embedding': image_features.tolist()},
                    metadata={'task': 'embed_image', 'dimension': image_features.shape[-1]},
                    timing={
                        'total_time': end_time - start_time,
                        'start_time': start_time,
                        'end_time': end_time
                    }
                )

            else:
                raise ValueError(f"Unknown task: {task}")

        except Exception as e:
            logger.error(f"Local CLIP error: {e}")
            end_time = time.time()
            return TargetResponse(
                content="",
                raw_response={},
                metadata={},
                timing={'total_time': end_time - start_time},
                error=str(e)
            )

    def health_check(self) -> bool:
        """Check if CLIP model is available."""
        try:
            if self.provider == 'openai':
                response = self.client.get("/models")
                return response.status_code == 200
            return self.model_obj is not None and self.processor is not None
        except:
            return False

    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'client'):
            self.client.close()
        if hasattr(self, 'model_obj'):
            del self.model_obj
        if hasattr(self, 'processor'):
            del self.processor