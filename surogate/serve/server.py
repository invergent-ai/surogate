from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

from surogate import _surogate
from surogate.core.config.serve_config import ServeConfig
from surogate.dsl.ir_builder import build_dsl_ir_for_model
from surogate.kernels.jit_compile import compile_jit_kernels
from surogate.utils.hf import get_model_weights_path
from surogate.utils.logger import get_logger

logger = get_logger()


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer", "tool"]
    content: Any


class ChatCompletionsRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    stream: bool = False
    stop: Optional[str | list[str]] = None

    # Non-standard but useful for parity with existing internal generation knobs.
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None


class CompletionsRequest(BaseModel):
    model: Optional[str] = None
    prompt: str | list[str]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    stream: bool = False
    stop: Optional[str | list[str]] = None

    # Non-standard sampling knobs.
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None


@dataclass
class GeneratedChoice:
    index: int
    text: str
    token_ids: list[int]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


def _normalize_stop(stop: Optional[str | list[str]]) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    return [s for s in stop if isinstance(s, str) and s]


def _apply_stop(text: str, stop_strings: Sequence[str]) -> tuple[str, bool]:
    if not stop_strings:
        return text, False
    min_pos = -1
    for s in stop_strings:
        p = text.find(s)
        if p >= 0 and (min_pos < 0 or p < min_pos):
            min_pos = p
    if min_pos >= 0:
        return text[:min_pos], True
    return text, False


def _sse(data: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


class NativeServingRuntime:
    def __init__(self, config: ServeConfig):
        self.config = config
        self.model_id = config.model_id or config.model or "native"
        self._lock = threading.Lock()

        if "SUROGATE_MIN_STACK_MB" not in os.environ:
            os.environ["SUROGATE_MIN_STACK_MB"] = str(config.min_stack_mb)

        model_dir = self._resolve_model_dir(config.model or "")
        logger.info(f"Loading tokenizer for {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=config.trust_remote_code
        )
        self.eos_id = self.tokenizer.eos_token_id or 151643

        logger.info(f"Building DSL IR for {model_dir}")
        ir_json = build_dsl_ir_for_model(model_dir)
        jit_manifests = compile_jit_kernels(ir_json)

        options = _surogate.RuntimeOptions()
        options.dsl_ir_json = ir_json
        if jit_manifests:
            options.jit_kernel_manifests = jit_manifests
        options.use_cuda_graphs = bool(config.use_cuda_graphs)
        options.doc_masking = True
        options.recompute = "true"

        logger.info(f"Creating native serving trainer for {model_dir} ({config.gpus} GPUs)")
        self.trainer = _surogate.SurogateTrainer(
            ngpu=config.gpus,
            config=_surogate.PretrainedConfig.from_pretrained(model_dir, config.dtype),
            options=options,
            batch_size=config.batch_size,
            seq_len=config.sequence_len,
            grad_accum=1,
            memcpy_all_gather=True,
            memcpy_send_recv=True,
        )

        model_weights_path = get_model_weights_path(model_dir)
        logger.info(f"Importing weights from {model_weights_path}")
        self.trainer.import_weights(model_weights_path)
        logger.info("Native inference runtime ready")

    @staticmethod
    def _resolve_model_dir(model: str) -> str:
        if os.path.isdir(model):
            return model
        if os.path.isfile(model):
            return model

        from huggingface_hub import snapshot_download

        logger.info(f"Resolving model from HF: {model}")
        return snapshot_download(model)

    def ensure_auth(self, request: Request):
        api_key = self.config.api_key
        if not api_key:
            return
        auth = request.headers.get("authorization", "")
        expected = f"Bearer {api_key}"
        if auth != expected:
            raise HTTPException(status_code=401, detail="Invalid API key")

    def _messages_to_prompt_text(self, messages: list[ChatMessage]) -> str:
        normalized: list[dict[str, str]] = []
        for m in messages:
            content = m.content
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict):
                        if "text" in part and isinstance(part["text"], str):
                            parts.append(part["text"])
                        elif part.get("type") == "text" and isinstance(
                            part.get("text"), str
                        ):
                            parts.append(part["text"])
                text = "\n".join(parts)
            else:
                text = str(content)
            normalized.append({"role": m.role, "content": text})

        try:
            return self.tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return "\n".join(
                f"{m['role']}: {m['content']}" for m in normalized
            ) + "\nassistant:"

    def _generate(
        self,
        prompt_ids: list[int],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
        stop_strings: Sequence[str],
    ) -> list[GeneratedChoice]:
        with self._lock:
            tokens, _, prompt_lens, completion_lens = self.trainer.generate(
                prompts=[prompt_ids],
                num_completions=n,
                max_gen_len=max_tokens,
                temperature=temperature,
                eos_token_id=self.eos_id,
                use_lora=self.config.use_lora,
                use_cuda_graphs=self.config.use_cuda_graphs,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                prefill_chunk_size=self.config.prefill_chunk_size,
                repetition_penalty=repetition_penalty,
            )

        choices: list[GeneratedChoice] = []
        for i in range(len(tokens)):
            pl = int(prompt_lens[i])
            cl = int(completion_lens[i])
            raw_ids = [int(t) for t in tokens[i][pl : pl + cl]]
            raw_text = self.tokenizer.decode(raw_ids, skip_special_tokens=True)

            trimmed_text, stop_hit = _apply_stop(raw_text, stop_strings)
            if stop_hit:
                out_ids = self.tokenizer.encode(
                    trimmed_text, add_special_tokens=False
                )
            else:
                out_ids = raw_ids

            if stop_hit:
                finish_reason = "stop"
            elif raw_ids and raw_ids[-1] == self.eos_id:
                finish_reason = "stop"
            elif cl >= max_tokens:
                finish_reason = "length"
            else:
                finish_reason = "stop"

            choices.append(
                GeneratedChoice(
                    index=i,
                    text=trimmed_text,
                    token_ids=[int(t) for t in out_ids],
                    finish_reason=finish_reason,
                    prompt_tokens=pl,
                    completion_tokens=len(out_ids),
                )
            )
        return choices

    def generate_for_chat(self, req: ChatCompletionsRequest) -> list[GeneratedChoice]:
        if req.model and req.model not in {self.model_id, self.config.model}:
            raise ValueError(f"Unknown model '{req.model}'")
        if req.n <= 0:
            raise ValueError("`n` must be > 0")

        prompt_text = self._messages_to_prompt_text(req.messages)
        prompt_ids = self.tokenizer.encode(prompt_text)

        max_tokens = int(
            req.max_completion_tokens
            if req.max_completion_tokens is not None
            else (req.max_tokens if req.max_tokens is not None else self.config.max_gen_len)
        )
        temperature = (
            float(req.temperature)
            if req.temperature is not None
            else self.config.temperature
        )
        top_p = float(req.top_p) if req.top_p is not None else self.config.top_p
        top_k = int(req.top_k) if req.top_k is not None else self.config.top_k
        min_p = float(req.min_p) if req.min_p is not None else self.config.min_p
        repetition_penalty = (
            float(req.repetition_penalty)
            if req.repetition_penalty is not None
            else self.config.repetition_penalty
        )

        return self._generate(
            prompt_ids,
            n=req.n,
            max_tokens=max(1, max_tokens),
            temperature=temperature,
            top_k=max(0, top_k),
            top_p=top_p,
            min_p=max(0.0, min_p),
            repetition_penalty=max(1e-6, repetition_penalty),
            stop_strings=_normalize_stop(req.stop),
        )

    def generate_for_completion(self, req: CompletionsRequest) -> list[GeneratedChoice]:
        if req.model and req.model not in {self.model_id, self.config.model}:
            raise ValueError(f"Unknown model '{req.model}'")
        if req.n <= 0:
            raise ValueError("`n` must be > 0")

        if isinstance(req.prompt, list):
            if len(req.prompt) == 0:
                raise ValueError("`prompt` cannot be empty")
            if len(req.prompt) > 1:
                raise ValueError("This phase-1 server supports one prompt per request")
            prompt_text = req.prompt[0]
        else:
            prompt_text = req.prompt
        prompt_ids = self.tokenizer.encode(prompt_text)

        max_tokens = int(req.max_tokens or self.config.max_gen_len)
        temperature = (
            float(req.temperature)
            if req.temperature is not None
            else self.config.temperature
        )
        top_p = float(req.top_p) if req.top_p is not None else self.config.top_p
        top_k = int(req.top_k) if req.top_k is not None else self.config.top_k
        min_p = float(req.min_p) if req.min_p is not None else self.config.min_p
        repetition_penalty = (
            float(req.repetition_penalty)
            if req.repetition_penalty is not None
            else self.config.repetition_penalty
        )

        return self._generate(
            prompt_ids,
            n=req.n,
            max_tokens=max(1, max_tokens),
            temperature=temperature,
            top_k=max(0, top_k),
            top_p=top_p,
            min_p=max(0.0, min_p),
            repetition_penalty=max(1e-6, repetition_penalty),
            stop_strings=_normalize_stop(req.stop),
        )


def _usage_from_choices(choices: list[GeneratedChoice]) -> dict[str, int]:
    prompt_tokens = choices[0].prompt_tokens if choices else 0
    completion_tokens = sum(c.completion_tokens for c in choices)
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens + completion_tokens),
    }


def _chat_stream_chunks(
    request_id: str, created: int, model_id: str, choices: list[GeneratedChoice]
):
    async def _gen():
        for c in choices:
            yield _sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": c.index,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            if c.text:
                yield _sse(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {
                                "index": c.index,
                                "delta": {"content": c.text},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            yield _sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": c.index,
                            "delta": {},
                            "finish_reason": c.finish_reason,
                        }
                    ],
                }
            )
        yield b"data: [DONE]\n\n"

    return _gen()


def _completion_stream_chunks(
    request_id: str, created: int, model_id: str, choices: list[GeneratedChoice]
):
    async def _gen():
        for c in choices:
            if c.text:
                yield _sse(
                    {
                        "id": request_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {
                                "index": c.index,
                                "text": c.text,
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            yield _sse(
                {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": c.index,
                            "text": "",
                            "finish_reason": c.finish_reason,
                        }
                    ],
                }
            )
        yield b"data: [DONE]\n\n"

    return _gen()


def create_app(runtime: NativeServingRuntime) -> FastAPI:
    app = FastAPI(title="Surogate Native OpenAI-Compatible Server", version="0.1.0")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
    async def v1_root():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(request: Request):
        runtime.ensure_auth(request)
        created = int(time.time())
        model_id = runtime.model_id
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "surogate",
                    "root": model_id,
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionsRequest, request: Request):
        runtime.ensure_auth(request)
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        try:
            choices = await asyncio.to_thread(runtime.generate_for_chat, req)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error("chat completion failed", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

        if req.stream:
            return StreamingResponse(
                _chat_stream_chunks(request_id, created, runtime.model_id, choices),
                media_type="text/event-stream",
            )

        usage = _usage_from_choices(choices)
        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": runtime.model_id,
            "choices": [
                {
                    "index": c.index,
                    "message": {"role": "assistant", "content": c.text},
                    "finish_reason": c.finish_reason,
                }
                for c in choices
            ],
            "usage": usage,
        }
        return JSONResponse(content=response)

    @app.post("/v1/completions")
    async def completions(req: CompletionsRequest, request: Request):
        runtime.ensure_auth(request)
        request_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        try:
            choices = await asyncio.to_thread(runtime.generate_for_completion, req)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error("completion failed", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

        if req.stream:
            return StreamingResponse(
                _completion_stream_chunks(request_id, created, runtime.model_id, choices),
                media_type="text/event-stream",
            )

        usage = _usage_from_choices(choices)
        response = {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": runtime.model_id,
            "choices": [
                {
                    "index": c.index,
                    "text": c.text,
                    "finish_reason": c.finish_reason,
                }
                for c in choices
            ],
            "usage": usage,
        }
        return JSONResponse(content=response)

    return app


def serve(config: ServeConfig):
    runtime = NativeServingRuntime(config)
    app = create_app(runtime)
    logger.info(f"Starting native inference server on {config.host}:{config.port}")
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)
