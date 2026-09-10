"""Text rollouts through the same native DSL model used for training.

Token steps are continuously batched on the trainer's workspace. Requests keep
independent paged attention history and recurrent/convolution states.
Weights and adapters belong to the trainer throughout generation.
"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from surogate.grpo.decode_scheduler import DecodeCapacityError, DecodeScheduler
from surogate.grpo.tool_protocol import request_tools, select_protocol


def shared_execution(config: dict, targets=(), tokenizer=None) -> str:
    """Select the optimized server where available, otherwise the training model."""
    from surogate.dsl import models  # noqa: F401
    from surogate.dsl.ir_builder import resolve_architecture
    from surogate.dsl.py_compiler import get_model_spec, list_registered_models

    architecture = resolve_architecture(config)
    text = config.get("text_config", config)
    identifiers = (architecture, config.get("model_type", ""), text.get("model_type", ""))
    if any("nemotron" in name.lower() for name in identifiers):
        raise ValueError("Nemotron is excluded from shared-model GRPO")
    # These definitions still contain deferred training operators. Registration
    # alone does not make them supported training models.
    deferred = {"deepseek_v4", "qwen4_exp", "qwen4_exp_text"}
    spec = next((s for name in list_registered_models() if (s := get_model_spec(name)).hf_config and
                 architecture in (s.hf_config.architecture, s.hf_config.model_type)), None)
    if spec is None or spec.hf_config.model_type in deferred:
        raise ValueError(f"{architecture} is not a supported training architecture for shared-model GRPO")
    if config.get("quantization_config") or text.get("quantization_config"):
        raise ValueError("shared-model GRPO requires an unquantized BF16 checkpoint")
    if spec.hf_config.architecture == "Gemma3TextModel" or text.get("use_bidirectional_attention"):
        raise ValueError("shared-model GRPO requires a causal language model")
    standard = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "all"}
    if spec.hf_config.model_type in ("qwen3", "qwen3_5", "qwen3_5_text") and not set(targets or ()) - standard:
        if tokenizer is not None:
            protocol = select_protocol(tokenizer, config, [dict(type="function", function=dict(name="f"))])
            if protocol.fallback or protocol.name not in ("json_xml", "qwen_coder"):
                return "training"
        return "serve"
    return "training"


class SharedModelServer:
    """HTTP rollout service with exclusive ownership of the resident trainer."""

    uses_live_adapter = True

    def __init__(self, trainer, tokenizer, config: dict, settings: dict):
        import torch

        self.trainer, self.tokenizer, self.config = trainer, tokenizer, config
        self.model = settings["model"]
        self.context = min(settings["max_context"], trainer.seq_length)
        self.capacity = settings["max_concurrency"]
        if self.context <= 1 or self.capacity < 1:
            raise ValueError("shared-model serving requires positive request capacity and context > 1")
        self.condition = threading.Condition()
        self.compute_lock = threading.Lock()
        self.sleeping, self.closed, self.active = True, False, 0
        self.version, self.adapter = -1, None
        # Check residency once and count unique underlying allocations, including
        # tied embedding/head aliases. The trainer owns every allocation.
        base = [torch.from_dlpack(t) for t in trainer.get_shared_base_weights().values()]
        allocations = {(t.data_ptr(), t.numel() * t.element_size()) for t in base}
        self.base_bytes = sum(size for _, size in allocations)
        text = config.get("text_config", config)
        self.persistent_decode = config.get("model_type") == "glm5_next" or text.get("model_type") in ("glm5_next", "glm5_next_text")
        if self.persistent_decode and not hasattr(trainer, "decode_logits"):
            raise ValueError("GLM persistent decode requires the current native training extension")
        self.vocab = text["vocab_size"]
        eos = settings.get("eos_token_id", text.get("eos_token_id", config.get("eos_token_id", tokenizer.eos_token_id)))
        self.eos = set(eos if isinstance(eos, list) else [eos]) - {None}
        self.scheduler = None
        if hasattr(trainer, "set_decode_cache_budget"):
            trainer.set_decode_cache_budget(settings.get("decode_cache_bytes", 0))
        if hasattr(trainer, "set_decode_memory_budget"):
            trainer.set_decode_memory_budget(settings.get("decode_memory_bytes", 0))
        if hasattr(trainer, "decode_batch_logits"):
            self.scheduler = DecodeScheduler(
                trainer, max_batch=self.capacity,
                prefill_chunk=max(1, settings.get("prefill_chunk", 256)),
                token_budget=settings.get("kv_capacity", self.context * self.capacity),
                prefix_entries=settings.get("decode_prefix_entries", 32),
            )
            self.persistent_decode = True
        # Some chat checkpoints keep the base-model EOS in config.json while
        # the tokenizer uses a different end-of-turn token (Qwen3.5-0.8B).
        if tokenizer.eos_token_id is not None:
            self.eos.add(tokenizer.eos_token_id)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(1200)

            def log_message(self, *args):
                pass

            def send_json(self, status, value):
                data = json.dumps(value, allow_nan=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path in ("/health", "/healthz"):
                    self.send_json(200, {"status": "ok"})
                elif self.path == "/v1/models":
                    names = [owner.model] + ([owner.adapter] if owner.adapter else [])
                    self.send_json(200, {"object": "list", "data": [
                        dict(id=name, object="model", created=0, owned_by="local") for name in names]})
                else:
                    self.send_json(404, {"error": {"message": "unknown endpoint"}})

            def do_POST(self):
                started = False
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    if length <= 0 or length > 8 << 20:
                        raise ValueError("invalid request body length")
                    body = json.loads(self.rfile.read(length))
                    if not isinstance(body, dict):
                        raise ValueError("request body must be an object")
                    if self.path in ("/tokenize", "/v1/tokenize"):
                        tokens = owner.tokenize(body)
                        self.send_json(200, dict(tokens=tokens, count=len(tokens), max_model_len=owner.context))
                        return
                    if self.path not in ("/v1/chat/completions", "/v1/chat/completions/tokens"):
                        self.send_json(404, {"error": {"message": "unknown endpoint"}})
                        return
                    with owner.admit(body.get("model")):
                        request = owner.prepare(body)
                        callback = None
                        if body.get("stream"):
                            self.send_response(200)
                            self.send_header("Content-Type", "text/event-stream")
                            self.end_headers()
                            started = True

                            def callback(chunk):
                                self.wfile.write(b"data: " + json.dumps(chunk, allow_nan=False).encode() + b"\n\n")
                                self.wfile.flush()

                        result = owner.generate(request, callback)
                        if callback:
                            callback(dict(id=result["id"], object="chat.completion.chunk", created=result["created"],
                                model=owner.adapter, choices=[dict(index=0, delta={},
                                finish_reason=result["choices"][0]["finish_reason"])], usage=result["usage"]))
                            self.wfile.write(b"data: [DONE]\n\n")
                        else:
                            self.send_json(200, result)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as exc:
                    status = 429 if isinstance(exc, (Busy, DecodeCapacityError)) else 400 if isinstance(exc, (ValueError, TypeError, KeyError)) else 500
                    error_type = {429: "capacity_error", 400: "invalid_request_error", 500: "server_error"}[status]
                    error = {"error": {"message": str(exc), "type": error_type, "code": status}}
                    try:
                        if started:
                            callback(error)
                            self.wfile.write(b"data: [DONE]\n\n")
                            self.wfile.flush()
                        else:
                            self.send_json(status, error)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

        self.http = ThreadingHTTPServer((settings["host"], settings["port"]), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()

    @contextmanager
    def admit(self, model):
        with self.condition:
            if self.sleeping or self.closed or self.active >= self.capacity:
                raise Busy("shared model is training or at request capacity; retry after publication")
            if model not in (self.model, self.adapter):
                raise ValueError("unknown model or adapter")
            self.active += 1
        try:
            yield
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()

    def begin_training(self):
        with self.condition:
            self.sleeping = True
            self.condition.notify_all()
            self.condition.wait_for(lambda: self.active == 0)
        if self.persistent_decode:
            self.trainer.reset_decode_state()
        if self.scheduler:
            self.scheduler.invalidate_prefixes()

    def publish(self, name, modules, version):
        with self.condition:
            if self.closed or not self.sleeping or self.active or version != self.version + 1:
                raise ValueError("publication requires a drained training phase and the next policy version")
            if modules:
                raise ValueError("shared training execution uses the live adapter")
            self.adapter, self.version, self.sleeping = name, version, False

    def summary(self):
        with self.condition:
            return dict(policy_version=self.version, shared_base_bytes=self.base_bytes,
                        serving_base_allocated_bytes=0, base_upload_bytes=0,
                        sleeping=self.sleeping, execution="training", persistent_decode=self.persistent_decode,
                        continuous_batching=self.scheduler is not None,
                        gpu_sampling=self.scheduler is not None and hasattr(self.trainer, "decode_batch_sample"),
                        **(self.scheduler.summary() if self.scheduler else {}))

    def close(self):
        with self.condition:
            if self.closed:
                return
            self.closed = self.sleeping = True
        self.http.shutdown()
        self.begin_training()
        if self.scheduler:
            self.scheduler.close()
        self.http.server_close()
        self.thread.join()
        self.trainer = None

    def tokenize(self, body):
        tools = request_tools(body)
        if "tokens" in body:
            tokens = body["tokens"]
        elif "messages" in body:
            messages = body["messages"]
            if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) for m in messages):
                raise ValueError("messages must be a nonempty list")
            kwargs = dict(body.get("chat_template_kwargs") or {})
            protocol = select_protocol(self.tokenizer, self.config, tools, kwargs)
            messages = protocol.messages(messages, tools)
            kwargs["chat_template"] = protocol.chat_template
            kwargs.update(tokenize=True, return_dict=False, add_generation_prompt=body.get("add_generation_prompt", True))
            # Use the same schema and template arguments on generation and on
            # Verifiers' /tokenize bridge calls between agent turns.
            kwargs["tools"] = tools or None
            if "reasoning_effort" in body:
                kwargs["reasoning_effort"] = body["reasoning_effort"]
            tokens = self.tokenizer.apply_chat_template(messages, **kwargs)
        else:
            tokens = self.tokenizer.encode(body["prompt"], add_special_tokens=body.get("add_special_tokens", True))
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or not 0 <= t < self.vocab for t in tokens):
            raise ValueError("prompt must contain valid model token IDs")
        return tokens

    def prepare(self, body):
        if body.get("n", 1) != 1 or body.get("response_format"):
            raise ValueError("shared training execution supports one completion per request, without response_format")
        tools = request_tools(body)
        prompt = self.tokenize(body)
        maximum = int(body.get("max_completion_tokens", body.get("max_tokens", 128)))
        minimum = int(body.get("min_tokens", 0))
        if len(prompt) >= self.context:
            raise ValueError("prompt exceeds the maximum context length")
        if maximum <= 0 or minimum < 0 or minimum > maximum:
            raise ValueError("invalid completion token limits")
        for key, default, low, high in (("temperature", 1., 0., math.inf), ("top_p", 1., 0., 1.),
                                      ("min_p", 0., 0., 1.), ("repetition_penalty", 1., 0., math.inf)):
            value = float(body.get(key, default))
            if not math.isfinite(value) or not low <= value <= high or (key in ("top_p", "repetition_penalty") and value == 0):
                raise ValueError(f"invalid {key}")
        for key in ("presence_penalty", "frequency_penalty"):
            if not math.isfinite(float(body.get(key, 0.))):
                raise ValueError(f"invalid {key}")
        if int(body.get("top_k", -1)) < -1:
            raise ValueError("top_k must be -1 or nonnegative")
        top_logprobs = int(body.get("top_logprobs", 0) or 0)
        if not 0 <= top_logprobs <= 20:
            raise ValueError("top_logprobs must be between 0 and 20")
        eos = self.eos | set(body.get("stop_token_ids") or [])
        if any(type(t) is not int or not 0 <= t < self.vocab for t in eos):
            raise ValueError("stop token is outside the vocabulary")
        bias = body.get("logit_bias") or {}
        if not isinstance(bias, dict) or any(not 0 <= int(t) < self.vocab or not math.isfinite(float(v)) for t, v in bias.items()):
            raise ValueError("invalid logit_bias")
        seed = body.get("seed")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("seed must be a nonnegative integer")
        stop = body.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]
        if not isinstance(stop, list) or any(not isinstance(s, str) or not s for s in stop):
            raise ValueError("stop must contain nonempty strings")
        protocol = select_protocol(self.tokenizer, self.config, tools, body.get("chat_template_kwargs"))
        tail = self.tokenizer.decode(prompt[-256:], skip_special_tokens=False)
        return body | dict(prompt_ids=prompt, maximum=min(maximum, self.context - len(prompt)),
                           minimum=minimum, stop=stop, parsed_tools=tools, protocol=protocol, prompt_tail=tail)

    def generate(self, request, callback=None):
        if self.scheduler:
            session = self.scheduler.new_session()
            try:
                return self._generate(request, callback,
                    lambda tokens, reset, sampling=None: self.scheduler.step(session, tokens, reset, sampling))
            finally:
                self.scheduler.release(session)
        # One workspace serves all admitted requests; begin_training waits for
        # both the running request and admitted requests waiting on this lock.
        with self.compute_lock:
            try:
                return self._generate(request, callback)
            finally:
                if self.persistent_decode:
                    self.trainer.reset_decode_state()

    def _generate(self, request, callback, decode=None):
        prompt = request["prompt_ids"]
        inputs = None if self.persistent_decode else np.zeros((self.trainer.batch_size, self.trainer.seq_length), dtype=np.int32)
        if inputs is not None:
            inputs[0, :len(prompt)] = prompt
        positions = np.zeros(self.trainer.batch_size, dtype=np.int32)
        rng = np.random.default_rng(request.get("seed"))
        ids, scores, text = [], [], ""
        emitted_fields = dict(content="", reasoning_content="")
        message = dict(role="assistant", content="")
        identifier, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
        finish = "length"
        eos = (set() if request.get("ignore_eos", False) else self.eos) | set(request.get("stop_token_ids") or [])
        gpu_sampling = decode is not None and hasattr(self.trainer, "decode_batch_sample")
        temperature = float(request.get("temperature", 1.))
        params = {key: float(request.get(key, default)) for key, default in (
            ("temperature", 1.), ("top_p", 1.), ("min_p", 0.), ("repetition_penalty", 1.),
            ("presence_penalty", 0.), ("frequency_penalty", 0.))}
        params.update(top_k=int(request.get("top_k", -1)), top_logprobs=int(request.get("top_logprobs", 0) or 0),
                      logit_bias={int(token): float(bias) for token, bias in (request.get("logit_bias") or {}).items()})
        for step in range(request["maximum"]):
            positions[0] = len(prompt) + step - 1
            if gpu_sampling:
                tokens = np.asarray(prompt if step == 0 else [ids[-1]], dtype=np.int32)
                sampled = decode(tokens, step == 0, params | dict(
                    uniform=float(rng.random()) if temperature > 0 else 0.,
                    blocked_tokens=sorted(eos) if step < request["minimum"] else []))
                if sampled["status"] == 1:
                    raise RuntimeError("model returned nonfinite logits")
                if sampled["status"]:
                    raise ValueError("sampling constraints removed every token")
            elif self.persistent_decode:
                tokens = np.asarray(prompt if step == 0 else [ids[-1]], dtype=np.int32)
                logits = (decode(tokens, step == 0) if decode else
                          self.trainer.decode_logits(tokens, reset=step == 0)).astype(np.float64)
            else:
                logits = self.trainer.next_token_logits(inputs, positions)[0].astype(np.float64)
            if not gpu_sampling:
                sampled = sample_logits(logits, prompt + ids, params, eos if step < request["minimum"] else (), rng)
            token = sampled["token"]
            ids.append(token)
            if inputs is not None:
                inputs[0, len(prompt) + step] = token
            piece = self.tokenizer.decode([token], skip_special_tokens=False)
            score = dict(token=piece, logprob=sampled["logprob"], bytes=list(piece.encode()), top_logprobs=[])
            for t, logprob in zip(sampled["top_ids"], sampled["top_logprobs"], strict=True):
                token_text = self.tokenizer.decode([int(t)], skip_special_tokens=False)
                score["top_logprobs"].append(dict(token=token_text, logprob=float(logprob), bytes=list(token_text.encode())))
            scores.append(score)
            stopped = token in eos
            # Tool/think delimiters may themselves be special tokens. Preserve
            # them for parsing, removing only a terminal model EOS from the
            # text view. The training token/logprob arrays retain every token.
            protocol = request["protocol"]
            # Harmony's terminal token identifies tool handoff versus a final
            # answer. LFM/Gemma may use their closing tool tag as EOS too.
            keep_terminal = protocol.name == "harmony" or piece in ("<|tool_call_end|>", "<tool_call|>", "</tool_call>")
            text_ids = ids[:-1] if stopped and token in self.eos and not keep_terminal else ids
            text = self.tokenizer.decode(text_ids, skip_special_tokens=False)
            if step + 1 >= request["minimum"]:
                matches = [text.index(stop) for stop in request["stop"] if stop in text]
                if matches:
                    text, stopped = text[:min(matches)], True
            final = stopped or step + 1 == request["maximum"]
            # Keep incomplete UTF-8 and potential stop-string prefixes buffered.
            stable = text if final else text[:len(text) - max((len(s) - 1 for s in request["stop"]), default=0)]
            if not final and "\ufffd" in stable:
                stable = stable[:stable.index("\ufffd")]
            parse_tools = request["parsed_tools"] if not final or stopped else []
            fields = protocol.parse(stable, parse_tools, prefix=request["prompt_tail"], final=final)
            if not request.get("parallel_tool_calls", True) and len(fields.get("tool_calls", [])) > 1:
                fields = protocol.parse(stable, [], prefix=request["prompt_tail"], final=final)
            delta = dict(role="assistant")
            for key in ("content", "reasoning_content"):
                value = fields[key]
                if value != emitted_fields[key]:
                    delta[key] = value[len(emitted_fields[key]):]
                emitted_fields[key] = value
            if fields.get("tool_calls"):
                delta["tool_calls"] = [dict(index=i, **call) for i, call in enumerate(fields["tool_calls"])]
            message = dict(role="assistant", content=fields["content"])
            if fields["reasoning_content"]:
                message["reasoning_content"] = fields["reasoning_content"]
            if fields.get("tool_calls"):
                message["tool_calls"] = fields["tool_calls"]
                message["content"] = fields["content"] or None
            if callback:
                callback(dict(id=identifier, created=created, object="chat.completion.chunk", model=self.adapter,
                    choices=[dict(index=0, delta=delta, token_ids=[token],
                                  logprobs=dict(content=[score]) if request.get("logprobs") else None, finish_reason=None)]))
            if stopped:
                finish = "stop"
                break
        if message.get("tool_calls"):
            finish = "tool_calls"
        return dict(id=identifier, created=created, object="chat.completion", model=self.adapter,
            prompt_token_ids=prompt, choices=[dict(index=0, message=message,
                token_ids=ids, logprobs=dict(content=scores) if request.get("logprobs") else None, finish_reason=finish)],
            usage=dict(prompt_tokens=len(prompt), completion_tokens=len(ids), total_tokens=len(prompt) + len(ids)))


class Busy(RuntimeError):
    pass


def sample_logits(logits, history, params, blocked, rng):
    """Compatibility sampler for trainers without the compact GPU sampling API."""
    logits = logits.astype(np.float64)
    if not np.isfinite(logits).all():
        raise RuntimeError("model returned nonfinite logits")
    temperature = params["temperature"]
    logits /= temperature if temperature > 0 else 1.
    logprobs = logits - np.logaddexp.reduce(logits)
    sampling = logits.copy()
    seen, counts = np.unique(history, return_counts=True)
    seen = seen.astype(np.int64)
    penalty = params["repetition_penalty"]
    sampling[seen] = np.where(sampling[seen] > 0, sampling[seen] / penalty, sampling[seen] * penalty)
    sampling[seen] -= params["presence_penalty"] + counts * params["frequency_penalty"]
    for token, bias in params["logit_bias"].items():
        sampling[token] += bias
    sampling[list(blocked)] = -np.inf
    # Stable ties prefer smaller token IDs on both CPU and GPU.
    order = np.argsort(-sampling, kind="stable")
    if params["top_k"] > 0:
        sampling[order[params["top_k"]:]] = -np.inf
    with np.errstate(invalid="ignore", divide="ignore"):
        probabilities = np.exp(sampling - np.max(sampling))
        probabilities /= probabilities.sum()
        cumulative = np.cumsum(probabilities[order]) - probabilities[order]
        sampling[order[cumulative >= params["top_p"]]] = -np.inf
        sampling[probabilities < probabilities.max() * params["min_p"]] = -np.inf
        probabilities = np.exp(sampling - np.max(sampling))
        probabilities /= probabilities.sum()
    if not np.isfinite(probabilities).all():
        raise ValueError("sampling constraints removed every token or model logits are nonfinite")
    token = int(np.argmax(sampling) if temperature == 0 else rng.choice(len(sampling), p=probabilities))
    top = np.argsort(-logprobs, kind="stable")[:params["top_logprobs"]] if params["top_logprobs"] else np.array([], dtype=np.int64)
    return dict(token=token, logprob=float(logprobs[token]), top_ids=top.tolist(), top_logprobs=logprobs[top].tolist())
