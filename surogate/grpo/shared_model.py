"""Text rollouts through the same native DSL model used for training.

This path recomputes the prefix and serializes generation on the trainer's
workspace. It needs neither a second model nor serving-specific weight layouts.
"""

from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
import time
import uuid

import numpy as np


def shared_execution(config: dict, targets=()) -> str:
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
    deferred = {"deepseek_v4", "qwen4_exp", "qwen4_exp_text", "glm5_next"}
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
        return "serve"
    return "training"


class SharedModelServer:
    """HTTP rollout service with exclusive ownership of the resident trainer."""

    uses_live_adapter = True

    def __init__(self, trainer, tokenizer, config: dict, settings: dict):
        import torch

        self.trainer, self.tokenizer = trainer, tokenizer
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
        self.vocab = text["vocab_size"]
        eos = settings.get("eos_token_id", text.get("eos_token_id", config.get("eos_token_id", tokenizer.eos_token_id)))
        self.eos = set(eos if isinstance(eos, list) else [eos]) - {None}
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
                    if not started:
                        status = 429 if isinstance(exc, Busy) else 400 if isinstance(exc, (ValueError, TypeError, KeyError)) else 500
                        self.send_json(status, {"error": {"message": str(exc), "type": "invalid_request_error"}})

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
                        sleeping=self.sleeping, execution="training")

    def close(self):
        with self.condition:
            if self.closed:
                return
            self.closed = self.sleeping = True
        self.http.shutdown()
        self.begin_training()
        self.http.server_close()
        self.thread.join()
        self.trainer = None

    def tokenize(self, body):
        if "tokens" in body:
            tokens = body["tokens"]
        elif "messages" in body:
            messages = body["messages"]
            if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) for m in messages):
                raise ValueError("messages must be a nonempty list")
            for message in messages:
                if not isinstance(message.get("content", ""), str):
                    raise ValueError("shared-model GRPO currently accepts text messages only")
            kwargs = dict(body.get("chat_template_kwargs") or {})
            kwargs.update(tokenize=True, return_dict=False, add_generation_prompt=body.get("add_generation_prompt", True))
            tokens = self.tokenizer.apply_chat_template(messages, **kwargs)
        else:
            tokens = self.tokenizer.encode(body["prompt"], add_special_tokens=body.get("add_special_tokens", True))
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or not 0 <= t < self.vocab for t in tokens):
            raise ValueError("prompt must contain valid model token IDs")
        return tokens

    def prepare(self, body):
        if body.get("n", 1) != 1 or body.get("tools") or body.get("response_format"):
            raise ValueError("shared training execution supports one text completion per request")
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
        return body | dict(prompt_ids=prompt, maximum=min(maximum, self.context - len(prompt)),
                           minimum=minimum, stop=stop)

    def generate(self, request, callback=None):
        # One workspace serves all admitted requests; begin_training waits for
        # both the running request and admitted requests waiting on this lock.
        with self.compute_lock:
            return self._generate(request, callback)

    def _generate(self, request, callback):
        prompt = request["prompt_ids"]
        inputs = np.zeros((self.trainer.batch_size, self.trainer.seq_length), dtype=np.int32)
        inputs[0, :len(prompt)] = prompt
        positions = np.zeros(self.trainer.batch_size, dtype=np.int32)
        rng = np.random.default_rng(request.get("seed"))
        ids, scores, text, emitted = [], [], "", ""
        identifier, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
        finish = "length"
        eos = (set() if request.get("ignore_eos", False) else self.eos) | set(request.get("stop_token_ids") or [])
        for step in range(request["maximum"]):
            positions[0] = len(prompt) + step - 1
            logits = self.trainer.next_token_logits(inputs, positions)[0].astype(np.float64)
            if not np.isfinite(logits).all():
                raise RuntimeError("model returned nonfinite logits")
            temperature = float(request.get("temperature", 1.))
            logits /= temperature if temperature > 0 else 1.
            logprobs = logits - np.logaddexp.reduce(logits)
            sampling = logits.copy()
            seen, counts = np.unique(prompt + ids, return_counts=True)
            penalty = float(request.get("repetition_penalty", 1.))
            sampling[seen] = np.where(sampling[seen] > 0, sampling[seen] / penalty, sampling[seen] * penalty)
            sampling[seen] -= float(request.get("presence_penalty", 0.)) + counts * float(request.get("frequency_penalty", 0.))
            for token, bias in (request.get("logit_bias") or {}).items():
                sampling[int(token)] += float(bias)
            if step < request["minimum"]:
                sampling[list(eos)] = -np.inf
            order = np.argsort(sampling)[::-1]
            top_k = int(request.get("top_k", -1))
            if top_k > 0:
                sampling[order[top_k:]] = -np.inf
            probabilities = np.exp(sampling - np.max(sampling))
            probabilities /= probabilities.sum()
            cumulative = np.cumsum(probabilities[order]) - probabilities[order]
            sampling[order[cumulative >= float(request.get("top_p", 1.))]] = -np.inf
            sampling[probabilities < probabilities.max() * float(request.get("min_p", 0.))] = -np.inf
            probabilities = np.exp(sampling - np.max(sampling))
            probabilities /= probabilities.sum()
            if not np.isfinite(probabilities).all():
                raise ValueError("sampling constraints removed every token or model logits are nonfinite")
            token = int(np.argmax(sampling) if temperature == 0 else rng.choice(len(sampling), p=probabilities))
            ids.append(token)
            inputs[0, len(prompt) + step] = token
            piece = self.tokenizer.decode([token], skip_special_tokens=False)
            score = dict(token=piece, logprob=float(logprobs[token]), bytes=list(piece.encode()), top_logprobs=[])
            top_count = int(request.get("top_logprobs", 0) or 0)
            top_tokens = np.argsort(logprobs)[-top_count:][::-1] if top_count else ()
            for t in top_tokens:
                token_text = self.tokenizer.decode([int(t)], skip_special_tokens=False)
                score["top_logprobs"].append(dict(token=token_text, logprob=float(logprobs[t]), bytes=list(token_text.encode())))
            scores.append(score)
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            stopped = token in eos
            if step + 1 >= request["minimum"]:
                matches = [text.index(stop) for stop in request["stop"] if stop in text]
                if matches:
                    text, stopped = text[:min(matches)], True
            final = stopped or step + 1 == request["maximum"]
            # Keep incomplete UTF-8 and potential stop-string prefixes buffered.
            stable = text if final else text[:len(text) - max((len(s) - 1 for s in request["stop"]), default=0)]
            if not final and "\ufffd" in stable:
                stable = stable[:stable.index("\ufffd")]
            delta, emitted = stable[len(emitted):], stable
            if callback:
                callback(dict(id=identifier, created=created, object="chat.completion.chunk", model=self.adapter,
                    choices=[dict(index=0, delta=dict(role="assistant", content=delta), token_ids=[token],
                                  logprobs=dict(content=[score]) if request.get("logprobs") else None, finish_reason=None)]))
            if stopped:
                finish = "stop"
                break
        return dict(id=identifier, created=created, object="chat.completion", model=self.adapter,
            prompt_token_ids=prompt, choices=[dict(index=0, message=dict(role="assistant", content=text),
                token_ids=ids, logprobs=dict(content=scores) if request.get("logprobs") else None, finish_reason=finish)],
            usage=dict(prompt_tokens=len(prompt), completion_tokens=len(ids), total_tokens=len(prompt) + len(ids)))


class Busy(RuntimeError):
    pass
