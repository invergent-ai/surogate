import json
import os
from dataclasses import asdict
from http import HTTPStatus
from typing import Optional, Union

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from swift.llm import get_template
from swift.llm import safe_snapshot_download, AdapterRequest
from swift.llm.infer.infer_engine import PtEngine, InferEngine
from swift.llm.infer.protocol import ModelList, Model, ChatCompletionRequest, MultiModalRequestMixin
from swift.llm.infer.utils import update_generation_config_eos_token
from swift.llm.model.register import get_model_name
from swift.llm.template import Template
from swift.plugin import InferStats
from swift.tuners import Swift
from swift.utils import seed_everything

from surogate.config.serve_config import ServeConfig
from surogate.loaders.loader import load_model_and_tokenizer
from surogate.utils.command import SurogateCommand
from surogate.utils.logger import get_logger

logger = get_logger()

class SurogateServe(SurogateCommand):
    config: ServeConfig
    template: Template

    def __init__(self, **kwargs):
        super().__init__(ServeConfig, **kwargs)

        if self.config.seed:
            seed_everything(self.config.seed, full_determinism=self.config.deterministic)

        self._init_adapters()
        self.infer_engine = self._get_infer_engine()
        self.infer_kwargs = {}
        self.infer_stats = InferStats()
        self.app = FastAPI()
        self._register_app()

    def run(self):
        uvicorn.run(
            self.app,
            host=self.args['host'],
            port=self.args['port'],
            log_level=logger.level().lower())

    def _register_app(self):
        self.app.get('/health')(self.health)
        self.app.get('/stats')(self.stats)
        self.app.get('/v1/models')(self.get_available_models)
        self.app.post('/v1/chat/completions')(self.create_chat_completion)

    async def health(self) -> Response:
        """Health check endpoint."""
        if self.infer_engine is not None:
            return Response(status_code=200)
        else:
            return Response(status_code=503)

    async def stats(self) -> JSONResponse:
        global_stats = self.infer_stats.compute()
        for k, v in global_stats.items():
            global_stats[k] = round(v, 8)
        return JSONResponse(content=global_stats)

    async def get_available_models(self):
        model_list = [self.config.served_name or get_model_name(self.config.model)]
        if self.config.adapters:
            model_list += [adapter.name for adapter in self.config.adapters]

        data = [Model(id=model_id, owned_by="surogate") for model_id in model_list]
        return ModelList(data=data)

    async def _check_model(self, request: ChatCompletionRequest) -> Optional[str]:
        available_models = await self.get_available_models()
        model_list = [model.id for model in available_models.data]
        if request.model not in model_list:
            return f'`{request.model}` is not in the model_list: `{model_list}`.'

        return None

    def get_adapter_path(self, adapter_name: str) -> Optional[str]:
        for adapter in self.config.adapters:
            if adapter.name == adapter_name:
                return adapter.path
        return None

    async def create_chat_completion(self,
                                     request: ChatCompletionRequest,
                                     raw_request: Request,
                                     *,
                                     return_cmpl_response: bool = False):
        error_msg = await self._check_model(request)
        if error_msg:
            return self.create_error_response(HTTPStatus.BAD_REQUEST, error_msg)

        infer_kwargs = self.infer_kwargs.copy()
        adapter_path = self.get_adapter_path(request.model)

        if adapter_path:
            infer_kwargs['adapter_request'] = AdapterRequest(request.model, adapter_path)

        infer_request, request_config = request.parse()
        request_info = {'response': '', 'infer_request': infer_request.to_printable()}

        try:
            res_or_gen = await self.infer_engine.infer_async(
                infer_request,
                request_config,
                template=self.template,
                **infer_kwargs
            )
        except Exception as e:
            import traceback
            logger.info(traceback.format_exc())
            return self.create_error_response(HTTPStatus.BAD_REQUEST, str(e))

        if request_config.stream:
            async def _gen_wrapper():
                async for res in res_or_gen:
                    res = self._post_process(request_info, res, return_cmpl_response)
                    yield f'data: {json.dumps(asdict(res), ensure_ascii=False)}\n\n'
                yield 'data: [DONE]\n\n'

            return StreamingResponse(_gen_wrapper(), media_type='text/event-stream')
        elif hasattr(res_or_gen, 'choices'):
            # instance of ChatCompletionResponse
            return self._post_process(request_info, res_or_gen, return_cmpl_response)
        else:
            return res_or_gen

    def _post_process(self, request_info, response, return_cmpl_response: bool = False):
        for i in range(len(response.choices)):
            if not hasattr(response.choices[i], 'message') or not isinstance(response.choices[i].message.content,
                                                                             (tuple, list)):
                continue
            for j, content in enumerate(response.choices[i].message.content):
                if isinstance(content, dict) and content['type'] == 'image':
                    b64_image = MultiModalRequestMixin.to_base64(content['image'])
                    response.choices[i].message.content[j]['image'] = f'data:image/jpg;base64,{b64_image}'

        is_finished = all(response.choices[i].finish_reason for i in range(len(response.choices)))
        if 'stream' in response.__class__.__name__.lower():
            request_info['response'] += response.choices[0].delta.content
        else:
            request_info['response'] = response.choices[0].message.content
        if return_cmpl_response:
            response = response.to_cmpl_response()
        if is_finished:
            self.infer_stats.update(response)
            if self.args.verbose:
                logger.info(request_info)
        return response

    @staticmethod
    def create_error_response(status_code: Union[int, str, HTTPStatus], message: str) -> JSONResponse:
        status_code = int(status_code)
        return JSONResponse({'message': message, 'object': 'error'}, status_code)

    def _get_infer_engine(self) -> InferEngine:
        if self.config.infer_backend == 'pytorch':
            model, tokenizer = load_model_and_tokenizer(self.config.model, self.config.model_type, self.args, True)
            self.template = get_template(model.model_meta.template, tokenizer, use_chat_template=self.config.use_chat_template)
            if self.template.use_model:
                self.template.model = model
            for adapter in self.config.adapters:
                model = Swift.from_pretrained(model, adapter.path)

            update_generation_config_eos_token(model.generation_config, self.template)
            return PtEngine.from_model_template(model, self.template)
        elif self.config.infer_backend == 'vllm':
            from swift.llm.infer import VllmEngine
            from vllm.config import KVTransferConfig

            if self.config.adapters:
                self.infer_kwargs['adapter_request'] = AdapterRequest('_lora', self.config.adapters[0].path)

            self.template = get_template(
                self.config.model_meta.template,
                processor=None,
                use_chat_template=self.config.use_chat_template
            )

            os.environ["VLLM_HAS_FLASHINFER_CUBIN"] = "1"

            if self.cache_enabled():
                os.environ["LMCACHE_TRACK_USAGE"] = "false"
                os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
                os.environ["LMCACHE_USAGE_TRACK_URL"] = "/dev/null"
                os.environ["PYTHONHASHSEED"] = "0"
                os.environ["LMCACHE_CHUNK_SIZE"] = str(self.config.cache.chunk_size)
                os.environ["LMCACHE_REMOTE_SERDE"] = "cachegen"
                if self.config.cache.max_memory_cache_gb > 0:
                    os.environ["LMCACHE_LOCAL_CPU"] = "True"
                    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(self.config.cache.max_memory_cache_gb)
                if self.config.cache.max_disk_cache_gb > 0:
                    os.environ["LMCACHE_LOCAL_CPU"] = "False"
                    os.environ["LMCACHE_LOCAL_DISK"] = self.config.cache.disk_cache_path
                    os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = str(self.config.cache.max_disk_cache_gb)

            return VllmEngine(
                use_async_engine=True,
                model_id_or_path=self.config.model,
                gpu_memory_utilization=self.config.max_memory,
                tensor_parallel_size=self.config.tensor_parallel,
                max_model_len=self.config.max_context,
                use_hf=True,
                hub_token=self.args['hub_token'],
                model_type=self.config.model_type,
                template=self.template,
                engine_kwargs={
                    'kv_transfer_config': KVTransferConfig(
                        kv_connector="LMCacheConnectorV1",
                        kv_role="kv_both",
                    )
                } if self.cache_enabled() else None
            )
        elif self.config.infer_backend == 'sglang':
            from swift.llm.infer import SglangEngine

            self.template = get_template(
                self.config.model_meta.template,
                processor=None,
                use_chat_template=self.config.use_chat_template
            )

            return SglangEngine(
                model_id_or_path=self.config.model,
                mem_fraction_static=0.9,
                tp_size=self.config.tensor_parallel,
                context_length=self.config.max_context,
                use_hf=True,
                hub_token=self.args['hub_token'],
                model_type=self.config.model_type,
                template=self.template,
                engine_kwargs={
                    'enable_hierarchical_cache': self.cache_enabled(),
                    'hicache_ratio': 1
                }
            )
        else:
            raise ValueError(f"Unsupported inference engine: {self.config['engine']}")


    def _init_adapters(self):
        if self.config.adapters:
            for adapter in self.config.adapters:
                safe_snapshot_download(adapter.path, use_hf=True, hub_token=self.args['hf_token'])

    def cache_enabled(self):
        return self.config.cache and self.config.cache.enabled

