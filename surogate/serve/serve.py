import asyncio
import json
from dataclasses import asdict
from http import HTTPStatus
from threading import Thread
from typing import Optional, Union, DefaultDict

import uvicorn
from fastapi import FastAPI, Response, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from surogate.loaders.loader import load_model_and_tokenizer
from surogate.utils.command import SurogateCommand
from surogate.utils.logger import get_logger
from swift.llm import safe_snapshot_download, AdapterRequest
from swift.llm.infer.infer_engine import PtEngine
from swift.llm.infer.protocol import ModelList, Model, ChatCompletionRequest, EmbeddingRequest, MultiModalRequestMixin
from swift.plugin import InferStats
from swift.llm.infer.utils import prepare_adapter, update_generation_config_eos_token
from swift.utils import seed_everything
from swift.llm import ModelMeta, get_template

from surogate.utils.seed import RAND_SEED

logger = get_logger()


class SurogateServe(SurogateCommand):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        seed = RAND_SEED + max(self.config.get('rank', -1), 0)
        seed_everything(seed)

        # self._init_adapters()

        self.infer_kwargs = {}
        if self.config['engine'] == 'vllm' and self.config['adapters']:
            self.infer_kwargs['adapter_request'] = AdapterRequest('_lora', self.config.adapters[0])

        self.infer_engine = self._get_infer_engine()

        self.infer_stats = InferStats()
        self.app = FastAPI(lifespan=self.lifespan)
        self.infer_engine = self._get_infer_engine()
        self._register_app()

    def _prepare_model_template(self):
        model, tokenizer = load_model_and_tokenizer(self.config, self.args)
        template = get_template(model.model_meta.template, tokenizer)
        if template.use_model:
            template.model = model

        # model = prepare_adapter(DictDefault(), model, adapters=self.adapters)
        update_generation_config_eos_token(model.generation_config, template)
        return model, template

    def run(self):
        uvicorn.run(
            self.app,
            host=self.args['host'],
            port=self.args['port'],
            log_level=logger.level().lower())

    def _register_app(self):
        self.app.get('/health')(self.health)
        self.app.get('/ping')(self.ping)
        self.app.post('/ping')(self.ping)
        self.app.get('/v1/models')(self.get_available_models)
        self.app.post('/v1/chat/completions')(self.create_chat_completion)
        # self.app.post('/v1/completions')(self.create_completion)
        # self.app.post('/v1/embeddings')(self.create_embedding)

    async def health(self) -> Response:
        """Health check endpoint."""
        if self.infer_engine is not None:
            return Response(status_code=200)
        else:
            return Response(status_code=503)

    async def ping(self) -> Response:
        """Ping check endpoint. Required for SageMaker compatibility."""
        return await self.health()

    async def get_available_models(self):
        model_list = [self.config['served_name']]
        if self.adapters:
            model_list += [name for name in self.adapters.keys()]

        data = [Model(id=model_id) for model_id in model_list]
        return ModelList(data=data)

    async def _check_model(self, request: ChatCompletionRequest) -> Optional[str]:
        available_models = await self.get_available_models()
        model_list = [model.id for model in available_models.data]
        if request.model not in model_list:
            return f'`{request.model}` is not in the model_list: `{model_list}`.'

        return None

    async def create_chat_completion(self,
                                     request: ChatCompletionRequest,
                                     raw_request: Request,
                                     *,
                                     return_cmpl_response: bool = False):
        args = self.args
        error_msg = await self._check_model(request)
        if error_msg:
            return self.create_error_response(HTTPStatus.BAD_REQUEST, error_msg)
        infer_kwargs = self.infer_kwargs.copy()
        adapter_path = args.adapter_mapping.get(request.model)
        if adapter_path:
            infer_kwargs['adapter_request'] = AdapterRequest(request.model, adapter_path)

        infer_request, request_config = request.parse()
        request_info = {'response': '', 'infer_request': infer_request.to_printable()}

        def pre_infer_hook(kwargs):
            request_info['generation_config'] = kwargs['generation_config']
            return kwargs

        infer_kwargs['pre_infer_hook'] = pre_infer_hook
        try:
            res_or_gen = await self.infer_async(infer_request, request_config, template=self.template, **infer_kwargs)
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
        args = self.args

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
            if args.log_interval > 0:
                self.infer_stats.update(response)
            if self.jsonl_writer:
                self.jsonl_writer.append(request_info)
            if self.args.verbose:
                logger.info(request_info)
        return response

    @staticmethod
    def create_error_response(status_code: Union[int, str, HTTPStatus], message: str) -> JSONResponse:
        status_code = int(status_code)
        return JSONResponse({'message': message, 'object': 'error'}, status_code)

    async def _log_stats_hook(self):
        while True:
            await asyncio.sleep(self.config['log_interval'])
            self._compute_infer_stats()
            self.infer_stats.reset()

    def lifespan(self, app: FastAPI):
        if self.config['log_interval'] > 0:
            thread = Thread(target=lambda: asyncio.run(self._log_stats_hook()), daemon=True)
            thread.start()

        try:
            yield
        finally:
            if self.config['log_interval'] > 0:
                self._compute_infer_stats()

    def _compute_infer_stats(self):
        global_stats = self.infer_stats.compute()
        for k, v in global_stats.items():
            global_stats[k] = round(v, 8)
        logger.info(global_stats)

    def _get_infer_engine(self):
        if self.config['engine'] == 'pytorch':
            model, self.template = self._prepare_model_template()
            return PtEngine.from_model_template(model, self.template)
        elif self.config['engine'] == 'vllm':
            from swift.llm.infer import VllmEngine
            return VllmEngine(
                model_id_or_path=self.config['model'],
                gpu_memory_utilization=self.config['max_memory'] or 0.9,
                tensor_parallel_size=self.config['tp'] or 1,
                max_model_len=self.config['max_context'] if self.config['max_context'] else None,
                use_hf=True,
                hub_token=self.args['hub_token'] if self.args['hub_token'] else None,
                model_type=self.config['model_type']
            )
        elif self.config['engine'] == 'sglang':
            from swift.llm.infer import SglangEngine
            return SglangEngine(
                model_id_or_path=self.config['model'],
                tp_size=self.config['tp'] or 1,
                context_length=self.config['max_context'] if self.config['max_context'] else None,
                use_hf=True,
                hub_token=self.args['hub_token'] if self.args['hub_token'] else None,
                model_type=self.config['model_type']
            )
        else:
            raise ValueError(f"Unsupported inference engine: {self.config['engine']}")

    def _init_adapters(self):
        self.adapters = {}
        if self.config['adapters']:
            for i, adapter in enumerate(self.config['adapters']):
                adapter_path = safe_snapshot_download(adapter['path'], use_hf=True, hub_token=self.args['hf_token'])
                self.adapters[adapter['name']] = adapter_path
