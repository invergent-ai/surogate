import asyncio
from threading import Thread

import uvicorn
from fastapi import FastAPI, Response
from swift import get_logger
from swift.llm import safe_snapshot_download
from swift.llm.infer.protocol import ModelList, Model
from swift.plugin import InferStats

from surogate.utils.config import load_config

logger = get_logger()


class SurogateServe:
    def __init__(self, **kwargs):
        self.args = kwargs
        self.config = load_config(self.args['config'])
        self._init_adapters()

        self.infer_stats = InferStats()
        self.app = FastAPI(lifespan=self.lifespan)
        self.infer_engine = self._get_infer_engine()
        self._register_app()

    def run(self):
        uvicorn.run(
            self.app,
            host=self.args['host'],
            port=self.args['port'],
            log_level=logger.level)

    def _register_app(self):
        self.app.get('/health')(self.health)
        self.app.get('/ping')(self.ping)
        self.app.post('/ping')(self.ping)
        self.app.get('/v1/models')(self.get_available_models)
        # self.app.post('/v1/chat/completions')(self.create_chat_completion)
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
        if self.config['engine'] == 'vllm':
            from swift.llm.infer import VllmEngine
            return VllmEngine(
                model_id_or_path=self.config['model'],
                gpu_memory_utilization=self.config['max_memory'] or 0.9,
                tensor_parallel_size=self.config['tp'] or 1,
                max_model_len=self.config['max_context'] if self.config['max_context'] else None,
                use_hf=True,
                hub_token=self.args['hub_token'] if self.args['hub_token'] else None,
            )
        elif self.config['engine'] == 'sglang':
            from swift.llm.infer import SglangEngine
            return SglangEngine(
                model_id_or_path=self.config['model'],
                tp_size=self.config['tp'] or 1,
                context_length=self.config['max_context'] if self.config['max_context'] else None,
                use_hf=True,
                hub_token=self.args['hub_token'] if self.args['hub_token'] else None,
            )
        elif self.config['engine'] == 'sglang':
            from swift.llm.infer import PtEngine
            return PtEngine(
                model_id_or_path=self.config['model'],
                use_hf=True,
                hub_token=self.args['hub_token'] if self.args['hub_token'] else None
            )
        else:
            raise ValueError(f"Unsupported inference engine: {self.config['engine']}")

    def _init_adapters(self):
        self.adapters = {}
        if self.config['adapters']:
            for i, adapter in enumerate(self.config['adapters']):
                adapter_path = safe_snapshot_download(adapter['path'], use_hf=True,
                                                      hub_token=self.args['hf_token'] if self.args[
                                                          'hf_token'] else None)
                self.adapters[adapter['name']] = adapter_path
