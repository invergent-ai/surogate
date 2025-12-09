from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator, List, Optional, Union

from transformers import PreTrainedTokenizerBase

from surogate.core.infer.protocol import RequestConfig, ChatCompletionResponse, ChatCompletionStreamResponse
from surogate.core.model.chat_templates.inputs import InferRequest
from surogate.core.model.chat_templates.processor import get_chat_template_processor
from surogate.utils.logger import get_logger
from surogate.utils.metric import Metric

logger = get_logger()

class BaseInferEngine(ABC):

    @abstractmethod
    def infer(self,
              infer_requests: List[InferRequest],
              request_config: Optional[RequestConfig] = None,
              metrics: Optional[List[Metric]] = None,
              *,
              use_tqdm: Optional[bool] = None,
              **kwargs) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
        """
        This method performs inference on a list of inference requests.

        The method takes a list of inference requests and processes them according to the provided configuration.
        It can optionally use tqdm for progress visualization and accept additional keyword arguments.

        Args:
            infer_requests (List[InferRequest]): A list of inference requests to be processed.
            request_config (Optional[RequestConfig]): Configuration for the request, if any.
            metrics (Optional[List[Metric]]): A list of usage information to return.
            use_tqdm (Optional[bool]): Whether to use tqdm for progress visualization.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
                The result of the inference.
        """
        pass

    @abstractmethod
    async def infer_async(self,
                          infer_request: InferRequest,
                          request_config: Optional[RequestConfig] = None,
                          **kwargs) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        """
        This method performs asynchronous inference on a single inference request.

        The method takes an inference request and processes it according to the provided configuration.
        It can accept additional keyword arguments.

        Args:
            infer_request (InferRequest): An inference request to be processed.
            request_config (Optional[RequestConfig]): Configuration for the request, if any.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]: The result of
                the asynchronous inference.
        """
        pass

class InferEngine(BaseInferEngine):
    def _post_init(self, template=None):
        processor = self.processor
        self.model_info = processor.model_info
        self.model_meta = processor.model_meta
        self.model_dir = self.model_info.model_dir
        self.model_name = self.model_info.model_name
        self.max_model_len = self.model_info.max_model_len
        self.task_type = self.model_info.task_type
        self.config = self.model_info.config
        if template is None:
                self.default_template = get_chat_template_processor(self.model_meta.template, self.processor)
        else:
            self.default_template = template
            self.default_template.init_processor(self.processor)

        self._adapters_pool = {}

    @property
    def tokenizer(self):
        tokenizer = self.processor
        if not isinstance(tokenizer, PreTrainedTokenizerBase) and hasattr(tokenizer, 'tokenizer'):
            tokenizer = tokenizer.tokenizer
        return tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        if self.processor is self.tokenizer:
            self.processor = value
        elif self.tokenizer is not value:
            raise AttributeError('Please use `self.processor` for assignment.')