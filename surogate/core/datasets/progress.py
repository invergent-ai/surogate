from tqdm.asyncio import tqdm_asyncio

from surogate.utils.logger import get_logger

logger = get_logger()


class LoggingIOWriter:
    def write(self, msg):
        if len(msg.strip()) > 0:
            logger.info(msg.strip())

class HfHubPrettyTqdmInner(tqdm_asyncio):
    def __init__(self, *args, **kwargs):
        self.prefix = kwargs.pop('prefix')
        super().__init__(*args, **kwargs)
        self.fp = LoggingIOWriter()
        self.sp = self.status_printer(self.fp)

    @staticmethod
    def status_printer(file):
        tqdm_asyncio.status_printer(LoggingIOWriter())

    @classmethod
    def write(cls, s, file = None, end = "\n", nolock = False):
        fp = file if file is not None else LoggingIOWriter()
        with cls.external_write_mode(file=file, nolock=nolock):
            # Write the message
            fp.write(s)
            fp.write(end)

    def display(self, msg = None, pos = None):
        if self.total > 0:
            logger.info(f"{self.prefix}{msg if msg is not None else self.__str__()}")

    def __iter__(self):
        # Ensure prefix is applied during iteration
        if hasattr(super(), '__iter__'):
            return super().__iter__()
        return self

def create_hfhub_tqdm(prefix=""):
    class HfHubPrettyTqdm(HfHubPrettyTqdmInner):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault('prefix', prefix)
            super().__init__(*args, **kwargs)

    return HfHubPrettyTqdm

import datasets
# datasets.utils.logging.tqdm = create_hfhub_tqdm("Dataset - ")
# datasets.utils.tqdm = create_hfhub_tqdm("Dataset - ")
# datasets.download.download_manager.tqdm = create_hfhub_tqdm("Downloading dataset - ")
# datasets.arrow_dataset.hf_tqdm = create_hfhub_tqdm("Processing dataset - ")
import transformers
# transformers.utils.logging.tqdm = create_hfhub_tqdm("Loading model - ")
