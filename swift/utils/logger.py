# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib.util
import logging
import os
from contextlib import contextmanager
from types import MethodType
from typing import Optional

from modelscope.utils.logger import get_logger as get_ms_logger


# Avoid circular reference
def _is_local_master():
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    return local_rank in {-1, 0}


init_loggers = {}

# old format
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_format = logging.Formatter('[%(levelname)s:%(name)s] %(message)s')

info_set = set()
warning_set = set()


def info_if(self, msg, cond, *args, **kwargs):
    if cond:
        with logger_context(self, logging.INFO):
            self.info(msg)


def warning_if(self, msg, cond, *args, **kwargs):
    if cond:
        with logger_context(self, logging.INFO):
            self.warning(msg)


def info_once(self, msg, *args, **kwargs):
    hash_id = kwargs.get('hash_id') or msg
    if hash_id in info_set:
        return
    info_set.add(hash_id)
    self.info(msg)


def warning_once(self, msg, *args, **kwargs):
    hash_id = kwargs.get('hash_id') or msg
    if hash_id in warning_set:
        return
    warning_set.add(hash_id)
    self.warning(msg)


def get_logger(log_file: Optional[str] = None, log_level: Optional[int] = None, file_mode: str = 'w'):
    from surogate.utils.logger import get_logger as get_surogate_logger
    return get_surogate_logger(log_file=log_file, log_level=log_level, file_mode=file_mode)

logger = get_logger()
ms_logger = get_ms_logger()

ms_logger.handlers[0].setFormatter(logger_format)
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
if _is_local_master():
    ms_logger.setLevel(log_level)
else:
    ms_logger.setLevel(logging.ERROR)


@contextmanager
def logger_context(logger, log_leval):
    origin_log_level = logger.level
    logger.setLevel(log_leval)
    try:
        yield
    finally:
        logger.setLevel(origin_log_level)


@contextmanager
def ms_logger_context(log_leval):
    with logger_context(get_ms_logger(), log_leval):
        yield


def add_file_handler_if_needed(logger, log_file, file_mode, log_level):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return

    if importlib.util.find_spec('torch') is not None:
        is_worker0 = int(os.getenv('LOCAL_RANK', -1)) in {-1, 0}
    else:
        is_worker0 = True

    if is_worker0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        file_handler.setFormatter(logger_format)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
