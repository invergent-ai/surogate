import json
import os
from collections.abc import Mapping, Sequence
from queue import Queue
from threading import Thread
from typing import Any, Literal

import json_repair
import torch
from torch.distributed import gather_object

from surogate.utils.logger import get_logger

logger = get_logger()


def check_json_format(obj: Any, token_safe: bool = True) -> Any:
    if obj is None or isinstance(obj, (int, float, str, complex)):  # bool is a subclass of int
        return obj
    if isinstance(obj, bytes):
        return "<<<bytes>>>"
    if isinstance(obj, (torch.dtype, torch.device)):
        obj = str(obj)
        return obj[len("torch.") :] if obj.startswith("torch.") else obj

    if isinstance(obj, Sequence):
        res = []
        for x in obj:
            res.append(check_json_format(x, token_safe))
    elif isinstance(obj, Mapping):
        res = {}
        for k, v in obj.items():
            if token_safe and isinstance(k, str) and "_token" in k and isinstance(v, str):
                res[k] = None
            else:
                res[k] = check_json_format(v, token_safe)
    else:
        if token_safe:
            unsafe_items = {}
            for k, v in obj.__dict__.items():
                if "_token" in k:
                    unsafe_items[k] = v
                    setattr(obj, k, None)
            res = repr(obj)
            # recover
            for k, v in unsafe_items.items():
                setattr(obj, k, v)
        else:
            res = repr(obj)  # e.g. function, object
    return res


class JsonlWriter:
    def __init__(
        self,
        fpath: str,
        *,
        encoding: str = "utf-8",
        strict: bool = True,
        enable_async: bool = False,
        write_on_rank: Literal["master", "last"] = "master",
    ):
        self.fpath = os.path.abspath(os.path.expanduser(fpath)) if self.is_write_rank else None
        self.encoding = encoding
        self.strict = strict
        self.enable_async = enable_async
        self._queue = Queue()
        self._thread = None

    def _append_worker(self):
        while True:
            item = self._queue.get()
            self._append(**item)

    def _append(self, obj: dict | list[dict], gather_obj: bool = False):
        if isinstance(obj, (list, tuple)) and all(isinstance(item, dict) for item in obj):
            obj_list = obj
        else:
            obj_list = [obj]
        if gather_obj and torch.distributed.is_initialized():
            obj_list = gather_object(obj_list)
        if not self.is_write_rank:
            return
        obj_list = check_json_format(obj_list)
        for i, _obj in enumerate(obj_list):
            obj_list[i] = json.dumps(_obj, ensure_ascii=False) + "\n"
        self._write_buffer("".join(obj_list))

    def append(self, obj: dict | list[dict], gather_obj: bool = False):
        if self.enable_async:
            if self._thread is None:
                self._thread = Thread(target=self._append_worker, daemon=True)
                self._thread.start()
            self._queue.put({"obj": obj, "gather_obj": gather_obj})
        else:
            self._append(obj, gather_obj=gather_obj)

    def _write_buffer(self, text: str):
        if not text:
            return
        assert self.is_write_rank, f"self.is_write_rank: {self.is_write_rank}"
        try:
            os.makedirs(os.path.dirname(self.fpath), exist_ok=True)
            with open(self.fpath, "a", encoding=self.encoding) as f:
                f.write(text)
        except Exception:
            if self.strict:
                raise
            logger.error(f"Cannot write content to jsonl file. text: {text}")


def append_to_jsonl(fpath: str, obj: dict | list[dict], *, encoding: str = "utf-8", strict: bool = True) -> None:
    jsonl_writer = JsonlWriter(fpath, encoding=encoding, strict=strict)
    jsonl_writer.append(obj)


def json_parse_to_dict(value: str | dict | None, strict: bool = True) -> str | dict:
    """Convert a JSON string or JSON file into a dict"""
    # If the value could potentially be a string, it is generally advisable to set strict to False.
    if value is None:
        value = {}
    elif isinstance(value, str):
        if os.path.exists(value):  # local path
            with open(value, encoding="utf-8") as f:
                value = json.load(f)
        else:  # json str
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                if strict:
                    try:
                        # fix malformed json string, e.g., incorrect quotation marks
                        old_value = value
                        value = json_repair.repair_json(value)
                        logger.warning(
                            f"Unable to parse json string, try to repair it, "
                            f"the string before and after repair are '{old_value}' | '{value}'"
                        )
                        value = json.loads(value)
                    except Exception:
                        logger.error(f"Unable to parse json string: '{value}', and try to repair failed")
                        raise
    return value
