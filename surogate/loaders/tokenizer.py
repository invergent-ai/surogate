from transformers import PreTrainedTokenizer, AutoTokenizer, AddedToken

from surogate.utils.dict import DictDefault
from surogate.utils.distributed import is_main_process
from surogate.utils.logger import get_logger

logger = get_logger()

def load_tokenizer(cfg: DictDefault) -> PreTrainedTokenizer:
    """Load and configure the tokenizer based on the provided config."""
    tokenizer_path = cfg.get('tokenizer') or cfg.get('model')
    if tokenizer_path is None:
        raise ValueError("Either 'tokenizer' or 'model' must be specified in the config.")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=True, trust_remote_code=True
    )

    additional_special_tokens = None
    if cfg.special_tokens:
        special_tokens = cfg.get('special_tokens').to_dict()
        additional_special_tokens = special_tokens.pop(
            "additional_special_tokens", None
        )
        for k, val in special_tokens.items():
            tokenizer.add_special_tokens(
                {k: AddedToken(val, rstrip=False, lstrip=False, normalized=False)}
            )

        bos_or_eos_in_special_tokens = (
                "bos_token" in cfg.special_tokens and "eos_token" in cfg.special_tokens
        )
        if (
                tokenizer.__class__.__name__
                in (
                "LlamaTokenizerFast",
                "CodeLlamaTokenizerFast",
        )
                and bos_or_eos_in_special_tokens
        ):
            tokenizer.update_post_processor()

    if cfg.tokens:
        tokenizer.add_tokens(
            [
                AddedToken(token, rstrip=False, lstrip=False, normalized=False)
                for token in cfg.tokens
            ]
        )

    # Additional special tokens are a List, and need to be treated differently than regular special
    # tokens. We add them after we have called `add_tokens` in case these additional special tokens
    # are new tokens.
    #
    # Usage:
    #
    # ```py
    # special_tokens:
    #   additional_special_tokens: ["<|im_start|>", "<|im_end|>"]
    # ```
    if additional_special_tokens is not None:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": additional_special_tokens}
        )

    if is_main_process():
        logger.debug(f"EOS: {tokenizer.eos_token_id} / {tokenizer.eos_token}")
        logger.debug(f"BOS: {tokenizer.bos_token_id} / {tokenizer.bos_token}")
        logger.debug(f"PAD: {tokenizer.pad_token_id} / {tokenizer.pad_token}")
        logger.debug(f"UNK: {tokenizer.unk_token_id} / {tokenizer.unk_token}")

    return tokenizer