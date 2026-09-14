"""Reconstruct EmbeddingGemma's SentencePiece resources from its GGUF vocabulary."""

import json
import math


def frontend_from_gguf(source) -> dict[str, bytes]:
    from sentencepiece import sentencepiece_model_pb2 as sentencepiece

    def read(name, default=None):
        return source.kv("tokenizer.ggml." + name, default)

    if read("model") != "llama":
        raise ValueError("EmbeddingGemma GGUF must contain a SentencePiece tokenizer")
    tokens, scores, kinds = (read(name) for name in ("tokens", "scores", "token_type"))
    if (not isinstance(tokens, (list, tuple)) or not tokens or
            not isinstance(scores, (list, tuple)) or not isinstance(kinds, (list, tuple)) or
            len(tokens) != len(scores) or len(tokens) != len(kinds)):
        raise ValueError("GGUF tokenizer tokens, scores and token_type must have matching lengths")

    model = sentencepiece.ModelProto()
    trainer = model.trainer_spec
    trainer.model_type = sentencepiece.TrainerSpec.BPE
    trainer.vocab_size = len(tokens)
    trainer.byte_fallback = 6 in kinds
    config = {"tokenizer_class": "GemmaTokenizer"}
    for gguf_name, name in (("unknown", "unk"), ("bos", "bos"), ("eos", "eos"), ("padding", "pad")):
        value = read(gguf_name + "_token_id", -1)
        if (isinstance(value, bool) or not isinstance(value, int) or
                value < (-1 if name == "pad" else 0) or value >= len(tokens)):
            raise ValueError(f"GGUF tokenizer {gguf_name}_token_id is missing or outside its vocabulary")
        setattr(trainer, name + "_id", value)
        if value >= 0:
            setattr(trainer, name + "_piece", tokens[value])
            config[name + "_token"] = tokens[value]

    # The encoder pools over BOS ... EOS. Reject incompatible metadata instead of
    # silently changing which tokens contribute to the embedding.
    for name in ("add_bos_token", "add_eos_token"):
        if read(name, True) is not True:
            raise ValueError(f"EmbeddingGemma GGUF requires tokenizer.ggml.{name}=true")
        config[name] = True

    normalizer = model.normalizer_spec
    normalizer.name = "identity"
    normalizer.add_dummy_prefix = read("add_space_prefix", True)
    normalizer.remove_extra_whitespaces = read("remove_extra_whitespaces", False)
    normalizer.escape_whitespaces = True
    for index, (token, score, kind) in enumerate(zip(tokens, scores, kinds)):
        if (not isinstance(token, str) or not token or isinstance(score, bool) or
                not isinstance(score, (int, float)) or not math.isfinite(score) or
                isinstance(kind, bool) or kind not in (1, 2, 3, 4, 5, 6)):
            raise ValueError(f"GGUF tokenizer has an invalid piece at index {index}")
        # GGUF stores added whitespace tokens literally. SentencePiece matches
        # its pieces after escaping spaces; keep their IDs while restoring that
        # representation. The GGUF exporter may also mark <unk> as CONTROL.
        model.pieces.add(piece=token.replace(" ", "▁"), score=score,
                         type=sentencepiece.ModelProto.SentencePiece.UNKNOWN
                         if index == trainer.unk_id else kind)
    context = source.kv("gemma-embedding.context_length")
    if context is not None:
        config["model_max_length"] = context
    return {"frontend/tokenizer.model": model.SerializeToString(),
            "frontend/tokenizer_config.json": json.dumps(config, ensure_ascii=False).encode()}
