"""Version of the prepared serving cache, independent of conversion dependencies."""

# 7 includes GLM pooled indexer weights and full checkpoint context metadata.
# 8: an unquantised Qwen3.5-family checkpoint is stored BF16 (was re-quantised to W8).
SERVING_CACHE_VERSION = 8
