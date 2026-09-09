"""Version of the prepared serving cache, independent of conversion dependencies."""

# Rebuild prepared frontends with corrected stop tokens and SentencePiece domains.
# 5: the vision tower is stored as the checkpoint ships it, not quantized. An artifact
#    cached before this carries the four-bit tower, and nothing about its source changed,
#    so only the version tells the two apart.
SERVING_CACHE_VERSION = 5
