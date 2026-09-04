"""Frontend, sampling, multimodal and vision helpers shared by the reference models.

Nothing here is specific to one architecture: it is the machinery around a decoder --
resource-backed tokenizer and chat template, the sampler, the multimodal batch, the
activation tap, and the vision-tower operators -- rather than any decoder itself.
"""
