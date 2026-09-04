"""One module per published quantised export of this architecture.

The object contract lives in `inventory.py` and is the same for every export: what differs
here is only where the numbers come from. Each export was written by a different quantiser,
so each reads a different source — ModelOpt states a scale as a multiplier, compressed-tensors
states it as a divisor, one export ships a second checkpoint holding the quantised weights,
another encodes its embedding rows in FP8. Those are file formats, not model variants, which
is why they are separate readers rather than branches in one.
"""
