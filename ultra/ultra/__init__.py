"""Fugu-Ultra: a generative workflow Conductor (parallel track to the Director router).

Standalone by design — Ultra vendors the worker-pool + grading infrastructure under
``ultra.workers`` / ``ultra.grading`` so the Conductor track never imports the
production ``director`` router package.
"""

__version__ = "0.0.1"
