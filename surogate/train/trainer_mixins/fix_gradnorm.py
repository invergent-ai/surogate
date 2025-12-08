from contextlib import contextmanager

import torch


class FixGradnormNan:
    @staticmethod
    @contextmanager
    def _fix_grad_norm_nan():
        from accelerate import Accelerator
        origin_clip_grad_norm_ = Accelerator.clip_grad_norm_

        def clip_grad_norm_(self, parameters, *args, **kwargs):
            # If NaN occurs, ignore weight updates.
            parameters = list(parameters)
            grad_norm = origin_clip_grad_norm_(self, parameters, *args, **kwargs)
            if isinstance(grad_norm, torch.Tensor) and grad_norm.isnan().item():
                for p in parameters:
                    p.grad = None
            return grad_norm

        Accelerator.clip_grad_norm_ = clip_grad_norm_
        try:
            yield
        finally:
            Accelerator.clip_grad_norm_ = origin_clip_grad_norm_