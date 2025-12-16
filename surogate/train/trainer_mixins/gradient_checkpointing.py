import inspect
from functools import wraps
from types import MethodType
from typing import Optional, List, Any, Callable, Tuple, ContextManager

import torch
import torch.utils.checkpoint
from peft import PeftModel
from transformers import Trainer
from torch.utils.checkpoint import (
    ContextManager,
    _DEFAULT_DETERMINISM_MODE,
    _checkpoint_without_reentrant_generator,
    noop_context_fn,
    _get_autocast_kwargs,
    _infer_device_type,
    get_device_states,
    set_device_states,
    contextlib,
    _get_device_module
)

from surogate.core.model.hf_config import HfConfigFactory
from surogate.core.model.registry import ModelTemplate
from surogate.core.model.utils import find_module_list
from surogate.utils.dist import is_dist
from surogate.utils.logger import get_logger
from surogate.utils.utils import deep_getattr

logger = get_logger()

global CPU_BUFFERS
global CPU_INDEX
global GPU_BUFFERS
global BACKWARD_PASS
global EXTRA_STREAMS
global MAIN_STREAMS
global MINIMUM_SIZE
global LAST_GC_INDEX
global FIRST_PASS
global CURRENT_GC_INDEX

CPU_BUFFERS = []
CPU_INDEX = None


class UnslothCheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        # All Unsloth Zoo code licensed under LGPLv3
        # check_backward_validity(args)
        # Check if no requires_grad in inputs
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.device_type = _infer_device_type(*args)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(
            ctx.device_type
        )
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device_type)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*args)

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        ctx._requires_gradient = False
        use_gpu_buffer = False

        for i, arg in enumerate(args):
            if torch.is_tensor(arg):

                if i == 0 and arg.requires_grad:
                    global FIRST_PASS
                    global LAST_GC_INDEX
                    if FIRST_PASS:
                        # Save last layer index so next run we do not offload activations
                        # Saves VRAM and saves some time
                        # See https://github.com/pytorch/torchtune/pull/1443
                        LAST_GC_INDEX += 1
                    pass
                    global CURRENT_GC_INDEX
                    CURRENT_GC_INDEX += 1

                    ctx._requires_gradient = True
                    new_size = arg.numel()

                    global MINIMUM_SIZE
                    global CPU_INDEX
                    if new_size > MINIMUM_SIZE and ((CURRENT_GC_INDEX != LAST_GC_INDEX) or FIRST_PASS):
                        use_gpu_buffer = True
                        global CPU_BUFFERS
                        global GPU_BUFFERS
                        global BACKWARD_PASS
                        global EXTRA_STREAMS
                        global MAIN_STREAMS
                        device = arg.device
                        device_index = device.index
                        GPU_BUFFER = GPU_BUFFERS[device_index]
                        MAIN_STREAM = MAIN_STREAMS[device_index]
                        EXTRA_STREAM = EXTRA_STREAMS[device_index]

                        # Handle interrupted training runs
                        if BACKWARD_PASS:
                            BACKWARD_PASS = False
                            CPU_INDEX = 0
                        pass

                        # Extend buffer size
                        if CPU_INDEX >= len(CPU_BUFFERS):
                            x = torch.empty(new_size, dtype=arg.dtype, device="cpu", pin_memory=True)
                            CPU_BUFFERS.append(x)
                        pass

                        x = CPU_BUFFERS[CPU_INDEX]
                        shape = arg.shape
                        if new_size > x.numel(): x.resize_(new_size)
                        if new_size > GPU_BUFFER.numel(): GPU_BUFFER.resize_(new_size)
                        x = x[:new_size].view(shape)

                        # See https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams
                        EXTRA_STREAM.wait_stream(MAIN_STREAM)
                        with torch.cuda.stream(EXTRA_STREAM):
                            x.copy_(arg, non_blocking=True)

                        ctx._saved_metadata = (new_size, shape, CPU_INDEX, device_index, MAIN_STREAM, EXTRA_STREAM,)
                        CPU_INDEX += 1
                        tensor_inputs.append(None)
                    else:
                        ctx._saved_metadata = (None, None, None, None, None, None,)
                        tensor_inputs.append(arg)
                    pass
                else:
                    tensor_inputs.append(arg)
                pass
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)
            pass
        pass
        if ctx._requires_gradient: ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = run_function(*args)

        if use_gpu_buffer: MAIN_STREAM.wait_stream(EXTRA_STREAM)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        # All Unsloth Zoo code licensed under LGPLv3
        if not ctx._requires_gradient: return None

        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "When use_reentrant=True, torch.utils.checkpoint is incompatible"
                " with .grad() or passing an `inputs` parameter to .backward()."
                " To resolve this error, you can either set use_reentrant=False,"
                " or call .backward() without passing the `inputs` argument."
            )

        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        new_size, shape, CPU_INDEX, device_index, MAIN_STREAM, EXTRA_STREAM = ctx._saved_metadata
        if CPU_INDEX is not None:
            global GPU_BUFFER
            buffer = GPU_BUFFERS[device_index][:new_size].view(shape)
            x = CPU_BUFFERS[CPU_INDEX][:new_size].view(shape)

            # See https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams
            EXTRA_STREAM.wait_stream(MAIN_STREAM)
            with torch.cuda.stream(EXTRA_STREAM):
                buffer.copy_(x, non_blocking=True)
        else:
            # No GPU buffer seen
            if len(tensor_indices) != 0:
                inputs[tensor_indices[0]] = tensors[0]

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices[1:], start=1):
            inputs[idx] = tensors[i]

        global BACKWARD_PASS
        BACKWARD_PASS = True
        global FIRST_PASS
        FIRST_PASS = False
        global CURRENT_GC_INDEX
        CURRENT_GC_INDEX = 0

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(
                devices=rng_devices, enabled=ctx.preserve_rng_state, device_type=ctx.device_type
        ):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(ctx.fwd_devices, ctx.fwd_device_states, device_type=ctx.device_type)

            device_autocast_ctx = torch.amp.autocast(
                device_type=ctx.device_type, **ctx.device_autocast_kwargs
            ) if torch.amp.is_autocast_available(ctx.device_type) else contextlib.nullcontext()

            # detached_inputs = detach_variable(tuple(inputs))
            detached_inputs = []
            for inp in inputs:
                if not isinstance(inp, torch.Tensor):
                    detached_inputs.append(inp)
                    continue
                x = inp.detach()
                x.requires_grad = inp.requires_grad
                detached_inputs.append(x)

            # Wait for GPU buffer to finish
            if CPU_INDEX is not None:
                MAIN_STREAM.wait_stream(EXTRA_STREAM)
                x = buffer.detach()
                x.requires_grad_(True)
                detached_inputs[0] = x

            with torch.enable_grad(), device_autocast_ctx, torch.amp.autocast("cpu",
                                                                              **ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])

        if len(outputs_with_grad) == 0:
            pass
        else:
            torch.autograd.backward(outputs_with_grad, args_with_grad)

        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in detached_inputs
        )
        # Clear all memory
        for i in range(len(detached_inputs)):
            detached_inputs[i] = None
            inputs[i] = None

        return (None, None) + grads


@torch._disable_dynamo
def unsloth_checkpoint(
        function,
        *args,
        use_reentrant: Optional[bool] = None,
        context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
        determinism_check: str = _DEFAULT_DETERMINISM_MODE,
        debug: bool = False,
        **kwargs
):
    if use_reentrant is None:
        use_reentrant = True

    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs and use_reentrant:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    if use_reentrant:
        if context_fn is not noop_context_fn or debug is not False:
            raise ValueError(
                "Passing `context_fn` or `debug` is only supported when "
                "use_reentrant=False."
            )
        return UnslothCheckpointFunction.apply(function, preserve, *args)
    else:
        gen = _checkpoint_without_reentrant_generator(
            function, preserve, context_fn, determinism_check, debug, *args, **kwargs
        )
        # Runs pre-forward logic
        next(gen)
        ret = function(*args, **kwargs)
        # Runs post-forward logic
        try:
            next(gen)
        except StopIteration:
            return ret


class GradientCheckpointingMixin(Trainer):
    def _fix_gradient_checkpointing(self):
        # fix use_reentrant
        if hasattr(torch.utils.checkpoint, '_old_checkpoint'):  # avoid double patching
            return
        args = self.args
        if args.gradient_checkpointing_kwargs:
            use_reentrant_ = args.gradient_checkpointing_kwargs.get('use_reentrant')
        else:
            use_reentrant_ = None
        if use_reentrant_ is None:
            if is_dist() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                use_reentrant_ = False
            else:
                use_reentrant_ = True

        _old_checkpoint = torch.utils.checkpoint.checkpoint

        @wraps(_old_checkpoint)
        def _new_checkpoint(*args, use_reentrant=None, **kwargs):
            return unsloth_checkpoint(*args, use_reentrant=use_reentrant_, **kwargs)

        self.initialize_unsloth_gradient_checkpointing(self.model.dtype)

        torch.utils.checkpoint._old_checkpoint = _old_checkpoint
        torch.utils.checkpoint.checkpoint = _new_checkpoint
        torch.utils.checkpoint.CheckpointFunction = UnslothCheckpointFunction
        try:
            # Fix the old version of transformers.
            import transformers.modeling_utils
            transformers.modeling_utils.checkpoint = _new_checkpoint
        except (ImportError, AttributeError):
            pass

    def _prepare_gradient_checkpointing(self, model) -> None:
        HfConfigFactory.set_model_config_attr(model, 'use_cache', False)
        args = self.args

        if args.gradient_checkpointing:
            self.dynamic_gradient_checkpointing(model)

        gc_kwargs = {}
        parameters = inspect.signature(model.gradient_checkpointing_enable).parameters
        if 'gradient_checkpointing_kwargs' in parameters:
            gc_kwargs['gradient_checkpointing_kwargs'] = args.gradient_checkpointing_kwargs

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(**gc_kwargs)
            model.enable_input_require_grads()

        model_template = model.model_template
        model_arch = model_template.model_arch
        if model_template.is_multimodal and model_arch:
            for vision_tower_name in model_arch.vision_tower:
                vision_tower = deep_getattr(model, vision_tower_name)
                if hasattr(vision_tower, 'enable_input_require_grads'):
                    try:
                        if args.vit_gradient_checkpointing:
                            vision_tower.gradient_checkpointing_enable(**gc_kwargs)
                            vision_tower.enable_input_require_grads()
                        else:
                            vision_tower.gradient_checkpointing_disable()
                            vision_tower.disable_input_require_grads()
                    except (NotImplementedError, AttributeError) as e:
                        logger.warning(f'prepare gradient_checkpointing failed: {e}')

        # Avoid vit_gradient_checkpointing being overwritten by transformers.Trainer.gradient_checkpointing_enable.
        self.args.gradient_checkpointing = False

    def dynamic_gradient_checkpointing(self, model, including_vit: bool = True) -> None:
        if isinstance(model, PeftModel):
            model = model.model

        model_template: ModelTemplate = getattr(model, 'model_template', None)
        if model_template is not None and model_template.is_multimodal and model_template.model_arch:
            tower_names = model_template.model_arch.language_model.copy()
            if including_vit:
                tower_names += model_template.model_arch.vision_tower
        else:
            tower_names = [None]

        model.supports_gradient_checkpointing = True
        for tower_name in tower_names:
            if tower_name is None:
                model_tower = model
            else:
                model_tower = deep_getattr(model, tower_name)
            model_tower.supports_gradient_checkpointing = True
            module_list = find_module_list(model_tower)
            if module_list is None:
                continue
            self._add_gradient_checkpointing(module_list)
            logger.info(f'Automatically add gradient_checkpointing to {model_tower.__class__}.')

    def _add_gradient_checkpointing(self, module_list):
        requires_grad = None

        def _new_forward(self, *args, **kwargs):
            nonlocal requires_grad
            if requires_grad is None:
                requires_grad = any(p.requires_grad for p in self.parameters())

            new_args = self._kwargs_to_args(self.__old_forward, args, kwargs)
            if new_args is not None and self.gradient_checkpointing and self.training:
                if new_args and isinstance(new_args[0], torch.Tensor) and requires_grad and not new_args[
                    0].requires_grad:
                    new_args[0].requires_grad_(True)
                layer_ret = self._gradient_checkpointing_func(self.__old_forward, *new_args)
                logger.info_once('Successfully using dynamic gradient checkpointing.')
            else:
                layer_ret = self.__old_forward(*args, **kwargs)
            return layer_ret

        for module in module_list:
            module.gradient_checkpointing = False
            if hasattr(module, '_old_forward'):  # device_map
                __old_forward = module._old_forward
                module._old_forward = MethodType(_new_forward, module)
            else:
                __old_forward = module.forward
                module.forward = MethodType(_new_forward, module)
            module.__old_forward = __old_forward

    def _kwargs_to_args(self, func, args, kwargs) -> Optional[List[Any]]:
        parameters = inspect.signature(func).parameters
        args = list(args)
        parameters = list(parameters.items())[len(args):]
        for key, param in parameters:
            if key in kwargs:
                args.append(kwargs[key])
            elif param.default != param.empty:
                args.append(param.default)
            else:
                return
        return args

    def initialize_unsloth_gradient_checkpointing(self, dtype):
        # All Unsloth Zoo code licensed under LGPLv3
        global CPU_BUFFERS
        global CPU_INDEX
        global GPU_BUFFERS
        global BACKWARD_PASS
        global EXTRA_STREAMS
        global MAIN_STREAMS
        global MINIMUM_SIZE
        global LAST_GC_INDEX
        global FIRST_PASS
        global CURRENT_GC_INDEX
        CPU_BUFFERS = []
        CPU_INDEX = 0

        for i in range(200):
            x = torch.empty(128 * 1024, dtype=dtype, device="cpu", pin_memory=True)
            CPU_BUFFERS.append(x)

        # Allocate buffers to how many GPUs
        n_gpus = torch.cuda.device_count()
        try:
            GPU_BUFFERS = tuple([torch.empty(2 * 256 * 2048, dtype=dtype, device=f"cuda:{i}") for i in range(n_gpus)])
        except Exception as e:
            print("=" * 10 + "\n")
            print("Unsloth: Your setup does not support `PYTORCH_CUDA_ALLOC_CONF`\n")
            print("Please set `import os; os.environ['PYTORCH_CUDA_ALLOC_CONF'] = '';`\n")
            print("Then re-run Unsloth from the start.")
            print("=" * 10 + "\n")
            raise

        BACKWARD_PASS = True
        EXTRA_STREAMS = tuple([torch.cuda.Stream() for i in range(n_gpus)])
        MAIN_STREAMS = tuple([torch.cuda.default_stream(torch.device(f"cuda:{i}")) for i in range(n_gpus)])

        # Minimum size to enable Unsloth GC is 2MB -> 32 layers = 64MB
        n_bytes = torch.finfo(dtype).bits // 8
        MINIMUM_SIZE = 2 * 1024 * 1024 // n_bytes

        # Disable offloading on the last layer - uses more VRAM and is slower
        # See https://github.com/pytorch/torchtune/pull/1443
        LAST_GC_INDEX = 0
        FIRST_PASS = True
        CURRENT_GC_INDEX = 0
