import torch

from surogate.utils.logger import get_logger

logger = get_logger()

TORCH_COMPILE_DEDBUG = False

import accelerate

def torch_compile_kwargs(*args, **kwargs):
    logger.debug("Enabled auto compiling")
    return {
        "dynamic": True,
        "fullgraph": False,
        "options": {
            "epilogue_fusion": True,
            "max_autotune": True,
            "shape_padding": True,
            "trace.enabled": TORCH_COMPILE_DEDBUG,
            "triton.cudagraphs": False,
        },
    }

accelerate.utils.dataclasses.TorchDynamoPlugin.to_kwargs = torch_compile_kwargs
accelerate.utils.TorchDynamoPlugin.to_kwargs = torch_compile_kwargs
accelerate.accelerator.TorchDynamoPlugin.to_kwargs = torch_compile_kwargs
del accelerate

def patch_torch_compile(O3=False, ignore_errors=True, debug=TORCH_COMPILE_DEDBUG):
    # All Unsloth Zoo code licensed under LGPLv3
    import os, logging

    if debug:
        os.environ["TORCHDYNAMO_VERBOSE"] = "1"
        os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        torch._logging.set_logs(
            dynamo=logging.WARN,
            inductor=logging.WARN,
            graph_breaks=True,
            recompiles=True,
            recompiles_verbose=True,
            compiled_autograd_verbose=False,  # Produces too much code
            aot_joint_graph=False,  # Produces too much code
            aot_graphs=False,  # Produces too much code
            perf_hints=True,  # Performance improvement hints
        )
        torch._dynamo.config.verbose = True
    else:
        os.environ.pop("TORCHDYNAMO_VERBOSE", None)
        os.environ.pop("TORCHINDUCTOR_COMPILE_THREADS", None)
        os.environ.pop("TORCHINDUCTOR_FORCE_DISABLE_CACHES", None)
        os.environ.pop("TORCH_LOGS", None)
        torch._logging.set_logs(all=logging.CRITICAL)
        torch._dynamo.config.verbose = False
    pass

    os.environ["UNSLOTH_PATCHED"] = "1"
    # See https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html
    # Caches kernel generations for faster restarts
    # https://dev-discuss.pytorch.org/t/impact-of-multithreading-and-local-caching-on-torch-compile/2498/3
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE"] = "1"
    os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)

    # Duplicate functions will cause hashing issues
    # os.environ["TORCHINDUCTOR_CACHE_DIR"] = UNSLOTH_COMPILE_LOCATION

    # https://github.com/sayakpaul/diffusers-torchao?tab=readme-ov-file#things-to-keep-in-mind-when-benchmarking
    os.environ["ENABLE_AOT_AUTOGRAD_CACHE"] = "1"

    # Torch compile arguments
    torch_compile_arguments = [
        f"config.debug = {debug}",
        "config.dce = True",
        "config.memory_planning = True",
        # Using 'combined' memory pool will cause re-compiles for dynamic shapres. We just re-use already allocated memory pools
        "config.memory_pool = 'none'",
        "config.efficient_conv_bn_eval_fx_passes = True",  # Reduces stability a little bit
        "config.dynamic_scale_rblock = True",  # Scale down RBLOCK for better occupancy
        # Disable reorder_for_compute_comm_overlap since it errors for non multi GPU systems
        # "config.reorder_for_compute_comm_overlap = True", # # enable reordering pass for increasing overlap between compute and communication
        f"config.max_autotune = {O3}",  # enable slow autotuning passes to select algorithms
        f"config.max_autotune_pointwise = {O3}",
        # enable slow autotuning passes to select pointwise/reductions algorithms
        f"config.max_autotune_gemm = False",  # GEMM is unnecessary
        "config.max_autotune_gemm_backends = 'ATEN,TRITON,CPP'",  # Not much faster
        "config.autotune_fallback_to_aten = True",  # Fallback to ATEN backend
        "config.autotune_multi_device = True",  # If autotuning in subprocess, whether to use multiple devices
        f"config.coordinate_descent_tuning = {O3}",
        f"config.aggressive_fusion = {O3}",  # Careful changes results!
        # [TODO] COMBO KERNELS makes everything slower!
        # "config.combo_kernels = True", # Experimental - enable the combo kernel that combines data-independent kernels
        # "config.combo_kernel_foreach_dynamic_shapes = True",
        "config.freezing = False",  # Freezes weights --> ** only useful for inference **
        # f"config.triton.multi_kernel = {O3}", # use tuning to pick between different subkernels
        "config.cuda.enable_cuda_lto = True",
        "config.cuda.use_fast_math = True",
        f"config.cuda.compile_opt_level = {'-O2' if O3 else '-O1'}",
        # See torch.compile, the missing manual
        # https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8
        # f"config.emulate_precision_casts = {not debug}", # Force X.to(f32).to(f16) instead of X.to(f16)
        # when setting to not debug aka True, we get errors on torch2.6
        # TypeError: ValueRangeAnalysis.to_dtype() got an unexpected keyword argument 'use_compute_types'
        # this keyword exists in torch2.7.0 but not in torch2.6.0 so set to False until torch2.6.0 is deprecated.
    ]
    # Torch dynamo arguments
    torch_dynamo_arguments = [
        "config.accumulated_cache_size_limit = 1024",  # Bump up a bit from 256
        f"config.suppress_errors = {not debug and ignore_errors}",  # Supress errors for now
        f"config.do_not_emit_runtime_asserts = {not debug}",
        "config.inline_inbuilt_nn_modules = True",  # Torch 2.5 Regional recompilation
        "config.numpy_default_float = 'float32'",
        # FAILS for Gemma!
        "config.compiled_autograd = False",  # New Torch 2.4 feature which can compile backwards passes
        # https://pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html
        # [NOTE] recompile_limit and cache_size_limit are equivalent!
        "config.recompile_limit = 1024",  # Increase recompile amounts to 1024 - then will do eager
        "config.cache_size_limit = 1024",  # Flex Attention
        # f"config.fail_on_recompile_limit_hit = {not debug and ignore_errors}", # Ignore recompiles CANNOT be used in tandem with suppress_errors
        "config.allow_unspec_int_on_nn_module = True",
        # Integers in modules will auto wrap torch.tensor(self.vocab_size)
        f"config.optimize_ddp = {not debug}",  # Optimizes DDP, but can error out so disable on debug
        # Captures .item() for eg
        # n_chunks = int(torch.ceil((torch.tensor(vocab_size) / 262144) * 8))
        "config.capture_scalar_outputs = True",
        # Capture torch.arange(...), torch.zeros(...)
        "config.capture_dynamic_output_shape_ops = True",
    ]
    if not debug and ignore_errors:
        # Have to explicitly set it!
        torch._dynamo.config.suppress_errors = True

    import torch._inductor.config as config
    for _try_compile_argument in torch_compile_arguments:
        try:
            exec(_try_compile_argument)
        except:
            pass

    import torch._dynamo.config as config
    for _try_dynamo_argument in torch_dynamo_arguments:
        try:
            exec(_try_dynamo_argument)
        except:
            pass

