import functools
import importlib
import inspect
import os
import pathlib
import re
import sys
import tempfile
from typing import Optional
from filelock import FileLock
import torch
from unsloth.models._utils import unsloth_compile_transformers

from surogate.utils.logger import get_logger

logger = get_logger()

UNSLOTH_COMPILE_LOCATION = "unsloth_compiled_cache"
UNSLOTH_COMPILE_USE_TEMP = False

def compile_transformers(model_type: str = "qwen3"):
    model_location = f"transformers.models.{model_type}.modeling_{model_type}"
    try:
        exec(f"import {model_location}", globals())
    except ModuleNotFoundError:
        return

    torch_compile_options = {
        "debug": False,
        "trace.graph_diagram": False,
        "trace.enabled": False,
        "disable_progress": False,
        "verbose_progress": True,

        "epilogue_fusion": True,
        "max_autotune": False,
        "shape_padding": True,
        "triton.cudagraphs": False,
        "dce": True,
        "memory_planning": True,
        "coordinate_descent_tuning": False,
        "compile_threads": 4,
        "group_fusion": True,
    }
    # patch_lora_forwards(torch_compile_options)

    unsloth_compile_transformers(
        "",
        "",
        model_types=["llama", "qwen3"],
        compile_attention=True,
        disable_causal_masks=True,
        compile_torch_modules=True,
        compile_custom_modules=True,
        compile_function_calls=True,
        fuse_lm_head=True,
        gradient_checkpointing=False,
        manual_replacements=True,
        fast_lora_forwards=True,
        fast_residual_stream=False,
        accurate_accumulation=True,
        epilogue_fusion=True,
        max_autotune=False,
        shape_padding=True,
        cudagraphs=False,
        fullgraph=True,
        import_from_cache=False,
        disable=False,
        return_logits=False,
        debug=False,
        sdpa_dynamic_compile=True,
        sdpa_gqa_replace=True,
        sdpa_bool_masks=True,
        sdpa_dynamic_mask=True
    )


# Torch.compiling makes things slower - rather just leave it as addmm
COMPILED_LORA_FORWARD = """
torch_addmm = torch.addmm
torch_add   = torch.add
# @torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def lora_forward(result, lora_A, lora_B, dropout, x, scaling):
    # Use result.dtype (bfloat16 from base layer) since x may have been cast to float32
    # by _cast_input_dtype when autocast is disabled
    target_dtype = result.dtype
    xA = dropout(x).to(target_dtype) @ lora_A.weight.to(target_dtype).t()
    # output = result + scaling * xA @ lora_B.weight.t()
    shape = result.shape
    output = torch_addmm(
        result.view(-1, shape[-1]),
        xA.view(-1, xA.shape[-1]),
        lora_B.weight.to(target_dtype).t(),
        alpha = scaling,
        beta = 1,
    ).view(shape)

    bias = lora_B.bias
    if bias is not None:
        output = torch_add(
            output,
            bias.to(target_dtype),
            alpha = scaling,
        )
    return output
pass

"""

def patch_lora_forwards(torch_compile_options):
    # All Unsloth Zoo code licensed under LGPLv3
    Linear_LoRA_Layers = get_lora_layer_modules()
    success = 0
    could_not_replace_modules = []
    for function, parent, child in Linear_LoRA_Layers:
        if not hasattr(function, "forward"): continue
        if function.forward.__name__ == "unsloth_forward": continue

        exec(f"import {parent}", locals(), globals())
        source = inspect.getsource(function.forward)

        spaces = source.find("def")
        source = source.split("\n")
        source = "\n".join(x[spaces:] for x in source)
        old_hash = hash(source)

        # Remove cloning
        source = source.replace("result = result.clone()", "")

        # Use addmm
        old1 = "output = lora_B(lora_A(dropout(x))) * scaling"
        old2 = "result = result + lora_B(lora_A(dropout(x))) * scaling"
        add = "result = result + output"

        if (old1 not in source and add not in source) and \
                (old2 not in source):
            pass
        else:
            replace = "return lora_forward(result, lora_A, lora_B, dropout, x, scaling)"
            source = source.replace(old1, replace)
            source = source.replace(old2, replace)

        # Update function name
        source = source.replace(
            "def forward",
            "def unsloth_forward",
            1,
        )

        # Remove variant_kwargs = {k: kwargs.pop(k, None) for k in VARIANT_KWARG_KEYS}
        # No need for alora for now
        variant_kwarg_keys = "variant_kwargs = {k: kwargs.pop(k, None) for k in VARIANT_KWARG_KEYS}"
        variant_found = source.find(variant_kwarg_keys)
        if variant_found != -1:
            variant_end = source.find("\n", variant_found + len(variant_kwarg_keys))
            source = source.replace(source[variant_found : variant_end], "")

        # Check failed upcasting
        replacements = [
            "x = x.to(lora_A.weight.dtype)",
            "x = self._cast_input_dtype(x, lora_A.weight.dtype)",
        ]

        if "torch.is_autocast_enabled()" not in source:
            new = "if not torch.is_autocast_enabled(): " \
                  "result, x = " \
                  "result.to(lora_A.weight.dtype), " \
                  "x.to(lora_A.weight.dtype)"
            for replace in replacements:
                source = source.replace(replace, new)

        source = source.replace(
            "self._check_forward_args(x, *args, **kwargs)",
            "",
        )

        if hash(source) != old_hash:
            success += 1
            forward = create_new_function(
                f"{child}_peft_forward",
                COMPILED_LORA_FORWARD + source,
                parent,
                dir(eval(parent)),
                prepend = \
                    f"\ntorch_compile_options = {torch_compile_options}\n"
            ).unsloth_forward
            exec(f"{parent}.{child}.forward = forward", globals(), locals())
        else:
            could_not_replace_modules.append(parent)

    if success <= 5:
        print("Unsloth: Not an error, but could not optimize some PEFT modules.")

def get_lora_layer_modules():
    # All Unsloth Zoo code licensed under LGPLv3
    import peft.tuners.lora
    path = os.path.split(peft.tuners.lora.__file__)[0]
    files = os.listdir(path)

    Linear_LoRA_Layers = []
    for file in files:
        if file == "__init__.py" or not file.endswith(".py"): continue
        item = f"peft.tuners.lora.{file[:-len('.py')]}"
        exec(f"import {item}", locals(), globals())
        modules = dir(eval(item))
        modules = [x for x in modules if x.startswith("Linear") or x.endswith("Linear")]
        if len(modules) == 0: continue
        exec(f"from {item} import ({', '.join(modules)})", locals(), globals())
        Linear_LoRA_Layers += [(eval(x), item, x,) for x in modules]
    pass
    return tuple(Linear_LoRA_Layers)

disble_use_cache_logging = """
if hasattr(logger, "addFilter"):
    import logging
    class HideLoggingMessage(logging.Filter):
        def __init__(self, text): self.text = text
        def filter(self, x): return not (self.text in x.getMessage())
    pass
    logger.addFilter(HideLoggingMessage("`use_cache=True`"))
"""

def create_new_function(
        name,
        new_source,
        model_location,
        functions,
        prepend = "",
        append = "",
        overwrite = True,
        add_torch_compile = False,
):
    # All Unsloth Zoo code licensed under LGPLv3
    old_new_source = new_source
    # Fix all softmax low precisions to float32
    new_source = higher_precision_softmax(new_source)
    if new_source[0] == " ":
        spaces = new_source.find("def")
        new_source = new_source.split("\n")
        new_source = "\n".join(x[spaces:] for x in new_source)
    if add_torch_compile:
        new_source = \
            "@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)\n" \
            f"{new_source}"
    # Import items to make the function executable
    items = [x for x in functions if ((x in new_source) and (x != name) and not (f"def {x}(" in new_source))]
    # Patch for SiglipEncoder and others
    if "SiglipEncoder" in new_source: items += ["SiglipEncoder"]
    # Check for create_causal_mask, create_masks_for_generate, create_sliding_window_causal_mask
    mask_functions = get_mask_functions()
    for mask_function in mask_functions:
        if mask_function in new_source: items += [mask_function]

    # Full import script
    imports = "from torch import Tensor\n"
    imports += "import torch\n"
    imports += "import torch.nn as nn\n"
    imports += "from torch.nn import functional as F\n"
    imports += "from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable\n"
    imports += f"from {model_location} import (" + ", ".join(x for x in items) + ")" if len(items) != 0 else ""
    new_source = imports + "\n\n" + new_source
    # Check logger and remove use_cache
    if "logger" in items:
        new_source = new_source + "\n" + disble_use_cache_logging + "\n"
    new_source = prepend + new_source + append

    file_source = None
    compile_folder, UNSLOTH_COMPILE_USE_TEMP = get_compile_folder(use_tempfile = False)
    function_location = os.path.join(compile_folder, f"{name}.py")

    if not overwrite and os.path.isfile(function_location):
        # Check if exactly equivalent
        with open(function_location, "r", encoding = "utf-8") as f:
            file_source = f.read()
        if file_source != new_source:
            overwrite = True

    def write_file(function_location, write_new_source):
        lock = get_lock(function_location)
        new_write_bytes = write_new_source.encode("utf-8")
        try:
            with lock:
                # existence check
                try:
                    st = os.stat(function_location)
                except Exception as e:
                    st = None

                need_write = False
                if st is None or st.st_size != len(new_write_bytes):
                    need_write = True
                else:
                    with open(function_location, "rb") as f:
                        need_write = f.read() != new_write_bytes

                if need_write:
                    with open(function_location, "wb", buffering = 0) as file:
                        file.write(new_write_bytes)
                        file.flush()
                        os.fsync(file.fileno())
            return None
        except Exception as e:
            # consider adding logging to main_process only
            # counterpoint: we may want to see errors on all processes
            if os.environ.get("UNSLOTH_LOGGING_ENABLED", "0") == "1":
                logger.error(f"Unsloth: Failed to write file {function_location} because {str(e)}")
            return None

    if overwrite or not os.path.isfile(function_location):
        try:
            distributed_function(1, write_file, function_location, new_source)
        except Exception as error:
            if UNSLOTH_COMPILE_USE_TEMP:
                raise RuntimeError(error)
            else:
                # Failed so instead use a temporary directory
                compile_folder, UNSLOTH_COMPILE_USE_TEMP = get_compile_folder(use_tempfile = True)
                function_location = os.path.join(compile_folder, f"{name}.py")
                distributed_function(1, write_file, function_location, new_source)


    # Now import modules! Use a tempfile if it fails on the first try!
    old_path = None
    new_module = None

    def import_module(compile_folder, name):
        target_name = os.path.join(compile_folder, f"{name}.py")
        lock = get_lock(target_name)
        # Add directory to sys.path temporarily if it's not already there
        if compile_folder not in sys.path:
            old_path = list(sys.path)
            # Fail if name already exists!
            if name in old_path:
                raise OSError(f"Unsloth: File {name} already exists")
            sys.path.insert(0, compile_folder)
        try:
            with lock:
                # Try standard import
                new_module = importlib.import_module(name)
                return new_module, old_path
        except Exception as e:
            if os.environ.get("UNSLOTH_LOGGING_ENABLED", "0") == "1":
                logger.error(f"Unsloth: Failed to import module {name} because {str(e)}")
            raise e

    try:
        new_module, old_path = import_module(compile_folder, name)
    except Exception as e:
        new_module = None
        # Try using temp directory instead!
        if not UNSLOTH_COMPILE_USE_TEMP:
            compile_folder, UNSLOTH_COMPILE_USE_TEMP = get_compile_folder(use_tempfile = True)
            function_location = os.path.join(compile_folder, f"{name}.py")
            distributed_function(1, write_file, function_location, new_source)
            if is_main_process():
                logger.info(f"Standard import failed for {name}: {e}. Using tempfile instead!")
            try:
                new_module, old_path = import_module(compile_folder, name)
            except Exception as e:
                new_module = None
                if is_main_process():
                    logger.info(f"Standard import failed for {name}: {e}. Using spec.loader.exec_module instead!")

        # Fallback to direct module loading
        if new_module is None:
            try:
                module_name = f"unsloth_cache_{name}"
                file_location = os.path.join(compile_folder, name) + ".py"
                lock = get_lock(file_location)
                with lock:
                    spec = importlib.util.spec_from_file_location(module_name, file_location)
                    new_module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = new_module
                    spec.loader.exec_module(new_module)
            except Exception as e:
                raise RuntimeError(f"Direct module loading failed for {name}: {e}")
    finally:
        # Restore original sys.path if we modified it
        if old_path is not None:
            sys.path = old_path

    if new_module is None:
        raise ImportError(f'Unsloth: Cannot import {name} from {UNSLOTH_COMPILE_LOCATION}')

    return new_module

def _get_compile_folder(use_tempfile = False):
    global UNSLOTH_COMPILE_USE_TEMP
    global UNSLOTH_COMPILE_LOCATION

    if use_tempfile:
        UNSLOTH_COMPILE_USE_TEMP = True
        leaf = os.path.basename(UNSLOTH_COMPILE_LOCATION)
        location = os.path.join(tempfile.gettempdir(), leaf)
        logger.info(
            f"Unsloth: We'll be using `{location}` for temporary Unsloth patches."
        )
        os.makedirs(location, exist_ok = True)
    else:
        location = UNSLOTH_COMPILE_LOCATION
        try:
            # Try creating the directory
            os.makedirs(location, exist_ok = True)
            return location, UNSLOTH_COMPILE_USE_TEMP
        except Exception as e:
            logger.error(f"Unsloth: Failed to create directory `{UNSLOTH_COMPILE_LOCATION}` because {str(e)}")

            # Instead use a temporary location!
            location, UNSLOTH_COMPILE_USE_TEMP = _get_compile_folder(use_tempfile = True)
    return location, UNSLOTH_COMPILE_USE_TEMP

def get_compile_folder(use_tempfile = False):
    location, UNSLOTH_COMPILE_USE_TEMP = distributed_function(2, _get_compile_folder, use_tempfile)
    return location, UNSLOTH_COMPILE_USE_TEMP


def distributed_function(n = 1, function = None, *args, **kwargs):
    assert function is not None

    # Not launched distributed at all
    if not is_distributed():
        out = function(*args, **kwargs)
        return out if n == 1 else out

    # Multi-process: only main executes the function
    if is_main_process():
        out = function(*args, **kwargs)
        obj_list = [out] if n == 1 else list(out)
    else:
        obj_list = [None for _ in range(n)]

    # If the process group is initialized, we can synchronize / share the result
    if torch.distributed.is_initialized():
        # Broadcast result to all ranks
        torch.distributed.broadcast_object_list(obj_list, src = 0)
        # Barrier to make sure everyone waits until main is done
        torch.distributed.barrier()

    return obj_list[0] if n == 1 else obj_list

def is_distributed():
    return torch.distributed.is_initialized() or torch.distributed.is_torchelastic_launched()

def is_main_process():
    if torch.distributed.is_initialized():
        # torch.distributed.init_process_group was run, so get_rank works
        return torch.distributed.get_rank() == 0
    elif torch.distributed.is_torchelastic_launched():
        # accelerate launch for example calls init_process_group later
        return os.environ.get("RANK", "0") == "0"
    return True


def _lock_path_for(target: str) -> str:
    """ str needs to be a valid file path """
    locks_dir = pathlib.Path(target).parent / ".locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    return str(locks_dir / f".lock.{pathlib.Path(target).name}")

def get_lock(target: str, timeout: Optional[int] = None) -> FileLock:
    """
    Get a lock for a target file.
    target: str, the path to the file to lock
    timeout: int, the timeout in seconds for the lock
    If timeout is not provided, it will use the value of
    the environment variable UNSLOTH_LOCK_TIMEOUT, otherwise 10 seconds.

    Returns:
        FileLock, the lock for the target file
    """
    lock_path = _lock_path_for(target)
    if timeout is None:
        timeout = int(os.environ.get("UNSLOTH_LOCK_TIMEOUT", "10"))
    return FileLock(lock_path, timeout=timeout)

@functools.lru_cache(1)
def get_mask_functions():
    try:
        import transformers.masking_utils
        masking_utils = dir(transformers.masking_utils)
        return [x for x in masking_utils if x.startswith("create")]
    except:
        return []


# Convert F.softmax(x, ...) to F.softmax(x, ..., dtype = torch.float32).to(x.dtype)
def higher_precision_softmax(source):
    """
    Converts all softmax to float32 for eg:
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
    """
    softmax_objects = re.finditer(
        r"(nn\.functional\.softmax|F\.softmax)" \
        r"\(" \
        r"([^,]{1,}), " \
        r"(dim[ ]?\=[ ]?[\-0-9]{1,2})" \
        r"(\,[ ]?dtype[^\)]{1,})?" \
        r"\)",
        source,
    )
    for item in softmax_objects:
        full_match, matches = item.group(0), item.groups()
        softmax, variable, dim, dtype = matches
        new = f"{softmax}({variable}, {dim}, dtype = torch.float32).to({variable}.dtype)"
        source = source.replace(full_match, new)
    return source
