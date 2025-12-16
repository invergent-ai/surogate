import os

os.environ["TRITON_DISABLE_LINE_INFO"] = "1" # Reduces Triton binary size
os.environ["TRITON_FRONT_END_DEBUGGING"] = "0" # Disables debugging

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

