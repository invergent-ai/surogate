nsys profile -o trace <command>

# Summary of GPU kernels by time
nsys stats --report cuda_gpu_kern_sum trace.nsys-rep

# Summary of CUDA API calls (shows CPU-side overhead)
nsys stats --report cuda_api_sum trace.nsys-rep

# Memory operations
nsys stats --report cuda_gpu_mem_size_sum trace.nsys-rep
