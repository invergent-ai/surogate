#pragma once

namespace sinfer {

// The CUDA architecture list libsinfer.so was compiled for, as CMake spelled it
// (e.g. "120a"). Reported by the products' --version output; a mismatch with the
// running device is why kernels fail to launch on an otherwise healthy card.
const char* built_cuda_architectures();

}  // namespace sinfer
