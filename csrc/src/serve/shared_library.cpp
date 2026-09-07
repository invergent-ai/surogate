// The translation unit of libsinfer.so.
//
// Every serve product -- surogate-engine, surogate-engine-cli, surogate-embed and
// the _surogate_serve Python module -- is the same engine behind a different front
// door. Linked against the static archives, each embedded its own copy of the
// device code: 490 MB of fatbin per product, 2.0 GB for the four, which a wheel
// would have to carry four times over. They link this library instead, so the
// fatbin exists once.
//
// The library has no callers of its own, so the archives go in under
// --whole-archive (csrc/CMakeLists.txt) and this file is what gives the target a
// source. It carries the one fact worth asking a built .so for: the CUDA
// architectures its device code was compiled for. A binary that reports 120a on a
// SM89 card is the answer to "why does every kernel launch fail".

#include "shared_library.h"

namespace sinfer {

const char* built_cuda_architectures() {
    return SINFER_BUILD_CUDA_ARCHS;
}

}  // namespace sinfer
