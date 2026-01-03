- SUROGATE_MATMUL_AUTOTUNE=1 (default off), plus SUROGATE_MATMUL_AUTOTUNE_TOPK, ..._ITERS, ..._WARMUP, ..._VERBOSE. csrc/src/kernels/  matmul.cpp:89
- Autotune is skipped during CUDA graph capture and for accumulate=true calls (to avoid perturbing accumulation buffers). csrc/src   kernels/matmul.cpp:333
- 
Implemented Proposal A “Matmul Plans”: MatmulPlanCache caches cuBLASLt descriptors/layouts + chosen algo per (shape/
    layout/dtype/epilogue/workspace) and optionally autotunes top‑K algos via env vars
    SUROGATE_MATMUL_AUTOTUNE{,_TOPK,_ITERS,_WARMUP,_VERBOSE} (autotune skips CUDA graph capture + accumulate=true). See csrc/
    src/kernels/matmul_plans.h:16, csrc/src/kernels/matmul.cpp:45.
  - Plumbed MatmulPlanCache* plan_cache through matmul APIs and passed the per-run-state cache (rs.MatmulPlans.get()) at call
    sites. See csrc/src/kernels/kernels.h:106, csrc/src/kernels/kernels.cpp:820, csrc/src/training/model.h:216, csrc/src/
    training/model.cpp:57.
  - Added tests under csrc/src/testing/test-matmul-plans.cu:100 and wired them into unit-tests in csrc/CMakeLists.txt:330.
  - Verified locally: make unit-tests and make test both pass