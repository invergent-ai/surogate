FlashAttention-2 v2.8.3 sources (Dao-AILab, BSD-3-Clause, LICENSE here), copied 2026-11-21 by lane g2 from gpubox2
build-inkling/build/integrator/_deps/flash_attn-src (the serving branch's fetch: our `csrc/cmake/patch_flash_attn.sh`
is NOT applied here; the fork ports its ordered-dQ semaphore itself). NOT compiled: the base for the Inkling fork in
`../` (namespace inkling_flash; rel-bias near part, far part with seqused_k, fp32 O, given lse/delta, GQA head-looped
backward, fp32 dK/dV, d_rel) -- lanes/g2/DESIGN-kernels.md A.1.
