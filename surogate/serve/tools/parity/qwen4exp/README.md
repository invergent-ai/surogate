# Qwen3.8-Flash-Next parity tools

Token 0 of a fresh sequence has no cross-token dependence, so every block of layer 0 (and the
n-gram PLE entering layer 1) is a state-free unit test. These scripts re-derive that token on
the CPU straight from the GGUF with llama.cpp's algebra and compare the engine's dumps
against it. They were the tools that found the two Flash-Next defects (SiLU instead of the
sigmoid GDN output gate; the RMSNorm kernels not loading `z` for the sigmoid epilogue).

Run from the repository root with the `.venv` interpreter. `SUROGATE_PARITY_DIR` (default `.`)
holds the dumps and the saved reference vectors.

1. Engine dumps: `SUROGATE_SERVE_DUMP_RESIDUAL=$SUROGATE_PARITY_DIR/dumps surogate-engine-cli
   <artifact> --prompt "The capital of France is" --greedy --max-new 2 --no-thinking
   --no-cuda-graph --max-context 2048 --kv-capacity auto` writes `f<N>_layer<L>.bin` (residual
   streams before each layer), `f<N>_L<L>_{mixer,mlp}_<stage>.bin` (hc mix/inject/block
   output/combined, GDN fused projection, conv, α/β/g, `o`, gated norm, out-projection, PLE
   in/out) for the first two forwards, layers 0-1, and `f<N>_final.bin`.
2. `compare_dumps.py [dir] [fN]` prints per-dump statistics (token 0 first/last values).
3. `hc_reference.py <templated prompt file>` — layer-0 attention-side hyper-connection mix;
   `gdn_reference.py` — the GDN block (also saves `ref_*.npy` in the engine's head order);
   `mlp_reference.py gdn_reference.py` — combine, MLP-side mix, softmax router top-10,
   experts, shared expert, layer output; `ple_reference.py mlp_reference.py gdn_reference.py`
   — layer-1 PLE rows/gather/gate/conv and layer 1's first mix. Each prints llama.cpp's
   `llama-eval-callback` values for the same prompt next to its own.
4. `compare_stage.py <dump.bin> <ref.npy> [block]` — full-vector, per-block (per-head)
   rel-L2 / cosine / scale of token 0; this is what localised the RMSNorm defect after
   first/last-3 spot checks had not.
5. `check_rows.py` decodes chosen row spans of a W8 artifact tensor (the artifact verifier
   samples only the first/last 512 rows); `verify_flash_artifact.py` is the full artifact
   check; `patch_conv.py` rewrites the GDN conv objects in place from the GGUF (kept as the
   template for in-place artifact patches).

llama.cpp reference: `llama-eval-callback -m <gguf> -p "$(cat prompt)" --no-warmup` prints
every tensor (prefix `common_debug_cb_eval:`); the hc mix / GDN / MoE / PLE algebra is in
`study/llama.cpp-master/src/models/qwen4exp.cpp`. Value heads: llama.cpp pairs value head h
with key head h mod 16 on the GGUF's tiled order; the converter un-tiles to HF grouped order
and the kernel pairs with ⌊h/3⌋ — equivalent, verified numerically.
