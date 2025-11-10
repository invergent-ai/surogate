uv pip install "sglang" --prerelease=allow --no-deps
uv pip install "diffusers==0.35.2" "imageio==2.36.0" "opencv-python==4.10.0.84" "checkpoint-engine==0.1.2" \
   "yunchang==0.6.3.post1" "opencv-python==4.10.0.84" "imageio-ffmpeg==0.5.1" "PyYAML==6.0.1" "moviepy>=2.0.0" \
   "cloudpickle" "remote-pdb" "torchcodec==0.5.0" "st_attn ==0.0.7"  "vsa==0.0.4" "flashinfer-cubin==0.5.2"

git clone https://github.com/sgl-project/sglang
cd sglang/sgl-kernel
rm -rf dist/* || true && \
  CMAKE_POLICY_VERSION_MINIMUM=3.5 MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 \
  uv build --wheel -Cbuild-dir=build . --verbose --color=always --no-build-isolation
uv pip install dist/sgl_kernel-*.whl