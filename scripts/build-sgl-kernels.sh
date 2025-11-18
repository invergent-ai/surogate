git clone https://github.com/sgl-project/sglang
cd sglang/sgl-kernel
rm -rf dist/* || true && \
  CMAKE_POLICY_VERSION_MINIMUM=3.5 MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 \
  uv build --wheel -Cbuild-dir=build . --verbose --color=always --no-build-isolation
uv pip install dist/sgl_kernel-*.whl