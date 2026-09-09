# Makefile for Surogate - LLM Training System
# Wraps CMake build system for convenience

BUILD_DIR ?= csrc/build
BUILD_TYPE ?= Release
PARALLEL_JOBS ?= $(shell nproc)

CCACHE := $(shell which ccache 2>/dev/null)
CUDA_HOME ?= $(or $(CUDA_PATH),$(shell dirname $$(dirname $$(which nvcc 2>/dev/null)) 2>/dev/null),/usr/local/cuda)
# Resolve version-switching symlinks such as /usr/local/cuda before handing the
# compiler and toolkit root to CMake.  Otherwise an existing CMake cache can
# retain one toolkit's libraries while the symlink starts compiling objects
# with another toolkit's cudaDeviceProp ABI.
CUDA_HOME := $(realpath $(CUDA_HOME))
CUDA_CMAKE_FLAGS := -DCMAKE_CUDA_COMPILER=$(CUDA_HOME)/bin/nvcc -DCUDAToolkit_ROOT=$(CUDA_HOME)
ifdef CCACHE
CCACHE_FLAGS := -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
 -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
export CCACHE_CUDA_PATHS := $(CUDA_HOME)
endif

# Link _surogate against the venv's pip NCCL (2.29.x, what torch expects) instead of the
# system libnccl (2.28.x): loading the system copy first breaks torch's libtorch_cuda
# (undefined ncclCommResume). The RPATH already points at $ORIGIN/../nvidia/nccl/lib.
PIP_NCCL := $(abspath .venv/lib/python3.12/site-packages/nvidia/nccl)
ifneq ($(wildcard $(PIP_NCCL)/lib/libnccl.so.2),)
NCCL_CMAKE_FLAGS := -DNCCL_INCLUDE_DIR=$(PIP_NCCL)/include -DNCCL_LIB_DIR=$(PIP_NCCL)/lib
endif

.PHONY: all build build-all export-checkpoint wheel wheel-cu128 wheel-cu130 configure clean clean-all build-tests test test-unit test-integration test-all regression-smoke regression-update-baseline regression-gpu help info format format-check format-cpp format-py lint-py

# Default target
all: build

# Trainer and serving engine in one command. They are separate targets because they
# are separate builds -- different build dirs, and `make build` deliberately does not
# pay for the engine -- but a tree that both trains and serves needs both, and asking
# people to remember two commands is how one of them goes missing.
build-all: build serve-build

# Configure the build
configure:
	cmake -S csrc -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CUDA_CMAKE_FLAGS) $(CCACHE_FLAGS) $(NCCL_CMAKE_FLAGS)

# Build all targets
build: configure
	cmake --build $(BUILD_DIR) --parallel $(PARALLEL_JOBS)
	cp -f $(BUILD_DIR)/_surogate*.so surogate/
	cp -f $(BUILD_DIR)/_surogate*.so .venv/lib/python3.12/site-packages/surogate/
	cp -f $(BUILD_DIR)/libsurogate-common.so surogate/
	cp -f $(BUILD_DIR)/libsurogate-common.so .venv/lib/python3.12/site-packages/surogate/

# ---------------------------------------------------------------------------
# Serving engine (csrc/src/serve, first-class targets of the main project;
# design/serve-engine-plan.md). Uses a separate build dir so serve builds
# never touch the live training build in $(BUILD_DIR).
SERVE_BUILD_DIR ?= csrc/build-serve
# The project's own venv, not `uv run`: `uv run` would rebuild the CUDA extension to satisfy
# the project dependency, and these tests need none of it.
PYTEST ?= .venv/bin/python -m pytest

# One engine binary for the RTX line: Ada (4090) and Blackwell (5090, PRO 6000). The kernels
# that only Blackwell can run carry only its cubin -- csrc/CMakeLists.txt pins those targets --
# so the second architecture costs what it actually uses rather than doubling the fatbin.
SERVE_CUDA_ARCHS ?= 89;120a

serve-configure:
	cmake -S csrc -B $(SERVE_BUILD_DIR) -G Ninja \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CUDA_ARCHITECTURES="$(SERVE_CUDA_ARCHS)" \
		-DSUROGATE_SERVE_CUDA_ARCHS="$(SERVE_CUDA_ARCHS)" \
		-DPYTHON_BINDING=ON $(CCACHE_FLAGS)

serve-build: serve-configure
	cmake --build $(SERVE_BUILD_DIR) --parallel $(PARALLEL_JOBS) \
		--target surogate-engine-cli surogate-engine surogate-embed _surogate_serve
	cp -f $(SERVE_BUILD_DIR)/_surogate_serve*.so surogate/ 2>/dev/null || true
	cp -f $(SERVE_BUILD_DIR)/_surogate_serve*.so .venv/lib/python3.12/site-packages/surogate/ 2>/dev/null || true
	# The module holds no device code of its own any more; libsinfer.so does, and
	# the module finds it through $$ORIGIN. Copying one without the other leaves an
	# import error about a missing library.
	cp -f $(SERVE_BUILD_DIR)/libsinfer.so surogate/ 2>/dev/null || true
	cp -f $(SERVE_BUILD_DIR)/libsinfer.so .venv/lib/python3.12/site-packages/surogate/ 2>/dev/null || true

# Build the engine *and* every registered test binary. The tests are excluded from `all`,
# so naming the aggregate is what makes them exist; without it ctest reports "Not Run"
# against binaries nobody compiled.
serve-test-build:
	cmake -S csrc -B $(SERVE_BUILD_DIR) -G Ninja \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CUDA_ARCHITECTURES=120a \
		-DSUROGATE_SERVE_TESTS=ON $(CCACHE_FLAGS)
	cmake --build $(SERVE_BUILD_DIR) --parallel $(PARALLEL_JOBS) \
		--target surogate-engine-cli surogate-engine serve-tests

# The quantiser `surogate quantize` drives. Fetched from a pinned llama.cpp and built CPU-only,
# then installed into the package where the command looks for it -- the same two steps the
# wheel takes, so a source tree and an installed wheel behave alike. Revision and rationale:
# csrc/cmake/llama_cpp_quantizer.cmake
QUANTIZER_BUILD_DIR ?= build/quantizer

quantizer:
	cmake -S csrc -B $(QUANTIZER_BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=Release \
		-DSUROGATE_BUILD_QUANTIZER=ON -DPYTHON_BINDING=OFF -DBUILD_TESTS=OFF
	cmake --build $(QUANTIZER_BUILD_DIR) --parallel $(PARALLEL_JOBS) --target llama-quantize
	cmake --install $(QUANTIZER_BUILD_DIR) --prefix . --component quantizer
	@echo "==> surogate/serve/_llama_cpp/bin/llama-quantize"

# The engine's own suite. A test whose fixture is absent exits 77 and ctest reports it
# skipped, so a machine without the real weights still gets a meaningful pass.
#
# Run in parallel on one card. Most of these are op tests that allocate little and spend
# their time on the host, so serialising them left the GPU idle and the suite took minutes.
# Lower CTEST_PARALLEL if a card is small or busy; 1 restores the old serial behaviour.
CTEST_PARALLEL ?= 8

serve-test: serve-test-build
	cd $(SERVE_BUILD_DIR) && ctest --output-on-failure -j $(CTEST_PARALLEL) $(CTEST_FLAGS)

# The Python half: converters, artifact container, declaration contract. No GPU, seconds.
serve-test-py:
	$(PYTEST) -q tests/serve tests/test_serve_contract.py

# The rollout contract, driven by the client GRPO actually uses, and the GRPO loop
# end to end. Both need a checkpoint and a GPU, so they are separate from the suites
# above and skip without one:
#
#   make grpo-test SUROGATE_TEST_MODEL=/path/to/Qwen3-0.6B
#
# The end-to-end one takes about a minute and is the gate a change to the rollout
# path should pass: it is the only test that would catch a refused adapter reload, a
# reward that is always zero, or a run that finishes and then exits non-zero.
SUROGATE_TEST_MODEL ?=
SUROGATE_TEST_GRPO_GPUS ?= 0,1
# Which card the single-engine contract test serves on; the end-to-end one uses the
# pair above.
SUROGATE_TEST_DEVICE ?= 0

grpo-test:
	SUROGATE_TEST_MODEL="$(SUROGATE_TEST_MODEL)" \
	SUROGATE_TEST_DEVICE="$(SUROGATE_TEST_DEVICE)" \
	SUROGATE_TEST_GRPO_GPUS="$(SUROGATE_TEST_GRPO_GPUS)" \
	$(PYTEST) -q tests/serve/test_grpo_rollout_contract.py tests/grpo/test_grpo_run_smoke.py

# Everything the serving engine has. This is the command a change to `csrc/src/serve` or
# `surogate/serve` has to pass.
serve-check: serve-test-py serve-test

.PHONY: grpo-test serve-configure serve-build serve-test-build serve-test serve-test-py serve-check quantizer

# Internal helper: build + repair wheel for a given CUDA tag
# Usage: $(call build_wheel,cu128)
define build_wheel
	cp pyproject.toml pyproject.toml.bak && \
	trap 'mv -f pyproject.toml.bak pyproject.toml' EXIT INT TERM; \
	uv run --no-project --with tomlkit python3 .github/scripts/set_cuda_version_tag.py $(1) && \
	CMAKE_ARGS="$(CUDA_CMAKE_FLAGS) $(CCACHE_FLAGS) $(NCCL_CMAKE_FLAGS)" CMAKE_BUILD_PARALLEL_LEVEL=$(PARALLEL_JOBS) uv build --wheel --out-dir dist && \
	uv run --no-project --with auditwheel --with patchelf auditwheel repair dist/*.whl \
		-w dist/repaired/ \
		--exclude libcuda.so.1 \
		--exclude libcudart.so.12 \
		--exclude libcudart.so.13 \
		--exclude libcudnn.so.9 \
		--exclude libcufile.so.0 \
		--exclude libnccl.so.2 \
		--exclude libcublas.so.12 \
		--exclude libcublas.so.13 \
		--exclude libcublasLt.so.12 \
		--exclude libcublasLt.so.13 \
		--exclude libnvidia-ml.so.1 && \
	mv dist/repaired/*.whl dist/ && \
	rm -rf dist/repaired/ dist/*linux_x86_64*.whl
	@echo "Wheel ready in dist/:"
	@ls -lh dist/*.whl
endef

wheel-cu128:
	$(call build_wheel,cu128)

wheel-cu130:
	$(call build_wheel,cu130)

wheel-dev: configure
	cmake --build $(BUILD_DIR) --parallel $(PARALLEL_JOBS) --target _surogate
	cp -f $(BUILD_DIR)/_surogate*.so surogate/
	cp -f $(BUILD_DIR)/_surogate*.so .venv/lib/python3.12/site-packages/surogate/
	cp -f $(BUILD_DIR)/libsurogate-common.so surogate/
	cp -f $(BUILD_DIR)/libsurogate-common.so .venv/lib/python3.12/site-packages/surogate/

# ==============================================================================
# Format / Lint
# ==============================================================================
# `make format`       - format C++/CUDA with clang-format + Python with ruff
# `make format-check` - verify everything is already formatted (CI-friendly)
#
# Requirements:
#   clang-format (>= 18 recommended)  apt install clang-format
#   ruff                              pip install ruff  (or uv tool install ruff)

CLANG_FORMAT ?= clang-format
RUFF         ?= ruff

# Files to format: C++/CUDA under csrc/src, excluding vendored + generated.
CPP_SRC_FIND := find csrc/src \
    \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \
       -o -name '*.h' -o -name '*.hpp' -o -name '*.hh' \
       -o -name '*.cu' -o -name '*.cuh' \) \
    ! -path 'csrc/src/third_party/*'

format-cpp:
	@echo "==> clang-format (C++/CUDA)"
	@$(CPP_SRC_FIND) -print0 | xargs -0 $(CLANG_FORMAT) -i

format-py:
	@echo "==> ruff format (Python)"
	@$(RUFF) format surogate tests 2>/dev/null || $(RUFF) format surogate
	@echo "==> ruff check --fix (Python; non-blocking for remaining lint issues)"
	@$(RUFF) check --fix surogate tests 2>/dev/null || \
	  $(RUFF) check --fix surogate || \
	  echo "    (some lint issues remain — run 'make lint-py' to see them)"

lint-py:
	@$(RUFF) check surogate tests 2>/dev/null || $(RUFF) check surogate

format: format-cpp format-py
	@echo "==> done"

format-check:
	@echo "==> clang-format --dry-run"
	@$(CPP_SRC_FIND) -print0 | xargs -0 $(CLANG_FORMAT) --dry-run --Werror
	@echo "==> ruff format --check"
	@$(RUFF) format --check surogate tests 2>/dev/null || $(RUFF) format --check surogate
	@echo "==> ruff check"
	@$(RUFF) check surogate tests 2>/dev/null || $(RUFF) check surogate

# ==============================================================================
# Testing Targets
# ==============================================================================

# Build test executables without running them
build-tests:
	cmake -S csrc -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DBUILD_TESTS=ON $(CUDA_CMAKE_FLAGS) $(CCACHE_FLAGS) $(NCCL_CMAKE_FLAGS)
	cmake --build $(BUILD_DIR) --parallel $(PARALLEL_JOBS) --target unit-tests integration-tests

# Build and run unit tests (kernels, modules, components)
# Fast feedback loop for development
test-unit: build-tests
	cd $(BUILD_DIR) && ctest -R unit-tests --output-on-failure

# Build and run integration tests (training loops, distributed)
# Slower tests for full system validation
test-integration: build-tests
	cd $(BUILD_DIR) && ctest -R integration-tests --output-on-failure

# Build and run all tests (unit + integration)
# Full test suite for CI and pre-release validation
test-all: build-tests
	cd $(BUILD_DIR) && ctest --output-on-failure

# Default test target (backward compatible, runs unit tests)
test: test-unit

# First-month refactor regression harness. These targets intentionally avoid
# tests/test_distributed.py; distributed GPU coverage is driven by the baseline
# runner matrix instead.
regression-smoke:
	uv run pytest -q tests/test_regression_baseline_runner.py tests/test_moe_monitor.py --no-gpu
	uv run python -m surogate.regression.baseline_runner --out /tmp/surogate-regression-current --compare --report

regression-update-baseline:
	uv run python -m surogate.regression.baseline_runner --out /tmp/surogate-regression-current --baseline regression_baselines/locked --update-baseline --report

regression-gpu:
	uv run python -m surogate.regression.baseline_runner --out regression_baselines/current --baseline regression_baselines/locked --run --steps $${STEPS:-5} --compare --report

# Clean build artifacts (keep build directory structure)
clean:
	@if [ -d "$(BUILD_DIR)" ] && [ -f "$(BUILD_DIR)/CMakeCache.txt" ]; then \
		cmake --build $(BUILD_DIR) --target clean 2>/dev/null || true; \
	fi
	rm -rf $(BUILD_DIR)/CMakeCache.txt $(BUILD_DIR)/CMakeFiles
	rm -rf dist wheelhouse *.egg-info surogate/*.so
	rm -rf build

# Full clean - remove build directory entirely
clean-all:
	rm -rf $(BUILD_DIR)
	rm -rf dist *.egg-info

# Rebuild from scratch
rebuild: clean-all build

# Show build configuration
info:
	@echo "Build configuration:"
	@echo "  BUILD_DIR:     $(BUILD_DIR)"
	@echo "  BUILD_TYPE:    $(BUILD_TYPE)"
	@echo "  PARALLEL_JOBS: $(PARALLEL_JOBS)"
ifdef CCACHE
	@echo "  ccache:        enabled ($(CCACHE))"
	@echo "  ccache CUDA:   $(if $(shell ccache --version | grep -q '^ccache version [4-9]' && echo yes),enabled (CCACHE_CUDA_PATHS=$(CCACHE_CUDA_PATHS)),disabled (requires ccache >= 4.0))"
	@ccache --show-stats 2>/dev/null || true
else
	@echo "  ccache:        disabled (not found in PATH)"
endif

# Help target
help:
	@echo "Surogate Build System"
	@echo ""
	@echo "Usage: make [target] [options]"
	@echo ""
	@echo "Build Targets:"
	@echo "  all              - Build all targets (default)"
	@echo "  build            - Build all targets"
	@echo "  wheel            - Build Python wheel using uv"
	@echo "  wheel-dev        - Build Python wheel in development mode"
	@echo "  configure        - Run CMake configuration"
	@echo ""
	@echo "Format Targets:"
	@echo "  format           - Format C++/CUDA (clang-format) and Python (ruff)"
	@echo "  format-cpp       - Only format C++/CUDA under csrc/src"
	@echo "  format-py        - Only format Python under surogate/"
	@echo "  format-check     - Verify formatting is clean (exits nonzero if not)"
	@echo ""
	@echo "Test Targets:"
	@echo "  build-tests      - Build test executables without running them"
	@echo "  test             - Build and run unit tests (default, fast feedback)"
	@echo "  test-unit        - Build and run unit tests (kernels, modules, components)"
	@echo "  test-integration - Build and run integration tests (training, distributed)"
	@echo "  test-all         - Build and run all tests (unit + integration)"
	@echo "  regression-smoke - Run no-GPU first-month regression harness checks"
	@echo "  regression-gpu   - Run GPU first-month regression matrix (STEPS=N optional)"
	@echo ""
	@echo "Cleanup Targets:"
	@echo "  clean            - Clean build artifacts"
	@echo "  clean-all        - Remove build directory entirely"
	@echo "  rebuild          - Clean and rebuild from scratch"
	@echo ""
	@echo "Options (environment variables):"
	@echo "  BUILD_TYPE=<type>    - CMake build type: Release, Debug, RelWithDebInfo (default: Release)"
	@echo "  PARALLEL_JOBS=<n>    - Number of parallel build jobs (default: nproc)"
	@echo ""
	@echo "Examples:"
	@echo "  make                 # Build everything"
	@echo "  make test            # Build and run unit tests"
	@echo "  make test-all        # Build and run all tests"
	@echo "  make clean-all build # Full rebuild"
