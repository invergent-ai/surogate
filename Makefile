# Makefile for Surogate - LLM Training System
# Wraps CMake build system for convenience

BUILD_DIR ?= csrc/build
BUILD_TYPE ?= Release
PARALLEL_JOBS ?= $(shell nproc)

.PHONY: all build export-checkpoint wheel configure clean clean-all build-tests test test-unit test-integration test-all help

# Default target
all: build

# Configure the build
configure:
	cmake -S csrc -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

# Build all targets
build: configure
	cmake --build $(BUILD_DIR) --parallel $(PARALLEL_JOBS)

# Build Python wheel
wheel:
	uv build --wheel

wheel-dev: configure
	cmake --build $(BUILD_DIR) --parallel $(PARALLEL_JOBS) --target _surogate
	cp -f $(BUILD_DIR)/_surogate*.so surogate/
	cp -f $(BUILD_DIR)/_surogate*.so .venv/lib/python3.12/site-packages/surogate/
	cp -f $(BUILD_DIR)/libsurogate-common.so surogate/
	cp -f $(BUILD_DIR)/libsurogate-common.so .venv/lib/python3.12/site-packages/surogate/

# ==============================================================================
# Testing Targets
# ==============================================================================

# Build test executables without running them
build-tests:
	cmake -S csrc -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DBUILD_TESTS=ON
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

# Clean build artifacts (keep build directory structure)
clean:
	@if [ -d "$(BUILD_DIR)" ] && [ -f "$(BUILD_DIR)/CMakeCache.txt" ]; then \
		cmake --build $(BUILD_DIR) --target clean 2>/dev/null || true; \
	fi
	rm -rf $(BUILD_DIR)/CMakeCache.txt $(BUILD_DIR)/CMakeFiles
	rm -rf dist *.egg-info surogate/*.so

# Full clean - remove build directory entirely
clean-all:
	rm -rf $(BUILD_DIR)
	rm -rf dist *.egg-info

# Rebuild from scratch
rebuild: clean-all build

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
	@echo "Test Targets:"
	@echo "  build-tests      - Build test executables without running them"
	@echo "  test             - Build and run unit tests (default, fast feedback)"
	@echo "  test-unit        - Build and run unit tests (kernels, modules, components)"
	@echo "  test-integration - Build and run integration tests (training, distributed)"
	@echo "  test-all         - Build and run all tests (unit + integration)"
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
