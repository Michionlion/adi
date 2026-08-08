.DEFAULT_GOAL := build

BUILD_DIR ?= build
BUILD_TYPE ?= Release
MODEL ?= models/Mach-1-Additive-35B.gguf

CMAKE ?= cmake
CTEST ?= ctest
UV ?= uv

CMAKE_ARGS ?=
BUILD_JOBS ?= 8
TEST_ARGS ?=
RUN_ARGS ?=

.PHONY: build test run clean

build:
	$(CMAKE) -S . -B "$(BUILD_DIR)" -DCMAKE_BUILD_TYPE="$(BUILD_TYPE)" -DADI_TEST_MODEL="$(MODEL)" $(CMAKE_ARGS)
	$(CMAKE) --build "$(BUILD_DIR)" --config "$(BUILD_TYPE)" --parallel "$(BUILD_JOBS)"

test: build
	$(CTEST) --test-dir "$(BUILD_DIR)" -C "$(BUILD_TYPE)" --output-on-failure $(TEST_ARGS)

run: build
	$(UV) run tools/adi_chat.py --model "$(MODEL)" $(RUN_ARGS)

clean:
	$(CMAKE) -E remove_directory "$(BUILD_DIR)"
