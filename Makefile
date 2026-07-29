PROJECT_DIR := $(CURDIR)
MEDIAPIPE_DIR := $(PROJECT_DIR)/..

APP_TARGET := //ddd_cpp:ddd_app

BAZEL_CACHE_DIR := $(PROJECT_DIR)/.bazel_cache

BAZEL_FLAGS := \
	--repo_env=CC=/usr/bin/gcc-15 \
	--repo_env=HERMETIC_PYTHON_VERSION=3.12 \
	--action_env=CCACHE_DISABLE=1 \
	--host_action_env=CCACHE_DISABLE=1 \
	--define=MEDIAPIPE_DISABLE_GPU=1 \
	--compilation_mode=fastbuild \
	--disk_cache=$(BAZEL_CACHE_DIR) \
	--experimental_convenience_symlinks=ignore

.PHONY: build run

build:
	cd "$(MEDIAPIPE_DIR)" && \
	bazel build $(BAZEL_FLAGS) $(APP_TARGET)

run: build
	@BAZEL_BIN="$$(cd "$(MEDIAPIPE_DIR)" && \
		bazel info $(BAZEL_FLAGS) bazel-bin)"; \
	cd "$(PROJECT_DIR)" && \
	LD_LIBRARY_PATH="$(PROJECT_DIR)/third_party/onnxruntime/lib:$${LD_LIBRARY_PATH:-}" \
	"$$BAZEL_BIN/ddd_cpp/ddd_app"
