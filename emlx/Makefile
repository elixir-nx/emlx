# Helper to escape spaces so that paths work as Make targets/prerequisites
# and survive expansion into shell recipes (backslash-escaped spaces are
# treated as literal spaces by both Make and the shell).
empty :=
space := $(empty) $(empty)
esc = $(subst $(space),\$(space),$(1))

# Private configuration
PRIV_DIR = $(MIX_APP_PATH)/priv
BUILD_DIR = $(EMLX_CACHE_DIR)/emlx-$(EMLX_VERSION)-mlx-$(MLX_VERSION)$(MLX_VARIANT)/objs
EMLX_SO = $(PRIV_DIR)/libemlx.so
EMLX_LIB_DIR = $(PRIV_DIR)/mlx/lib
# Headers are copied alongside the lib (unlike the lib copy, these are not
# needed by EMLX itself at runtime) so out-of-tree native plugins loaded via
# `EMLX.NIF.load_plugin/2` (e.g. emlx_axon's qwen3 compute plugin) can build
# against the exact same libmlx this EMLX was compiled against, without
# duplicating EMLX's libmlx version/cache-dir resolution logic — see
# emlx_axon's Makefile/mix.exs, which locate these via
# `Application.app_dir(:emlx, "priv/mlx/...")`.
EMLX_INCLUDE_DIR = $(PRIV_DIR)/mlx/include
EMLX_PLUGIN_ABI_HEADER = $(PRIV_DIR)/include/emlx/plugin/abi.hpp

# Only used if MLX_BUILD is true
MLX_SRC_ARCHIVE = $(EMLX_CACHE_DIR)/mlx/$(MLX_VERSION)$(MLX_VARIANT).tar.gz
MLX_SRC_DIR = $(EMLX_CACHE_DIR)/mlx/src-$(MLX_VERSION)$(MLX_VARIANT)
MLX_BUILD_DIR = $(EMLX_CACHE_DIR)/mlx/build-$(MLX_VERSION)$(MLX_VARIANT)
MLX_INSTALL_DIR = $(MLX_DIR)

$(info LIBMLX_ENABLE_DEBUG=$(LIBMLX_ENABLE_DEBUG))
# Build flags
CFLAGS = -fPIC -I c_src -I$(call esc,$(ERTS_INCLUDE_DIR)) -I$(call esc,$(MLX_INCLUDE_DIR)) \
         -I$(call esc,$(FINE_INCLUDE_DIR)) -Wall -std=c++20
ifeq ($(LIBMLX_ENABLE_DEBUG),true)
		CFLAGS += -g
		CMAKE_BUILD_TYPE = Debug
else
		CFLAGS += -O3
		CMAKE_BUILD_TYPE = Release
endif

LDFLAGS = -L$(call esc,$(MLX_LIB_DIR)) -lmlx -shared

# Platform-specific settings
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    MLX_SO ?= $(MLX_LIB_DIR)/libmlx.dylib
    LDFLAGS += -undefined dynamic_lookup -flat_namespace -rpath @loader_path/mlx/lib
    LDFLAGS += -framework Metal -framework Foundation -framework Accelerate
    # Match the precompiled libmlx deployment target so ld does not warn about
    # linking a newer minos dylib (mlx-build Darwin variants).
    ifneq ($(MACOSX_DEPLOYMENT_TARGET),)
      CFLAGS += -mmacosx-version-min=$(MACOSX_DEPLOYMENT_TARGET)
      LDFLAGS += -mmacosx-version-min=$(MACOSX_DEPLOYMENT_TARGET)
    endif
		MAKE_DEFAULT_JOBS = $(shell sysctl -n hw.ncpu)
else
    MLX_SO ?= $(MLX_LIB_DIR)/libmlx.so
    LDFLAGS += -Wl,-rpath,'$$ORIGIN/mlx/lib'
		MAKE_DEFAULT_JOBS = $(shell nproc)
endif

MAKE_JOBS ?= $(MAKE_DEFAULT_JOBS)

# Source files
SOURCES = c_src/emlx_nif.cpp c_src/emlx_fast.cpp c_src/emlx/compiler.cpp c_src/emlx/plugin/registry.cpp
HEADERS = c_src/nx_nif_utils.hpp c_src/emlx_worker.hpp c_src/emlx_async.hpp c_src/emlx_nif_shared.hpp c_src/emlx/plugin/abi.hpp c_src/emlx/plugin/registry.hpp c_src/emlx/compiler.hpp c_src/emlx_runtime_call_bridge.hpp
OBJECTS = $(patsubst c_src/%.cpp,$(call esc,$(BUILD_DIR))/%.o,$(SOURCES))

# Main targets
all: $(call esc,$(MLX_SO)) $(call esc,$(EMLX_PLUGIN_ABI_HEADER)) $(call esc,$(EMLX_SO)) $(if $(filter true,$(EMLX_MLXC)),mlx_c,)
	@ echo > /dev/null

$(call esc,$(PRIV_DIR)):
	@ mkdir -p "$(PRIV_DIR)"

$(call esc,$(EMLX_PLUGIN_ABI_HEADER)): c_src/emlx/plugin/abi.hpp
	@ mkdir -p "$(dir $(EMLX_PLUGIN_ABI_HEADER))"
	@ cp "$<" "$@"

$(call esc,$(BUILD_DIR))/%.o: c_src/%.cpp $(HEADERS)
	@ mkdir -p "$(dir $@)"
	$(CXX) $(CFLAGS) -c "$<" -o "$@"

$(call esc,$(MLX_SRC_ARCHIVE)):
	@ if [ "$(MLX_BUILD)" = "true" ]; then \
		mkdir -p "$(EMLX_CACHE_DIR)/mlx" && \
		curl -fSL "https://github.com/ml-explore/mlx/archive/refs/tags/v$(MLX_VERSION).tar.gz" -o "$(MLX_SRC_ARCHIVE)" ; \
	fi

$(call esc,$(MLX_SRC_DIR)): $(call esc,$(MLX_SRC_ARCHIVE))
	@ if [ "$(MLX_BUILD)" = "true" ]; then \
		mkdir -p "$(MLX_SRC_DIR)" && \
		tar -xzf "$(MLX_SRC_ARCHIVE)" -C "$(MLX_SRC_DIR)" --strip-components=1 ; \
	fi

$(call esc,$(MLX_SO)): $(call esc,$(MLX_SRC_DIR))
	@ if [ "$(MLX_BUILD)" = "true" ]; then \
		cd "$(MLX_SRC_DIR)" && \
		export DESTDIR="$(MLX_INSTALL_DIR)" && \
		cmake -B "$(MLX_BUILD_DIR)" \
			-D CMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
			-D CMAKE_INSTALL_PREFIX=/ \
			-D MLX_BUILD_TESTS=OFF \
			-D MLX_BUILD_EXAMPLES=OFF \
			-D MLX_BUILD_BENCHMARKS=OFF \
			-D MLX_BUILD_PYTHON_BINDINGS=OFF \
			-D MLX_METAL_DEBUG=$(LIBMLX_ENABLE_DEBUG) \
			-D MLX_METAL_JIT=$(LIBMLX_ENABLE_JIT) \
			-D BUILD_SHARED_LIBS=ON \
			. && \
		cmake --build "$(MLX_BUILD_DIR)" --config "$(CMAKE_BUILD_TYPE)" -j$(MAKE_JOBS) && \
		cmake --install "$(MLX_BUILD_DIR)" --config "$(CMAKE_BUILD_TYPE)" && \
		if [ -f "$(MLX_BUILD_DIR)/jaccl/libjaccl.dylib" ]; then \
			cp "$(MLX_BUILD_DIR)/jaccl/libjaccl.dylib" "$(MLX_LIB_DIR)/" ; \
		fi ; \
	fi

$(call esc,$(EMLX_SO)): $(call esc,$(PRIV_DIR)) $(call esc,$(MLX_SO)) $(call esc,$(EMLX_PLUGIN_ABI_HEADER)) $(OBJECTS)
	@ echo "Copying MLX library to $(EMLX_LIB_DIR)"
	@ mkdir -p "$(EMLX_LIB_DIR)"
	@ cp -a "$(MLX_LIB_DIR)"/* "$(EMLX_LIB_DIR)"
	@ echo "Copying MLX headers to $(EMLX_INCLUDE_DIR)"
	@ mkdir -p "$(EMLX_INCLUDE_DIR)"
	@ cp -a "$(MLX_INCLUDE_DIR)"/* "$(EMLX_INCLUDE_DIR)"
	$(CXX) $(OBJECTS) -o "$(EMLX_SO)" $(LDFLAGS)

clean:
	rm -rf "$(PRIV_DIR)"
	rm -rf "$(BUILD_DIR)"
	rm -rf "$(BUILD_DIR)_c"

.PHONY: all clean mlx_c

# ─── mlx-c lane (opt-in: `make mlx_c`, or EMLX_MLXC=true mix compile) ────
#
# Fetches the official mlx-c bindings (pinned tag; its CMake FetchContent
# pins the matching libmlx minor version and builds everything itself) and
# links two lane libraries exposing the same binary-in/binary-out op surface:
#
#   priv/libemlx_c.so    - ops via the mlx-c API    (EMLX.C.NIF)
#   priv/libemlx_ref.so  - same ops via direct C++    (EMLX.C.RefNIF)
#
# Each .so statically links its own libmlx, so the lanes run side-by-side
# with the main libemlx.so for differential testing and benchmarking.
# Selected at runtime via `config :emlx, :mlx_api, :c | :cpp`.
#
# Currently verified on Apple Silicon; contributions for other platforms welcome.

MLXC_VERSION ?= v0.7.0
MLXC_VER = $(patsubst v%,%,$(MLXC_VERSION))
MLXC_CACHE = $(EMLX_CACHE_DIR)/mlx-c/$(MLXC_VERSION)
MLXC_SRC = $(MLXC_CACHE)/mlx-c-$(MLXC_VER)
MLXC_BUILD = $(MLXC_CACHE)/build
MLXC_ARCHIVE = $(MLXC_CACHE)/mlx-c-$(MLXC_VERSION).tar.gz
EMLX_C_SO = $(PRIV_DIR)/libemlx_c.so
EMLX_REF_SO = $(PRIV_DIR)/libemlx_ref.so

ifeq ($(UNAME_S),Darwin)
MLXC_LIBS_C = -Wl,-force_load,$(call esc,$(MLXC_BUILD)/libmlxc.a) \
              -Wl,-force_load,$(call esc,$(MLXC_BUILD)/_deps/mlx-build/libmlx.a) \
              -Wl,-force_load,$(call esc,$(MLXC_BUILD)/jaccl/libjaccl.a) \
              -Wl,-force_load,$(call esc,$(MLXC_BUILD)/_deps/mlx-build/mlx/io/libgguflib.a)
MLXC_LIBS_REF = -Wl,-force_load,$(call esc,$(MLXC_BUILD)/_deps/mlx-build/libmlx.a) \
                -Wl,-force_load,$(call esc,$(MLXC_BUILD)/jaccl/libjaccl.a) \
                -Wl,-force_load,$(call esc,$(MLXC_BUILD)/_deps/mlx-build/mlx/io/libgguflib.a)
MLXC_FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate
MLXC_METALLIB = $(firstwildcard $(MLXC_BUILD)/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib)
else
MLXC_LIBS_C = $(MLXC_BUILD)/libmlxc.a $(MLXC_BUILD)/_deps/mlx-build/libmlx.a
MLXC_LIBS_REF = $(MLXC_BUILD)/_deps/mlx-build/libmlx.a
MLXC_FRAMEWORKS =
MLXC_METALLIB =
endif

mlx_c: $(MLXC_BUILD)/libmlxc.a $(EMLX_C_SO) $(EMLX_REF_SO)
	@ if [ -f "$(MLXC_BUILD)/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib" ]; then \
		cp -f "$(MLXC_BUILD)/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib" "$(PRIV_DIR)/"; \
	fi
	@ echo "mlx-c lane built: $(EMLX_C_SO), $(EMLX_REF_SO)"

$(call esc,$(MLXC_ARCHIVE)):
	@ mkdir -p "$(MLXC_CACHE)"
	curl -fSL "https://github.com/ml-explore/mlx-c/archive/refs/tags/$(MLXC_VERSION).tar.gz" -o "$@"

$(call esc,$(MLXC_SRC)): $(call esc,$(MLXC_ARCHIVE))
	tar -xzf "$(MLXC_ARCHIVE)" -C "$(MLXC_CACHE)"

$(call esc,$(MLXC_BUILD))/libmlxc.a: $(call esc,$(MLXC_SRC))
	cmake -B "$(MLXC_BUILD)" -S "$(MLXC_SRC)" \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DMLX_C_BUILD_EXAMPLES=OFF
	cmake --build "$(MLXC_BUILD)" -j$(MAKE_JOBS)

$(call esc,$(BUILD_DIR))_c/emlx_c_nif.o: c_src/emlx_c/emlx_c_nif.cpp
	@ mkdir -p "$(BUILD_DIR)_c"
	$(CXX) $(CFLAGS) -I$(call esc,$(MLXC_SRC)) -c "$<" -o "$@"

$(call esc,$(BUILD_DIR))_c/emlx_ref_nif.o: c_src/emlx_c/emlx_ref_nif.cpp
	@ mkdir -p "$(BUILD_DIR)_c"
	$(CXX) $(CFLAGS) -c "$<" -o "$@"

$(call esc,$(EMLX_C_SO)): $(call esc,$(BUILD_DIR))_c/emlx_c_nif.o $(call esc,$(MLXC_BUILD))/libmlxc.a $(PRIV_DIR)
	$(CXX) $(CFLAGS) -shared "$<" -o "$@" $(MLXC_LIBS_C) $(MLXC_FRAMEWORKS) -undefined dynamic_lookup $(if $(MACOSX_DEPLOYMENT_TARGET),-mmacosx-version-min=$(MACOSX_DEPLOYMENT_TARGET),)

# The reference lane links the very same libmlx EMLX itself builds against.
$(call esc,$(EMLX_REF_SO)): $(call esc,$(BUILD_DIR))_c/emlx_ref_nif.o $(PRIV_DIR)
	$(CXX) $(CFLAGS) -shared "$<" -o "$@" $(LDFLAGS)
