# Helper to escape spaces so that paths work as Make targets/prerequisites
# and survive expansion into shell recipes (backslash-escaped spaces are
# treated as literal spaces by both Make and the shell).
empty :=
space := $(empty) $(empty)
esc = $(subst $(space),\$(space),$(1))

# Private configuration
PRIV_DIR = $(MIX_APP_PATH)/priv
ifeq ($(strip $(EMLX_SOURCE_CACHE_ID)),)
$(error EMLX_SOURCE_CACHE_ID is required)
endif
BUILD_DIR = $(EMLX_CACHE_DIR)/emlx-$(EMLX_VERSION)-mlx-$(MLX_VERSION)$(MLX_VARIANT)-source-$(EMLX_SOURCE_CACHE_ID)/objs
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
EMLX_GENERATED_INCLUDE_DIR = $(PRIV_DIR)/include
EMLX_COMPAT_HEADER = $(EMLX_GENERATED_INCLUDE_DIR)/emlx/plugin/build_compat.hpp
EMLX_PLUGIN_ABI_HEADER = $(EMLX_GENERATED_INCLUDE_DIR)/emlx/plugin/abi.hpp
EMLX_PLUGIN_TOOLCHAIN_HEADER = $(EMLX_GENERATED_INCLUDE_DIR)/emlx/plugin/toolchain.hpp
EMLX_STAGING_DIR = $(BUILD_DIR)/compat-staging
EMLX_STAGED_COMPAT_HEADER = $(EMLX_STAGING_DIR)/emlx/plugin/build_compat.hpp
EMLX_DEPFILE_DIR = $(BUILD_DIR)/deps
EMLX_BUILD_ID_TOOL = $(BUILD_DIR)/tools/emlx_build_identity
EMLX_PACKAGED_BUILD_ID_TOOL = $(PRIV_DIR)/build_support/emlx_build_identity
EMLX_PLUGIN_TOOL_WRAPPER = $(PRIV_DIR)/build_support/emlx_plugin_tool_wrapper

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
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
    MLX_SO ?= $(MLX_LIB_DIR)/libmlx.dylib
    EMLX_TARGET_TRIPLE = $(UNAME_M)-apple-darwin
    LDFLAGS += -undefined dynamic_lookup -flat_namespace -rpath @loader_path/mlx/lib
    LDFLAGS += -framework Metal -framework Foundation -framework Accelerate
		MAKE_DEFAULT_JOBS = $(shell sysctl -n hw.ncpu)
else
    MLX_SO ?= $(MLX_LIB_DIR)/libmlx.so
    EMLX_TARGET_TRIPLE = $(UNAME_M)-unknown-linux-gnu
    LDFLAGS += -Wl,-rpath,'$$ORIGIN/mlx/lib'
		MAKE_DEFAULT_JOBS = $(shell nproc)
endif

MAKE_JOBS ?= $(MAKE_DEFAULT_JOBS)

# Source files
SOURCES = c_src/emlx_nif.cpp c_src/emlx_fast.cpp c_src/emlx/compiler.cpp c_src/emlx/plugin/registry.cpp c_src/emlx_native_image.cpp c_src/emlx_nif_lifecycle.cpp c_src/emlx_sha256.cpp
HEADERS = c_src/nx_nif_utils.hpp c_src/emlx_worker.hpp c_src/emlx_async.hpp c_src/emlx_nif_shared.hpp c_src/emlx/plugin/abi.hpp c_src/emlx/plugin/toolchain.hpp c_src/emlx/plugin/registry.hpp c_src/emlx_native_image.hpp c_src/emlx_nif_lifecycle.hpp c_src/emlx_sha256.hpp c_src/emlx_depfile.hpp c_src/emlx/compiler.hpp c_src/emlx_runtime_call_bridge.hpp
OBJECTS = $(patsubst c_src/%.cpp,$(call esc,$(BUILD_DIR))/%.o,$(SOURCES))
DEPFILES = $(patsubst c_src/%.cpp,$(call esc,$(EMLX_DEPFILE_DIR))/%.d,$(SOURCES))
HOST_DEPFILE_ARGS = $(foreach source,$(SOURCES),"$(source)" "$(EMLX_DEPFILE_DIR)/$(patsubst c_src/%.cpp,%.d,$(source))")

# Main targets
all: $(call esc,$(MLX_SO)) $(call esc,$(EMLX_COMPAT_HEADER)) $(call esc,$(EMLX_PLUGIN_ABI_HEADER)) $(call esc,$(EMLX_PLUGIN_TOOLCHAIN_HEADER)) $(call esc,$(EMLX_PACKAGED_BUILD_ID_TOOL)) $(call esc,$(EMLX_PLUGIN_TOOL_WRAPPER)) $(call esc,$(EMLX_SO))
	@ echo > /dev/null

$(call esc,$(PRIV_DIR)):
	@ mkdir -p "$(PRIV_DIR)"

$(call esc,$(EMLX_BUILD_ID_TOOL)): c_src/tools/emlx_build_identity.cpp c_src/emlx_sha256.cpp c_src/emlx_sha256.hpp c_src/emlx_depfile.cpp c_src/emlx_depfile.hpp c_src/emlx/plugin/toolchain.hpp
	@ mkdir -p "$(BUILD_DIR)/tools"
	$(CXX) -std=c++20 -O2 -I c_src c_src/tools/emlx_build_identity.cpp c_src/emlx_sha256.cpp c_src/emlx_depfile.cpp -o "$@"

$(call esc,$(EMLX_PACKAGED_BUILD_ID_TOOL)): $(call esc,$(EMLX_BUILD_ID_TOOL))
	@ mkdir -p "$(PRIV_DIR)/build_support"
	@ cp "$<" "$@"

$(call esc,$(EMLX_PLUGIN_TOOL_WRAPPER)): c_src/tools/emlx_plugin_tool_wrapper.cpp
	@ mkdir -p "$(PRIV_DIR)/build_support"
	$(CXX) -std=c++20 -O2 c_src/tools/emlx_plugin_tool_wrapper.cpp -o "$@"

$(call esc,$(EMLX_PLUGIN_ABI_HEADER)): c_src/emlx/plugin/abi.hpp
	@ mkdir -p "$(dir $(EMLX_PLUGIN_ABI_HEADER))"
	@ cp "$<" "$@"

$(call esc,$(EMLX_PLUGIN_TOOLCHAIN_HEADER)): c_src/emlx/plugin/toolchain.hpp
	@ mkdir -p "$(dir $(EMLX_PLUGIN_TOOLCHAIN_HEADER))"
	@ cp "$<" "$@"

$(call esc,$(EMLX_STAGED_COMPAT_HEADER)): $(call esc,$(MLX_SO)) $(call esc,$(EMLX_BUILD_ID_TOOL)) FORCE
	@ mkdir -p "$(dir $(EMLX_STAGED_COMPAT_HEADER))"
	"$(EMLX_BUILD_ID_TOOL)" "$(MLX_SO)" "$(MLX_INCLUDE_DIR)" "$(MLX_VERSION)" "$(MLX_VARIANT)" "$(EMLX_TARGET_TRIPLE)" "$(EMLX_STAGED_COMPAT_HEADER)"

$(call esc,$(EMLX_DEPFILE_DIR))/%.d: c_src/%.cpp $(HEADERS) $(call esc,$(EMLX_STAGED_COMPAT_HEADER))
	@ mkdir -p "$(BUILD_DIR)/$(dir $*)"
	@ mkdir -p "$(EMLX_DEPFILE_DIR)/$(dir $*)"
	$(CXX) -I"$(EMLX_STAGING_DIR)" $(CFLAGS) -include "$(EMLX_STAGED_COMPAT_HEADER)" -MD -MF "$@" -MT "$<" -c "$<" -o "$(BUILD_DIR)/$*.o"

$(call esc,$(BUILD_DIR))/%.o: $(call esc,$(EMLX_DEPFILE_DIR))/%.d
	@ test -f "$@"

$(call esc,$(EMLX_COMPAT_HEADER)): $(OBJECTS) $(DEPFILES) $(call esc,$(EMLX_BUILD_ID_TOOL))
	@ mkdir -p "$(dir $(EMLX_COMPAT_HEADER))"
	@ rm -f "$(EMLX_COMPAT_HEADER)"
	"$(EMLX_BUILD_ID_TOOL)" verify-host-deps "$(MLX_INCLUDE_DIR)" "$(EMLX_STAGED_COMPAT_HEADER)" "$(EMLX_COMPAT_HEADER)" $(HOST_DEPFILE_ARGS)

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

$(call esc,$(EMLX_SO)): $(call esc,$(PRIV_DIR)) $(call esc,$(MLX_SO)) $(OBJECTS)
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

print-build-dir:
	@ printf '%s\n' "$(BUILD_DIR)"

FORCE:

.PHONY: all clean print-build-dir FORCE
