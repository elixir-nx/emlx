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

# Only used if MLX_BUILD is true
MLX_SRC_ARCHIVE = $(EMLX_CACHE_DIR)/mlx/$(MLX_VERSION)$(MLX_VARIANT).tar.gz
MLX_SRC_DIR = $(EMLX_CACHE_DIR)/mlx/src-$(MLX_VERSION)$(MLX_VARIANT)
MLX_BUILD_DIR = $(EMLX_CACHE_DIR)/mlx/build-$(MLX_VERSION)$(MLX_VARIANT)
MLX_INSTALL_DIR = $(MLX_DIR)
MLX_SO ?= $(MLX_LIB_DIR)/libmlx.dylib

$(info LIBMLX_ENABLE_DEBUG=$(LIBMLX_ENABLE_DEBUG))
# Build flags
CFLAGS = -fPIC -I$(call esc,$(ERTS_INCLUDE_DIR)) -I$(call esc,$(MLX_INCLUDE_DIR)) -Wall \
         -std=c++20
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
    LDFLAGS += -undefined dynamic_lookup -flat_namespace -rpath @loader_path/mlx/lib
    LDFLAGS += -framework Metal -framework Foundation -framework Accelerate
		MAKE_DEFAULT_JOBS = $(shell sysctl -n hw.ncpu)
else
    LDFLAGS += -Wl,-rpath,'$$ORIGIN/mlx/lib'
		MAKE_DEFAULT_JOBS = $(shell nproc)
endif

MAKE_JOBS ?= $(MAKE_DEFAULT_JOBS)

# Source files
SOURCES = c_src/emlx_nif.cpp c_src/emlx_fast.cpp c_src/emlx_compiler.cpp
HEADERS = c_src/nx_nif_utils.hpp c_src/emlx_worker.hpp c_src/emlx_async.hpp c_src/emlx_nif_shared.hpp c_src/emlx_compiler.hpp
OBJECTS = $(patsubst c_src/%.cpp,$(call esc,$(BUILD_DIR))/%.o,$(SOURCES))

# Main targets
all: $(call esc,$(MLX_SO)) $(call esc,$(EMLX_SO))
	@ echo > /dev/null

$(call esc,$(PRIV_DIR)):
	@ mkdir -p "$(PRIV_DIR)"

$(call esc,$(BUILD_DIR))/%.o: c_src/%.cpp $(HEADERS)
	@ mkdir -p "$(BUILD_DIR)"
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

$(call esc,$(EMLX_SO)): $(call esc,$(PRIV_DIR)) $(call esc,$(MLX_SO)) $(OBJECTS)
	@ echo "Copying MLX library to $(EMLX_LIB_DIR)"
	@ mkdir -p "$(EMLX_LIB_DIR)"
	@ cp -a "$(MLX_LIB_DIR)"/* "$(EMLX_LIB_DIR)"
	$(CXX) $(OBJECTS) -o "$(EMLX_SO)" $(LDFLAGS)

clean:
	rm -rf "$(PRIV_DIR)"
	rm -rf "$(BUILD_DIR)"

.PHONY: all clean
