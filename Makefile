# Private configuration
PRIV_DIR = $(MIX_APP_PATH)/priv
BUILD_DIR = $(EMLX_CACHE_DIR)/emlx-$(EMLX_VERSION)$(MLX_VARIANT)/objs
EMLX_SO = $(PRIV_DIR)/libemlx.so
EMLX_LIB_DIR = $(PRIV_DIR)/mlx/lib

# Only used if MLX_BUILD is true
MLX_SRC_ARCHIVE = $(EMLX_CACHE_DIR)/mlx/$(MLX_VERSION)$(MLX_VARIANT).tar.gz
MLX_SRC_DIR = $(EMLX_CACHE_DIR)/mlx/src-$(MLX_VERSION)$(MLX_VARIANT)
MLX_BUILD_DIR = $(EMLX_CACHE_DIR)/mlx/build-$(MLX_VERSION)$(MLX_VARIANT)
MLX_INSTALL_DIR = $(MLX_DIR)
MLX_SO = $(MLX_LIB_DIR)/libmlx.dylib

# Build flags
CFLAGS = -fPIC -I$(ERTS_INCLUDE_DIR) -I$(MLX_INCLUDE_DIR) -Wall \
         -std=c++17
ifeq ($(LIBMLX_ENABLE_DEBUG),true)
		CFLAGS += -g
		CMAKE_BUILD_TYPE = Debug
else
		CFLAGS += -O3
		CMAKE_BUILD_TYPE = Release
endif

LDFLAGS = -L$(MLX_LIB_DIR) -lmlx -shared

# Platform-specific settings
UNAME_S = $(shell uname -s)

ifeq ($(UNAME_S), Darwin)
    LDFLAGS += -flat_namespace -undefined dynamic_lookup -rpath @loader_path/mlx/lib
		MAKE_DEFAULT_JOBS = $(shell sysctl -n hw.ncpu)
else
    LDFLAGS += -Wl,-rpath,'$$ORIGIN/mlx/lib'
		MAKE_DEFAULT_JOBS = $(shell nproc)
endif

MAKE_JOBS ?= $(MAKE_DEFAULT_JOBS)

# Source files
SOURCES = c_src/emlx_nif.cpp
OBJECTS = $(patsubst c_src/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))

# Main targets
all: $(MLX_SO) $(EMLX_SO)
	@ echo > /dev/null

$(PRIV_DIR):
	@ mkdir -p $(PRIV_DIR)

$(BUILD_DIR)/%.o: c_src/%.cpp
	@ mkdir -p $(BUILD_DIR)
	$(CXX) $(CFLAGS) -c $< -o $@

$(MLX_SRC_ARCHIVE):
	@ if [ "$(MLX_BUILD)" = "true" ]; then \
		mkdir -p "$(EMLX_CACHE_DIR)/mlx" && \
		curl -fSL "https://github.com/ml-explore/mlx/archive/refs/tags/v$(MLX_VERSION).tar.gz" -o "$(MLX_SRC_ARCHIVE)" ; \
	fi

$(MLX_SRC_DIR): $(MLX_SRC_ARCHIVE)
	@ if [ "$(MLX_BUILD)" = "true" ]; then \
		mkdir -p "$(MLX_SRC_DIR)" && \
		tar -xzf "$(MLX_SRC_ARCHIVE)" -C "$(MLX_SRC_DIR)" --strip-components=1 ; \
	fi

$(MLX_SO): $(MLX_SRC_DIR)
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
		cmake --install "$(MLX_BUILD_DIR)" --config "$(CMAKE_BUILD_TYPE)" ; \
	fi

$(EMLX_SO): $(PRIV_DIR) $(MLX_SO) $(OBJECTS)
	@ echo "Copying MLX library to $(EMLX_LIB_DIR)"
	@ mkdir -p $(EMLX_LIB_DIR)
	@ cp -a $(MLX_LIB_DIR)/* $(EMLX_LIB_DIR)
	$(CXX) $(OBJECTS) -o $(EMLX_SO) $(LDFLAGS)

clean:
	rm -rf $(PRIV_DIR)
	rm -rf $(BUILD_DIR)

.PHONY: all clean
