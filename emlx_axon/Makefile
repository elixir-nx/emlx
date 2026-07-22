# Builds the standalone model compute plugins.
empty :=
space := $(empty) $(empty)
esc = $(subst $(space),\$(space),$(1))

PRIV_DIR = $(MIX_APP_PATH)/priv
QWEN3_SO = $(PRIV_DIR)/libemlx_qwen3.so
LLAMA_SO = $(PRIV_DIR)/libemlx_llama.so
OBJ_DIR = $(PRIV_DIR)/objs
QWEN3_OBJECT = $(OBJ_DIR)/qwen3_plugin.o
LLAMA_OBJECT = $(OBJ_DIR)/llama_plugin.o
QWEN3_SOURCE = c_src/qwen3_plugin.cpp
LLAMA_SOURCE = c_src/llama_plugin.cpp
PLUGIN_ABI = $(EMLX_PLUGIN_INCLUDE_DIR)/emlx/plugin/abi.hpp
# Rebuild plugin objects when MLX's ABI surface changes — not only when our
# sources change. A stale .o compiled against an older ops.h keeps undefined
# refs like the pre-0.32 3-arg `astype` and fails dlopen against newer libmlx.
MLX_OPS_H = $(MLX_INCLUDE_DIR)/mlx/ops.h

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    PLATFORM_CFLAGS = -Wglobal-constructors -Wexit-time-destructors -Werror=global-constructors -Werror=exit-time-destructors
    PLATFORM_LDFLAGS = -undefined dynamic_lookup -flat_namespace -rpath @loader_path/mlx/lib -framework Metal -framework Foundation -framework Accelerate
    MLX_SHARED_LIB = $(MLX_LIB_DIR)/libmlx.dylib
else
    PLATFORM_CFLAGS =
    PLATFORM_LDFLAGS = -Wl,-rpath,'$$ORIGIN/mlx/lib'
    MLX_SHARED_LIB = $(MLX_LIB_DIR)/libmlx.so
endif

CFLAGS = -fPIC -isystem $(call esc,$(MLX_INCLUDE_DIR)) -I$(call esc,$(EMLX_PLUGIN_INCLUDE_DIR)) -Wall -std=c++20 -O3 -fvisibility=hidden $(PLATFORM_CFLAGS)
LDFLAGS = -L$(call esc,$(MLX_LIB_DIR)) -lmlx -shared $(PLATFORM_LDFLAGS)

all: $(call esc,$(QWEN3_SO)) $(call esc,$(LLAMA_SO))
	@ echo > /dev/null

$(call esc,$(QWEN3_OBJECT)): $(QWEN3_SOURCE) $(call esc,$(PLUGIN_ABI)) $(call esc,$(MLX_OPS_H)) $(call esc,$(MLX_SHARED_LIB))
	@ mkdir -p "$(OBJ_DIR)"
	$(CXX) $(CFLAGS) -c "$<" -o "$@"

$(call esc,$(LLAMA_OBJECT)): $(LLAMA_SOURCE) $(call esc,$(PLUGIN_ABI)) $(call esc,$(MLX_OPS_H)) $(call esc,$(MLX_SHARED_LIB))
	@ mkdir -p "$(OBJ_DIR)"
	$(CXX) $(CFLAGS) -c "$<" -o "$@"

$(call esc,$(QWEN3_SO)): $(call esc,$(QWEN3_OBJECT))
	@ mkdir -p "$(PRIV_DIR)/mlx/lib"
	@ cp -a "$(MLX_LIB_DIR)"/* "$(PRIV_DIR)/mlx/lib"
	$(CXX) "$<" -o "$@" $(LDFLAGS)

$(call esc,$(LLAMA_SO)): $(call esc,$(LLAMA_OBJECT))
	@ mkdir -p "$(PRIV_DIR)/mlx/lib"
	@ cp -a "$(MLX_LIB_DIR)"/* "$(PRIV_DIR)/mlx/lib"
	$(CXX) "$<" -o "$@" $(LDFLAGS)

clean:
	rm -rf "$(PRIV_DIR)"

.PHONY: all clean
