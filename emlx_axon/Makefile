# Builds the standalone Qwen3 compute plugin.
empty :=
space := $(empty) $(empty)
esc = $(subst $(space),\$(space),$(1))

PRIV_DIR = $(MIX_APP_PATH)/priv
QWEN3_SO = $(PRIV_DIR)/libemlx_qwen3.so
OBJ_DIR = $(PRIV_DIR)/objs
QWEN3_OBJECT = $(OBJ_DIR)/qwen3_plugin.o
QWEN3_SOURCE = c_src/qwen3_plugin.cpp
PLUGIN_ABI = $(EMLX_PLUGIN_INCLUDE_DIR)/emlx/plugin/abi.hpp

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    PLATFORM_CFLAGS = -Wglobal-constructors -Wexit-time-destructors -Werror=global-constructors -Werror=exit-time-destructors
    PLATFORM_LDFLAGS = -undefined dynamic_lookup -flat_namespace -rpath @loader_path/mlx/lib -framework Metal -framework Foundation -framework Accelerate
else
    PLATFORM_CFLAGS =
    PLATFORM_LDFLAGS = -Wl,-rpath,'$$ORIGIN/mlx/lib'
endif

CFLAGS = -fPIC -isystem $(call esc,$(MLX_INCLUDE_DIR)) -I$(call esc,$(EMLX_PLUGIN_INCLUDE_DIR)) -Wall -std=c++20 -O3 -fvisibility=hidden $(PLATFORM_CFLAGS)
LDFLAGS = -L$(call esc,$(MLX_LIB_DIR)) -lmlx -shared $(PLATFORM_LDFLAGS)

all: $(call esc,$(QWEN3_SO))
	@ echo > /dev/null

$(call esc,$(QWEN3_OBJECT)): $(QWEN3_SOURCE) $(call esc,$(PLUGIN_ABI))
	@ mkdir -p "$(OBJ_DIR)"
	$(CXX) $(CFLAGS) -c "$<" -o "$@"

$(call esc,$(QWEN3_SO)): $(call esc,$(QWEN3_OBJECT))
	@ mkdir -p "$(PRIV_DIR)/mlx/lib"
	@ cp -a "$(MLX_LIB_DIR)"/* "$(PRIV_DIR)/mlx/lib"
	$(CXX) "$<" -o "$@" $(LDFLAGS)

clean:
	rm -rf "$(PRIV_DIR)"

.PHONY: all clean
