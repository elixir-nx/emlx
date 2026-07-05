# Helper to escape spaces so that paths work as Make targets/prerequisites
# and survive expansion into shell recipes (backslash-escaped spaces are
# treated as literal spaces by both Make and the shell).
empty :=
space := $(empty) $(empty)
esc = $(subst $(space),\$(space),$(1))

# Builds the standalone qwen3 compute plugin — pure MLX graph-building
# code, no erl_nif dependency whatsoever (see c_src/qwen3_plugin.cpp and
# emlx's c_src/emlx_fast/qwen3_plugin_abi.hpp). `dlopen`'d at runtime by
# emlx's host NIF via `EMLX.NIF.load_plugin("qwen3", path)` — see
# lib/emlx_axon/application.ex.
#
# MLX_INCLUDE_DIR/MLX_LIB_DIR point at emlx's own already-resolved libmlx
# copy (priv/mlx/{include,lib}) so this plugin links against the exact
# same libmlx, without duplicating emlx's version/cache-dir resolution
# logic — see mix.exs's `make_env`.
PRIV_DIR = $(MIX_APP_PATH)/priv
QWEN3_SO = $(PRIV_DIR)/libemlx_qwen3.so

CFLAGS = -fPIC -I$(call esc,$(MLX_INCLUDE_DIR)) -I$(call esc,$(QWEN3_ABI_INCLUDE_DIR)) -Wall -std=c++20 -O3
LDFLAGS = -L$(call esc,$(MLX_LIB_DIR)) -lmlx -shared

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS += -undefined dynamic_lookup -flat_namespace -rpath @loader_path/mlx/lib
    LDFLAGS += -framework Metal -framework Foundation -framework Accelerate
else
    LDFLAGS += -Wl,-rpath,'$$ORIGIN/mlx/lib'
endif

SOURCES = c_src/qwen3_plugin.cpp
HEADERS = $(call esc,$(QWEN3_ABI_INCLUDE_DIR))/qwen3_plugin_abi.hpp
OBJECTS = $(patsubst c_src/%.cpp,$(call esc,$(PRIV_DIR))/objs/%.o,$(SOURCES))

all: $(call esc,$(QWEN3_SO))
	@ echo > /dev/null

$(call esc,$(PRIV_DIR)):
	@ mkdir -p "$(PRIV_DIR)"

$(call esc,$(PRIV_DIR))/objs/%.o: c_src/%.cpp $(HEADERS)
	@ mkdir -p "$(dir $@)"
	$(CXX) $(CFLAGS) -c "$<" -o "$@"

# Emlx's own mlx lib copy (priv/mlx/lib, see MLX_LIB_DIR in mix.exs) must
# already exist — this plugin's rpath (`@loader_path/mlx/lib`) resolves
# relative to *its own* location, so we also copy the lib alongside it
# here rather than relying on emlx's priv dir at runtime.
$(call esc,$(QWEN3_SO)): $(call esc,$(PRIV_DIR)) $(OBJECTS)
	@ mkdir -p "$(PRIV_DIR)/mlx/lib"
	@ cp -a "$(MLX_LIB_DIR)"/* "$(PRIV_DIR)/mlx/lib"
	$(CXX) $(OBJECTS) -o "$(QWEN3_SO)" $(LDFLAGS)

clean:
	rm -rf "$(PRIV_DIR)"

.PHONY: all clean
