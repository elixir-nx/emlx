# Builds the standalone Qwen3 compute plugin. Preliminary and final compiler
# invocations use separate generated identity headers and depfile trees so the
# published build ID is derived from the complete compiler-resolved closure.
empty :=
space := $(empty) $(empty)
esc = $(subst $(space),\$(space),$(1))

PRIV_DIR = $(MIX_APP_PATH)/priv
QWEN3_SO = $(PRIV_DIR)/libemlx_qwen3.so
GENERATED_INCLUDE_DIR = $(PRIV_DIR)/include
QWEN3_MLX_ID_HEADER = $(GENERATED_INCLUDE_DIR)/emlx_qwen3_mlx_headers_build_id.hpp
QWEN3_BUILD_ID_HEADER = $(GENERATED_INCLUDE_DIR)/emlx_qwen3_plugin_build_id.hpp
QWEN3_BUILD_ID_FILE = $(PRIV_DIR)/emlx_qwen3.build_id
SCAN_DIR = $(MIX_APP_PATH)/qwen3-plugin-scan
SCAN_INCLUDE_DIR = $(SCAN_DIR)/include
SCAN_BUILD_ID_HEADER = $(SCAN_INCLUDE_DIR)/emlx_qwen3_plugin_build_id.hpp
SCAN_DEP_DIR = $(SCAN_DIR)/deps
SCAN_OBJ_DIR = $(SCAN_DIR)/objs
FINAL_DEP_DIR = $(PRIV_DIR)/deps
FINAL_OBJ_DIR = $(PRIV_DIR)/objs
VERIFY_STAMP = $(PRIV_DIR)/.qwen3-plugin-deps-verified
PLUGIN_INSPECTOR = c_src/inspect_plugin.sh
PLUGIN_DEPENDENCY_POLICY = c_src/plugin_dependency_policy_v1.txt

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Darwin)
    TARGET_TRIPLE = $(UNAME_M)-apple-darwin
    PLATFORM_CFLAGS = -Wglobal-constructors -Wexit-time-destructors -Werror=global-constructors -Werror=exit-time-destructors
    PLATFORM_LDFLAGS = -undefined dynamic_lookup -flat_namespace -rpath @loader_path/mlx/lib -framework Metal -framework Foundation -framework Accelerate
else
    TARGET_TRIPLE = $(UNAME_M)-unknown-linux-gnu
    PLATFORM_CFLAGS =
    PLATFORM_LDFLAGS = -Wl,-rpath,'$$ORIGIN/mlx/lib'
endif

BASE_CFLAGS = -fPIC -isystem $(call esc,$(MLX_INCLUDE_DIR)) -I$(call esc,$(EMLX_PLUGIN_INCLUDE_DIR)) -Wall -std=c++20 -O3 -fvisibility=hidden $(PLATFORM_CFLAGS)
LDFLAGS = -L$(call esc,$(MLX_LIB_DIR)) -lmlx -shared $(PLATFORM_LDFLAGS)

SOURCES = c_src/qwen3_plugin.cpp
HEADERS = $(call esc,$(EMLX_PLUGIN_INCLUDE_DIR))/emlx/plugin/abi.hpp $(call esc,$(EMLX_PLUGIN_INCLUDE_DIR))/emlx/plugin/toolchain.hpp $(call esc,$(EMLX_PLUGIN_INCLUDE_DIR))/emlx/plugin/build_compat.hpp
SCAN_DEPFILES = $(patsubst c_src/%.cpp,$(call esc,$(SCAN_DEP_DIR))/%.d,$(SOURCES))
FINAL_DEPFILES = $(patsubst c_src/%.cpp,$(call esc,$(FINAL_DEP_DIR))/%.d,$(SOURCES))
OBJECTS = $(patsubst c_src/%.cpp,$(call esc,$(FINAL_OBJ_DIR))/%.o,$(SOURCES))
PLUGIN_SCAN_ARGS = $(foreach source,$(SOURCES),--dep "$(source)" "$(SCAN_DEP_DIR)/$(patsubst c_src/%.cpp,%.d,$(source))")
PLUGIN_VERIFY_ARGS = $(foreach source,$(SOURCES),"$(source)" "$(SCAN_DEP_DIR)/$(patsubst c_src/%.cpp,%.d,$(source))" "$(FINAL_DEP_DIR)/$(patsubst c_src/%.cpp,%.d,$(source))")

all: $(call esc,$(QWEN3_SO))
	@ echo > /dev/null

$(call esc,$(PRIV_DIR)):
	@ mkdir -p "$@"

$(call esc,$(QWEN3_MLX_ID_HEADER)): $(call esc,$(EMLX_PLUGIN_INCLUDE_DIR))/emlx/plugin/build_compat.hpp
	@ mkdir -p "$(GENERATED_INCLUDE_DIR)"
	"$(EMLX_BUILD_ID_TOOL)" write-mlx-header-id "$(MLX_INCLUDE_DIR)" "$@" "$<"

$(call esc,$(SCAN_BUILD_ID_HEADER)): $(call esc,$(QWEN3_MLX_ID_HEADER)) FORCE
	@ rm -rf "$(SCAN_DIR)"
	@ mkdir -p "$(SCAN_INCLUDE_DIR)"
	"$(EMLX_BUILD_ID_TOOL)" write-id-header "$@" EMLX_QWEN3_PLUGIN_BUILD_ID 0000000000000000000000000000000000000000000000000000000000000000

$(call esc,$(SCAN_DEP_DIR))/%.d: c_src/%.cpp $(HEADERS) $(PLUGIN_INSPECTOR) $(call esc,$(SCAN_BUILD_ID_HEADER)) $(call esc,$(QWEN3_MLX_ID_HEADER))
	@ mkdir -p "$(SCAN_OBJ_DIR)/$(dir $*)" "$(dir $@)"
	"$(EMLX_PLUGIN_TOOL_WRAPPER)" --role compiler --mode scan --real "$(EMLX_PLUGIN_REAL_CXX)" -- $(BASE_CFLAGS) -I"$(SCAN_INCLUDE_DIR)" -I"$(GENERATED_INCLUDE_DIR)" -MD -MF "$@" -MT "$<" -c "$<" -o "$(SCAN_OBJ_DIR)/$*.o"
	@ "$(PLUGIN_INSPECTOR)" object "$(SCAN_OBJ_DIR)/$*.o" || { rm -f "$@" "$(SCAN_OBJ_DIR)/$*.o"; exit 1; }

$(call esc,$(QWEN3_BUILD_ID_HEADER)): $(SCAN_DEPFILES) $(call esc,$(QWEN3_MLX_ID_HEADER)) $(PLUGIN_DEPENDENCY_POLICY) Makefile
	"$(EMLX_BUILD_ID_TOOL)" plugin-build-id "$(CURDIR)" "$(EMLX_SOURCE_DIR)" "$(MLX_INCLUDE_DIR)" "$(SCAN_BUILD_ID_HEADER)" "$(QWEN3_MLX_ID_HEADER)" "$(EMLX_PLUGIN_INCLUDE_DIR)/emlx/plugin/build_compat.hpp" "$@" "$(QWEN3_BUILD_ID_FILE)" --policy "$(EMLX_SOURCE_DIR)/c_src/emlx/plugin/tool_wrapper.cpp" --policy "$(EMLX_SOURCE_DIR)/c_src/emlx/plugin/toolchain.hpp" --policy "$(CURDIR)/$(PLUGIN_DEPENDENCY_POLICY)" --policy "$(CURDIR)/$(PLUGIN_INSPECTOR)" --policy "$(CURDIR)/Makefile" --flag "compiler-family=$(EMLX_PLUGIN_COMPILER_FAMILY)" --flag "target=$(TARGET_TRIPLE)" --flag "language=c++20" --flag "optimization=O3" --flag "visibility=hidden" --flag "exceptions=enabled" --flag "rtti=enabled" $(PLUGIN_SCAN_ARGS)

$(call esc,$(QWEN3_BUILD_ID_FILE)): $(call esc,$(QWEN3_BUILD_ID_HEADER))
	@ test -f "$@"

$(call esc,$(FINAL_DEP_DIR))/%.d: c_src/%.cpp $(HEADERS) $(call esc,$(QWEN3_BUILD_ID_HEADER)) $(call esc,$(QWEN3_MLX_ID_HEADER))
	@ mkdir -p "$(FINAL_OBJ_DIR)/$(dir $*)" "$(dir $@)"
	"$(EMLX_PLUGIN_TOOL_WRAPPER)" --role compiler --mode compile --real "$(EMLX_PLUGIN_REAL_CXX)" -- $(BASE_CFLAGS) -I"$(GENERATED_INCLUDE_DIR)" -MD -MF "$@" -MT "$<" -c "$<" -o "$(FINAL_OBJ_DIR)/$*.o"

$(call esc,$(FINAL_OBJ_DIR))/%.o: $(call esc,$(FINAL_DEP_DIR))/%.d $(PLUGIN_INSPECTOR)
	@ test -f "$@"
	@ "$(PLUGIN_INSPECTOR)" object "$@"

$(call esc,$(VERIFY_STAMP)): $(SCAN_DEPFILES) $(FINAL_DEPFILES) $(call esc,$(QWEN3_BUILD_ID_HEADER)) $(call esc,$(QWEN3_BUILD_ID_FILE))
	"$(EMLX_BUILD_ID_TOOL)" verify-plugin-deps "$(MLX_INCLUDE_DIR)" "$(SCAN_BUILD_ID_HEADER)" "$(QWEN3_BUILD_ID_HEADER)" $(PLUGIN_VERIFY_ARGS)
	@ build_id=$$(tr -d '\n' < "$(QWEN3_BUILD_ID_FILE)"); \
	  "$(EMLX_BUILD_ID_TOOL)" write-id-header "$@" EMLX_QWEN3_VERIFIED_BUILD_ID "$$build_id"

$(call esc,$(QWEN3_SO)): $(call esc,$(PRIV_DIR)) $(OBJECTS) $(call esc,$(VERIFY_STAMP)) $(call esc,$(QWEN3_BUILD_ID_FILE))
	@ mkdir -p "$(PRIV_DIR)/mlx/lib"
	@ cp -a "$(MLX_LIB_DIR)"/* "$(PRIV_DIR)/mlx/lib"
	"$(EMLX_PLUGIN_TOOL_WRAPPER)" --role linker --mode link --real "$(EMLX_PLUGIN_REAL_CXX)" -- $(OBJECTS) -o "$@" $(LDFLAGS)
	@ "$(PLUGIN_INSPECTOR)" image "$@" "$(PLUGIN_DEPENDENCY_POLICY)" || { rm -f "$@"; exit 1; }

clean:
	rm -rf "$(PRIV_DIR)" "$(SCAN_DIR)"

FORCE:

.PHONY: all clean FORCE
