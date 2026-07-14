#pragma once

// Generic, immutable name-keyed native plugin registry. It loads one
// versioned descriptor symbol, copies validated callback records into
// EMLX-owned storage, and keeps the shared object alive for the VM lifetime.

#include "emlx/plugin/abi.hpp"
#include "emlx_native_image.hpp"
#include "erl_nif.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

// The two-argument loader is the expert path and validates the generic ABI
// without requiring a caller-provided package build ID. The three-argument
// loader additionally verifies the packaged build identity and controlled
// loader environment.
ERL_NIF_TERM load_plugin(ErlNifEnv *, int, const ERL_NIF_TERM[]);
ERL_NIF_TERM load_plugin_with_build_id(ErlNifEnv *, int,
                                       const ERL_NIF_TERM[]);
ERL_NIF_TERM call_plugin(ErlNifEnv *, int, const ERL_NIF_TERM[]);
ERL_NIF_TERM call_plugin_async(ErlNifEnv *, int, const ERL_NIF_TERM[]);

struct EMLXLoadedPluginCallback {
  std::string name;
  std::string debug_name;
  uint32_t schema_version;
  uint32_t attr_schema_version;
  uint32_t operand_count;
  EMLXOperandCountFn operand_count_from_attrs;
  uint32_t output_count;
  EMLXOutputCountFn output_count_from_attrs;
  uint32_t device_capabilities;
  EMLXPluginCallback callback;
};

struct EMLXLoadedPlugin {
  std::string name;
  std::string canonical_path;
  std::string plugin_build_id;
  std::string mlx_build_id;
  std::string mlx_headers_build_id;
  EMLXNativeImageIdentity mlx_runtime_image;
  void *shared_object_handle;
  std::unordered_map<std::string,
                     std::shared_ptr<const EMLXLoadedPluginCallback>>
      callbacks;

  ~EMLXLoadedPlugin();
};

struct EMLXResolvedPluginCallback {
  std::shared_ptr<const EMLXLoadedPlugin> plugin;
  std::shared_ptr<const EMLXLoadedPluginCallback> callback;
};

EMLXResolvedPluginCallback
emlx_resolve_plugin_callback(const std::string &plugin,
                             const std::string &callback);

bool emlx_valid_plugin_name(const std::string &value);

std::string emlx_plugin_callback_failure_error(
    const std::string &plugin, const std::string &callback,
    const std::string &detail, size_t limit = 4096);

uint32_t emlx_invoke_plugin_count_policy(
    EMLXOperandCountFn policy, EMLXPluginInt64View attrs,
    uint32_t fixed_count, const char *kind, const std::string &plugin,
    const std::string &callback, size_t error_limit = 4096);

std::vector<mlx::core::array> emlx_invoke_plugin_callback(
    const std::string &plugin, const std::string &callback,
    const std::vector<mlx::core::array> &operands,
    const std::vector<int64_t> &attrs, const mlx::core::Device &device);
