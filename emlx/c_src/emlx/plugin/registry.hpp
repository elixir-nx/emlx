#pragma once

// Generic, immutable name-keyed native plugin registry. It loads one
// versioned descriptor symbol, copies validated callback records into
// EMLX-owned storage, and keeps the shared object alive for the VM lifetime.

#include "emlx/plugin/abi.hpp"
#include "erl_nif.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Loads and validates the generic plugin descriptor, then keeps the accepted
// shared object alive for the VM lifetime.
ERL_NIF_TERM load_plugin(ErlNifEnv *, int, const ERL_NIF_TERM[]);
ERL_NIF_TERM call_plugin(ErlNifEnv *, int, const ERL_NIF_TERM[]);
ERL_NIF_TERM call_plugin_async(ErlNifEnv *, int, const ERL_NIF_TERM[]);

class EMLXSharedObjectHandle {
public:
  explicit EMLXSharedObjectHandle(void *value = nullptr) : value_(value) {}
  ~EMLXSharedObjectHandle();

  EMLXSharedObjectHandle(const EMLXSharedObjectHandle &) = delete;
  EMLXSharedObjectHandle &operator=(const EMLXSharedObjectHandle &) = delete;
  EMLXSharedObjectHandle(EMLXSharedObjectHandle &&other) noexcept;
  EMLXSharedObjectHandle &operator=(EMLXSharedObjectHandle &&other) noexcept;

  explicit operator bool() const { return value_ != nullptr; }
  void *get() const { return value_; }

private:
  void *value_;
};

struct EMLXLoadedPluginCallback {
  explicit EMLXLoadedPluginCallback(
      const emlx::plugin::callback_descriptor_t &source);

  std::string name;
  uint32_t schema_version;
  uint32_t attr_schema_version;
  uint32_t operand_count;
  emlx::plugin::operand_count_fn_t operand_count_from_attrs;
  uint32_t output_count;
  emlx::plugin::output_count_fn_t output_count_from_attrs;
  std::vector<emlx::plugin::device_type_t> supported_devices;
  emlx::plugin::callback_fn_t callback;
};

struct EMLXLoadedPlugin {
  EMLXSharedObjectHandle shared_object;
  std::string name;
  std::string canonical_path;
  std::unordered_map<std::string,
                     std::shared_ptr<const EMLXLoadedPluginCallback>>
      callbacks;
};

struct EMLXResolvedPluginCallback {
  std::shared_ptr<const EMLXLoadedPlugin> plugin;
  std::shared_ptr<const EMLXLoadedPluginCallback> callback;
};

EMLXResolvedPluginCallback
emlx_resolve_plugin_callback(const std::string &plugin,
                             const std::string &callback);

bool emlx_valid_plugin_name(const std::string &value);

bool emlx_plugin_callback_supports_device(
    const EMLXLoadedPluginCallback &callback,
    emlx::plugin::device_type_t device_type);

std::string emlx_plugin_callback_failure_error(
    const std::string &plugin, const std::string &callback,
    const std::string &detail, size_t limit = 4096);

uint32_t emlx_invoke_plugin_count_policy(
    emlx::plugin::operand_count_fn_t policy, emlx::plugin::int64_view_t attrs,
    uint32_t fixed_count, const char *kind, const std::string &plugin,
    const std::string &callback, size_t error_limit = 4096);

std::vector<mlx::core::array> emlx_invoke_plugin_callback(
    const std::string &plugin, const std::string &callback,
    const std::vector<mlx::core::array> &operands,
    const std::vector<int64_t> &attrs, const mlx::core::Device &device);
