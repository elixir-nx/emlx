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

namespace emlx::plugin {

// Loads and validates the generic plugin descriptor, then keeps the accepted
// shared object alive for the VM lifetime.
ERL_NIF_TERM load_plugin(ErlNifEnv *, int, const ERL_NIF_TERM[]);
ERL_NIF_TERM call_plugin(ErlNifEnv *, int, const ERL_NIF_TERM[]);
ERL_NIF_TERM call_plugin_async(ErlNifEnv *, int, const ERL_NIF_TERM[]);

class shared_object_handle_t {
public:
  explicit shared_object_handle_t(void *value = nullptr) : value_(value) {}
  ~shared_object_handle_t();

  shared_object_handle_t(const shared_object_handle_t &) = delete;
  shared_object_handle_t &operator=(const shared_object_handle_t &) = delete;
  shared_object_handle_t(shared_object_handle_t &&other) noexcept;
  shared_object_handle_t &operator=(shared_object_handle_t &&other) noexcept;

  explicit operator bool() const { return value_ != nullptr; }
  void *get() const { return value_; }

private:
  void *value_;
};

struct loaded_callback_t {
  explicit loaded_callback_t(const callback_descriptor_t &source);

  std::string name;
  uint32_t schema_version;
  uint32_t attr_schema_version;
  uint32_t operand_count;
  operand_count_fn_t operand_count_from_attrs;
  uint32_t output_count;
  output_count_fn_t output_count_from_attrs;
  std::vector<device_type_t> supported_devices;
  callback_fn_t callback;
};

struct loaded_plugin_t {
  shared_object_handle_t shared_object;
  std::string name;
  std::string canonical_path;
  std::unordered_map<std::string, std::shared_ptr<const loaded_callback_t>>
      callbacks;
};

struct resolved_callback_t {
  std::shared_ptr<const loaded_plugin_t> plugin;
  std::shared_ptr<const loaded_callback_t> callback;
};

resolved_callback_t resolve_callback(const std::string &plugin,
                                     const std::string &callback);

bool valid_name(const std::string &value);

bool callback_supports_device(const loaded_callback_t &callback,
                              device_type_t device_type);

std::string callback_failure_error(
    const std::string &plugin, const std::string &callback,
    const std::string &detail, size_t limit = 4096);

uint32_t invoke_count_policy(
    operand_count_fn_t policy, int64_view_t attrs,
    uint32_t fixed_count, const char *kind, const std::string &plugin,
    const std::string &callback, size_t error_limit = 4096);

std::vector<mlx::core::array> invoke_callback(
    const std::string &plugin, const std::string &callback,
    std::vector<mlx::core::array> operands, std::vector<int64_t> attrs,
    const mlx::core::Device &device);

} // namespace emlx::plugin
