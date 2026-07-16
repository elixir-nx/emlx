#include "emlx/plugin/registry.hpp"
#include "emlx_nif_shared.hpp"
#include "nx_nif_utils.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {

constexpr size_t kNameMax = 128;
constexpr uint32_t kCallbacksMax = 256;
constexpr uint64_t kDeviceTypesMax = 2;
constexpr size_t kErrorMax = 4096;
constexpr size_t kCallPluginNifSuffixSize = sizeof(" in NIF.call_plugin/5") - 1;

std::unordered_map<std::string, std::shared_ptr<const EMLXLoadedPlugin>>
    g_plugins;
std::shared_mutex g_plugin_mutex;

bool valid_name(const std::string &value) {
  if (value.empty() || value.size() > kNameMax) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
           (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
  });
}

bool valid_device_type(emlx::plugin::device_type_t device_type) {
  return device_type == mlx::core::Device::DeviceType::cpu ||
         device_type == mlx::core::Device::DeviceType::gpu;
}

bool valid_utf8(const std::string &value) {
  size_t index = 0;
  while (index < value.size()) {
    const uint8_t first = static_cast<uint8_t>(value[index++]);
    if (first <= 0x7f) {
      continue;
    }
    uint32_t codepoint = 0;
    size_t continuation = 0;
    if (first >= 0xc2 && first <= 0xdf) {
      codepoint = first & 0x1f;
      continuation = 1;
    } else if (first >= 0xe0 && first <= 0xef) {
      codepoint = first & 0x0f;
      continuation = 2;
    } else if (first >= 0xf0 && first <= 0xf4) {
      codepoint = first & 0x07;
      continuation = 3;
    } else {
      return false;
    }
    if (index + continuation > value.size()) {
      return false;
    }
    for (size_t offset = 0; offset < continuation; ++offset) {
      const uint8_t next = static_cast<uint8_t>(value[index++]);
      if ((next & 0xc0) != 0x80) {
        return false;
      }
      codepoint = (codepoint << 6) | (next & 0x3f);
    }
    if ((continuation == 2 && codepoint < 0x800) ||
        (continuation == 3 && codepoint < 0x10000) ||
        (codepoint >= 0xd800 && codepoint <= 0xdfff) || codepoint > 0x10ffff) {
      return false;
    }
  }
  return true;
}

std::string bounded_error(const std::string &value, size_t limit = kErrorMax) {
  static constexpr char kTruncationMarker[] = "... [truncated]";
  constexpr size_t marker_size = sizeof(kTruncationMarker) - 1;

  if (limit == 0) {
    return {};
  }
  if (!valid_utf8(value)) {
    return "plugin returned invalid UTF-8 error detail";
  }

  const bool truncated = value.size() > limit;
  size_t prefix_size = value.size();
  if (truncated) {
    if (limit <= marker_size) {
      return std::string(kTruncationMarker, limit);
    }
    prefix_size = limit - marker_size;
    while (prefix_size > 0 && prefix_size < value.size() &&
           (static_cast<uint8_t>(value[prefix_size]) & 0xc0U) == 0x80U) {
      --prefix_size;
    }
  }

  std::string result = value.substr(0, prefix_size);
  for (char &byte : result) {
    const unsigned char unsigned_byte = static_cast<unsigned char>(byte);
    if ((unsigned_byte < 0x20 && byte != '\n' && byte != '\t') ||
        unsigned_byte == 0x7f) {
      byte = '?';
    }
  }
  if (truncated) {
    result.append(kTruncationMarker);
  }
  return result;
}

std::string callback_failure_error(const std::string &plugin_name,
                                   const std::string &callback_name,
                                   const std::string &detail, size_t limit) {
  const std::string prefix =
      "plugin callback \"" + plugin_name + "/" + callback_name + "\" failed";
  if (detail.empty()) {
    return prefix + " without error detail";
  }

  const std::string separator = ": ";
  const size_t detail_limit = prefix.size() + separator.size() < limit
                                  ? limit - prefix.size() - separator.size()
                                  : 0;
  return prefix + separator + bounded_error(detail, detail_limit);
}

uint32_t invoke_count_policy(emlx::plugin::operand_count_fn_t policy,
                             emlx::plugin::int64_view_t attrs,
                             uint32_t fixed_count, const char *kind,
                             const std::string &plugin_name,
                             const std::string &callback_name,
                             size_t error_limit) {
  if (!policy) {
    return fixed_count;
  }

  uint32_t count = 0;
  std::string error;
  bool ok = false;
  try {
    ok = policy(attrs, count, error);
  } catch (const std::bad_alloc &) {
    throw std::runtime_error(std::string("plugin callback ") + kind +
                             " policy allocation failed");
  } catch (const std::exception &exception) {
    error = exception.what();
  } catch (...) {
    error = std::string("unknown plugin ") + kind + " policy exception";
  }
  if (!ok) {
    const std::string policy_detail =
        error.empty() ? std::string(kind) + " policy returned no error detail"
                      : std::string(kind) + " policy failed: " + error;
    throw std::runtime_error(callback_failure_error(
        plugin_name, callback_name, policy_detail, error_limit));
  }
  if (count == 0) {
    throw std::runtime_error("plugin callback \"" + plugin_name + "/" +
                             callback_name + "\" derived zero " + kind +
                             " count");
  }
  return count;
}

std::string copy_required_string(const std::string &value, size_t limit,
                                 const char *field) {
  if (value.empty()) {
    throw std::runtime_error(std::string("plugin field ") + field +
                             " is empty");
  }
  if (value.size() > limit) {
    throw std::runtime_error(std::string("plugin field ") + field +
                             " exceeds its length limit");
  }
  return value;
}

std::string canonical_path(const std::string &path) {
  char *resolved = realpath(path.c_str(), nullptr);
  if (!resolved) {
    throw std::runtime_error(
        "plugin path does not resolve to an existing file");
  }
  std::string result(resolved);
  std::free(resolved);
  return result;
}

std::shared_ptr<const EMLXLoadedPlugin>
load_generic_candidate(const std::string &requested_name,
                       const std::string &path) {
  const std::string resolved_path = canonical_path(path);

  {
    std::shared_lock lock(g_plugin_mutex);
    auto existing = g_plugins.find(requested_name);
    if (existing != g_plugins.end()) {
      if (existing->second->canonical_path == resolved_path) {
        return existing->second;
      }
      throw std::runtime_error(
          "plugin registration conflicts with an accepted plugin");
    }
  }

  EMLXSharedObjectHandle handle{
      dlopen(resolved_path.c_str(), RTLD_NOW | RTLD_LOCAL)};
  if (!handle) {
    const char *detail = dlerror();
    throw std::runtime_error(std::string("failed to load plugin: ") +
                             (detail ? detail : "unknown loader error"));
  }

  dlerror();
  auto discovery = reinterpret_cast<emlx::plugin::discovery_v1_fn_t>(
      dlsym(handle.get(), "emlx_plugin_descriptor_v1"));
  const char *symbol_error = dlerror();
  if (symbol_error) {
    throw std::runtime_error(std::string("failed to resolve ") +
                             "emlx_plugin_descriptor_v1: " + symbol_error);
  }
  if (!discovery) {
    throw std::runtime_error("emlx_plugin_descriptor_v1 resolved to null");
  }

  const emlx::plugin::bootstrap_v1_t *plugin_bootstrap = discovery();
  if (!plugin_bootstrap) {
    throw std::runtime_error("plugin returned a null bootstrap");
  }

  emlx::plugin::bootstrap_v1_t bootstrap{};
  std::memcpy(&bootstrap, plugin_bootstrap, sizeof(bootstrap));
  if (bootstrap.magic != emlx::plugin::magic_v1) {
    throw std::runtime_error("plugin bootstrap has an invalid magic value");
  }
  if (bootstrap.bootstrap_size != sizeof(emlx::plugin::bootstrap_v1_t)) {
    throw std::runtime_error("plugin bootstrap size does not match ABI v1");
  }
  if (bootstrap.plugin_abi_version != emlx::plugin::abi_v1) {
    throw std::runtime_error("plugin bootstrap ABI version is unsupported");
  }
  if (bootstrap.descriptor_size != sizeof(emlx::plugin::descriptor_t)) {
    throw std::runtime_error(
        "plugin bootstrap descriptor size does not match ABI v1");
  }
  if (!bootstrap.descriptor) {
    throw std::runtime_error("plugin bootstrap has a null descriptor pointer");
  }

  if (reinterpret_cast<uintptr_t>(bootstrap.descriptor) %
          alignof(emlx::plugin::descriptor_t) !=
      0) {
    throw std::runtime_error("plugin descriptor pointer is not aligned");
  }

  const auto &descriptor =
      *static_cast<const emlx::plugin::descriptor_t *>(bootstrap.descriptor);
  if (descriptor.descriptor_size != sizeof(emlx::plugin::descriptor_t)) {
    throw std::runtime_error("plugin descriptor size does not match ABI v1");
  }
  if (descriptor.callback_descriptor_size !=
      sizeof(emlx::plugin::callback_descriptor_t)) {
    throw std::runtime_error(
        "plugin callback descriptor size does not match ABI v1");
  }

  const std::string descriptor_name =
      copy_required_string(descriptor.name, kNameMax, "name");
  if (!valid_name(descriptor_name)) {
    throw std::runtime_error(
        "plugin descriptor name contains invalid characters");
  }
  if (descriptor_name != requested_name) {
    throw std::runtime_error(
        "plugin descriptor name does not match requested name");
  }

  if (descriptor.callback_count > kCallbacksMax) {
    throw std::runtime_error("plugin callback count exceeds its limit");
  }
  if (descriptor.callback_count > 0 && !descriptor.callbacks) {
    throw std::runtime_error("plugin callback table pointer is null");
  }
  if (descriptor.callback_count > 0 &&
      reinterpret_cast<uintptr_t>(descriptor.callbacks) %
              alignof(emlx::plugin::callback_descriptor_t) !=
          0) {
    throw std::runtime_error("plugin callback table pointer is not aligned");
  }
  const size_t callback_span = static_cast<size_t>(descriptor.callback_count) *
                               sizeof(emlx::plugin::callback_descriptor_t);
  if (descriptor.callback_count > 0 &&
      callback_span / descriptor.callback_count !=
          sizeof(emlx::plugin::callback_descriptor_t)) {
    throw std::runtime_error("plugin callback table span overflows");
  }

  auto loaded = std::make_shared<EMLXLoadedPlugin>();
  loaded->shared_object = std::move(handle);
  loaded->name = descriptor_name;
  loaded->canonical_path = resolved_path;

  for (uint32_t i = 0; i < descriptor.callback_count; ++i) {
    auto callback =
        std::make_shared<EMLXLoadedPluginCallback>(descriptor.callbacks[i]);
    if (!loaded->callbacks.emplace(callback->name, callback).second) {
      throw std::runtime_error("plugin callback names must be unique");
    }
  }

  {
    std::lock_guard lock(g_plugin_mutex);
    auto [it, inserted] = g_plugins.emplace(requested_name, loaded);
    if (!inserted) {
      if (it->second->canonical_path == resolved_path) {
        return it->second;
      }
      throw std::runtime_error(
          "plugin registration conflicts with an accepted plugin");
    }
  }
  return loaded;
}

fine::Term load_plugin_impl(ErlNifEnv *env, std::string name,
                            std::string path) {
  try {
    if (!valid_name(name)) {
      return nx::nif::error(env, "load_plugin expects a valid name");
    }
    load_generic_candidate(name, path);
    return nx::nif::ok(env);
  } catch (const std::bad_alloc &) {
    return nx::nif::error(env, "plugin loader allocation failed");
  } catch (const std::exception &error) {
    return nx::nif::error(env, bounded_error(error.what()).c_str());
  } catch (...) {
    return nx::nif::error(env, "internal plugin loader error");
  }
}

} // namespace

EMLXSharedObjectHandle::~EMLXSharedObjectHandle() {
  if (value_) {
    dlclose(value_);
  }
}

EMLXSharedObjectHandle::EMLXSharedObjectHandle(
    EMLXSharedObjectHandle &&other) noexcept
    : value_(std::exchange(other.value_, nullptr)) {}

EMLXSharedObjectHandle &EMLXSharedObjectHandle::operator=(
    EMLXSharedObjectHandle &&other) noexcept {
  if (this != &other) {
    if (value_) {
      dlclose(value_);
    }
    value_ = std::exchange(other.value_, nullptr);
  }
  return *this;
}

EMLXLoadedPluginCallback::EMLXLoadedPluginCallback(
    const emlx::plugin::callback_descriptor_t &source)
    : name(copy_required_string(source.name, kNameMax, "callback name")),
      schema_version(source.schema_version),
      attr_schema_version(source.attr_schema_version),
      operand_count(source.operand_count),
      operand_count_from_attrs(source.operand_count_from_attrs),
      output_count(source.output_count),
      output_count_from_attrs(source.output_count_from_attrs),
      callback(source.callback) {
  if (!valid_name(name)) {
    throw std::runtime_error(
        "plugin callback name contains invalid characters");
  }
  if (schema_version != 1) {
    throw std::runtime_error("plugin callback schema version is unsupported");
  }
  if (attr_schema_version != 1) {
    throw std::runtime_error(
        "plugin callback attribute schema version is unsupported");
  }
  if (!callback) {
    throw std::runtime_error("plugin callback function pointer is null");
  }
  if (!source.supported_devices.data) {
    throw std::runtime_error(
        "plugin callback supported device pointer is null");
  }
  if (source.supported_devices.size == 0) {
    throw std::runtime_error("plugin callback supports no devices");
  }
  if (source.supported_devices.size > kDeviceTypesMax) {
    throw std::runtime_error(
        "plugin callback supported device count exceeds its limit");
  }
  if (reinterpret_cast<uintptr_t>(source.supported_devices.data.get()) %
          alignof(emlx::plugin::device_type_t) !=
      0) {
    throw std::runtime_error(
        "plugin callback supported device pointer is not aligned");
  }
  if ((operand_count == 0) == (operand_count_from_attrs == nullptr)) {
    throw std::runtime_error("plugin callback operand policy is invalid");
  }
  if ((output_count == 0) == (output_count_from_attrs == nullptr)) {
    throw std::runtime_error("plugin callback output policy is invalid");
  }

  supported_devices.reserve(source.supported_devices.size);
  for (uint64_t device_index = 0;
       device_index < source.supported_devices.size; ++device_index) {
    emlx::plugin::device_type_t device_type{};
    std::memcpy(&device_type,
                source.supported_devices.data.get() + device_index,
                sizeof(device_type));
    if (!valid_device_type(device_type)) {
      throw std::runtime_error(
          "plugin callback contains an invalid device type");
    }
    if (std::find(supported_devices.begin(), supported_devices.end(),
                  device_type) != supported_devices.end()) {
      throw std::runtime_error(
          "plugin callback contains a duplicate device type");
    }
    supported_devices.push_back(device_type);
  }
}

bool emlx_valid_plugin_name(const std::string &value) {
  return valid_name(value);
}

bool emlx_plugin_callback_supports_device(
    const EMLXLoadedPluginCallback &callback,
    emlx::plugin::device_type_t device_type) {
  return std::find(callback.supported_devices.begin(),
                   callback.supported_devices.end(),
                   device_type) != callback.supported_devices.end();
}

EMLXResolvedPluginCallback
emlx_resolve_plugin_callback(const std::string &plugin,
                             const std::string &callback) {
  std::shared_lock lock(g_plugin_mutex);
  auto plugin_it = g_plugins.find(plugin);
  if (plugin_it == g_plugins.end()) {
    throw std::runtime_error("plugin \"" + plugin + "\" is not loaded");
  }
  auto callback_it = plugin_it->second->callbacks.find(callback);
  if (callback_it == plugin_it->second->callbacks.end()) {
    throw std::runtime_error("plugin callback \"" + plugin + "/" + callback +
                             "\" is not registered");
  }
  return {plugin_it->second, callback_it->second};
}

std::string emlx_plugin_callback_failure_error(const std::string &plugin,
                                               const std::string &callback,
                                               const std::string &detail,
                                               size_t limit) {
  return callback_failure_error(plugin, callback, detail, limit);
}

uint32_t emlx_invoke_plugin_count_policy(
    emlx::plugin::operand_count_fn_t policy, emlx::plugin::int64_view_t attrs,
    uint32_t fixed_count, const char *kind, const std::string &plugin,
    const std::string &callback, size_t error_limit) {
  return invoke_count_policy(policy, attrs, fixed_count, kind, plugin, callback,
                             error_limit);
}

std::vector<mlx::core::array> emlx_invoke_plugin_callback(
    const std::string &plugin_name, const std::string &callback_name,
    std::vector<mlx::core::array> operands, std::vector<int64_t> attrs,
    const mlx::core::Device &device) {
  auto resolved = emlx_resolve_plugin_callback(plugin_name, callback_name);
  const auto &callback = *resolved.callback;
  auto operand_view = emlx::plugin::make_view(std::move(operands));
  auto attr_view = emlx::plugin::make_view(std::move(attrs));
  constexpr size_t callback_error_max = kErrorMax - kCallPluginNifSuffixSize;
  const uint32_t expected_operands = emlx_invoke_plugin_count_policy(
      callback.operand_count_from_attrs, attr_view, callback.operand_count,
      "operand", plugin_name, callback_name, callback_error_max);
  if (operand_view.size != expected_operands) {
    throw std::runtime_error(
        "plugin callback received " + std::to_string(operand_view.size) +
        " operands, expected " + std::to_string(expected_operands));
  }
  const uint32_t expected_outputs = emlx_invoke_plugin_count_policy(
      callback.output_count_from_attrs, attr_view, callback.output_count,
      "output", plugin_name, callback_name, callback_error_max);
  if (!emlx::g_current_worker) {
    throw std::runtime_error("plugin execution has no current worker");
  }
  if (!emlx_plugin_callback_supports_device(callback, device.type)) {
    throw std::runtime_error(
        "plugin callback does not support the worker device");
  }
  const auto stream = emlx::g_current_worker->stream();
  emlx::plugin::call_t call{std::move(operand_view), std::move(attr_view),
                            device, stream};
  std::vector<mlx::core::array> outputs;
  std::optional<std::string> error;
  try {
    error = callback.callback(call, outputs);
  } catch (const std::bad_alloc &) {
    throw std::runtime_error("plugin callback allocation failed");
  } catch (const std::exception &exception) {
    error = exception.what();
  } catch (...) {
    error = "unknown plugin callback exception";
  }
  if (error) {
    throw std::runtime_error(emlx_plugin_callback_failure_error(
        plugin_name, callback_name, *error, callback_error_max));
  }
  if (outputs.size() != expected_outputs) {
    throw std::runtime_error(
        "plugin callback returned wrong output count: got " +
        std::to_string(outputs.size()) + ", expected " +
        std::to_string(expected_outputs));
  }
  return outputs;
}

#ifndef EMLX_PLUGIN_REGISTRY_TESTING
std::vector<mlx::core::array>
call_plugin_impl(ErlNifEnv *env, std::string plugin_name,
                 std::string callback_name, std::vector<TensorArg> operands,
                 std::vector<int64_t> attrs, mlx::core::Device device) {
  (void)env;
  std::vector<mlx::core::array> arrays;
  arrays.reserve(operands.size());
  for (const auto &operand : operands) {
    arrays.push_back(*operand);
  }
  return emlx_invoke_plugin_callback(plugin_name, callback_name,
                                     std::move(arrays), std::move(attrs),
                                     device);
}
FINE_ASYNC_NIF(call_plugin)
#endif

ERL_NIF_TERM load_plugin(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  return fine::nif(env, argc, argv, load_plugin_impl);
}
