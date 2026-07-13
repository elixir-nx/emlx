#include "emlx_plugin_registry.hpp"
#include "emlx_nif_shared.hpp"
#include "emlx_plugin_build_compat.hpp"
#include "nx_nif_utils.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {

constexpr size_t kNameMax = 128;
constexpr size_t kDebugNameMax = 256;
constexpr uint32_t kCallbacksMax = 256;
constexpr uint32_t kOperandsMax = 4096;
constexpr uint32_t kOutputsMax = 1024;
constexpr size_t kErrorMax = 4096;
constexpr size_t kCallPluginNifSuffixSize =
    sizeof(" in NIF.call_plugin/5") - 1;

std::unordered_map<std::string, std::shared_ptr<const EMLXLoadedPlugin>>
    g_plugins;
std::shared_mutex g_plugin_mutex;

struct CandidateHandle {
  void *value = nullptr;
  ~CandidateHandle() {
    if (value)
      dlclose(value);
  }
  void release() { value = nullptr; }
};

bool valid_name(const std::string &value) {
  if (value.empty() || value.size() > kNameMax)
    return false;
  return std::all_of(value.begin(), value.end(), [](unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
           (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
  });
}

bool valid_build_id(const std::string &value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

bool valid_ascii(const std::string &value) {
  return std::all_of(value.begin(), value.end(), [](unsigned char byte) {
    return byte >= 0x20 && byte <= 0x7e;
  });
}

bool valid_utf8(const std::string &value) {
  size_t index = 0;
  while (index < value.size()) {
    const uint8_t first = static_cast<uint8_t>(value[index++]);
    if (first <= 0x7f)
      continue;
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
    if (index + continuation > value.size())
      return false;
    for (size_t offset = 0; offset < continuation; ++offset) {
      const uint8_t next = static_cast<uint8_t>(value[index++]);
      if ((next & 0xc0) != 0x80)
        return false;
      codepoint = (codepoint << 6) | (next & 0x3f);
    }
    if ((continuation == 2 && codepoint < 0x800) ||
        (continuation == 3 && codepoint < 0x10000) ||
        (codepoint >= 0xd800 && codepoint <= 0xdfff) || codepoint > 0x10ffff)
      return false;
  }
  return true;
}

std::string bounded_error(const std::string &value, size_t limit = kErrorMax) {
  static constexpr char kTruncationMarker[] = "... [truncated]";
  constexpr size_t marker_size = sizeof(kTruncationMarker) - 1;

  if (limit == 0)
    return {};
  if (!valid_utf8(value))
    return "plugin returned invalid UTF-8 error detail";

  const bool truncated = value.size() > limit;
  size_t prefix_size = value.size();
  if (truncated) {
    if (limit <= marker_size)
      return std::string(kTruncationMarker, limit);
    prefix_size = limit - marker_size;
    while (prefix_size > 0 && prefix_size < value.size() &&
           (static_cast<uint8_t>(value[prefix_size]) & 0xc0U) == 0x80U)
      --prefix_size;
  }

  std::string result = value.substr(0, prefix_size);
  for (char &byte : result) {
    const unsigned char unsigned_byte = static_cast<unsigned char>(byte);
    if ((unsigned_byte < 0x20 && byte != '\n' && byte != '\t') ||
        unsigned_byte == 0x7f)
      byte = '?';
  }
  if (truncated)
    result.append(kTruncationMarker);
  return result;
}

std::string callback_failure_error(const std::string &plugin_name,
                                   const std::string &callback_name,
                                   const std::string &detail, size_t limit) {
  const std::string prefix = "plugin callback \"" + plugin_name + "/" +
                             callback_name + "\" failed";
  if (detail.empty())
    return prefix + " without error detail";

  const std::string separator = ": ";
  const size_t detail_limit =
      prefix.size() + separator.size() < limit
          ? limit - prefix.size() - separator.size()
          : 0;
  return prefix + separator + bounded_error(detail, detail_limit);
}

uint32_t invoke_count_policy(EMLXOperandCountFn policy,
                             EMLXPluginInt64View attrs,
                             uint32_t fixed_count, const char *kind,
                             const std::string &plugin_name,
                             const std::string &callback_name,
                             size_t error_limit) {
  if (!policy)
    return fixed_count;

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
    throw std::runtime_error(
        callback_failure_error(plugin_name, callback_name, policy_detail,
                               error_limit));
  }
  return count;
}

constexpr uint32_t host_endianness() {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return EMLX_ENDIAN_BIG;
#else
  return EMLX_ENDIAN_LITTLE;
#endif
}

std::string copy_required_string(EMLXPluginStringView view, size_t limit,
                                 const char *field) {
  if (!view.data || view.size == 0 || view.size > limit ||
      view.size > std::numeric_limits<size_t>::max())
    throw std::runtime_error(std::string("plugin field ") + field +
                             " is missing or exceeds its limit");
  return std::string(view.data, static_cast<size_t>(view.size));
}

std::string copy_optional_string(EMLXPluginStringView view, size_t limit,
                                 const char *field) {
  if (!view.data && view.size == 0)
    return {};
  if (!view.data || view.size == 0 || view.size > limit ||
      view.size > std::numeric_limits<size_t>::max())
    throw std::runtime_error(std::string("plugin field ") + field +
                             " is noncanonical or exceeds its limit");
  return std::string(view.data, static_cast<size_t>(view.size));
}

std::string canonical_path(const std::string &path) {
  char *resolved = realpath(path.c_str(), nullptr);
  if (!resolved)
    throw std::runtime_error("plugin path does not resolve to an existing file");
  std::string result(resolved);
  std::free(resolved);
  return result;
}

const char *forbidden_loader_override() {
#if defined(__APPLE__)
  static constexpr const char *vars[] = {
      "DYLD_LIBRARY_PATH",       "DYLD_FALLBACK_LIBRARY_PATH",
      "DYLD_INSERT_LIBRARIES",  "DYLD_FRAMEWORK_PATH",
      "DYLD_FALLBACK_FRAMEWORK_PATH", "DYLD_ROOT_PATH"};
#else
  static constexpr const char *vars[] = {"LD_LIBRARY_PATH", "LD_PRELOAD",
                                          "LD_AUDIT"};
#endif
  for (const char *var : vars) {
    if (std::getenv(var) != nullptr)
      return var;
  }
  return nullptr;
}

std::shared_ptr<const EMLXLoadedPlugin>
load_generic_candidate(const std::string &requested_name,
                       const std::string &path,
                       const std::string &expected_build_id) {
  const std::string resolved_path = canonical_path(path);

  {
    std::shared_lock lock(g_plugin_mutex);
    auto existing = g_plugins.find(requested_name);
    if (existing != g_plugins.end()) {
      if (existing->second->canonical_path == resolved_path &&
          (expected_build_id.empty() ||
           existing->second->plugin_build_id == expected_build_id))
        return existing->second;
      throw std::runtime_error("plugin registration conflicts with an accepted plugin");
    }
  }

  CandidateHandle handle{dlopen(resolved_path.c_str(), RTLD_NOW | RTLD_LOCAL)};
  if (!handle.value) {
    const char *detail = dlerror();
    throw std::runtime_error(std::string("failed to load plugin: ") +
                             (detail ? detail : "unknown loader error"));
  }

  dlerror();
  auto discovery = reinterpret_cast<EMLXPluginDiscoveryV1>(
      dlsym(handle.value, "emlx_plugin_descriptor_v1"));
  const char *symbol_error = dlerror();
  if (!discovery || symbol_error)
    throw std::runtime_error("plugin is missing emlx_plugin_descriptor_v1");

  const EMLXPluginBootstrapV1 *plugin_bootstrap = discovery();
  if (!plugin_bootstrap)
    throw std::runtime_error("plugin returned a null bootstrap");

  EMLXPluginBootstrapV1 bootstrap{};
  std::memcpy(&bootstrap, plugin_bootstrap, sizeof(bootstrap));
  if (bootstrap.magic != EMLX_PLUGIN_MAGIC_V1 ||
      bootstrap.bootstrap_size != sizeof(EMLXPluginBootstrapV1) ||
      bootstrap.plugin_abi_version != EMLX_PLUGIN_ABI_V1 ||
      bootstrap.header_abi_hash != EMLX_PLUGIN_HEADER_ABI_HASH_V1 ||
      bootstrap.layout_abi_hash != EMLX_PLUGIN_LAYOUT_ABI_HASH_V1 ||
      bootstrap.pointer_width_bits != sizeof(void *) * 8 ||
      bootstrap.endianness != host_endianness() ||
      bootstrap.descriptor_size != sizeof(EMLXPluginDescriptor) ||
      !bootstrap.descriptor)
    throw std::runtime_error("plugin bootstrap is incompatible with EMLX");

  if (reinterpret_cast<uintptr_t>(bootstrap.descriptor) %
          alignof(EMLXPluginDescriptor) !=
      0)
    throw std::runtime_error("plugin descriptor pointer is not aligned");

  EMLXPluginDescriptor descriptor{};
  std::memcpy(&descriptor, bootstrap.descriptor, sizeof(descriptor));
  if (descriptor.compatibility.plugin_abi_version != EMLX_PLUGIN_ABI_V1 ||
      descriptor.compatibility.header_abi_version !=
          EMLX_PLUGIN_HEADER_ABI_V1 ||
      descriptor.compatibility.header_abi_hash !=
          EMLX_PLUGIN_HEADER_ABI_HASH_V1 ||
      descriptor.compatibility.plugin_descriptor_size !=
          sizeof(EMLXPluginDescriptor) ||
      descriptor.compatibility.callback_descriptor_size !=
          sizeof(EMLXPluginCallbackDescriptor) ||
      descriptor.compatibility.pointer_width_bits != sizeof(void *) * 8 ||
      descriptor.compatibility.endianness != host_endianness())
    throw std::runtime_error("plugin descriptor is incompatible with EMLX");

  const std::string descriptor_name =
      copy_required_string(descriptor.name, kNameMax, "name");
  if (!valid_name(descriptor_name) || descriptor_name != requested_name)
    throw std::runtime_error("plugin descriptor name does not match requested name");

  const std::string build_id = copy_required_string(
      descriptor.compatibility.plugin_build_id, 64, "plugin_build_id");
  if (!valid_build_id(build_id) ||
      (!expected_build_id.empty() && build_id != expected_build_id))
    throw std::runtime_error("plugin build identity does not match expected identity");

  const std::string mlx_version = copy_required_string(
      descriptor.compatibility.mlx_version, 256, "mlx_version");
  const std::string mlx_variant = copy_required_string(
      descriptor.compatibility.mlx_variant, 256, "mlx_variant");
  const std::string mlx_build_id = copy_required_string(
      descriptor.compatibility.mlx_build_id, 64, "mlx_build_id");
  const std::string mlx_headers_build_id = copy_required_string(
      descriptor.compatibility.mlx_headers_build_id, 64,
      "mlx_headers_build_id");
  const std::string target = copy_required_string(
      descriptor.compatibility.target_triple, 256, "target_triple");
  const std::string compiler = copy_required_string(
      descriptor.compatibility.compiler_abi_family, 256,
      "compiler_abi_family");
  const std::string standard_library = copy_required_string(
      descriptor.compatibility.cxx_standard_library_abi, 256,
      "cxx_standard_library_abi");
  if (!valid_build_id(mlx_build_id) || !valid_build_id(mlx_headers_build_id) ||
      !valid_ascii(mlx_version) || !valid_ascii(mlx_variant) ||
      !valid_ascii(target) || !valid_ascii(compiler) ||
      !valid_ascii(standard_library))
    throw std::runtime_error("plugin compatibility identity has invalid bytes");
  if (mlx_version != EMLX_EXPECTED_MLX_VERSION ||
      mlx_variant != EMLX_EXPECTED_MLX_VARIANT ||
      mlx_build_id != EMLX_EXPECTED_MLX_BUILD_ID ||
      mlx_headers_build_id != EMLX_EXPECTED_MLX_HEADERS_BUILD_ID ||
      target != EMLX_EXPECTED_TARGET_TRIPLE ||
      compiler != EMLX_EXPECTED_COMPILER_FAMILY ||
      standard_library != EMLX_EXPECTED_CXX_STDLIB_ABI)
    throw std::runtime_error("plugin MLX compatibility identity does not match EMLX");

  auto host_identity = emlx_host_runtime_identity();
  if (!host_identity)
    throw std::runtime_error("EMLX host MLX identity is unavailable");
  if (!descriptor.mlx_runtime_anchor ||
      descriptor.mlx_runtime_anchor != host_identity->mlx_runtime_anchor)
    throw std::runtime_error("plugin resolves a different MLX runtime anchor");

  if (descriptor.callback_count > kCallbacksMax ||
      (descriptor.callback_count > 0 && !descriptor.callbacks))
    throw std::runtime_error("plugin callback table is invalid");
  if (descriptor.callback_count > 0 &&
      reinterpret_cast<uintptr_t>(descriptor.callbacks) %
              alignof(EMLXPluginCallbackDescriptor) !=
          0)
    throw std::runtime_error("plugin callback table pointer is not aligned");
  const size_t callback_span =
      static_cast<size_t>(descriptor.callback_count) *
      sizeof(EMLXPluginCallbackDescriptor);
  if (descriptor.callback_count > 0 &&
      callback_span / descriptor.callback_count !=
          sizeof(EMLXPluginCallbackDescriptor))
    throw std::runtime_error("plugin callback table span overflows");

  auto loaded = std::make_shared<EMLXLoadedPlugin>();
  loaded->name = descriptor_name;
  loaded->canonical_path = resolved_path;
  loaded->plugin_build_id = build_id;
  loaded->mlx_build_id = mlx_build_id;
  loaded->mlx_headers_build_id = mlx_headers_build_id;
  loaded->mlx_runtime_image = host_identity->mlx_runtime_image;
  loaded->shared_object_handle = handle.value;

  for (uint32_t i = 0; i < descriptor.callback_count; ++i) {
    EMLXPluginCallbackDescriptor source{};
    std::memcpy(&source, descriptor.callbacks + i, sizeof(source));
    auto callback = std::make_shared<EMLXLoadedPluginCallback>();
    callback->name = copy_required_string(source.name, kNameMax, "callback name");
    callback->debug_name =
        copy_optional_string(source.debug_name, kDebugNameMax, "debug_name");
    if (!callback->debug_name.empty() && !valid_utf8(callback->debug_name))
      throw std::runtime_error("plugin callback debug_name is not valid UTF-8");
    if (!valid_name(callback->name) || source.schema_version != 1 ||
        source.attr_schema_version != 1 || !source.callback ||
        source.operand_count > kOperandsMax || source.output_count > kOutputsMax ||
        source.device_capabilities == 0 ||
        (source.device_capabilities & ~EMLX_PLUGIN_DEVICE_KNOWN_V1) != 0)
      throw std::runtime_error("plugin callback descriptor is invalid");
    if ((source.operand_count == 0) ==
        (source.operand_count_from_attrs == nullptr))
      throw std::runtime_error("plugin callback operand policy is invalid");
    if ((source.output_count == 0) ==
        (source.output_count_from_attrs == nullptr))
      throw std::runtime_error("plugin callback output policy is invalid");
    callback->schema_version = source.schema_version;
    callback->attr_schema_version = source.attr_schema_version;
    callback->operand_count = source.operand_count;
    callback->operand_count_from_attrs = source.operand_count_from_attrs;
    callback->output_count = source.output_count;
    callback->output_count_from_attrs = source.output_count_from_attrs;
    callback->device_capabilities = source.device_capabilities;
    callback->callback = source.callback;
    if (!loaded->callbacks.emplace(callback->name, callback).second)
      throw std::runtime_error("plugin callback names must be unique");
  }

  {
    std::unique_lock lock(g_plugin_mutex);
    auto [it, inserted] = g_plugins.emplace(requested_name, loaded);
    if (!inserted) {
      if (it->second->canonical_path == resolved_path &&
          (expected_build_id.empty() ||
           it->second->plugin_build_id == expected_build_id))
        return it->second;
      throw std::runtime_error("plugin registration conflicts with an accepted plugin");
    }
  }
  handle.release();
  return loaded;
}

} // namespace

EMLXLoadedPlugin::~EMLXLoadedPlugin() = default;

bool emlx_valid_plugin_name(const std::string &value) {
  return valid_name(value);
}

EMLXResolvedPluginCallback
emlx_resolve_plugin_callback(const std::string &plugin,
                             const std::string &callback) {
  std::shared_lock lock(g_plugin_mutex);
  auto plugin_it = g_plugins.find(plugin);
  if (plugin_it == g_plugins.end())
    throw std::runtime_error("plugin \"" + plugin + "\" is not loaded");
  auto callback_it = plugin_it->second->callbacks.find(callback);
  if (callback_it == plugin_it->second->callbacks.end())
    throw std::runtime_error("plugin callback \"" + plugin + "/" + callback +
                             "\" is not registered");
  return {plugin_it->second, callback_it->second};
}

std::string emlx_plugin_callback_failure_error(
    const std::string &plugin, const std::string &callback,
    const std::string &detail, size_t limit) {
  return callback_failure_error(plugin, callback, detail, limit);
}

uint32_t emlx_invoke_plugin_count_policy(
    EMLXOperandCountFn policy, EMLXPluginInt64View attrs,
    uint32_t fixed_count, const char *kind, const std::string &plugin,
    const std::string &callback, size_t error_limit) {
  return invoke_count_policy(policy, attrs, fixed_count, kind, plugin, callback,
                             error_limit);
}

std::vector<mlx::core::array> emlx_invoke_plugin_callback(
    const std::string &plugin_name, const std::string &callback_name,
    const std::vector<mlx::core::array> &operands,
    const std::vector<int64_t> &attrs, const mlx::core::Device &device) {
  auto resolved = emlx_resolve_plugin_callback(plugin_name, callback_name);
  const auto &callback = *resolved.callback;
  const EMLXPluginInt64View attr_view{attrs.data(), attrs.size()};
  constexpr size_t callback_error_max = kErrorMax - kCallPluginNifSuffixSize;
  const uint32_t expected_operands =
      emlx_invoke_plugin_count_policy(
          callback.operand_count_from_attrs, attr_view, callback.operand_count,
          "operand", plugin_name, callback_name, callback_error_max);
  if (expected_operands > kOperandsMax || operands.size() != expected_operands)
    throw std::runtime_error("plugin callback operand count mismatch");
  if (!emlx::g_current_worker)
    throw std::runtime_error("plugin execution has no current worker");
  const uint32_t device_bit = device.type == mlx::core::Device::DeviceType::cpu
                                  ? EMLX_PLUGIN_DEVICE_CPU_V1
                                  : EMLX_PLUGIN_DEVICE_GPU_METAL_V1;
  if ((callback.device_capabilities & device_bit) == 0)
    throw std::runtime_error("plugin callback does not support the worker device");
  const auto stream = emlx::g_current_worker->stream();
  EMLXPluginExecutionContext execution{&device, &stream};
  EMLXPluginCall call{{operands.data(), operands.size()},
                      attr_view, &execution};
  std::vector<mlx::core::array> outputs;
  std::string error;
  bool ok = false;
  try {
    ok = callback.callback(call, outputs, error);
  } catch (const std::bad_alloc &) {
    throw std::runtime_error("plugin callback allocation failed");
  } catch (const std::exception &exception) {
    error = exception.what();
  } catch (...) {
    error = "unknown plugin callback exception";
  }
  if (!ok)
    throw std::runtime_error(
        emlx_plugin_callback_failure_error(
            plugin_name, callback_name, error, callback_error_max));
  const uint32_t expected_outputs =
      emlx_invoke_plugin_count_policy(
          callback.output_count_from_attrs, attr_view, callback.output_count,
          "output", plugin_name, callback_name, callback_error_max);
  if (expected_outputs > kOutputsMax || outputs.size() != expected_outputs)
    throw std::runtime_error("plugin callback returned the wrong output count");
  return outputs;
}

#if !defined(EMLX_PLUGIN_REGISTRY_TESTING)
std::vector<mlx::core::array>
call_plugin_impl(ErlNifEnv *env, std::string plugin_name,
                 std::string callback_name, std::vector<TensorArg> operands,
                 std::vector<int64_t> attrs, mlx::core::Device device) {
  (void)env;
  std::vector<mlx::core::array> arrays;
  arrays.reserve(operands.size());
  for (const auto &operand : operands)
    arrays.push_back(*operand);
  return emlx_invoke_plugin_callback(plugin_name, callback_name, arrays, attrs,
                                     device);
}
FINE_ASYNC_NIF(call_plugin)
#endif

ERL_NIF_TERM load_plugin(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  (void)argc;
  try {
    std::string name;
    std::string path;
    if (!nx::nif::get(env, argv[0], name) || !valid_name(name))
      return nx::nif::error(env, "load_plugin expects a valid name");
    if (!nx::nif::get(env, argv[1], path))
      return nx::nif::error(env, "load_plugin expects a path string");
    load_generic_candidate(name, path, "");
    return nx::nif::ok(env);
  } catch (const std::bad_alloc &) {
    return nx::nif::error(env, "plugin loader allocation failed");
  } catch (const std::exception &error) {
    return nx::nif::error(env, bounded_error(error.what()).c_str());
  } catch (...) {
    return nx::nif::error(env, "internal plugin loader error");
  }
}

ERL_NIF_TERM load_plugin_with_build_id(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  (void)argc;
  try {
    std::string name;
    std::string path;
    std::string expected_build_id;
    if (!nx::nif::get(env, argv[0], name) || !valid_name(name))
      return nx::nif::error(env, "load_plugin expects a valid name");
    if (!nx::nif::get(env, argv[1], path))
      return nx::nif::error(env, "load_plugin expects a path string");
    if (!nx::nif::get(env, argv[2], expected_build_id) ||
        !valid_build_id(expected_build_id))
      return nx::nif::error(
          env, "expected plugin build identity must contain exactly 64 lowercase hexadecimal bytes");
    if (const char *override_name = forbidden_loader_override())
      return nx::nif::error(
          env, (std::string("plugin cannot be loaded while runtime loader override ") +
                override_name + " is present")
                   .c_str());
    load_generic_candidate(name, path, expected_build_id);
    return nx::nif::ok(env);
  } catch (const std::bad_alloc &) {
    return nx::nif::error(env, "plugin loader allocation failed");
  } catch (const std::exception &error) {
    return nx::nif::error(env, bounded_error(error.what()).c_str());
  } catch (...) {
    return nx::nif::error(env, "internal plugin loader error");
  }
}
