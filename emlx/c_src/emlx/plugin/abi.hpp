#pragma once

#include "mlx/mlx.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define EMLX_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#define EMLX_PLUGIN_EXPORT
#endif

namespace emlx::plugin {

// magic_v1 is the ASCII tag "EMLXPLG1" packed into a uint64_t:
// 45 4D 4C 58 50 4C 47 31. V1 constants and layouts are immutable. An
// incompatible ABI revision should add parallel V2 constants and types, use
// the "EMLXPLG2" tag (0x454D4C58504C4732), set its ABI version to 2, and
// expose a versioned discovery symbol rather than changing V1 in place.
inline constexpr uint64_t magic_v1 = 0x454D4C58504C4731ULL;
inline constexpr uint32_t abi_v1 = 1;

// Plugin-owned views are borrowed by EMLX only while loading the descriptor.
// Call-owned views remain valid only for the count-policy or callback
// invocation that receives them. Neither side may retain borrowed pointers.
template <typename T> struct view_t {
  const T *data;
  uint64_t size;
};

using string_view_t = view_t<char>;
using array_view_t = view_t<mlx::core::array>;
using int64_view_t = view_t<int64_t>;
using device_type_t = mlx::core::Device::DeviceType;
using device_view_t = view_t<device_type_t>;

struct execution_context_t {
  const mlx::core::Device *device;
  const mlx::core::Stream *stream;
};

struct call_t {
  array_view_t operands;
  int64_view_t attrs;
  const execution_context_t *execution;
};

// Policies and callbacks may run concurrently on EMLX worker threads and must
// therefore be reentrant. Count policies return false and populate their error
// string on failure. Callback success returns std::nullopt; expected callback
// failures return an error string. Exceptions must not cross the plugin
// boundary.
using output_count_fn_t = bool (*)(int64_view_t, uint32_t &, std::string &);
using operand_count_fn_t = bool (*)(int64_view_t, uint32_t &, std::string &);
using callback_fn_t = std::optional<std::string> (*)(
    const call_t &, std::vector<mlx::core::array> &);

struct callback_descriptor_t {
  string_view_t name;
  uint32_t schema_version;
  uint32_t attr_schema_version;
  uint32_t operand_count;
  operand_count_fn_t operand_count_from_attrs;
  uint32_t output_count;
  output_count_fn_t output_count_from_attrs;
  device_view_t supported_devices;
  callback_fn_t callback;
};

struct descriptor_t {
  // Strings, the callback table, policy functions, and callbacks must remain
  // valid for the lifetime of the VM process.
  string_view_t name;
  uint64_t descriptor_size;
  uint64_t callback_descriptor_size;
  uint32_t callback_count;
  const callback_descriptor_t *callbacks;
};

struct bootstrap_v1_t {
  uint64_t magic;
  uint32_t bootstrap_size;
  uint32_t plugin_abi_version;
  uint64_t descriptor_size;
  const void *descriptor;
};

using discovery_v1_fn_t = const bootstrap_v1_t *(*)() noexcept;

static_assert(std::is_standard_layout_v<bootstrap_v1_t>);
static_assert(std::is_trivially_copyable_v<bootstrap_v1_t>);
static_assert(sizeof(bootstrap_v1_t) == 32);
static_assert(alignof(bootstrap_v1_t) == 8);
static_assert(offsetof(bootstrap_v1_t, magic) == 0);
static_assert(offsetof(bootstrap_v1_t, bootstrap_size) == 8);
static_assert(offsetof(bootstrap_v1_t, plugin_abi_version) == 12);
static_assert(offsetof(bootstrap_v1_t, descriptor_size) == 16);
static_assert(offsetof(bootstrap_v1_t, descriptor) == 24);

static_assert(std::is_standard_layout_v<string_view_t>);
static_assert(std::is_trivially_copyable_v<string_view_t>);
static_assert(sizeof(string_view_t) == 16);
static_assert(alignof(string_view_t) == 8);

static_assert(std::is_standard_layout_v<array_view_t>);
static_assert(std::is_trivially_copyable_v<array_view_t>);
static_assert(sizeof(array_view_t) == 16);

static_assert(std::is_standard_layout_v<int64_view_t>);
static_assert(std::is_trivially_copyable_v<int64_view_t>);
static_assert(sizeof(int64_view_t) == 16);

static_assert(std::is_enum_v<device_type_t>);
static_assert(std::is_standard_layout_v<device_view_t>);
static_assert(std::is_trivially_copyable_v<device_view_t>);
static_assert(sizeof(device_view_t) == 16);

static_assert(std::is_standard_layout_v<execution_context_t>);
static_assert(std::is_trivially_copyable_v<execution_context_t>);
static_assert(sizeof(execution_context_t) == 16);

static_assert(std::is_standard_layout_v<call_t>);
static_assert(std::is_trivially_copyable_v<call_t>);
static_assert(sizeof(call_t) == 40);

static_assert(std::is_standard_layout_v<callback_descriptor_t>);
static_assert(std::is_trivially_copyable_v<callback_descriptor_t>);
static_assert(sizeof(callback_descriptor_t) == 80);
static_assert(alignof(callback_descriptor_t) == 8);

static_assert(std::is_standard_layout_v<descriptor_t>);
static_assert(std::is_trivially_copyable_v<descriptor_t>);
static_assert(sizeof(descriptor_t) == 48);
static_assert(alignof(descriptor_t) == 8);
static_assert(offsetof(descriptor_t, name) == 0);
static_assert(offsetof(descriptor_t, descriptor_size) == 16);
static_assert(offsetof(descriptor_t, callback_descriptor_size) == 24);
static_assert(offsetof(descriptor_t, callback_count) == 32);
static_assert(offsetof(descriptor_t, callbacks) == 40);

} // namespace emlx::plugin

extern "C" EMLX_PLUGIN_EXPORT const emlx::plugin::bootstrap_v1_t *
emlx_plugin_descriptor_v1() noexcept;
