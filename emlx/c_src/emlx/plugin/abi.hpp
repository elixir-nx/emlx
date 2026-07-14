#pragma once

#include "mlx/mlx.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define EMLX_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#define EMLX_PLUGIN_EXPORT
#endif

inline constexpr uint64_t EMLX_PLUGIN_MAGIC_V1 = 0x454D4C58504C4731ULL;
inline constexpr uint32_t EMLX_PLUGIN_ABI_V1 = 1;
inline constexpr uint32_t EMLX_PLUGIN_DEVICE_CPU_V1 = 1U << 0;
inline constexpr uint32_t EMLX_PLUGIN_DEVICE_GPU_METAL_V1 = 1U << 1;
inline constexpr uint32_t EMLX_PLUGIN_DEVICE_KNOWN_V1 =
    EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1;
inline constexpr uint32_t EMLX_PLUGIN_OPERAND_COUNT_MAX_V1 = 8192;
inline constexpr uint32_t EMLX_PLUGIN_OUTPUT_COUNT_MAX_V1 = 1024;

// Plugin-owned views are borrowed by EMLX only while loading the descriptor.
// Call-owned views remain valid only for the count-policy or callback
// invocation that receives them. Neither side may retain borrowed pointers.
template <typename T> struct EMLXPluginView {
  const T *data;
  uint64_t size;
};

struct EMLXPluginStringView {
  const char *data;
  uint64_t size;
};

using EMLXPluginArrayView = EMLXPluginView<mlx::core::array>;
using EMLXPluginInt64View = EMLXPluginView<int64_t>;

struct EMLXPluginExecutionContext {
  const mlx::core::Device *device;
  const mlx::core::Stream *stream;
};

struct EMLXPluginCall {
  EMLXPluginArrayView operands;
  EMLXPluginInt64View attrs;
  const EMLXPluginExecutionContext *execution;
};

// Policies and callbacks may run concurrently on EMLX worker threads and must
// therefore be reentrant. Expected failures return false and populate error;
// exceptions must not cross the plugin boundary.
using EMLXOutputCountFn = bool (*)(EMLXPluginInt64View, uint32_t &,
                                   std::string &);
using EMLXOperandCountFn = bool (*)(EMLXPluginInt64View, uint32_t &,
                                    std::string &);
using EMLXPluginCallback = bool (*)(const EMLXPluginCall &,
                                    std::vector<mlx::core::array> &,
                                    std::string &);

struct EMLXPluginCallbackDescriptor {
  EMLXPluginStringView name;
  uint32_t schema_version;
  uint32_t attr_schema_version;
  uint32_t operand_count;
  EMLXOperandCountFn operand_count_from_attrs;
  uint32_t output_count;
  EMLXOutputCountFn output_count_from_attrs;
  uint32_t device_capabilities;
  EMLXPluginCallback callback;
};

struct EMLXPluginDescriptor {
  // Strings, the callback table, policy functions, and callbacks must remain
  // valid for the lifetime of the VM process.
  EMLXPluginStringView name;
  uint64_t descriptor_size;
  uint64_t callback_descriptor_size;
  uint32_t callback_count;
  const EMLXPluginCallbackDescriptor *callbacks;
};

struct EMLXPluginBootstrapV1 {
  uint64_t magic;
  uint32_t bootstrap_size;
  uint32_t plugin_abi_version;
  uint64_t descriptor_size;
  const void *descriptor;
};

using EMLXPluginDiscoveryV1 = const EMLXPluginBootstrapV1 *(*)() noexcept;

static_assert(std::is_standard_layout_v<EMLXPluginBootstrapV1>);
static_assert(std::is_trivially_copyable_v<EMLXPluginBootstrapV1>);
static_assert(sizeof(EMLXPluginBootstrapV1) == 32);
static_assert(alignof(EMLXPluginBootstrapV1) == 8);
static_assert(offsetof(EMLXPluginBootstrapV1, magic) == 0);
static_assert(offsetof(EMLXPluginBootstrapV1, bootstrap_size) == 8);
static_assert(offsetof(EMLXPluginBootstrapV1, plugin_abi_version) == 12);
static_assert(offsetof(EMLXPluginBootstrapV1, descriptor_size) == 16);
static_assert(offsetof(EMLXPluginBootstrapV1, descriptor) == 24);

static_assert(std::is_standard_layout_v<EMLXPluginStringView>);
static_assert(std::is_trivially_copyable_v<EMLXPluginStringView>);
static_assert(sizeof(EMLXPluginStringView) == 16);
static_assert(alignof(EMLXPluginStringView) == 8);

static_assert(std::is_standard_layout_v<EMLXPluginArrayView>);
static_assert(std::is_trivially_copyable_v<EMLXPluginArrayView>);
static_assert(sizeof(EMLXPluginArrayView) == 16);

static_assert(std::is_standard_layout_v<EMLXPluginInt64View>);
static_assert(std::is_trivially_copyable_v<EMLXPluginInt64View>);
static_assert(sizeof(EMLXPluginInt64View) == 16);

static_assert(std::is_standard_layout_v<EMLXPluginExecutionContext>);
static_assert(std::is_trivially_copyable_v<EMLXPluginExecutionContext>);
static_assert(sizeof(EMLXPluginExecutionContext) == 16);

static_assert(std::is_standard_layout_v<EMLXPluginCall>);
static_assert(std::is_trivially_copyable_v<EMLXPluginCall>);
static_assert(sizeof(EMLXPluginCall) == 40);

static_assert(std::is_standard_layout_v<EMLXPluginCallbackDescriptor>);
static_assert(std::is_trivially_copyable_v<EMLXPluginCallbackDescriptor>);
static_assert(sizeof(EMLXPluginCallbackDescriptor) == 72);
static_assert(alignof(EMLXPluginCallbackDescriptor) == 8);

static_assert(std::is_standard_layout_v<EMLXPluginDescriptor>);
static_assert(std::is_trivially_copyable_v<EMLXPluginDescriptor>);
static_assert(sizeof(EMLXPluginDescriptor) == 48);
static_assert(alignof(EMLXPluginDescriptor) == 8);
static_assert(offsetof(EMLXPluginDescriptor, name) == 0);
static_assert(offsetof(EMLXPluginDescriptor, descriptor_size) == 16);
static_assert(offsetof(EMLXPluginDescriptor, callback_descriptor_size) == 24);
static_assert(offsetof(EMLXPluginDescriptor, callback_count) == 32);
static_assert(offsetof(EMLXPluginDescriptor, callbacks) == 40);

extern "C" EMLX_PLUGIN_EXPORT const EMLXPluginBootstrapV1 *
emlx_plugin_descriptor_v1() noexcept;
