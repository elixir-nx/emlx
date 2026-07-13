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
inline constexpr uint32_t EMLX_PLUGIN_HEADER_ABI_V1 = 1;
inline constexpr uint32_t EMLX_ENDIAN_LITTLE = 1;
inline constexpr uint32_t EMLX_ENDIAN_BIG = 2;
inline constexpr uint32_t EMLX_PLUGIN_DEVICE_CPU_V1 = 1U << 0;
inline constexpr uint32_t EMLX_PLUGIN_DEVICE_GPU_METAL_V1 = 1U << 1;
inline constexpr uint32_t EMLX_PLUGIN_DEVICE_KNOWN_V1 =
    EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1;

using EMLXMLXRuntimeAnchor = const char *(*)();

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
  EMLXPluginStringView debug_name;
};

struct EMLXPluginCompatibility {
  uint32_t plugin_abi_version;
  uint32_t header_abi_version;
  uint64_t header_abi_hash;
  EMLXPluginStringView mlx_version;
  EMLXPluginStringView mlx_variant;
  EMLXPluginStringView mlx_build_id;
  EMLXPluginStringView mlx_headers_build_id;
  EMLXPluginStringView target_triple;
  uint32_t pointer_width_bits;
  uint32_t endianness;
  EMLXPluginStringView compiler_abi_family;
  EMLXPluginStringView cxx_standard_library_abi;
  uint64_t plugin_descriptor_size;
  uint64_t callback_descriptor_size;
  EMLXPluginStringView plugin_build_id;
};

struct EMLXPluginDescriptor {
  EMLXPluginStringView name;
  EMLXPluginCompatibility compatibility;
  EMLXMLXRuntimeAnchor mlx_runtime_anchor;
  uint32_t callback_count;
  const EMLXPluginCallbackDescriptor *callbacks;
};

struct EMLXPluginBootstrapV1 {
  uint64_t magic;
  uint32_t bootstrap_size;
  uint32_t plugin_abi_version;
  uint64_t header_abi_hash;
  uint64_t layout_abi_hash;
  uint32_t pointer_width_bits;
  uint32_t endianness;
  uint64_t descriptor_size;
  const void *descriptor;
};

using EMLXPluginDiscoveryV1 = const EMLXPluginBootstrapV1 *(*)() noexcept;

constexpr uint64_t emlx_plugin_fnv1a_byte(uint64_t hash, uint8_t value) {
  return (hash ^ value) * 1099511628211ULL;
}

constexpr uint64_t emlx_plugin_fnv1a_string(const char *value, size_t size) {
  uint64_t hash = 14695981039346656037ULL;
  for (size_t i = 0; i < size; ++i)
    hash = emlx_plugin_fnv1a_byte(hash, static_cast<uint8_t>(value[i]));
  return hash;
}

template <typename UInt>
constexpr uint64_t emlx_plugin_fnv1a_le(uint64_t hash, UInt value) {
  for (size_t i = 0; i < sizeof(UInt); ++i) {
    hash = emlx_plugin_fnv1a_byte(hash, static_cast<uint8_t>(value & 0xffU));
    value >>= 8U;
  }
  return hash;
}

inline constexpr char EMLX_PLUGIN_ABI_SIGNATURE_V1[] =
    "EMLXPluginBootstrapV1{u64 magic,u32 bootstrap_size,u32 plugin_abi_version,"
    "u64 header_abi_hash,u64 layout_abi_hash,u32 pointer_width_bits,u32 "
    "endianness,u64 descriptor_size,const void* descriptor};"
    "EMLXPluginStringView{const char* data,u64 size};"
    "EMLXPluginArrayView{const mlx::core::array* data,u64 size};"
    "EMLXPluginInt64View{const i64* data,u64 size};"
    "EMLXPluginExecutionContext{const mlx::core::Device* device,const "
    "mlx::core::Stream* stream};"
    "EMLXPluginCall{EMLXPluginArrayView operands,EMLXPluginInt64View attrs,"
    "const EMLXPluginExecutionContext* execution};"
    "EMLXPluginCallbackDescriptor{name,schema_version,attr_schema_version,"
    "operand_count,operand_count_from_attrs,output_count,"
    "output_count_from_attrs,device_capabilities,callback,debug_name};"
    "EMLXPluginCompatibility{plugin_abi_version,header_abi_version,"
    "header_abi_hash,mlx_version,mlx_variant,mlx_build_id,mlx_headers_build_id,"
    "target_triple,pointer_width_bits,endianness,compiler_abi_family,"
    "cxx_standard_library_abi,plugin_descriptor_size,callback_descriptor_size,"
    "plugin_build_id};"
    "EMLXPluginDescriptor{name,compatibility,mlx_runtime_anchor,callback_count,"
    "callbacks};"
    "EMLXPluginDiscoveryV1=const EMLXPluginBootstrapV1*(*)() noexcept";

inline constexpr uint64_t EMLX_PLUGIN_HEADER_ABI_HASH_V1 =
    emlx_plugin_fnv1a_string(EMLX_PLUGIN_ABI_SIGNATURE_V1,
                             sizeof(EMLX_PLUGIN_ABI_SIGNATURE_V1) - 1);

constexpr uint64_t emlx_plugin_layout_record(uint64_t hash, uint32_t id,
                                             uint64_t size, uint64_t alignment,
                                             const uint64_t *offsets,
                                             uint32_t count) {
  hash = emlx_plugin_fnv1a_le(hash, id);
  hash = emlx_plugin_fnv1a_le(hash, size);
  hash = emlx_plugin_fnv1a_le(hash, alignment);
  hash = emlx_plugin_fnv1a_le(hash, count);
  for (uint32_t i = 0; i < count; ++i)
    hash = emlx_plugin_fnv1a_le(hash, offsets[i]);
  return hash;
}

constexpr uint64_t emlx_plugin_layout_hash_v1() {
  uint64_t hash = 14695981039346656037ULL;
  constexpr uint64_t bootstrap[] = {
      offsetof(EMLXPluginBootstrapV1, magic),
      offsetof(EMLXPluginBootstrapV1, bootstrap_size),
      offsetof(EMLXPluginBootstrapV1, plugin_abi_version),
      offsetof(EMLXPluginBootstrapV1, header_abi_hash),
      offsetof(EMLXPluginBootstrapV1, layout_abi_hash),
      offsetof(EMLXPluginBootstrapV1, pointer_width_bits),
      offsetof(EMLXPluginBootstrapV1, endianness),
      offsetof(EMLXPluginBootstrapV1, descriptor_size),
      offsetof(EMLXPluginBootstrapV1, descriptor)};
  constexpr uint64_t string_view[] = {offsetof(EMLXPluginStringView, data),
                                      offsetof(EMLXPluginStringView, size)};
  constexpr uint64_t array_view[] = {offsetof(EMLXPluginArrayView, data),
                                     offsetof(EMLXPluginArrayView, size)};
  constexpr uint64_t int64_view[] = {offsetof(EMLXPluginInt64View, data),
                                     offsetof(EMLXPluginInt64View, size)};
  constexpr uint64_t execution[] = {
      offsetof(EMLXPluginExecutionContext, device),
      offsetof(EMLXPluginExecutionContext, stream)};
  constexpr uint64_t call[] = {offsetof(EMLXPluginCall, operands),
                               offsetof(EMLXPluginCall, attrs),
                               offsetof(EMLXPluginCall, execution)};
  constexpr uint64_t callback[] = {
      offsetof(EMLXPluginCallbackDescriptor, name),
      offsetof(EMLXPluginCallbackDescriptor, schema_version),
      offsetof(EMLXPluginCallbackDescriptor, attr_schema_version),
      offsetof(EMLXPluginCallbackDescriptor, operand_count),
      offsetof(EMLXPluginCallbackDescriptor, operand_count_from_attrs),
      offsetof(EMLXPluginCallbackDescriptor, output_count),
      offsetof(EMLXPluginCallbackDescriptor, output_count_from_attrs),
      offsetof(EMLXPluginCallbackDescriptor, device_capabilities),
      offsetof(EMLXPluginCallbackDescriptor, callback),
      offsetof(EMLXPluginCallbackDescriptor, debug_name)};
  constexpr uint64_t compatibility[] = {
      offsetof(EMLXPluginCompatibility, plugin_abi_version),
      offsetof(EMLXPluginCompatibility, header_abi_version),
      offsetof(EMLXPluginCompatibility, header_abi_hash),
      offsetof(EMLXPluginCompatibility, mlx_version),
      offsetof(EMLXPluginCompatibility, mlx_variant),
      offsetof(EMLXPluginCompatibility, mlx_build_id),
      offsetof(EMLXPluginCompatibility, mlx_headers_build_id),
      offsetof(EMLXPluginCompatibility, target_triple),
      offsetof(EMLXPluginCompatibility, pointer_width_bits),
      offsetof(EMLXPluginCompatibility, endianness),
      offsetof(EMLXPluginCompatibility, compiler_abi_family),
      offsetof(EMLXPluginCompatibility, cxx_standard_library_abi),
      offsetof(EMLXPluginCompatibility, plugin_descriptor_size),
      offsetof(EMLXPluginCompatibility, callback_descriptor_size),
      offsetof(EMLXPluginCompatibility, plugin_build_id)};
  constexpr uint64_t descriptor[] = {
      offsetof(EMLXPluginDescriptor, name),
      offsetof(EMLXPluginDescriptor, compatibility),
      offsetof(EMLXPluginDescriptor, mlx_runtime_anchor),
      offsetof(EMLXPluginDescriptor, callback_count),
      offsetof(EMLXPluginDescriptor, callbacks)};

  hash = emlx_plugin_layout_record(hash, 1, sizeof(EMLXPluginBootstrapV1),
                                   alignof(EMLXPluginBootstrapV1), bootstrap, 9);
  hash = emlx_plugin_layout_record(hash, 2, sizeof(EMLXPluginStringView),
                                   alignof(EMLXPluginStringView), string_view, 2);
  hash = emlx_plugin_layout_record(hash, 3, sizeof(EMLXPluginArrayView),
                                   alignof(EMLXPluginArrayView), array_view, 2);
  hash = emlx_plugin_layout_record(hash, 4, sizeof(EMLXPluginInt64View),
                                   alignof(EMLXPluginInt64View), int64_view, 2);
  hash = emlx_plugin_layout_record(hash, 5, sizeof(EMLXPluginExecutionContext),
                                   alignof(EMLXPluginExecutionContext), execution, 2);
  hash = emlx_plugin_layout_record(hash, 6, sizeof(EMLXPluginCall),
                                   alignof(EMLXPluginCall), call, 3);
  hash = emlx_plugin_layout_record(hash, 7, sizeof(EMLXPluginCallbackDescriptor),
                                   alignof(EMLXPluginCallbackDescriptor), callback, 10);
  hash = emlx_plugin_layout_record(hash, 8, sizeof(EMLXPluginCompatibility),
                                   alignof(EMLXPluginCompatibility), compatibility, 15);
  return emlx_plugin_layout_record(hash, 9, sizeof(EMLXPluginDescriptor),
                                   alignof(EMLXPluginDescriptor), descriptor, 5);
}

inline constexpr uint64_t EMLX_PLUGIN_LAYOUT_ABI_HASH_V1 =
    emlx_plugin_layout_hash_v1();

static_assert(std::is_standard_layout_v<EMLXPluginBootstrapV1>);
static_assert(std::is_trivially_copyable_v<EMLXPluginBootstrapV1>);
static_assert(sizeof(EMLXPluginBootstrapV1) == 56);
static_assert(alignof(EMLXPluginBootstrapV1) == 8);
static_assert(offsetof(EMLXPluginBootstrapV1, descriptor) == 48);
static_assert(std::is_standard_layout_v<EMLXPluginStringView>);
static_assert(std::is_trivially_copyable_v<EMLXPluginStringView>);
static_assert(sizeof(EMLXPluginStringView) == 16);
static_assert(alignof(EMLXPluginStringView) == 8);
static_assert(std::is_standard_layout_v<EMLXPluginArrayView>);
static_assert(std::is_standard_layout_v<EMLXPluginInt64View>);
static_assert(std::is_standard_layout_v<EMLXPluginExecutionContext>);
static_assert(std::is_standard_layout_v<EMLXPluginCall>);
static_assert(std::is_standard_layout_v<EMLXPluginCallbackDescriptor>);
static_assert(std::is_standard_layout_v<EMLXPluginCompatibility>);
static_assert(std::is_standard_layout_v<EMLXPluginDescriptor>);

extern "C" EMLX_PLUGIN_EXPORT const EMLXPluginBootstrapV1 *
emlx_plugin_descriptor_v1() noexcept;
