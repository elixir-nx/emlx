#include "emlx_plugin_abi.hpp"
#include "emlx_plugin_build_compat.hpp"
#include "emlx_plugin_toolchain.hpp"

#include <cstring>
#include <stdexcept>

namespace {

#ifndef EMLX_FIXTURE_PLUGIN_NAME
#define EMLX_FIXTURE_PLUGIN_NAME "proof"
#endif

inline constexpr char kPluginName[] = EMLX_FIXTURE_PLUGIN_NAME;
inline constexpr char kBuildId[] =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
inline constexpr char kScaleAdd[] = "scale_add";
inline constexpr char kPartialFailure[] = "partial_failure";
inline constexpr char kWrongShape[] = "wrong_shape";
inline constexpr char kCpuOnly[] = "cpu_only_scale_add";
inline constexpr char kGpuOnly[] = "gpu_only_scale_add";
inline constexpr char kThrowingOperandPolicy[] = "throwing_operand_policy";
inline constexpr char kThrowingOutputPolicy[] = "throwing_output_policy";
inline constexpr char kOversizedError[] = "oversized_error";
inline constexpr char kInvalidUtf8Error[] = "invalid_utf8_error";
inline constexpr char kEmptyError[] = "empty_error";
inline constexpr char kThrowAfterOutput[] = "throw_after_output";
inline constexpr char kUnknownThrowAfterOutput[] = "unknown_throw_after_output";
inline constexpr char kWrongOutputCount[] = "wrong_output_count";
#if defined(EMLX_FIXTURE_BAD_CALLBACK_NAME)
inline constexpr char kPrimaryCallbackName[] = "invalid/name";
#else
inline constexpr char kPrimaryCallbackName[] = "scale_add";
#endif
#if defined(EMLX_FIXTURE_DUPLICATE_CALLBACK)
inline constexpr char kPartialFailureName[] = "scale_add";
#else
inline constexpr char kPartialFailureName[] = "partial_failure";
#endif
#if defined(EMLX_FIXTURE_BAD_DEBUG_UTF8)
inline constexpr char kBadDebugName[] = "\xff";
#endif

template <size_t N>
constexpr EMLXPluginStringView string_view(const char (&value)[N]) {
  return {value, N - 1};
}

double f64_from_bits(int64_t bits) {
  uint64_t raw = static_cast<uint64_t>(bits);
  double value;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
}

bool scale_add(const EMLXPluginCall &call,
               std::vector<mlx::core::array> &outputs,
               std::string &error) {
  try {
    if (call.operands.size != 1 || call.attrs.size != 2 || !call.execution ||
        !call.execution->stream) {
      error = "scale_add expects one operand and two attributes";
      return false;
    }
    const auto &input = call.operands.data[0];
    auto scale = mlx::core::array(f64_from_bits(call.attrs.data[0]), input.dtype());
    auto bias = mlx::core::array(f64_from_bits(call.attrs.data[1]), input.dtype());
    outputs.push_back(mlx::core::add(
        mlx::core::multiply(input, scale, *call.execution->stream), bias,
        *call.execution->stream));
    return true;
  } catch (const std::exception &exception) {
    error = exception.what();
    return false;
  } catch (...) {
    error = "unknown proof callback failure";
    return false;
  }
}

bool partial_failure(const EMLXPluginCall &call,
                     std::vector<mlx::core::array> &outputs,
                     std::string &error) {
  if (call.operands.size == 1)
    outputs.push_back(call.operands.data[0]);
  error = "intentional partial failure";
  return false;
}

bool wrong_shape(const EMLXPluginCall &call,
                 std::vector<mlx::core::array> &outputs,
                 std::string &error) {
  if (call.operands.size != 1 || !call.execution || !call.execution->stream) {
    error = "wrong_shape expects one operand";
    return false;
  }
  outputs.push_back(
      mlx::core::sum(call.operands.data[0], false, *call.execution->stream));
  return true;
}

bool oversized_error(const EMLXPluginCall &,
                     std::vector<mlx::core::array> &, std::string &error) {
  error.assign(4080, 'a');
  for (size_t i = 0; i < 32; ++i)
    error.append("\xE2\x82\xAC");
  return false;
}

bool invalid_utf8_error(const EMLXPluginCall &,
                        std::vector<mlx::core::array> &,
                        std::string &error) {
  error = std::string("invalid byte: ") + static_cast<char>(0xff);
  return false;
}

bool empty_error(const EMLXPluginCall &, std::vector<mlx::core::array> &,
                 std::string &) {
  return false;
}

bool throw_after_output(const EMLXPluginCall &call,
                        std::vector<mlx::core::array> &outputs,
                        std::string &) {
  outputs.push_back(call.operands.data[0]);
  throw std::runtime_error("intentional callback exception");
}

bool unknown_throw_after_output(const EMLXPluginCall &call,
                                std::vector<mlx::core::array> &outputs,
                                std::string &) {
  outputs.push_back(call.operands.data[0]);
  throw 42;
}

bool wrong_output_count(const EMLXPluginCall &call,
                        std::vector<mlx::core::array> &outputs,
                        std::string &) {
  outputs.push_back(call.operands.data[0]);
  outputs.push_back(call.operands.data[0]);
  return true;
}

bool throwing_operand_policy(EMLXPluginInt64View, uint32_t &,
                             std::string &) {
  throw std::runtime_error("intentional operand policy exception");
}

bool throwing_output_policy(EMLXPluginInt64View, uint32_t &,
                            std::string &) {
  throw std::runtime_error("intentional output policy exception");
}

constinit const EMLXPluginCallbackDescriptor kCallbacks[] = {
#if defined(EMLX_FIXTURE_NULL_CALLBACK)
    {string_view(kPrimaryCallbackName), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1, nullptr,
     {nullptr, 0}},
#else
    {string_view(kPrimaryCallbackName),
#if defined(EMLX_FIXTURE_BAD_CALLBACK_SCHEMA)
     2,
#else
     1,
#endif
#if defined(EMLX_FIXTURE_BAD_ATTR_SCHEMA)
     2,
#else
     1,
#endif
     1,
#if defined(EMLX_FIXTURE_BAD_OPERAND_POLICY)
     throwing_operand_policy,
#else
     nullptr,
#endif
     1,
#if defined(EMLX_FIXTURE_BAD_OUTPUT_POLICY)
     throwing_output_policy,
#else
     nullptr,
#endif
#if defined(EMLX_FIXTURE_BAD_DEVICE)
     1U << 12,
#else
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1,
#endif
     scale_add,
#if defined(EMLX_FIXTURE_BAD_DEBUG_UTF8)
     string_view(kBadDebugName)},
#else
     {nullptr, 0}},
#endif
#endif
    {string_view(kPartialFailureName), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1,
     partial_failure, {nullptr, 0}},
    {string_view(kWrongShape), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1, wrong_shape,
     {nullptr, 0}},
    {string_view(kCpuOnly), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1, scale_add, {nullptr, 0}},
    {string_view(kGpuOnly), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_GPU_METAL_V1, scale_add, {nullptr, 0}},
    {string_view(kThrowingOperandPolicy), 1, 1, 0, throwing_operand_policy, 1,
     nullptr, EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1,
     scale_add, {nullptr, 0}},
    {string_view(kThrowingOutputPolicy), 1, 1, 1, nullptr, 0,
     throwing_output_policy,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1, scale_add,
     {nullptr, 0}},
    {string_view(kOversizedError), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1,
     oversized_error, {nullptr, 0}},
    {string_view(kInvalidUtf8Error), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1,
     invalid_utf8_error, {nullptr, 0}},
    {string_view(kEmptyError), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1, empty_error,
     {nullptr, 0}},
    {string_view(kThrowAfterOutput), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1,
     throw_after_output, {nullptr, 0}},
    {string_view(kUnknownThrowAfterOutput), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1,
     unknown_throw_after_output, {nullptr, 0}},
    {string_view(kWrongOutputCount), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1,
     wrong_output_count, {nullptr, 0}},
};

#if defined(EMLX_FIXTURE_MISALIGNED_CALLBACKS)
#define EMLX_FIXTURE_CALLBACKS_PTR                                             \
  reinterpret_cast<const EMLXPluginCallbackDescriptor *>(                    \
      reinterpret_cast<const char *>(kCallbacks) + 1)
#else
#define EMLX_FIXTURE_CALLBACKS_PTR kCallbacks
#endif

#if defined(EMLX_FIXTURE_MISALIGNED_CALLBACKS)
const EMLXPluginDescriptor kDescriptor{
#else
constinit const EMLXPluginDescriptor kDescriptor{
#endif
#if defined(EMLX_FIXTURE_NULL_PLUGIN_NAME)
    {nullptr, 5},
#else
    string_view(kPluginName),
#endif
    {
#if defined(EMLX_FIXTURE_BAD_DESCRIPTOR_ABI)
     2,
#else
     EMLX_PLUGIN_ABI_V1,
#endif
#if defined(EMLX_FIXTURE_BAD_DESCRIPTOR_HEADER_ABI)
     2,
#else
     EMLX_PLUGIN_HEADER_ABI_V1,
#endif
#if defined(EMLX_FIXTURE_BAD_DESCRIPTOR_HEADER_HASH)
     EMLX_PLUGIN_HEADER_ABI_HASH_V1 + 1,
#else
     EMLX_PLUGIN_HEADER_ABI_HASH_V1,
#endif
     string_view(EMLX_EXPECTED_MLX_VERSION),
     string_view(EMLX_EXPECTED_MLX_VARIANT),
     string_view(EMLX_EXPECTED_MLX_BUILD_ID),
     string_view(EMLX_EXPECTED_MLX_HEADERS_BUILD_ID),
     string_view(EMLX_EXPECTED_TARGET_TRIPLE),
     sizeof(void *) * 8,
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
     EMLX_ENDIAN_BIG,
#else
     EMLX_ENDIAN_LITTLE,
#endif
     string_view(EMLX_ACTUAL_COMPILER_FAMILY),
     string_view(EMLX_ACTUAL_CXX_STDLIB_ABI),
     sizeof(EMLXPluginDescriptor),
#if defined(EMLX_FIXTURE_BAD_CALLBACK_DESCRIPTOR_SIZE)
     sizeof(EMLXPluginCallbackDescriptor) + 1,
#else
     sizeof(EMLXPluginCallbackDescriptor),
#endif
     string_view(kBuildId)},
    static_cast<EMLXMLXRuntimeAnchor>(&mlx::core::version),
#if defined(EMLX_FIXTURE_TOO_MANY_CALLBACKS)
    257,
#else
    static_cast<uint32_t>(sizeof(kCallbacks) / sizeof(kCallbacks[0])),
#endif
#if defined(EMLX_FIXTURE_NULL_CALLBACKS)
    nullptr};
#else
    EMLX_FIXTURE_CALLBACKS_PTR};
#endif

#if defined(EMLX_FIXTURE_MISALIGNED_DESCRIPTOR)
const EMLXPluginBootstrapV1 kBootstrap{
#else
constinit const EMLXPluginBootstrapV1 kBootstrap{
#endif
#if defined(EMLX_FIXTURE_BAD_MAGIC)
    0,
#else
    EMLX_PLUGIN_MAGIC_V1,
#endif
#if defined(EMLX_FIXTURE_BAD_BOOTSTRAP_SIZE)
    sizeof(EMLXPluginBootstrapV1) + 1,
#else
    sizeof(EMLXPluginBootstrapV1),
#endif
#if defined(EMLX_FIXTURE_BAD_BOOTSTRAP_ABI)
    2,
#else
    EMLX_PLUGIN_ABI_V1,
#endif
#if defined(EMLX_FIXTURE_BAD_HEADER_HASH)
    EMLX_PLUGIN_HEADER_ABI_HASH_V1 + 1,
#else
    EMLX_PLUGIN_HEADER_ABI_HASH_V1,
#endif
#if defined(EMLX_FIXTURE_BAD_LAYOUT)
    EMLX_PLUGIN_LAYOUT_ABI_HASH_V1 + 1,
#else
    EMLX_PLUGIN_LAYOUT_ABI_HASH_V1,
#endif
#if defined(EMLX_FIXTURE_BAD_POINTER_WIDTH)
    sizeof(void *) * 8 + 1,
#else
    sizeof(void *) * 8,
#endif
#if defined(EMLX_FIXTURE_BAD_ENDIANNESS)
    99,
#else
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    EMLX_ENDIAN_BIG,
#else
    EMLX_ENDIAN_LITTLE,
#endif
#endif
#if defined(EMLX_FIXTURE_BAD_DESCRIPTOR_SIZE)
    sizeof(EMLXPluginDescriptor) + 1,
#else
    sizeof(EMLXPluginDescriptor),
#endif
#if defined(EMLX_FIXTURE_NULL_DESCRIPTOR)
    nullptr};
#elif defined(EMLX_FIXTURE_MISALIGNED_DESCRIPTOR)
    reinterpret_cast<const char *>(&kDescriptor) + 1};
#else
    &kDescriptor};
#endif

} // namespace

extern "C" EMLX_PLUGIN_EXPORT const EMLXPluginBootstrapV1 *
emlx_plugin_descriptor_v1() noexcept {
  return &kBootstrap;
}
