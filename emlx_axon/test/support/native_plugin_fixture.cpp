#include "emlx/plugin/abi.hpp"

#include <cstring>
#include <stdexcept>

namespace {

#ifndef EMLX_FIXTURE_PLUGIN_NAME
#define EMLX_FIXTURE_PLUGIN_NAME "proof"
#endif

inline constexpr char kPluginName[] = EMLX_FIXTURE_PLUGIN_NAME;
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
inline constexpr char kOversizedOperandPolicy[] = "oversized_operand_policy";
inline constexpr char kOversizedOutputPolicy[] = "oversized_output_policy";
inline constexpr char kZeroOperandPolicy[] = "zero_operand_policy";
inline constexpr char kZeroOutputPolicy[] = "zero_output_policy";
inline constexpr char kDynamicCounts[] = "dynamic_counts";
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

std::optional<std::string>
scale_add(const EMLXPluginCall &call,
          std::vector<mlx::core::array> &outputs) {
  try {
    if (call.operands.size != 1 || call.attrs.size != 2 || !call.execution ||
        !call.execution->stream) {
      return "scale_add expects one operand and two attributes";
    }
    const auto &input = call.operands.data[0];
    auto scale = mlx::core::array(f64_from_bits(call.attrs.data[0]), input.dtype());
    auto bias = mlx::core::array(f64_from_bits(call.attrs.data[1]), input.dtype());
    outputs.push_back(mlx::core::add(
        mlx::core::multiply(input, scale, *call.execution->stream), bias,
        *call.execution->stream));
    return std::nullopt;
  } catch (const std::exception &exception) {
    return exception.what();
  } catch (...) {
    return "unknown proof callback failure";
  }
}

std::optional<std::string>
partial_failure(const EMLXPluginCall &call,
                std::vector<mlx::core::array> &outputs) {
  if (call.operands.size == 1)
    outputs.push_back(call.operands.data[0]);
  return "intentional partial failure";
}

std::optional<std::string>
wrong_shape(const EMLXPluginCall &call,
            std::vector<mlx::core::array> &outputs) {
  if (call.operands.size != 1 || !call.execution || !call.execution->stream) {
    return "wrong_shape expects one operand";
  }
  outputs.push_back(
      mlx::core::sum(call.operands.data[0], false, *call.execution->stream));
  return std::nullopt;
}

std::optional<std::string>
oversized_error(const EMLXPluginCall &,
                std::vector<mlx::core::array> &) {
  std::string error;
  error.assign(4080, 'a');
  for (size_t i = 0; i < 32; ++i)
    error.append("\xE2\x82\xAC");
  return error;
}

std::optional<std::string>
invalid_utf8_error(const EMLXPluginCall &,
                   std::vector<mlx::core::array> &) {
  return std::string("invalid byte: ") + static_cast<char>(0xff);
}

std::optional<std::string>
empty_error(const EMLXPluginCall &, std::vector<mlx::core::array> &) {
  return std::string{};
}

std::optional<std::string>
throw_after_output(const EMLXPluginCall &call,
                   std::vector<mlx::core::array> &outputs) {
  outputs.push_back(call.operands.data[0]);
  throw std::runtime_error("intentional callback exception");
}

std::optional<std::string>
unknown_throw_after_output(const EMLXPluginCall &call,
                           std::vector<mlx::core::array> &outputs) {
  outputs.push_back(call.operands.data[0]);
  throw 42;
}

std::optional<std::string>
wrong_output_count(const EMLXPluginCall &call,
                   std::vector<mlx::core::array> &outputs) {
  outputs.push_back(call.operands.data[0]);
  outputs.push_back(call.operands.data[0]);
  return std::nullopt;
}

bool throwing_operand_policy(EMLXPluginInt64View, uint32_t &,
                             std::string &) {
  throw std::runtime_error("intentional operand policy exception");
}

bool throwing_output_policy(EMLXPluginInt64View, uint32_t &,
                            std::string &) {
  throw std::runtime_error("intentional output policy exception");
}

bool oversized_operand_policy(EMLXPluginInt64View, uint32_t &count,
                               std::string &) {
  count = EMLX_PLUGIN_OPERAND_COUNT_MAX_V1 + 1;
  return true;
}

bool oversized_output_policy(EMLXPluginInt64View, uint32_t &count,
                              std::string &) {
  count = EMLX_PLUGIN_OUTPUT_COUNT_MAX_V1 + 1;
  return true;
}

bool zero_count_policy(EMLXPluginInt64View, uint32_t &count, std::string &) {
  count = 0;
  return true;
}

bool one_count_policy(EMLXPluginInt64View, uint32_t &count, std::string &) {
  count = 1;
  return true;
}

std::optional<std::string>
callback_must_not_run(const EMLXPluginCall &,
                      std::vector<mlx::core::array> &) {
  throw std::runtime_error("oversized policy callback ran");
}

constinit const EMLXPluginCallbackDescriptor kCallbacks[] = {
#if defined(EMLX_FIXTURE_NULL_CALLBACK)
    {string_view(kPrimaryCallbackName), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1, nullptr},
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
     scale_add},
#endif
    {string_view(kPartialFailureName), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1,
     partial_failure},
    {string_view(kWrongShape), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1, wrong_shape},
    {string_view(kCpuOnly), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1, scale_add},
    {string_view(kGpuOnly), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_GPU_METAL_V1, scale_add},
    {string_view(kThrowingOperandPolicy), 1, 1, 0, throwing_operand_policy, 1,
     nullptr, EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1,
     scale_add},
    {string_view(kThrowingOutputPolicy), 1, 1, 1, nullptr, 0,
     throwing_output_policy,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1, scale_add},
    {string_view(kOversizedError), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1,
     oversized_error},
    {string_view(kInvalidUtf8Error), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1,
     invalid_utf8_error},
    {string_view(kEmptyError), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1, empty_error},
    {string_view(kThrowAfterOutput), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1,
     throw_after_output},
    {string_view(kUnknownThrowAfterOutput), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1,
     unknown_throw_after_output},
    {string_view(kWrongOutputCount), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1,
     wrong_output_count},
    {string_view(kOversizedOperandPolicy), 1, 1, 0,
     oversized_operand_policy, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1,
     callback_must_not_run},
    {string_view(kOversizedOutputPolicy), 1, 1, 1, nullptr, 0,
     oversized_output_policy,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1,
     callback_must_not_run},
    {string_view(kZeroOperandPolicy), 1, 1, 0, zero_count_policy, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1,
     callback_must_not_run},
    {string_view(kZeroOutputPolicy), 1, 1, 1, nullptr, 0, zero_count_policy,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1,
     callback_must_not_run},
    {string_view(kDynamicCounts), 1, 1, 0, one_count_policy, 0,
     one_count_policy,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1, scale_add},
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
#if defined(EMLX_FIXTURE_BAD_DESCRIPTOR_INNER_SIZE)
    sizeof(EMLXPluginDescriptor) + 1,
#else
    sizeof(EMLXPluginDescriptor),
#endif
#if defined(EMLX_FIXTURE_BAD_CALLBACK_DESCRIPTOR_SIZE)
    sizeof(EMLXPluginCallbackDescriptor) + 1,
#else
    sizeof(EMLXPluginCallbackDescriptor),
#endif
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
