#include "emlx_plugin_abi.hpp"
#include "emlx_plugin_build_compat.hpp"
#include "emlx_plugin_toolchain.hpp"

#include <cstring>

namespace {

inline constexpr char kPluginName[] = "proof";
inline constexpr char kBuildId[] =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
inline constexpr char kScaleAdd[] = "scale_add";
inline constexpr char kPartialFailure[] = "partial_failure";
inline constexpr char kWrongShape[] = "wrong_shape";
inline constexpr char kCpuOnly[] = "cpu_only_scale_add";
inline constexpr char kGpuOnly[] = "gpu_only_scale_add";

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

constinit const EMLXPluginCallbackDescriptor kCallbacks[] = {
    {string_view(kScaleAdd), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1, scale_add,
     {nullptr, 0}},
    {string_view(kPartialFailure), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1,
     partial_failure, {nullptr, 0}},
    {string_view(kWrongShape), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1 | EMLX_PLUGIN_DEVICE_GPU_METAL_V1, wrong_shape,
     {nullptr, 0}},
    {string_view(kCpuOnly), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_CPU_V1, scale_add, {nullptr, 0}},
    {string_view(kGpuOnly), 1, 1, 1, nullptr, 1, nullptr,
     EMLX_PLUGIN_DEVICE_GPU_METAL_V1, scale_add, {nullptr, 0}},
};

constinit const EMLXPluginDescriptor kDescriptor{
    string_view(kPluginName),
    {EMLX_PLUGIN_ABI_V1,
     EMLX_PLUGIN_HEADER_ABI_V1,
     EMLX_PLUGIN_HEADER_ABI_HASH_V1,
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
     sizeof(EMLXPluginCallbackDescriptor),
     string_view(kBuildId)},
    static_cast<EMLXMLXRuntimeAnchor>(&mlx::core::version),
    static_cast<uint32_t>(sizeof(kCallbacks) / sizeof(kCallbacks[0])),
    kCallbacks};

constinit const EMLXPluginBootstrapV1 kBootstrap{
    EMLX_PLUGIN_MAGIC_V1,
    sizeof(EMLXPluginBootstrapV1),
    EMLX_PLUGIN_ABI_V1,
    EMLX_PLUGIN_HEADER_ABI_HASH_V1,
    EMLX_PLUGIN_LAYOUT_ABI_HASH_V1,
    sizeof(void *) * 8,
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    EMLX_ENDIAN_BIG,
#else
    EMLX_ENDIAN_LITTLE,
#endif
    sizeof(EMLXPluginDescriptor),
    &kDescriptor};

} // namespace

extern "C" EMLX_PLUGIN_EXPORT const EMLXPluginBootstrapV1 *
emlx_plugin_descriptor_v1() noexcept {
  return &kBootstrap;
}
