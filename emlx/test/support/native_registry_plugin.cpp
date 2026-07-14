#include "emlx/plugin/abi.hpp"
#include "emlx/plugin/build_compat.hpp"
#include "emlx/plugin/toolchain.hpp"

namespace {

inline constexpr char kPluginName[] = "lifecycle-proof";
inline constexpr char kBuildId[] =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

template <size_t N>
constexpr EMLXPluginStringView string_view(const char (&value)[N]) {
  return {value, N - 1};
}

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
    0,
    nullptr};

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
