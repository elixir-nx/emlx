#pragma once

#define EMLX_STRINGIFY_INNER(value) #value
#define EMLX_STRINGIFY(value) EMLX_STRINGIFY_INNER(value)

#if defined(__clang__)
inline constexpr char EMLX_ACTUAL_COMPILER_FAMILY[] = "clang";
inline constexpr char EMLX_ACTUAL_COMPILER_MAJOR[] =
    EMLX_STRINGIFY(__clang_major__);
#elif defined(__GNUC__)
inline constexpr char EMLX_ACTUAL_COMPILER_FAMILY[] = "gcc";
inline constexpr char EMLX_ACTUAL_COMPILER_MAJOR[] = EMLX_STRINGIFY(__GNUC__);
#else
#error "plugin ABI v1 does not recognize this compiler ABI"
#endif

#if defined(__aarch64__) && defined(__APPLE__)
inline constexpr char EMLX_ACTUAL_TARGET_TRIPLE[] = "arm64-apple-darwin";
inline constexpr char EMLX_ACTUAL_ARCHITECTURE_ABI[] = "arm64";
#elif defined(__x86_64__) && defined(__linux__)
inline constexpr char EMLX_ACTUAL_TARGET_TRIPLE[] = "x86_64-unknown-linux-gnu";
inline constexpr char EMLX_ACTUAL_ARCHITECTURE_ABI[] = "x86_64-sysv";
#elif defined(__aarch64__) && defined(__linux__)
inline constexpr char EMLX_ACTUAL_TARGET_TRIPLE[] = "aarch64-unknown-linux-gnu";
inline constexpr char EMLX_ACTUAL_ARCHITECTURE_ABI[] = "aarch64-sysv";
#else
#error "plugin ABI v1 does not recognize this target ABI"
#endif

#if !defined(__cpp_constinit) || __cpp_constinit < 201907L
#error "plugin ABI v1 requires C++20 constinit support"
#endif
inline constexpr char EMLX_ACTUAL_CXX_LANGUAGE_MODE[] = "c++20";

#if defined(_LIBCPP_VERSION)
inline constexpr char EMLX_ACTUAL_CXX_STDLIB_FAMILY[] = "libc++";
#if defined(_LIBCPP_ABI_VERSION)
inline constexpr char EMLX_ACTUAL_CXX_STDLIB_ABI[] =
    "libc++-abi-" EMLX_STRINGIFY(_LIBCPP_ABI_VERSION);
#else
#error "plugin ABI v1 requires _LIBCPP_ABI_VERSION"
#endif
#elif defined(__GLIBCXX__)
inline constexpr char EMLX_ACTUAL_CXX_STDLIB_FAMILY[] = "libstdc++";
#if defined(_GLIBCXX_USE_CXX11_ABI)
inline constexpr char EMLX_ACTUAL_CXX_STDLIB_ABI[] =
    "libstdc++-cxx11-abi-" EMLX_STRINGIFY(_GLIBCXX_USE_CXX11_ABI);
#else
#error "plugin ABI v1 requires _GLIBCXX_USE_CXX11_ABI"
#endif
#else
#error "plugin ABI v1 does not recognize this C++ standard library"
#endif

#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
inline constexpr char EMLX_ACTUAL_EXCEPTIONS_MODE[] = "exceptions-on";
#else
inline constexpr char EMLX_ACTUAL_EXCEPTIONS_MODE[] = "exceptions-off";
#endif

#if defined(__GXX_RTTI)
inline constexpr char EMLX_ACTUAL_RTTI_MODE[] = "rtti-on";
#else
inline constexpr char EMLX_ACTUAL_RTTI_MODE[] = "rtti-off";
#endif

constexpr bool emlx_plugin_string_equal(const char *left, const char *right) {
  while (*left != '\0' && *right != '\0') {
    if (*left++ != *right++)
      return false;
  }
  return *left == *right;
}
