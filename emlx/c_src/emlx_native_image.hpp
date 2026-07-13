#pragma once

#include "emlx_plugin_abi.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>

inline constexpr uint32_t EMLX_NATIVE_IMAGE_PATH_MAX = 4096;
inline constexpr uint32_t EMLX_NATIVE_IMAGE_ERROR_MAX = 512;

enum class EMLXNativeImageErrorCode : uint32_t {
  none,
  already_initialized,
  invalid_anchor,
  unsupported_platform,
  resolve_failed,
  path_too_long,
  canonicalize_failed,
  open_failed,
  not_regular_file,
  stat_failed,
  changed_while_hashing,
  hash_failed,
  build_mismatch,
  allocation_failed,
  internal_error
};

struct EMLXNativeImageError {
  EMLXNativeImageErrorCode code = EMLXNativeImageErrorCode::none;
  std::string detail;
};

struct EMLXNativeImageIdentity {
  std::string canonical_path;
  uint64_t device = 0;
  uint64_t inode = 0;
  uint64_t size = 0;
  std::array<uint8_t, 32> sha256{};
};

struct EMLXNativeFileSnapshot {
  uint64_t device = 0;
  uint64_t inode = 0;
  uint64_t size = 0;
  int64_t modification_seconds = 0;
  int64_t modification_nanoseconds = 0;

  bool operator==(const EMLXNativeFileSnapshot &) const = default;
};

struct EMLXHostRuntimeIdentity {
  EMLXMLXRuntimeAnchor mlx_runtime_anchor = nullptr;
  EMLXNativeImageIdentity mlx_runtime_image;
};

bool emlx_native_image_identity(EMLXMLXRuntimeAnchor anchor,
                                EMLXNativeImageIdentity &identity,
                                EMLXNativeImageError &error);

bool emlx_capture_host_runtime_identity(
    const std::array<uint8_t, 32> &expected_mlx_sha256,
    std::shared_ptr<const EMLXHostRuntimeIdentity> &candidate,
    EMLXNativeImageError &error);

bool emlx_publish_host_runtime_identity(
    std::shared_ptr<const EMLXHostRuntimeIdentity> candidate,
    EMLXNativeImageError &error);

std::shared_ptr<const EMLXHostRuntimeIdentity> emlx_host_runtime_identity();

const char *emlx_native_image_error_name(EMLXNativeImageErrorCode code);

#if defined(EMLX_NATIVE_IMAGE_TESTING)
using EMLXNativeImageTestBarrier = void (*)(const char *path, void *context);

void emlx_native_image_set_test_barrier(EMLXNativeImageTestBarrier barrier,
                                        void *context);
uint64_t emlx_native_image_test_identity_count();
uint64_t emlx_native_image_test_capture_count();
#endif
