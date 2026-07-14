#include "emlx/native_image.hpp"
#include "emlx/sha256.hpp"

#include "mlx/version.h"

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>

namespace {

std::mutex g_host_identity_mutex;
std::shared_ptr<const EMLXHostRuntimeIdentity> g_host_identity;

#if defined(EMLX_NATIVE_IMAGE_TESTING)
EMLXNativeImageTestBarrier g_test_barrier = nullptr;
void *g_test_barrier_context = nullptr;
std::atomic<uint64_t> g_test_identity_count{0};
std::atomic<uint64_t> g_test_capture_count{0};
#endif

void set_error(EMLXNativeImageError &error, EMLXNativeImageErrorCode code,
               const std::string &detail) {
  error.code = code;
  error.detail = detail.substr(0, EMLX_NATIVE_IMAGE_ERROR_MAX);
}

template <typename Source>
bool checked_unsigned(Source value, uint64_t &output) {
  if constexpr (std::numeric_limits<Source>::is_signed) {
    if (value < 0)
      return false;
  }
  using Unsigned = std::make_unsigned_t<Source>;
  const auto converted = static_cast<Unsigned>(value);
  if constexpr (sizeof(Unsigned) > sizeof(uint64_t)) {
    if (converted > std::numeric_limits<uint64_t>::max())
      return false;
  }
  output = static_cast<uint64_t>(converted);
  return true;
}

template <typename Source>
bool checked_signed(Source value, int64_t &output) {
  if constexpr (std::numeric_limits<Source>::is_signed) {
    if constexpr (sizeof(Source) > sizeof(int64_t)) {
      if (value < std::numeric_limits<int64_t>::min() ||
          value > std::numeric_limits<int64_t>::max())
        return false;
    }
  } else if (value > static_cast<std::make_unsigned_t<int64_t>>(
                         std::numeric_limits<int64_t>::max())) {
    return false;
  }
  output = static_cast<int64_t>(value);
  return true;
}

bool snapshot_from_stat(const struct stat &value,
                        EMLXNativeFileSnapshot &snapshot,
                        EMLXNativeImageError &error) {
  if (!checked_unsigned(value.st_dev, snapshot.device) ||
      !checked_unsigned(value.st_ino, snapshot.inode) ||
      !checked_unsigned(value.st_size, snapshot.size)) {
    set_error(error, EMLXNativeImageErrorCode::stat_failed,
              "native image file metadata is outside supported ranges");
    return false;
  }
#if defined(__APPLE__)
  if (!checked_signed(value.st_mtimespec.tv_sec,
                      snapshot.modification_seconds)) {
    set_error(error, EMLXNativeImageErrorCode::stat_failed,
              "native image timestamp is outside supported ranges");
    return false;
  }
  snapshot.modification_nanoseconds = value.st_mtimespec.tv_nsec;
#elif defined(__linux__)
  if (!checked_signed(value.st_mtim.tv_sec, snapshot.modification_seconds)) {
    set_error(error, EMLXNativeImageErrorCode::stat_failed,
              "native image timestamp is outside supported ranges");
    return false;
  }
  snapshot.modification_nanoseconds = value.st_mtim.tv_nsec;
#else
  set_error(error, EMLXNativeImageErrorCode::unsupported_platform,
            "native image identity is unsupported on this platform");
  return false;
#endif
  if (snapshot.modification_nanoseconds < 0 ||
      snapshot.modification_nanoseconds > 999999999) {
    set_error(error, EMLXNativeImageErrorCode::stat_failed,
              "native image nanoseconds are outside the valid range");
    return false;
  }
  return true;
}

class FileDescriptor {
public:
  explicit FileDescriptor(int value) : value_(value) {}
  ~FileDescriptor() {
    if (value_ >= 0)
      close(value_);
  }
  int get() const { return value_; }

private:
  int value_;
};

} // namespace

#if defined(EMLX_NATIVE_IMAGE_TESTING)
void emlx_native_image_set_test_barrier(EMLXNativeImageTestBarrier barrier,
                                        void *context) {
  g_test_barrier = barrier;
  g_test_barrier_context = context;
}

uint64_t emlx_native_image_test_identity_count() {
  return g_test_identity_count.load();
}

uint64_t emlx_native_image_test_capture_count() {
  return g_test_capture_count.load();
}
#endif

const char *emlx_native_image_error_name(EMLXNativeImageErrorCode code) {
  switch (code) {
  case EMLXNativeImageErrorCode::none:
    return "none";
  case EMLXNativeImageErrorCode::already_initialized:
    return "already_initialized";
  case EMLXNativeImageErrorCode::invalid_anchor:
    return "invalid_anchor";
  case EMLXNativeImageErrorCode::unsupported_platform:
    return "unsupported_platform";
  case EMLXNativeImageErrorCode::resolve_failed:
    return "resolve_failed";
  case EMLXNativeImageErrorCode::path_too_long:
    return "path_too_long";
  case EMLXNativeImageErrorCode::canonicalize_failed:
    return "canonicalize_failed";
  case EMLXNativeImageErrorCode::open_failed:
    return "open_failed";
  case EMLXNativeImageErrorCode::not_regular_file:
    return "not_regular_file";
  case EMLXNativeImageErrorCode::stat_failed:
    return "stat_failed";
  case EMLXNativeImageErrorCode::changed_while_hashing:
    return "changed_while_hashing";
  case EMLXNativeImageErrorCode::hash_failed:
    return "hash_failed";
  case EMLXNativeImageErrorCode::build_mismatch:
    return "build_mismatch";
  case EMLXNativeImageErrorCode::allocation_failed:
    return "allocation_failed";
  case EMLXNativeImageErrorCode::internal_error:
    return "internal_error";
  }
  return "internal_error";
}

bool emlx_native_image_identity(EMLXMLXRuntimeAnchor anchor,
                                EMLXNativeImageIdentity &identity,
                                EMLXNativeImageError &error) {
#if defined(EMLX_NATIVE_IMAGE_TESTING)
  ++g_test_identity_count;
#endif
#if !defined(__APPLE__) && !defined(__linux__)
  set_error(error, EMLXNativeImageErrorCode::unsupported_platform,
            "native image identity is unsupported on this platform");
  return false;
#else
  if (!anchor) {
    set_error(error, EMLXNativeImageErrorCode::invalid_anchor,
              "native image anchor is null");
    return false;
  }
  Dl_info info{};
  if (dladdr(reinterpret_cast<const void *>(anchor), &info) == 0 ||
      !info.dli_fname) {
    set_error(error, EMLXNativeImageErrorCode::resolve_failed,
              "could not resolve the native image anchor");
    return false;
  }
  const size_t path_size = std::strlen(info.dli_fname);
  if (path_size == 0 || path_size >= EMLX_NATIVE_IMAGE_PATH_MAX) {
    set_error(error, EMLXNativeImageErrorCode::path_too_long,
              "resolved native image path is missing or too long");
    return false;
  }
  char *resolved = realpath(info.dli_fname, nullptr);
  if (!resolved) {
    set_error(error, EMLXNativeImageErrorCode::canonicalize_failed,
              std::strerror(errno));
    return false;
  }
  std::string path(resolved);
  std::free(resolved);
  if (path.size() >= EMLX_NATIVE_IMAGE_PATH_MAX) {
    set_error(error, EMLXNativeImageErrorCode::path_too_long,
              "canonical native image path is too long");
    return false;
  }

  FileDescriptor fd(open(path.c_str(), O_RDONLY | O_CLOEXEC));
  if (fd.get() < 0) {
    set_error(error, EMLXNativeImageErrorCode::open_failed,
              std::strerror(errno));
    return false;
  }
  struct stat before_raw {};
  if (fstat(fd.get(), &before_raw) != 0) {
    set_error(error, EMLXNativeImageErrorCode::stat_failed,
              std::strerror(errno));
    return false;
  }
  if (!S_ISREG(before_raw.st_mode)) {
    set_error(error, EMLXNativeImageErrorCode::not_regular_file,
              "native image is not a regular file");
    return false;
  }
  EMLXNativeFileSnapshot before;
  if (!snapshot_from_stat(before_raw, before, error))
    return false;

#if defined(EMLX_NATIVE_IMAGE_TESTING)
  if (g_test_barrier)
    g_test_barrier(path.c_str(), g_test_barrier_context);
#endif

  std::array<uint8_t, 32> digest{};
  std::string hash_error;
  if (!emlx_sha256_file_descriptor(fd.get(), digest, hash_error)) {
    set_error(error, EMLXNativeImageErrorCode::hash_failed, hash_error);
    return false;
  }
  struct stat after_raw {};
  if (fstat(fd.get(), &after_raw) != 0) {
    set_error(error, EMLXNativeImageErrorCode::stat_failed,
              std::strerror(errno));
    return false;
  }
  EMLXNativeFileSnapshot after;
  if (!snapshot_from_stat(after_raw, after, error))
    return false;
  if (!(before == after)) {
    set_error(error, EMLXNativeImageErrorCode::changed_while_hashing,
              "native image changed while hashing");
    return false;
  }

  identity = {std::move(path), before.device, before.inode, before.size, digest};
  return true;
#endif
}

bool emlx_capture_host_runtime_identity(
    const std::array<uint8_t, 32> &expected_mlx_sha256,
    std::shared_ptr<const EMLXHostRuntimeIdentity> &candidate,
    EMLXNativeImageError &error) {
#if defined(EMLX_NATIVE_IMAGE_TESTING)
  ++g_test_capture_count;
#endif
  {
    std::lock_guard lock(g_host_identity_mutex);
    if (g_host_identity) {
      set_error(error, EMLXNativeImageErrorCode::already_initialized,
                "host runtime identity is already initialized");
      return false;
    }
  }
  auto identity = std::make_shared<EMLXHostRuntimeIdentity>();
  identity->mlx_runtime_anchor =
      static_cast<EMLXMLXRuntimeAnchor>(&mlx::core::version);
  if (!emlx_native_image_identity(identity->mlx_runtime_anchor,
                                  identity->mlx_runtime_image, error))
    return false;
  if (identity->mlx_runtime_image.sha256 != expected_mlx_sha256) {
    set_error(error, EMLXNativeImageErrorCode::build_mismatch,
              "loaded MLX image does not match the expected build identity");
    return false;
  }
  candidate = std::move(identity);
  return true;
}

bool emlx_publish_host_runtime_identity(
    std::shared_ptr<const EMLXHostRuntimeIdentity> candidate,
    EMLXNativeImageError &error) {
  if (!candidate) {
    set_error(error, EMLXNativeImageErrorCode::internal_error,
              "host runtime identity candidate is null");
    return false;
  }
  std::lock_guard lock(g_host_identity_mutex);
  if (g_host_identity) {
    set_error(error, EMLXNativeImageErrorCode::already_initialized,
              "host runtime identity is already initialized");
    return false;
  }
  g_host_identity = std::move(candidate);
  return true;
}

std::shared_ptr<const EMLXHostRuntimeIdentity> emlx_host_runtime_identity() {
  std::lock_guard lock(g_host_identity_mutex);
  return g_host_identity;
}
