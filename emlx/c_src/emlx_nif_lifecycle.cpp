#include "emlx_nif_lifecycle.hpp"

#include "emlx_native_image.hpp"
#include "emlx_sha256.hpp"

#include <array>
#include <cstdio>
#include <memory>
#include <new>
#include <stdexcept>

namespace {

int fail_load(const char *reason) noexcept {
  std::fputs("EMLX NIF load failed: ", stderr);
  std::fputs(reason, stderr);
  std::fputc('\n', stderr);
  std::fflush(stderr);
  return -1;
}

} // namespace

int emlx_initialize_nif_runtime(ErlNifEnv *env,
                                EMLXOpenResources open_resources,
                                const char *expected_mlx_build_id) {
  try {
    std::array<uint8_t, 32> expected_hash{};
    if (!expected_mlx_build_id ||
        !emlx_sha256_parse_hex(expected_mlx_build_id, expected_hash))
      return fail_load("invalid_expected_mlx_build_identity");

    std::shared_ptr<const EMLXHostRuntimeIdentity> candidate;
    EMLXNativeImageError error;
    if (!emlx_capture_host_runtime_identity(expected_hash, candidate, error))
      return fail_load(emlx_native_image_error_name(error.code));

    if (!open_resources || open_resources(env) != 0)
      return fail_load("resource_open_failed");

    const auto accepted = candidate;
    if (!emlx_publish_host_runtime_identity(std::move(candidate), error))
      return fail_load(emlx_native_image_error_name(error.code));
    if (emlx_host_runtime_identity() != accepted)
      return fail_load("publication_verification_failed");

    return 0;
  } catch (const std::bad_alloc &) {
    return fail_load("allocation_failed");
  } catch (const std::exception &) {
    return fail_load("internal_error");
  } catch (...) {
    return fail_load("internal_error");
  }
}

int emlx_reject_nif_upgrade(ErlNifEnv *env, void **priv_data,
                            void **old_priv_data,
                            ERL_NIF_TERM load_info) noexcept {
  (void)env;
  (void)priv_data;
  (void)old_priv_data;
  (void)load_info;
  return -1;
}
