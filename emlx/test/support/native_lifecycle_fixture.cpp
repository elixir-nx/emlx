#include "emlx/native_image.hpp"
#include "emlx/nif/lifecycle.hpp"
#include "emlx/plugin/build_compat.hpp"
#include "emlx/plugin/registry.hpp"

#include "erl_nif.h"

#include <new>
#include <stdexcept>

namespace {

int open_ok(ErlNifEnv *) { return 0; }
int open_failure(ErlNifEnv *) { return -1; }
int open_bad_alloc(ErlNifEnv *) { throw std::bad_alloc(); }
int open_standard_exception(ErlNifEnv *) {
  throw std::runtime_error("fixture standard exception");
}
int open_unknown_exception(ErlNifEnv *) { throw 42; }

bool mode_is(ErlNifEnv *env, ERL_NIF_TERM load_info, const char *name) {
  return enif_is_identical(load_info, enif_make_atom(env, name));
}

int load(ErlNifEnv *env, void **, ERL_NIF_TERM load_info) {
  if (mode_is(env, load_info, "registry_only"))
    return 0;
  if (mode_is(env, load_info, "resource_failure"))
    return emlx_initialize_nif_runtime(env, open_failure,
                                       EMLX_EXPECTED_MLX_BUILD_ID);
  if (mode_is(env, load_info, "build_mismatch"))
    return emlx_initialize_nif_runtime(
        env, open_ok,
        "0000000000000000000000000000000000000000000000000000000000000000");
  if (mode_is(env, load_info, "bad_alloc"))
    return emlx_initialize_nif_runtime(env, open_bad_alloc,
                                       EMLX_EXPECTED_MLX_BUILD_ID);
  if (mode_is(env, load_info, "standard_exception"))
    return emlx_initialize_nif_runtime(env, open_standard_exception,
                                       EMLX_EXPECTED_MLX_BUILD_ID);
  if (mode_is(env, load_info, "unknown_exception"))
    return emlx_initialize_nif_runtime(env, open_unknown_exception,
                                       EMLX_EXPECTED_MLX_BUILD_ID);
  return emlx_initialize_nif_runtime(env, open_ok,
                                     EMLX_EXPECTED_MLX_BUILD_ID);
}

ERL_NIF_TERM status(ErlNifEnv *env, int, const ERL_NIF_TERM[]) {
  const bool ready = static_cast<bool>(emlx_host_runtime_identity());
  return enif_make_tuple3(
      env, enif_make_atom(env, ready ? "ready" : "uninitialized"),
      enif_make_uint64(env, emlx_native_image_test_capture_count()),
      enif_make_uint64(env, emlx_native_image_test_identity_count()));
}

ERL_NIF_TERM duplicate_publication(ErlNifEnv *env, int,
                                   const ERL_NIF_TERM[]) {
  auto accepted = emlx_host_runtime_identity();
  EMLXNativeImageError error;
  if (emlx_publish_host_runtime_identity(std::move(accepted), error))
    return enif_make_atom(env, "unexpected_success");
  return enif_make_atom(env, emlx_native_image_error_name(error.code));
}

ErlNifFunc functions[] = {{"status", 0, status},
                          {"duplicate_publication", 0,
                           duplicate_publication},
                          {"register_plugin", 2, load_plugin}};

} // namespace

ERL_NIF_INIT(Elixir.EMLX.NativeLifecycleFixture, functions, load, nullptr,
             emlx_upgrade_nif_runtime, nullptr)
