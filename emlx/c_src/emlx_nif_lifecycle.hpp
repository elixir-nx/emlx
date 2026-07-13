#pragma once

#include "erl_nif.h"

using EMLXOpenResources = int (*)(ErlNifEnv *env);

int emlx_initialize_nif_runtime(ErlNifEnv *env,
                                EMLXOpenResources open_resources,
                                const char *expected_mlx_build_id);

int emlx_reject_nif_upgrade(ErlNifEnv *env, void **priv_data,
                            void **old_priv_data,
                            ERL_NIF_TERM load_info) noexcept;
