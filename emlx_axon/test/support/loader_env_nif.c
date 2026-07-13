#include "erl_nif.h"

#include <stdlib.h>

static ERL_NIF_TERM put_env(ErlNifEnv *env, int argc,
                            const ERL_NIF_TERM argv[]) {
  char name[128];
  char value[1024];
  if (argc != 2 || enif_get_string(env, argv[0], name, sizeof(name), ERL_NIF_LATIN1) <= 0 ||
      enif_get_string(env, argv[1], value, sizeof(value), ERL_NIF_LATIN1) <= 0 ||
      setenv(name, value, 1) != 0)
    return enif_make_badarg(env);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM delete_env(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]) {
  char name[128];
  if (argc != 1 || enif_get_string(env, argv[0], name, sizeof(name), ERL_NIF_LATIN1) <= 0 ||
      unsetenv(name) != 0)
    return enif_make_badarg(env);
  return enif_make_atom(env, "ok");
}

static ErlNifFunc functions[] = {{"put", 2, put_env},
                                 {"delete", 1, delete_env}};

ERL_NIF_INIT(Elixir.EMLXAxon.PluginMetadataTest.LoaderEnvFixture, functions,
             NULL, NULL, NULL, NULL)
