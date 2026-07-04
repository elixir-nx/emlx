#pragma once

// Generic, name-keyed native plugin loader — `dlopen`s a standalone shared
// library (no erl_nif dependency) and caches the `void*` it returns from a
// well-known exported symbol, keyed by an arbitrary caller-chosen name.
//
// This has no knowledge of any specific plugin's ABI: callers (e.g.
// `emlx_fast/qwen3.cpp`) look their plugin up by name via `emlx_get_plugin`
// and `reinterpret_cast` the result to whatever vtable struct they expect —
// that contract lives entirely in the caller (see
// `emlx_fast/qwen3_plugin_abi.hpp`), not here.

#include "erl_nif.h"

#include <string>

// load_plugin(name, path) — `dlopen`s `path` and `dlsym`s the fixed
// `emlx_plugin_vtable` entry point, storing the resulting pointer under
// `name` for later retrieval via `emlx_get_plugin`. Reloading the same
// `name` replaces the previous entry (and `dlclose`s its handle). Not
// worker-routed: `dlopen`/`dlsym` do no MLX graph work.
ERL_NIF_TERM load_plugin(ErlNifEnv *, int, const ERL_NIF_TERM[]);

// Returns the cached vtable pointer for `name`, or `nullptr` if no plugin
// has been successfully loaded under that name yet.
const void *emlx_get_plugin(const std::string &name);
