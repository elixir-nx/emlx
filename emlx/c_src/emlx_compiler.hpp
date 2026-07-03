#pragma once

// emlx_compiler.hpp — emlx::native namespace: Op enum, Expr program struct,
// and NIF implementation declarations for compile_program / eval_program /
// native_expr_opcode_table.
//
// The three *_impl functions are implemented in emlx_compiler.cpp and called
// from thin NIF wrappers in emlx_nif.cpp.

#include "emlx_nif_shared.hpp"

namespace emlx {
namespace native {

// Compiled representation of an EMLX.Native.Expr program.
// Stored as an opaque BEAM resource; one instance per compiled defn cache entry.
// compile_program bakes the program into a capturing lambda, wraps it with
// mlx::core::detail::compile() (using a unique per-Expr ID) so MLX traces and
// caches the graph.  eval_program just calls compiled_fn(inputs).
// The destructor evicts the per-ID entry from MLX's global compile cache.
struct Expr {
  int num_inputs = 0;
  std::uintptr_t compile_id = 0;  // unique key for mlx::core::detail compile cache
  emlx::function compiled_fn;

  ~Expr();
};

// NIF implementation functions — thin wrappers in emlx_nif.cpp delegate here.
ERL_NIF_TERM compile_program(ErlNifEnv *env, int argc,
                             const ERL_NIF_TERM argv[]);
ERL_NIF_TERM eval_program(ErlNifEnv *env, int argc,
                          const ERL_NIF_TERM argv[]);

// Stage 32a Procedures #2-#3 — the production ":host_callback" opcode. See
// the "Host callback opcode" section in emlx_compiler.cpp. Callback
// *identity* (which MFA a callback_slot maps to) lives entirely on the
// Elixir side (EMLX.Native.Expr's per-program callback table); the target
// pid for each call is resolved from emlx::current_caller_pid()
// (emlx_async.hpp), not registered here — there is no registration NIF.
ERL_NIF_TERM host_callback_resume(ErlNifEnv *env, int argc,
                                  const ERL_NIF_TERM argv[]);

// Stage 32a spike — see the "Stage 32a spike" section in emlx_compiler.cpp.
// Not part of the production op registry; throwaway once the stage's
// go/no-go verdict is written up.
namespace spike32a {
ERL_NIF_TERM run_program(ErlNifEnv *env, ErlNifPid target_pid,
                         mlx::core::Device device, double input_value,
                         uint64_t compile_id);
ERL_NIF_TERM resume_call(ErlNifEnv *env, uint64_t call_id, double value);
} // namespace spike32a

} // namespace native
} // namespace emlx
