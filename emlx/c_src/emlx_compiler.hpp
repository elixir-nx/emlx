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
  // True when this program contains an inlined `:runtime_call` node — see
  // eval_program: such a program is force-evaluated (mlx::core::eval on the
  // outputs) before returning, instead of the usual deferred/lazy return, so
  // EMLXRuntimeCall::eval_cpu/eval_gpu fires while this NIF call's caller
  // pid (emlx::g_current_caller_pid) is still in scope.
  bool has_runtime_call = false;
  emlx::function compiled_fn;

  ~Expr();
};

// NIF implementation functions — thin wrappers in emlx_nif.cpp delegate here.
ERL_NIF_TERM compile_program(ErlNifEnv *env, int argc,
                             const ERL_NIF_TERM argv[]);
ERL_NIF_TERM eval_program(ErlNifEnv *env, int argc,
                          const ERL_NIF_TERM argv[]);

} // namespace native
} // namespace emlx
