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

// Opcode enum — wire values must stay in lockstep with
// EMLX.Native.Expr.wire_opcodes/0 in lib/emlx/native/expr.ex.
// The NIF native_expr_opcode_table/0 exposes this enum at runtime so the
// Elixir test can verify both tables match.
enum class Op : int {
  Add = 0,
};

// Compiled representation of an EMLX.Native.Expr program.
// Stored as an opaque BEAM resource; one instance per compiled defn cache entry.
// compile_program bakes the program into a capturing lambda, wraps it with
// mlx::core::compile() so MLX traces and caches the graph, and stores the
// result here.  eval_program just calls compiled_fn(inputs).
struct Expr {
  int num_inputs = 0;
  emlx::function compiled_fn;  // mlx::core::compile()-wrapped interpreter lambda
};

// NIF implementation functions — thin wrappers in emlx_nif.cpp delegate here.
ERL_NIF_TERM compile_program_impl(ErlNifEnv *env, int argc,
                                  const ERL_NIF_TERM argv[]);
ERL_NIF_TERM eval_program_impl(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]);
ERL_NIF_TERM opcode_table_impl(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]);

} // namespace native
} // namespace emlx
