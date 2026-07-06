#pragma once

// emlx_compiler.hpp — emlx::native namespace: Op enum, Expr program struct,
// and NIF implementation declarations for compile_program / eval_program /
// native_expr_opcode_table.
//
// The three *_impl functions are implemented in emlx_compiler.cpp and called
// from thin NIF wrappers in emlx_nif.cpp.

#include "emlx_nif_shared.hpp"
#include <variant>

namespace emlx {
namespace native {

// ── Program wire types ────────────────────────────────────────────────────
//
// Elixir counterparts: EMLX.Native.Instruction / EMLX.Native.Program
// (lib/emlx/native/{instruction,program}.ex), produced by EMLX.Native.Expr.to_native/1. Decoded
// by the fine::Decoder specializations below instead of the old to_native/1
// format's positional NIF args + bit-packed-int refs.

// A reference to an already-produced value ({:input, i} / {:capture, i} /
// {:const, i} / {:result, i} / {:carry, i} on the Elixir side), resolved
// against the runtime inputs / closed-over captures / closed-over constants /
// the flat per-eval results accumulator / a `:while` sub-program's current
// loop-carry slot, respectively. `Carry` only ever appears inside a
// `SubProgram` (a `:while` instruction's `cond`/`body`) — never in the
// top-level program's own instruction list, and vice versa for `Input`
// (see EMLXWhile's doc comment in emlx_compiler.cpp).
enum class RefKind { Input, Capture, Const, Result, Carry };

struct Ref {
  RefKind kind;
  int64_t index;
};

// One instruction attribute. Most are plain integers (shapes, axes, flags,
// f64_bits-encoded floats); a handful are MLX dtypes or quantized_matmul
// mode strings, sent as atoms so no int<->meaning lookup table needs to be
// kept in sync with Elixir (see string2dtype/dtype_map in
// emlx_nif_shared.hpp). The implicit int64_t conversion below keeps every
// existing `attrs[i]`-as-int64_t use site in emlx_compiler.cpp's op registry
// compiling unchanged.
class Attr {
public:
  Attr(int64_t v) : value_(v) {}
  Attr(fine::Atom a) : value_(std::move(a)) {}

  operator int64_t() const { return std::get<int64_t>(value_); }

  mlx::core::Dtype as_dtype() const {
    return string2dtype(std::get<fine::Atom>(value_).to_string());
  }

  std::string as_mode() const { return std::get<fine::Atom>(value_).to_string(); }

private:
  std::variant<int64_t, fine::Atom> value_;
};

struct Instruction;

// A `:while` instruction's condition or body, lowered as its own
// self-contained flat instruction list (not inlined into the parent
// program) — see EMLX.Native.Expr's `:while` moduledoc section and
// EMLXWhile (emlx_compiler.cpp). `instructions`' own `Ref::Result` entries
// are local to this sub-program (a fresh flat accumulator per interpretation
// — see `interpret_instructions`), distinct from the parent program's own
// `{:result, i}` numbering. `Ref::Carry` entries resolve against whatever
// loop-carry vector the interpreting primitive passes in for that
// iteration. `Ref::Capture`/`Ref::Const` resolve against the *same* shared
// captures/constants tables as the parent program (global, not
// re-instantiated per sub-program). `Ref::Input` never appears here.
struct SubProgram {
  std::vector<Instruction> instructions;
  std::vector<Ref> outputs;
};

struct Instruction {
  fine::Atom op;
  std::vector<Ref> operands;
  std::vector<Attr> attrs;
  // Only non-empty for `op == "while"`: exactly two entries, `[cond, body]`.
  // Empty (the common case) for every other op.
  std::vector<SubProgram> subprograms;
};

struct Program {
  int num_inputs;
  std::vector<mlx::core::array> captures;
  std::vector<std::tuple<double, mlx::core::Dtype>> constants;
  std::vector<Instruction> instructions;
  // Full ref list: real outputs followed by any keepalive tail (see
  // EMLX.Native.Expr.t/0's `keepalive_refs` doc on the Elixir side) —
  // `num_real_outputs` (below) marks the boundary.
  std::vector<Ref> outputs;
  int num_real_outputs;
};

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
  // How many of compiled_fn's returned arrays are real outputs (to be
  // converted to resource terms and returned to Elixir) — the rest is a
  // keepalive tail forced via mlx::core::eval alongside them (when
  // has_runtime_call) purely for its side effects, then dropped. See
  // emlx::native::Program::num_real_outputs.
  int num_real_outputs = 0;
  emlx::function compiled_fn;

  ~Expr();
};

// compile_program is defined via FINE_ASYNC_NIF(compile_program) in
// emlx_compiler.cpp (declares `compile_program`/`compile_program_async`
// here); eval_program is a plain hand-written NIF, called from a thin
// wrapper in emlx_nif.cpp.
ERL_NIF_TERM compile_program(ErlNifEnv *env, int argc,
                             const ERL_NIF_TERM argv[]);
ERL_NIF_TERM compile_program_async(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]);
ERL_NIF_TERM eval_program(ErlNifEnv *env, int argc,
                          const ERL_NIF_TERM argv[]);

} // namespace native
} // namespace emlx

namespace fine {

template <> struct Decoder<emlx::native::Ref> {
  static emlx::native::Ref decode(ErlNifEnv *env, const ERL_NIF_TERM &term) {
    auto [kind_atom, index] =
        fine::decode<std::tuple<fine::Atom, int64_t>>(env, term);
    if (kind_atom == "input")
      return {emlx::native::RefKind::Input, index};
    if (kind_atom == "capture")
      return {emlx::native::RefKind::Capture, index};
    if (kind_atom == "const")
      return {emlx::native::RefKind::Const, index};
    if (kind_atom == "result")
      return {emlx::native::RefKind::Result, index};
    if (kind_atom == "carry")
      return {emlx::native::RefKind::Carry, index};
    throw std::invalid_argument("decode failed, unknown ref kind: " +
                                kind_atom.to_string());
  }
};

template <> struct Decoder<emlx::native::Attr> {
  static emlx::native::Attr decode(ErlNifEnv *env, const ERL_NIF_TERM &term) {
    // Try int64 first (the overwhelmingly common case: shapes, axes, flags,
    // f64_bits-encoded floats); fall back to atom (dtypes, quant modes).
    ErlNifSInt64 v;
    if (enif_get_int64(env, term, &v)) {
      return emlx::native::Attr(static_cast<int64_t>(v));
    }
    return emlx::native::Attr(fine::decode<fine::Atom>(env, term));
  }
};

// Declared ahead of Decoder<Instruction> (which needs it) since
// emlx::native::SubProgram embeds emlx::native::Instruction — mutually
// recursive, so the two decoders are mutually recursive too.
template <> struct Decoder<emlx::native::SubProgram> {
  static emlx::native::SubProgram decode(ErlNifEnv *env, const ERL_NIF_TERM &term) {
    static const fine::Atom instructions_atom("instructions");
    static const fine::Atom outputs_atom("outputs");

    ERL_NIF_TERM instructions_term, outputs_term;
    if (!enif_get_map_value(env, term, fine::encode(env, instructions_atom),
                            &instructions_term) ||
        !enif_get_map_value(env, term, fine::encode(env, outputs_atom),
                            &outputs_term)) {
      throw std::invalid_argument(
          "decode failed, expected an EMLX.Native.SubProgram struct, got: " +
          format_term(env, term));
    }

    return emlx::native::SubProgram{
        fine::decode<std::vector<emlx::native::Instruction>>(env, instructions_term),
        fine::decode<std::vector<emlx::native::Ref>>(env, outputs_term)};
  }
};

// Custom (rather than the generic T::module/T::fields struct-decode
// mechanism) so field lookup failures produce a clear, specific error
// message without needing constexpr member-pointer/Atom-pointer tables.
template <> struct Decoder<emlx::native::Instruction> {
  static emlx::native::Instruction decode(ErlNifEnv *env,
                                          const ERL_NIF_TERM &term) {
    static const fine::Atom op_atom("op");
    static const fine::Atom operands_atom("operands");
    static const fine::Atom attrs_atom("attrs");
    static const fine::Atom subprograms_atom("subprograms");

    ERL_NIF_TERM op_term, operands_term, attrs_term, subprograms_term;
    if (!enif_get_map_value(env, term, fine::encode(env, op_atom), &op_term) ||
        !enif_get_map_value(env, term, fine::encode(env, operands_atom),
                            &operands_term) ||
        !enif_get_map_value(env, term, fine::encode(env, attrs_atom),
                            &attrs_term) ||
        !enif_get_map_value(env, term, fine::encode(env, subprograms_atom),
                            &subprograms_term)) {
      throw std::invalid_argument(
          "decode failed, expected an EMLX.Native.Instruction struct, "
          "got: " +
          format_term(env, term));
    }

    return emlx::native::Instruction{
        fine::decode<fine::Atom>(env, op_term),
        fine::decode<std::vector<emlx::native::Ref>>(env, operands_term),
        fine::decode<std::vector<emlx::native::Attr>>(env, attrs_term),
        fine::decode<std::vector<emlx::native::SubProgram>>(env, subprograms_term)};
  }
};

template <> struct Decoder<emlx::native::Program> {
  static emlx::native::Program decode(ErlNifEnv *env, const ERL_NIF_TERM &term) {
    static const fine::Atom num_inputs_atom("num_inputs");
    static const fine::Atom captures_atom("captures");
    static const fine::Atom constants_atom("constants");
    static const fine::Atom instructions_atom("instructions");
    static const fine::Atom outputs_atom("outputs");
    static const fine::Atom num_real_outputs_atom("num_real_outputs");

    auto get_field = [&](const fine::Atom &key) -> ERL_NIF_TERM {
      ERL_NIF_TERM value;
      if (!enif_get_map_value(env, term, fine::encode(env, key), &value)) {
        throw std::invalid_argument(
            "decode failed, expected an EMLX.Native.Program struct with "
            "field " +
            key.to_string() + ", got: " + format_term(env, term));
      }
      return value;
    };

    emlx::native::Program program;
    program.num_inputs = fine::decode<int>(env, get_field(num_inputs_atom));
    program.captures = fine::decode<std::vector<mlx::core::array>>(
        env, get_field(captures_atom));
    program.constants =
        fine::decode<std::vector<std::tuple<double, mlx::core::Dtype>>>(
            env, get_field(constants_atom));
    program.instructions = fine::decode<std::vector<emlx::native::Instruction>>(
        env, get_field(instructions_atom));
    program.outputs =
        fine::decode<std::vector<emlx::native::Ref>>(env, get_field(outputs_atom));
    program.num_real_outputs =
        fine::decode<int>(env, get_field(num_real_outputs_atom));
    return program;
  }
};

} // namespace fine
