#pragma once

// emlx/compiler.hpp — emlx::native namespace: Op enum, Expr program struct,
// and NIF implementation declarations for compile_program / eval_program /
// native_expr_opcode_table.
//
// The three *_impl functions are implemented in emlx/compiler.cpp and called
// from thin NIF wrappers in emlx_nif.cpp.

#include "emlx_nif_shared.hpp"
#include "emlx/plugin/registry.hpp"
#include "mlx/compile_impl.h"
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
// (see EMLXWhile's doc comment in emlx/compiler.cpp).
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
// existing `attrs[i]`-as-int64_t use site in emlx/compiler.cpp's op registry
// compiling unchanged.
class Attr {
public:
  Attr(int64_t v) : value_(v) {}
  Attr(fine::Atom a) : value_(std::move(a)) {}
  Attr(std::string binary) : value_(std::move(binary)) {}

  operator int64_t() const { return std::get<int64_t>(value_); }

  mlx::core::Dtype as_dtype() const {
    return string2dtype(std::get<fine::Atom>(value_).to_string());
  }

  std::string as_mode() const { return std::get<fine::Atom>(value_).to_string(); }
  const std::string &as_binary() const { return std::get<std::string>(value_); }
  bool is_int() const { return std::holds_alternative<int64_t>(value_); }
  bool is_atom() const { return std::holds_alternative<fine::Atom>(value_); }
  bool is_binary() const { return std::holds_alternative<std::string>(value_); }

private:
  std::variant<int64_t, fine::Atom, std::string> value_;
};

struct PluginOutputTemplate {
  mlx::core::Dtype dtype;
  mlx::core::Shape shape;
};

struct Instruction;

// A `:while` instruction's condition or body, lowered as its own
// self-contained flat instruction list (not inlined into the parent
// program) — see EMLX.Native.Expr's `:while` moduledoc section and
// EMLXWhile (emlx/compiler.cpp). `instructions`' own `Ref::Result` entries
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
  emlx::plugin::resolved_callback_t resolved_plugin;
  std::vector<int64_t> plugin_attrs;
  std::vector<PluginOutputTemplate> plugin_outputs;
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
// The destructor evicts the per-ID entry from each MLX compile cache it is in.
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
  // MLX keeps one compile cache per thread, and compiled_fn adds its graph to
  // the cache of the thread that calls it. These are the caches to erase from.
  std::vector<mlx::core::detail::CompileCacheWeakPtr> compile_caches;

  ~Expr();
};

// compile_program is defined via FINE_ASYNC_NIF(compile_program) in
// emlx/compiler.cpp (declares `compile_program`/`compile_program_async`
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

namespace emlx::native {

// Decodes the field `key` of the Elixir struct `term` as T.
template <typename T>
fine::DecodeResult<T> decode_field(ErlNifEnv *env, const ERL_NIF_TERM &term,
                                   const char *struct_name,
                                   const fine::Atom &key) {
  ERL_NIF_TERM value;
  if (!enif_get_map_value(env, term, fine::encode(env, key), &value)) {
    return fine::DecodeError{std::string("decode failed, expected an ") +
                                 struct_name + " struct with field " +
                                 key.to_string(),
                             term};
  }
  return fine::try_decode<T>(env, value);
}

} // namespace emlx::native

namespace fine {

// Decode a constant entry {value, mlx_dtype} where value is either an Erlang
// float (for finite constants) or one of the atoms :infinity, :neg_infinity,
// :nan (for non-finite IEEE 754 values).  Erlang's bit-syntax float decoder
// rejects NaN and infinity, so the Elixir side passes atoms instead and this
// specialisation maps them to the appropriate C double.
template <>
struct Decoder<std::tuple<double, mlx::core::Dtype>> {
  using Constant = std::tuple<double, mlx::core::Dtype>;

  static DecodeResult<Constant> try_decode(ErlNifEnv *env,
                                           const ERL_NIF_TERM &term) {
    int arity;
    const ERL_NIF_TERM *elems;
    if (!enif_get_tuple(env, term, &arity, &elems) || arity != 2) {
      return DecodeError{"decode failed, expected a 2-tuple for constant", term};
    }
    double value;
    char atom[64];
    if (enif_get_atom(env, elems[0], atom, sizeof(atom), ERL_NIF_LATIN1)) {
      if (std::strcmp(atom, "infinity") == 0)
        value = std::numeric_limits<double>::infinity();
      else if (std::strcmp(atom, "neg_infinity") == 0)
        value = -std::numeric_limits<double>::infinity();
      else if (std::strcmp(atom, "nan") == 0)
        value = std::numeric_limits<double>::quiet_NaN();
      else
        return DecodeError{"decode failed, unexpected atom for constant value",
                           elems[0]};
    } else {
      auto decoded = fine::try_decode<double>(env, elems[0]);
      if (!decoded) {
        return std::move(decoded).error();
      }
      value = decoded.value();
    }
    auto dtype = fine::try_decode<mlx::core::Dtype>(env, elems[1]);
    if (!dtype) {
      return std::move(dtype).error();
    }
    return Constant{value, dtype.value()};
  }
};

template <> struct Decoder<emlx::native::Ref> {
  static DecodeResult<emlx::native::Ref> try_decode(ErlNifEnv *env,
                                                    const ERL_NIF_TERM &term) {
    using emlx::native::Ref;
    using emlx::native::RefKind;

    auto decoded = fine::try_decode<std::tuple<fine::Atom, int64_t>>(env, term);
    if (!decoded) {
      return std::move(decoded).error();
    }
    auto [kind_atom, index] = std::move(decoded).value();
    if (kind_atom == "input")
      return Ref{RefKind::Input, index};
    if (kind_atom == "capture")
      return Ref{RefKind::Capture, index};
    if (kind_atom == "const")
      return Ref{RefKind::Const, index};
    if (kind_atom == "result")
      return Ref{RefKind::Result, index};
    if (kind_atom == "carry")
      return Ref{RefKind::Carry, index};
    return DecodeError{"decode failed, unknown ref kind", term};
  }
};

template <> struct Decoder<emlx::native::Attr> {
  static DecodeResult<emlx::native::Attr> try_decode(ErlNifEnv *env,
                                                     const ERL_NIF_TERM &term) {
    // Try int64 first (the overwhelmingly common case: shapes, axes, flags,
    // f64_bits-encoded floats); fall back to atom (dtypes, quant modes).
    ErlNifSInt64 v;
    if (enif_get_int64(env, term, &v)) {
      return emlx::native::Attr(static_cast<int64_t>(v));
    }
    ErlNifBinary binary;
    if (enif_inspect_binary(env, term, &binary)) {
      return emlx::native::Attr(
          std::string(reinterpret_cast<const char *>(binary.data), binary.size));
    }
    auto atom = fine::try_decode<fine::Atom>(env, term);
    if (!atom) {
      return DecodeError{
          "decode failed, expected an integer, binary or atom attribute", term};
    }
    return emlx::native::Attr(std::move(atom).value());
  }
};

// Declared ahead of Decoder<Instruction> (which needs it) since
// emlx::native::SubProgram embeds emlx::native::Instruction — mutually
// recursive, so the two decoders are mutually recursive too.
template <> struct Decoder<emlx::native::SubProgram> {
  static DecodeResult<emlx::native::SubProgram>
  try_decode(ErlNifEnv *env, const ERL_NIF_TERM &term) {
    using emlx::native::decode_field;
    static const char *name = "EMLX.Native.SubProgram";
    static const fine::Atom instructions_atom("instructions");
    static const fine::Atom outputs_atom("outputs");

    auto instructions = decode_field<std::vector<emlx::native::Instruction>>(
        env, term, name, instructions_atom);
    if (!instructions) {
      return std::move(instructions).error();
    }
    auto outputs =
        decode_field<std::vector<emlx::native::Ref>>(env, term, name, outputs_atom);
    if (!outputs) {
      return std::move(outputs).error();
    }
    return emlx::native::SubProgram{std::move(instructions).value(),
                                    std::move(outputs).value()};
  }
};

// Custom (rather than the generic T::module/T::fields struct-decode
// mechanism) so field lookup failures produce a clear, specific error
// message without needing constexpr member-pointer/Atom-pointer tables.
template <> struct Decoder<emlx::native::Instruction> {
  static DecodeResult<emlx::native::Instruction>
  try_decode(ErlNifEnv *env, const ERL_NIF_TERM &term) {
    using emlx::native::decode_field;
    static const char *name = "EMLX.Native.Instruction";
    static const fine::Atom op_atom("op");
    static const fine::Atom operands_atom("operands");
    static const fine::Atom attrs_atom("attrs");
    static const fine::Atom subprograms_atom("subprograms");

    auto op = decode_field<fine::Atom>(env, term, name, op_atom);
    if (!op) {
      return std::move(op).error();
    }
    auto operands =
        decode_field<std::vector<emlx::native::Ref>>(env, term, name, operands_atom);
    if (!operands) {
      return std::move(operands).error();
    }
    auto attrs =
        decode_field<std::vector<emlx::native::Attr>>(env, term, name, attrs_atom);
    if (!attrs) {
      return std::move(attrs).error();
    }
    auto subprograms = decode_field<std::vector<emlx::native::SubProgram>>(
        env, term, name, subprograms_atom);
    if (!subprograms) {
      return std::move(subprograms).error();
    }
    return emlx::native::Instruction{
        std::move(op).value(), std::move(operands).value(),
        std::move(attrs).value(), std::move(subprograms).value()};
  }
};

template <> struct Decoder<emlx::native::Program> {
  static DecodeResult<emlx::native::Program> try_decode(ErlNifEnv *env,
                                                        const ERL_NIF_TERM &term) {
    using emlx::native::decode_field;
    static const char *name = "EMLX.Native.Program";
    static const fine::Atom num_inputs_atom("num_inputs");
    static const fine::Atom captures_atom("captures");
    static const fine::Atom constants_atom("constants");
    static const fine::Atom instructions_atom("instructions");
    static const fine::Atom outputs_atom("outputs");
    static const fine::Atom num_real_outputs_atom("num_real_outputs");

    auto num_inputs = decode_field<int>(env, term, name, num_inputs_atom);
    if (!num_inputs) {
      return std::move(num_inputs).error();
    }
    auto captures =
        decode_field<std::vector<mlx::core::array>>(env, term, name, captures_atom);
    if (!captures) {
      return std::move(captures).error();
    }
    auto constants =
        decode_field<std::vector<std::tuple<double, mlx::core::Dtype>>>(
            env, term, name, constants_atom);
    if (!constants) {
      return std::move(constants).error();
    }
    auto instructions = decode_field<std::vector<emlx::native::Instruction>>(
        env, term, name, instructions_atom);
    if (!instructions) {
      return std::move(instructions).error();
    }
    auto outputs =
        decode_field<std::vector<emlx::native::Ref>>(env, term, name, outputs_atom);
    if (!outputs) {
      return std::move(outputs).error();
    }
    auto num_real_outputs =
        decode_field<int>(env, term, name, num_real_outputs_atom);
    if (!num_real_outputs) {
      return std::move(num_real_outputs).error();
    }

    emlx::native::Program program;
    program.num_inputs = num_inputs.value();
    program.captures = std::move(captures).value();
    program.constants = std::move(constants).value();
    program.instructions = std::move(instructions).value();
    program.outputs = std::move(outputs).value();
    program.num_real_outputs = num_real_outputs.value();
    return program;
  }
};

} // namespace fine
