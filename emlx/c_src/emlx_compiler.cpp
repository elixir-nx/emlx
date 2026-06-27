// emlx_compiler.cpp — implements emlx::native compile/eval NIF logic.
//
// This file owns all of the EMLX.Native.Expr compiler substrate: the packed-ref
// helpers, interpreter dispatch loop, and the three NIF implementations.  The thin
// NIF wrappers in emlx_nif.cpp forward directly to the *_impl functions here.
//
// compile_program_impl bakes the program into a capturing lambda and wraps it with
// mlx::core::compile(), so MLX traces the computation graph on first eval and
// replays the cached compiled graph on subsequent calls.  eval_program_impl is a
// thin caller: compiled_fn(inputs) → eval → wrap outputs.

#include "emlx_compiler.hpp"

namespace emlx {
namespace native {

// ── Packed-ref helpers ────────────────────────────────────────────────────────
//
// Ref encoding: kind in bits [61:60], index in bits [59:0].
// Mirrors to_wire/1 in lib/emlx/native/expr.ex.
//
//   kind=0  input    — indexed into the runtime inputs vector
//   kind=1  capture  — indexed into closed-over captures
//   kind=2  constant — indexed into closed-over constants
//   kind=3  result   — indexed into the per-eval results accumulator

static constexpr uint64_t KIND_SHIFT = 60;
static constexpr uint64_t IDX_MASK = (uint64_t(1) << KIND_SHIFT) - 1;

static int ref_kind(int64_t packed) {
  return static_cast<int>((static_cast<uint64_t>(packed) >> KIND_SHIFT) & 3);
}

static int64_t ref_idx(int64_t packed) {
  return static_cast<int64_t>(static_cast<uint64_t>(packed) & IDX_MASK);
}

// ── NIF argument parsing helpers ──────────────────────────────────────────────

// Parse a list of int64 lists (one sub-list per instruction) from an
// ERL_NIF_TERM.  Used for operands and attrs.
static bool parse_nested_int64_list(ErlNifEnv *env, ERL_NIF_TERM list,
                                    std::vector<std::vector<int64_t>> &out) {
  unsigned length;
  if (!enif_get_list_length(env, list, &length))
    return false;
  out.reserve(length);
  ERL_NIF_TERM head, tail;
  while (enif_get_list_cell(env, list, &head, &tail)) {
    std::vector<int64_t> inner;
    if (!nx::nif::get_list(env, head, inner))
      return false;
    out.push_back(std::move(inner));
    list = tail;
  }
  return true;
}

// ── NIF implementations ───────────────────────────────────────────────────────

// compile_program — decodes the serialised EMLX.Native.Expr wire format, builds
// a capturing interpreter lambda, wraps it with mlx::core::compile(), and stores
// the result as an opaque emlx::native::Expr BEAM resource.
//
// MLX will trace the lambda on the first eval_program call, build a compiled
// computation graph, and replay that cached graph on all subsequent calls.
//
// argv[0] : num_inputs    (int)
// argv[1] : capture_refs  (list of MLX array resource refs)
// argv[2] : const_values  (list of doubles or ints)
// argv[3] : const_types   (list of dtype atoms, e.g. :float32)
// argv[4] : opcodes       (list of ints)
// argv[5] : operands      (list of list of int64 — packed refs per instr)
// argv[6] : attrs         (list of list of int64 — integer attrs per instr)
// argv[7] : output_refs   (list of int64 — packed output refs)
ERL_NIF_TERM compile_program_impl(ErlNifEnv *env, int argc,
                                  const ERL_NIF_TERM argv[]) {
  try {
    PARAM(0, int, num_inputs_val);
    LIST_PARAM(1, std::vector<mlx::core::array>, captures);

    // Parse const_values: doubles (or integers coerced to double).
    unsigned const_count;
    if (!enif_get_list_length(env, argv[2], &const_count))
      return nx::nif::error(env, "Unable to get const_values length");
    std::vector<double> const_values;
    const_values.reserve(const_count);
    {
      ERL_NIF_TERM head, tail, clist = argv[2];
      while (enif_get_list_cell(env, clist, &head, &tail)) {
        double v;
        if (!enif_get_double(env, head, &v)) {
          int64_t iv;
          if (!enif_get_int64(env, head, reinterpret_cast<ErlNifSInt64 *>(&iv)))
            return nx::nif::error(env,
                                  "Unable to parse const value as double or int");
          v = static_cast<double>(iv);
        }
        const_values.push_back(v);
        clist = tail;
      }
    }

    LIST_PARAM(3, std::vector<std::string>, const_types);
    LIST_PARAM(4, std::vector<int>, opcodes);

    std::vector<std::vector<int64_t>> operands;
    if (!parse_nested_int64_list(env, argv[5], operands))
      return nx::nif::error(env, "Unable to get operands nested list");

    std::vector<std::vector<int64_t>> attrs;
    if (!parse_nested_int64_list(env, argv[6], attrs))
      return nx::nif::error(env, "Unable to get attrs nested list");

    LIST_PARAM(7, std::vector<int64_t>, output_refs);

    // Build constant arrays on the current (worker) thread using its default stream.
    std::vector<mlx::core::array> constants;
    constants.reserve(const_values.size());
    for (size_t i = 0; i < const_values.size(); i++) {
      auto dtype = string2dtype(const_types[i]);
      constants.push_back(mlx::core::full({}, const_values[i], dtype));
    }

    // Build the interpreter lambda capturing all program data, then pass it
    // through mlx::core::compile().  MLX traces the lambda on the first
    // eval_program call (building a compiled computation graph) and replays
    // the cached graph on every subsequent call — no repeated graph construction.
    emlx::function fn =
        [captures = std::move(captures),
         constants = std::move(constants),
         opcodes = std::move(opcodes),
         operands = std::move(operands),
         attrs = std::move(attrs),
         output_refs = std::move(output_refs)](
            const std::vector<mlx::core::array> &inputs)
        -> std::vector<mlx::core::array> {
      std::vector<mlx::core::array> results;
      results.reserve(opcodes.size());

      // Resolve a packed ref to a concrete array.
      auto resolve = [&](int64_t packed) -> mlx::core::array {
        int kind = ref_kind(packed);
        int64_t idx = ref_idx(packed);
        switch (kind) {
        case 0:
          return inputs.at(static_cast<size_t>(idx));
        case 1:
          return captures.at(static_cast<size_t>(idx));
        case 2:
          return constants.at(static_cast<size_t>(idx));
        case 3:
          return results.at(static_cast<size_t>(idx));
        default:
          throw std::runtime_error("emlx::native: invalid ref kind " +
                                   std::to_string(kind));
        }
      };

      for (size_t i = 0; i < opcodes.size(); i++) {
        auto op = static_cast<Op>(opcodes[i]);
        const auto &ops = operands[i];

        switch (op) {
        case Op::Add: {
          auto a = resolve(ops[0]);
          auto b = resolve(ops[1]);
          // No explicit device: uses default stream on the current worker thread.
          results.push_back(mlx::core::add(a, b));
          break;
        }
        default:
          throw std::runtime_error(
              "emlx::native: unknown opcode " +
              std::to_string(static_cast<int>(op)));
        }
      }

      std::vector<mlx::core::array> outputs;
      outputs.reserve(output_refs.size());
      for (int64_t ref : output_refs) {
        outputs.push_back(resolve(ref));
      }
      return outputs;
    };

    // Allocate the program resource.
    auto *ptr = static_cast<Expr *>(
        enif_alloc_resource(resource_object<Expr>::type, sizeof(Expr)));
    if (!ptr)
      return nx::nif::error(env, "Failed to allocate Expr resource");

    new (ptr) Expr();
    ptr->num_inputs = num_inputs_val;
    ptr->compiled_fn = mlx::core::compile(std::move(fn));

    ERL_NIF_TERM ret = enif_make_resource(env, ptr);
    enif_release_resource(ptr);
    return nx::nif::ok(env, ret);
  }
  CATCH()
}

// eval_program — calls the MLX-compiled function against runtime inputs.
// MLX traces on the first call and replays the cached graph on subsequent calls.
// Returns the output arrays as a list of MLX array resource refs.
//
// argv[0] : program_ref  (emlx::native::Expr resource)
// argv[1] : input_refs   (list of MLX array resource refs — runtime inputs)
ERL_NIF_TERM eval_program_impl(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]) {
  try {
    Expr *prog;
    if (!enif_get_resource(env, argv[0], resource_object<Expr>::type,
                           reinterpret_cast<void **>(&prog)))
      return nx::nif::error(env, "Invalid Expr resource");

    LIST_PARAM(1, std::vector<mlx::core::array>, inputs);

    // Call the MLX-compiled function.  First call traces; subsequent calls
    // replay the cached compiled graph without rebuilding it.
    auto outputs = prog->compiled_fn(inputs);

    // Materialise the lazy graph on the worker's stream.
    mlx::core::eval(outputs);

    // Return as a list of tensor resource refs.
    size_t n = outputs.size();
    std::vector<ERL_NIF_TERM> terms;
    terms.reserve(n);
    for (size_t i = 0; i < n; i++) {
      terms.push_back(create_tensor_resource(env, outputs[i]));
    }
    ERL_NIF_TERM list =
        enif_make_list_from_array(env, terms.data(), static_cast<unsigned>(n));
    return nx::nif::ok(env, list);
  }
  CATCH()
}

// native_expr_opcode_table/0 — returns [{:add, 0}, ...] so Elixir tests can
// verify the Elixir opcode table and C++ enum are in lockstep.
// Non-worker-routed: pure metadata, no MLX graph access.
ERL_NIF_TERM opcode_table_impl(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]) {
  ERL_NIF_TERM pairs[] = {
      enif_make_tuple2(env, enif_make_atom(env, "add"),
                       enif_make_int(env, static_cast<int>(Op::Add))),
  };
  ERL_NIF_TERM list = enif_make_list_from_array(env, pairs, 1);
  return nx::nif::ok(env, list);
}

} // namespace native
} // namespace emlx
