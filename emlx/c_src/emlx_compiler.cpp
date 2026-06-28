// emlx_compiler.cpp — implements emlx::native compile/eval NIF logic.
//
// compile_program bakes the program into a capturing lambda and wraps it with
// mlx::core::compile(), so MLX traces the computation graph on first eval and
// replays the cached compiled graph on subsequent calls.  eval_program is a
// thin caller: compiled_fn(inputs) → eval → wrap outputs.
//
// Op dispatch uses a string→function registry instead of an integer opcode enum.
// Adding a new op: register it in `op_registry` below.  No enum, no wire
// integers, no lockstep parity table to maintain.

#include "emlx_compiler.hpp"
#include "mlx/compile_impl.h"
#include <atomic>
#include <unordered_map>

namespace emlx {
namespace native {

// ── Op registry ───────────────────────────────────────────────────────────────
//
// Each entry maps an op name string (matching the atom used in the Elixir IR)
// to a C++ function: (resolved_operands, integer_attrs) → result array.
// `operands` are already-resolved mlx::core::arrays; `attrs` are the integer
// attribute channel passed verbatim from the IR.
//
// This is the single source of truth for op semantics on the compiler path.
// No explicit device: ops run on the default stream of the current worker thread.

using OpFn = std::function<
    mlx::core::array(const std::vector<mlx::core::array> &ops,
                     const std::vector<int64_t> &attrs)>;

// Maps the dtype integer (from EMLX.Native.Expr.@mlx_type_to_int) to mlx Dtype.
// Must stay in sync with the @mlx_type_to_int map in emlx/lib/emlx/native/expr.ex.
static mlx::core::Dtype int_to_dtype(int64_t val) {
  static const mlx::core::Dtype table[] = {
      mlx::core::bool_,    // 0
      mlx::core::uint8,    // 1
      mlx::core::uint16,   // 2
      mlx::core::uint32,   // 3
      mlx::core::uint64,   // 4
      mlx::core::int8,     // 5
      mlx::core::int16,    // 6
      mlx::core::int32,    // 7
      mlx::core::int64,    // 8
      mlx::core::float16,  // 9
      mlx::core::bfloat16, // 10
      mlx::core::float32,  // 11
      mlx::core::complex64 // 12
  };
  if (val < 0 || val > 12)
    throw std::runtime_error("int_to_dtype: invalid dtype int " +
                             std::to_string(val));
  return table[static_cast<size_t>(val)];
}

static const std::unordered_map<std::string, OpFn> op_registry = {
    // ── cast ──────────────────────────────────────────────────────────────
    {"astype",
     [](const auto &ops, const auto &attrs) {
       return mlx::core::astype(ops[0], int_to_dtype(attrs[0]));
     }},

    // ── unary ops ─────────────────────────────────────────────────────────
    {"abs", [](const auto &ops, const auto &) { return mlx::core::abs(ops[0]); }},
    {"ceil", [](const auto &ops, const auto &) { return mlx::core::ceil(ops[0]); }},
    {"floor", [](const auto &ops, const auto &) { return mlx::core::floor(ops[0]); }},
    {"negate", [](const auto &ops, const auto &) { return mlx::core::negative(ops[0]); }},
    {"round", [](const auto &ops, const auto &) { return mlx::core::round(ops[0]); }},
    {"sign", [](const auto &ops, const auto &) { return mlx::core::sign(ops[0]); }},
    {"real", [](const auto &ops, const auto &) { return mlx::core::real(ops[0]); }},
    {"imag", [](const auto &ops, const auto &) { return mlx::core::imag(ops[0]); }},
    {"is_nan", [](const auto &ops, const auto &) { return mlx::core::isnan(ops[0]); }},
    {"is_infinity", [](const auto &ops, const auto &) { return mlx::core::isinf(ops[0]); }},
    {"bitwise_not",
     [](const auto &ops, const auto &) {
       return mlx::core::bitwise_invert(ops[0]);
     }},
    {"conjugate",
     [](const auto &ops, const auto &) { return mlx::core::conjugate(ops[0]); }},
    {"logical_not",
     [](const auto &ops, const auto &) { return mlx::core::logical_not(ops[0]); }},
    {"sigmoid", [](const auto &ops, const auto &) { return mlx::core::sigmoid(ops[0]); }},
    {"asin", [](const auto &ops, const auto &) { return mlx::core::arcsin(ops[0]); }},
    {"asinh", [](const auto &ops, const auto &) { return mlx::core::arcsinh(ops[0]); }},
    {"acos", [](const auto &ops, const auto &) { return mlx::core::arccos(ops[0]); }},
    {"acosh", [](const auto &ops, const auto &) { return mlx::core::arccosh(ops[0]); }},
    {"atan", [](const auto &ops, const auto &) { return mlx::core::arctan(ops[0]); }},
    {"atanh", [](const auto &ops, const auto &) { return mlx::core::arctanh(ops[0]); }},
    {"cos", [](const auto &ops, const auto &) { return mlx::core::cos(ops[0]); }},
    {"cosh", [](const auto &ops, const auto &) { return mlx::core::cosh(ops[0]); }},
    {"erf", [](const auto &ops, const auto &) { return mlx::core::erf(ops[0]); }},
    {"erf_inv", [](const auto &ops, const auto &) { return mlx::core::erfinv(ops[0]); }},
    {"exp", [](const auto &ops, const auto &) { return mlx::core::exp(ops[0]); }},
    {"expm1", [](const auto &ops, const auto &) { return mlx::core::expm1(ops[0]); }},
    {"log", [](const auto &ops, const auto &) { return mlx::core::log(ops[0]); }},
    {"log1p", [](const auto &ops, const auto &) { return mlx::core::log1p(ops[0]); }},
    {"rsqrt", [](const auto &ops, const auto &) { return mlx::core::rsqrt(ops[0]); }},
    {"sin", [](const auto &ops, const auto &) { return mlx::core::sin(ops[0]); }},
    {"sinh", [](const auto &ops, const auto &) { return mlx::core::sinh(ops[0]); }},
    {"sqrt", [](const auto &ops, const auto &) { return mlx::core::sqrt(ops[0]); }},
    {"tan", [](const auto &ops, const auto &) { return mlx::core::tan(ops[0]); }},
    {"tanh", [](const auto &ops, const auto &) { return mlx::core::tanh(ops[0]); }},

    // cbrt = x^(1/3); returns NaN for negative real inputs (matches EMLX.Backend)
    {"cbrt",
     [](const auto &ops, const auto &) {
       return mlx::core::power(
           ops[0], mlx::core::full({}, static_cast<float>(1.0 / 3.0), mlx::core::float32));
     }},

    // erfc = 1 - erf(x)
    {"erfc",
     [](const auto &ops, const auto &) {
       auto e = mlx::core::erf(ops[0]);
       return mlx::core::subtract(mlx::core::ones({}, e.dtype()), e);
     }},

    // ── binary ops ────────────────────────────────────────────────────────
    {"add", [](const auto &ops, const auto &) { return mlx::core::add(ops[0], ops[1]); }},
    {"subtract",
     [](const auto &ops, const auto &) { return mlx::core::subtract(ops[0], ops[1]); }},
    {"multiply",
     [](const auto &ops, const auto &) { return mlx::core::multiply(ops[0], ops[1]); }},
    {"divide",
     [](const auto &ops, const auto &) { return mlx::core::divide(ops[0], ops[1]); }},
    {"pow",
     [](const auto &ops, const auto &) { return mlx::core::power(ops[0], ops[1]); }},
    {"atan2",
     [](const auto &ops, const auto &) { return mlx::core::arctan2(ops[0], ops[1]); }},
    {"min",
     [](const auto &ops, const auto &) { return mlx::core::minimum(ops[0], ops[1]); }},
    {"max",
     [](const auto &ops, const auto &) { return mlx::core::maximum(ops[0], ops[1]); }},
    {"quotient",
     [](const auto &ops, const auto &) { return mlx::core::floor_divide(ops[0], ops[1]); }},
    {"left_shift",
     [](const auto &ops, const auto &) { return mlx::core::left_shift(ops[0], ops[1]); }},
    {"right_shift",
     [](const auto &ops, const auto &) { return mlx::core::right_shift(ops[0], ops[1]); }},
    {"bitwise_and",
     [](const auto &ops, const auto &) { return mlx::core::bitwise_and(ops[0], ops[1]); }},
    {"bitwise_or",
     [](const auto &ops, const auto &) { return mlx::core::bitwise_or(ops[0], ops[1]); }},
    {"bitwise_xor",
     [](const auto &ops, const auto &) { return mlx::core::bitwise_xor(ops[0], ops[1]); }},

    // remainder: MLX uses floor-division semantics; fix up to truncation (same sign as dividend)
    // to match EMLX.Backend.remainder which applies this same correction.
    {"remainder",
     [](const auto &ops, const auto &) {
       auto rem = mlx::core::remainder(ops[0], ops[1]);
       auto zero = mlx::core::full({}, 0, ops[0].dtype());
       auto neg_dividend = mlx::core::less(ops[0], zero);
       auto adjusted = mlx::core::subtract(rem, ops[1]);
       return mlx::core::where(neg_dividend, adjusted, rem);
     }},

    // compare — result is MLX bool_; Elixir lowerer emits astype(bool_, uint8) after
    {"equal",
     [](const auto &ops, const auto &) { return mlx::core::equal(ops[0], ops[1]); }},
    {"not_equal",
     [](const auto &ops, const auto &) { return mlx::core::not_equal(ops[0], ops[1]); }},
    {"greater",
     [](const auto &ops, const auto &) { return mlx::core::greater(ops[0], ops[1]); }},
    {"less",
     [](const auto &ops, const auto &) { return mlx::core::less(ops[0], ops[1]); }},
    {"greater_equal",
     [](const auto &ops, const auto &) { return mlx::core::greater_equal(ops[0], ops[1]); }},
    {"less_equal",
     [](const auto &ops, const auto &) { return mlx::core::less_equal(ops[0], ops[1]); }},

    // logical binary
    {"logical_and",
     [](const auto &ops, const auto &) { return mlx::core::logical_and(ops[0], ops[1]); }},
    {"logical_or",
     [](const auto &ops, const auto &) { return mlx::core::logical_or(ops[0], ops[1]); }},
    // logical_xor = (a || b) && !(a && b)
    {"logical_xor",
     [](const auto &ops, const auto &) {
       auto t1 = mlx::core::logical_or(ops[0], ops[1]);
       auto t2 = mlx::core::logical_not(mlx::core::logical_and(ops[0], ops[1]));
       return mlx::core::logical_and(t1, t2);
     }},

    // ── shape / movement ops ──────────────────────────────────────────────────
    //
    // iattrs encoding (must stay in sync with EMLX.Native.Expr moduledoc):
    //   reshape:     attrs = [d0, d1, …]                       — target shape dims (flat)
    //   squeeze:     attrs = [a0, a1, …]                       — axes to remove (non-negative)
    //   transpose:   attrs = [p0, p1, …]                       — permutation (non-negative)
    //   bitcast:     attrs = [dtype_int]                       — target dtype
    //   broadcast:   attrs = [n, d0..dn-1, m, a0..am-1]       — shape then axes (length-delimited)
    //   pad:         attrs = [n_dims, lo0,hi0,int0, …]        — n_dims triples per dim
    //   reverse:     attrs = [a0, a1, …]                       — axes to flip (non-negative)
    //   concatenate: attrs = [axis]; ops = all input tensors
    //   stack:       attrs = [axis]; ops = all input tensors

    {"reshape",
     [](const auto &ops, const auto &attrs) {
       std::vector<int> shape;
       shape.reserve(attrs.size());
       for (auto d : attrs)
         shape.push_back(static_cast<int>(d));
       return mlx::core::reshape(ops[0], to_shape(shape));
     }},

    {"squeeze",
     [](const auto &ops, const auto &attrs) {
       if (attrs.empty())
         return ops[0];
       std::vector<int> axes;
       axes.reserve(attrs.size());
       for (auto a : attrs)
         axes.push_back(static_cast<int>(a));
       return mlx::core::squeeze(ops[0], axes);
     }},

    {"transpose",
     [](const auto &ops, const auto &attrs) {
       std::vector<int> axes;
       axes.reserve(attrs.size());
       for (auto a : attrs)
         axes.push_back(static_cast<int>(a));
       return mlx::core::transpose(ops[0], axes);
     }},

    {"bitcast",
     [](const auto &ops, const auto &attrs) {
       return mlx::core::view(ops[0], int_to_dtype(attrs[0]));
     }},

    // broadcast: reshape input to place source dims at the axis positions, then broadcast_to.
    // Mirrors EMLX.Backend.broadcast / maybe_reshape logic exactly.
    {"broadcast",
     [](const auto &ops, const auto &attrs) {
       int n_shape = static_cast<int>(attrs[0]);
       std::vector<int> target_shape;
       target_shape.reserve(n_shape);
       for (int i = 0; i < n_shape; i++)
         target_shape.push_back(static_cast<int>(attrs[1 + i]));

       int n_axes = static_cast<int>(attrs[1 + n_shape]);
       std::vector<int> axes;
       axes.reserve(n_axes);
       for (int i = 0; i < n_axes; i++)
         axes.push_back(static_cast<int>(attrs[1 + n_shape + 1 + i]));

       auto tensor = ops[0];
       auto in_shape = tensor.shape();

       // Build broadcast_shape: 1s everywhere, input dims at axis positions.
       std::vector<int> broadcast_shape(n_shape, 1);
       for (int i = 0; i < n_axes; i++) {
         if (!in_shape.empty())
           broadcast_shape[axes[i]] = static_cast<int>(in_shape[i]);
       }
       auto reshaped = mlx::core::reshape(tensor, to_shape(broadcast_shape));
       return mlx::core::broadcast_to(reshaped, to_shape(target_shape));
     }},

    // pad: non-negative lo/hi, interior always 0 (Elixir raises otherwise).
    {"pad",
     [](const auto &ops, const auto &attrs) {
       int n_dims = static_cast<int>(attrs[0]);
       std::vector<int> axes, low_pads, high_pads;
       axes.reserve(n_dims);
       low_pads.reserve(n_dims);
       high_pads.reserve(n_dims);
       for (int i = 0; i < n_dims; i++) {
         axes.push_back(i);
         low_pads.push_back(static_cast<int>(attrs[1 + i * 3 + 0]));
         high_pads.push_back(static_cast<int>(attrs[1 + i * 3 + 1]));
         // attrs[1 + i*3 + 2] = interior, always 0 (validated in Elixir lowerer)
       }
       return mlx::core::pad(ops[0], axes, to_shape(low_pads), to_shape(high_pads), ops[1],
                             "constant");
     }},

    // reverse: implemented via slice with negative strides, matching EMLX.Backend.reverse_mlx.
    {"reverse",
     [](const auto &ops, const auto &attrs) {
       auto tensor = ops[0];
       auto shape = tensor.shape();
       int rank = static_cast<int>(shape.size());

       std::vector<int> starts(rank), stops(rank), strides(rank);
       for (int i = 0; i < rank; i++) {
         starts[i] = 0;
         stops[i] = static_cast<int>(shape[i]);
         strides[i] = 1;
       }
       for (auto a : attrs) {
         int ax = static_cast<int>(a);
         int d = static_cast<int>(shape[ax]);
         starts[ax] = d - 1;
         stops[ax] = -(d + 1);
         strides[ax] = -1;
       }
       return mlx::core::slice(tensor, to_shape(starts), to_shape(stops), to_shape(strides));
     }},

    {"concatenate",
     [](const auto &ops, const auto &attrs) {
       int axis = static_cast<int>(attrs[0]);
       return mlx::core::concatenate(ops, axis);
     }},

    {"stack",
     [](const auto &ops, const auto &attrs) {
       int axis = static_cast<int>(attrs[0]);
       return mlx::core::stack(ops, axis);
     }},
};

// ── Expr destructor ───────────────────────────────────────────────────────────
//
// Evicts the per-Expr entry from MLX's global compile cache so stale compiled
// graphs don't accumulate.  Called by default_dtor<Expr> when the BEAM resource
// is GC'd.

Expr::~Expr() {
  if (compile_id != 0) {
    mlx::core::detail::compile_erase(compile_id);
  }
}

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
// a capturing interpreter lambda backed by the op registry, wraps it with
// mlx::core::compile(), and stores the result as an opaque Expr BEAM resource.
//
// argv[0] : num_inputs    (int)
// argv[1] : capture_refs  (list of MLX array resource refs)
// argv[2] : const_values  (list of doubles or ints)
// argv[3] : const_types   (list of dtype atoms, e.g. :float32)
// argv[4] : op_names      (list of strings — atom names matching op_registry keys)
// argv[5] : operands      (list of list of int64 — packed refs per instr)
// argv[6] : attrs         (list of list of int64 — integer attrs per instr)
// argv[7] : output_refs   (list of int64 — packed output refs)
ERL_NIF_TERM compile_program(ErlNifEnv *env, int argc,
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
    LIST_PARAM(4, std::vector<std::string>, op_names);

    std::vector<std::vector<int64_t>> operands;
    if (!parse_nested_int64_list(env, argv[5], operands))
      return nx::nif::error(env, "Unable to get operands nested list");

    std::vector<std::vector<int64_t>> attrs;
    if (!parse_nested_int64_list(env, argv[6], attrs))
      return nx::nif::error(env, "Unable to get attrs nested list");

    LIST_PARAM(7, std::vector<int64_t>, output_refs);

    // Validate all op names against the registry at compile time so that any
    // unknown op surfaces here rather than inside the lambda at eval time.
    for (const auto &name : op_names) {
      if (op_registry.find(name) == op_registry.end())
        return nx::nif::error(
            env, ("emlx::native: unknown op \"" + name + "\"").c_str());
    }

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
         op_names = std::move(op_names),
         operands = std::move(operands),
         attrs = std::move(attrs),
         output_refs = std::move(output_refs)](
            const std::vector<mlx::core::array> &inputs)
        -> std::vector<mlx::core::array> {
      std::vector<mlx::core::array> results;
      results.reserve(op_names.size());

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

      for (size_t i = 0; i < op_names.size(); i++) {
        std::vector<mlx::core::array> op_inputs;
        op_inputs.reserve(operands[i].size());
        for (int64_t ref : operands[i]) {
          op_inputs.push_back(resolve(ref));
        }
        results.push_back(op_registry.at(op_names[i])(op_inputs, attrs[i]));
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

    // Assign a unique ID so MLX's global compile cache has a distinct entry per
    // Expr resource.  All our lambdas share the same C++ type (same capture
    // types), so the public mlx::core::compile() would map them all to the same
    // cache key — causing stale graph reuse across different compiled programs.
    static std::atomic<std::uintptr_t> next_id{1};
    std::uintptr_t unique_id = next_id.fetch_add(1, std::memory_order_relaxed);

    new (ptr) Expr();
    ptr->num_inputs = num_inputs_val;
    ptr->compile_id = unique_id;
    ptr->compiled_fn = mlx::core::detail::compile(std::move(fn), unique_id);

    ERL_NIF_TERM ret = enif_make_resource(env, ptr);
    enif_release_resource(ptr);
    return nx::nif::ok(env, ret);
  }
  CATCH()
}

// eval_program — calls the MLX-compiled function against runtime inputs.
// MLX traces on the first call and replays the cached graph on subsequent calls.
// Returns lazy output array refs — materialization is deferred to the caller
// (to_binary / Nx.to_number), matching the Evaluator's deferred-eval pattern.
//
// argv[0] : program_ref  (emlx::native::Expr resource)
// argv[1] : input_refs   (list of MLX array resource refs — runtime inputs)
ERL_NIF_TERM eval_program(ErlNifEnv *env, int argc,
                          const ERL_NIF_TERM argv[]) {
  try {
    Expr *prog;
    if (!enif_get_resource(env, argv[0], resource_object<Expr>::type,
                           reinterpret_cast<void **>(&prog)))
      return nx::nif::error(env, "Invalid Expr resource");

    LIST_PARAM(1, std::vector<mlx::core::array>, inputs);

    // Tracing (first call) and graph replay (subsequent calls) both happen
    // inside compiled_fn.  Outputs are lazy — no eval needed here.
    auto outputs = prog->compiled_fn(inputs);

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

} // namespace native
} // namespace emlx
