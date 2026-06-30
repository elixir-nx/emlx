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
#include <mutex>
#include <numeric>
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

// Float opts (eps/scale/base) ride the int64 attr channel as their IEEE-754
// double bits (see EMLX.Native.Expr.f64_bits/1).  Reverse the reinterpret here.
static inline float attr_to_float(int64_t bits) {
  double d;
  std::memcpy(&d, &bits, sizeof(double));
  return static_cast<float>(d);
}

// ── Window-op helpers ─────────────────────────────────────────────────────
//
// These mirror the Elixir backend's sliding-window algorithm, which uses
// as_strided to build a view then reduces over the window dims.
// No explicit device needed: MLX ops inherit stream from input arrays.

// Build a sliding window view.  padded has shape [...]; window and strides
// have length n=padded.ndim().  Result shape: [o0,...,on-1, w0,...,wn-1].
static mlx::core::array compiler_sliding_window_view(const mlx::core::array &padded,
                                                     const std::vector<int> &window,
                                                     const std::vector<int> &strides) {
  int n = padded.ndim();
  auto ps = padded.shape();

  // Doubled element strides: output dims and window dims share the same strides.
  auto orig_strides = padded.strides();
  std::vector<int64_t> view_strides(orig_strides.begin(), orig_strides.end());
  for (auto s : orig_strides)
    view_strides.push_back(static_cast<int64_t>(s));

  std::vector<int> view_shape;
  for (int i = 0; i < n; ++i)
    view_shape.push_back(ps[i] - window[i] + 1);
  for (int w : window)
    view_shape.push_back(w);

  auto strided = mlx::core::as_strided(padded, to_shape(view_shape),
                                       to_strides(view_strides), 0);

  std::vector<int> starts(2 * n, 0);
  std::vector<int> stops = view_shape;
  std::vector<int> sl_strides = strides;
  for (int i = 0; i < n; ++i)
    sl_strides.push_back(1);

  return mlx::core::slice(strided, to_shape(starts), to_shape(stops), to_shape(sl_strides));
}

// Decode attrs and run a window reduction (sum/product/max/min).
// op_code: 0=sum, 1=product, 2=max, 3=min.
// attrs = [n_dims, op_int, lo0, hi0, …, s0, …, w0, …, wd0, …]
static mlx::core::array window_reduce_impl(const mlx::core::array &t,
                                           const std::vector<int64_t> &attrs,
                                           int op_code) {
  int n = static_cast<int>(attrs[0]);
  // attrs[1] is op_int — matches op_code parameter; keep in sync.

  std::vector<int> low_pad(n), high_pad(n), strides(n), window(n), wd(n);
  int off = 2;
  for (int i = 0; i < n; ++i) {
    low_pad[i] = static_cast<int>(attrs[off++]);
    high_pad[i] = static_cast<int>(attrs[off++]);
  }
  for (int i = 0; i < n; ++i)
    strides[i] = static_cast<int>(attrs[off++]);
  for (int i = 0; i < n; ++i)
    window[i] = static_cast<int>(attrs[off++]);
  for (int i = 0; i < n; ++i)
    wd[i] = static_cast<int>(attrs[off++]);

  // Apply window dilations by interior-padding the window mask.
  // The mask is a bool_ array of shape `window` with interior zeros inserted.
  // We achieve this via interior_padding: expanded window dims are
  //   expanded[i] = window[i] + (window[i]-1)*(wd[i]-1).
  std::vector<int> expanded_window(n);
  for (int i = 0; i < n; ++i)
    expanded_window[i] = window[i] + (window[i] - 1) * (wd[i] - 1);

  // Build a 1-filled window mask (bools), dilated via pad + interior zeros.
  auto one_bool = mlx::core::ones(to_shape(window), mlx::core::bool_);
  auto zero_bool = mlx::core::zeros({}, mlx::core::bool_);
  auto window_mask = one_bool;
  for (int i = 0; i < n; ++i) {
    if (wd[i] > 1) {
      // Interior-pad axis i by (wd[i]-1).
      std::vector<int> ax = {i};
      std::vector<int> lo_i(n, 0), hi_i(n, 0);
      auto cur_shape = window_mask.shape();
      int interior = wd[i] - 1;
      // Reconstruct the shape after interior padding.
      // Use as_strided trick: this is complex — use a simpler approach:
      // create full expanded shape, fill with zeros, scatter ones.
      // Alternatively: concatenate [1, 0, 0, ...] along axis repeatedly.
      // Simplest: use pad with interior param — but mlx::core::pad doesn't
      // support interior padding.  Instead use reshape+broadcast trick:
      // reshape from [w] to [w, 1], broadcast to [w, wd], reshape to [w*wd],
      // then slice [0, wd-1, 2*wd-1, …] — i.e. stride-wd indices.
      // We'll do this per axis sequentially.
      int w_i = (int)cur_shape[i];
      // New size after dilation on axis i.
      int new_size = w_i + (w_i - 1) * interior;
      // Build dilated axis: start with zeros, then set positions 0, wd, 2*wd, ...
      auto arange_full = mlx::core::arange(0, new_size, 1, mlx::core::int32);
      // Positions occupied by real values: 0, wd, 2*wd, ...
      auto divisible = mlx::core::equal(
          mlx::core::remainder(arange_full,
                               mlx::core::full({}, wd[i], mlx::core::int32)),
          mlx::core::zeros({}, mlx::core::int32));
      // Build dilated from current using take at strided positions.
      // Take the [0, wd, 2*wd, ...] positions from the window_mask axis i.
      auto src_idx = mlx::core::astype(
          mlx::core::divide(arange_full, mlx::core::full({}, wd[i], mlx::core::int32)),
          mlx::core::uint32);
      // For interior positions, take from mask (value 1); replace interior zeros.
      // It's simpler to: take all, then where(divisible, val, 0).
      auto taken = mlx::core::take(window_mask, src_idx, i);
      // Mask out non-original positions.
      // Broadcast divisible to match taken shape.
      std::vector<int> div_shape(window_mask.ndim(), 1);
      div_shape[i] = new_size;
      auto div_nd = mlx::core::reshape(divisible, to_shape(div_shape));
      auto div_bc = mlx::core::broadcast_to(div_nd, taken.shape());
      window_mask = mlx::core::where(div_bc, taken, mlx::core::zeros({}, mlx::core::bool_));
    }
  }
  // expanded_window is now the shape of window_mask.

  // Pad the input tensor.
  std::vector<int> all_axes(n);
  std::iota(all_axes.begin(), all_axes.end(), 0);

  // Build pad_value for this op.
  auto make_pad_val = [&]() -> mlx::core::array {
    switch (op_code) {
    case 0: // sum: 0
      return mlx::core::full({}, 0, t.dtype());
    case 1: // product: 1
      return mlx::core::full({}, 1, t.dtype());
    case 2: // max: lowest representable value
      if (mlx::core::issubdtype(t.dtype(), mlx::core::floating)) {
        return mlx::core::full({}, -std::numeric_limits<float>::infinity(), t.dtype());
      } else {
        return mlx::core::full({}, std::numeric_limits<int32_t>::min(), t.dtype());
      }
    case 3: // min: highest representable value
      if (mlx::core::issubdtype(t.dtype(), mlx::core::floating)) {
        return mlx::core::full({}, std::numeric_limits<float>::infinity(), t.dtype());
      } else {
        return mlx::core::full({}, std::numeric_limits<int32_t>::max(), t.dtype());
      }
    default:
      throw std::runtime_error("window_reduce_impl: unknown op_code");
    }
  };
  auto pad_val = make_pad_val();

  auto padded = mlx::core::pad(t, all_axes, to_shape(low_pad), to_shape(high_pad), pad_val,
                               "constant");

  // Sliding window view: shape [o0,...,on-1, w0,...,wn-1].
  auto view = compiler_sliding_window_view(padded, expanded_window, strides);

  // Broadcast window_mask (shape expanded_window) to match view: all batch dims are 1.
  std::vector<int> mask_bc_shape(n, 1);
  for (int w : expanded_window)
    mask_bc_shape.push_back(w);
  auto mask_reshaped = mlx::core::reshape(window_mask, to_shape(mask_bc_shape));
  auto mask_bc = mlx::core::broadcast_to(mask_reshaped, view.shape());

  // Apply mask: replace masked-out positions with pad_val.
  auto masked_view = mlx::core::where(mask_bc, view, mlx::core::broadcast_to(
      mlx::core::reshape(pad_val, to_shape(std::vector<int>{})), view.shape()));

  // Reduce over the last n dims (the window axes).
  std::vector<int> window_axes;
  for (int i = n; i < 2 * n; ++i)
    window_axes.push_back(i);

  switch (op_code) {
  case 0:
    return mlx::core::sum(masked_view, window_axes, false);
  case 1:
    return mlx::core::prod(masked_view, window_axes, false);
  case 2:
    return mlx::core::max(masked_view, window_axes, false);
  case 3:
    return mlx::core::min(masked_view, window_axes, false);
  default:
    throw std::runtime_error("window_reduce_impl: unknown op_code");
  }
}

// Window scatter (max or min): replicate window_scatter_impl from emlx_nif.cpp.
// attrs = [n_dims, lo0, hi0, …, s0, …, w0, …]
// ops = [tensor_t, source, init_value]
static mlx::core::array window_scatter_impl_compiler(const mlx::core::array &tensor_t,
                                                     const mlx::core::array &tensor_source,
                                                     const mlx::core::array &tensor_init_value,
                                                     const std::vector<int64_t> &attrs,
                                                     bool scatter_max) {
  int n = static_cast<int>(attrs[0]);
  std::vector<int> low_pad(n), high_pad(n), strides(n), window(n);
  int off = 1;
  for (int i = 0; i < n; ++i) {
    low_pad[i] = static_cast<int>(attrs[off++]);
    high_pad[i] = static_cast<int>(attrs[off++]);
  }
  for (int i = 0; i < n; ++i)
    strides[i] = static_cast<int>(attrs[off++]);
  for (int i = 0; i < n; ++i)
    window[i] = static_cast<int>(attrs[off++]);

  auto init_casted = mlx::core::astype(tensor_init_value, tensor_t.dtype());
  std::vector<int> all_axes(n);
  std::iota(all_axes.begin(), all_axes.end(), 0);
  auto padded = mlx::core::pad(tensor_t, all_axes, to_shape(low_pad), to_shape(high_pad),
                               init_casted, "constant");

  auto padded_shape = padded.shape();
  std::vector<int> padded_shape_vec(padded_shape.begin(), padded_shape.end());

  auto window_view = compiler_sliding_window_view(padded, window, strides);

  std::vector<int> out_shape(window_view.shape().begin(), window_view.shape().begin() + n);

  int K = 1;
  for (int w : window)
    K *= w;

  std::vector<int> flat_shape = out_shape;
  flat_shape.push_back(K);
  auto windows_flat = mlx::core::reshape(window_view, to_shape(flat_shape));

  auto arg_idx = [&]() -> mlx::core::array {
    if (scatter_max) {
      return mlx::core::argmax(windows_flat, n, false);
    }
    auto m = mlx::core::min(windows_flat, std::vector<int>{n}, true);
    auto mask =
        mlx::core::astype(mlx::core::equal(windows_flat, m), mlx::core::uint32);
    auto arange_k = mlx::core::astype(mlx::core::arange(0, K, 1), mlx::core::uint32);
    std::vector<int> arange_shape(n + 1, 1);
    arange_shape[n] = K;
    auto arange_nd = mlx::core::reshape(arange_k, to_shape(arange_shape));
    auto weighted = mlx::core::multiply(mask, arange_nd);
    return mlx::core::argmax(weighted, n, false);
  }();

  std::vector<int> arg_exp_shape = out_shape;
  arg_exp_shape.push_back(1);
  auto arg_idx_exp = mlx::core::reshape(arg_idx, to_shape(arg_exp_shape));

  std::vector<mlx::core::array> abs_indices;
  for (int a = 0; a < n; ++a) {
    auto arange_a = mlx::core::astype(mlx::core::arange(0, (int)padded_shape[a], 1),
                                      mlx::core::int32);
    std::vector<int> iota_shape(n, 1);
    iota_shape[a] = (int)padded_shape[a];
    auto iota_nd = mlx::core::reshape(arange_a, to_shape(iota_shape));
    auto iota_bc = mlx::core::broadcast_to(iota_nd, to_shape(padded_shape_vec));
    auto iota_view = compiler_sliding_window_view(iota_bc, window, strides);
    auto iota_flat = mlx::core::reshape(iota_view, to_shape(flat_shape));
    auto abs_a = mlx::core::take_along_axis(iota_flat, arg_idx_exp, n);
    abs_indices.push_back(mlx::core::reshape(abs_a, to_shape(out_shape)));
  }

  auto source_shape_2n = std::vector<int>(tensor_source.shape().begin(),
                                          tensor_source.shape().end());
  for (int i = 0; i < n; ++i)
    source_shape_2n.push_back(1);
  auto updates = mlx::core::reshape(tensor_source, to_shape(source_shape_2n));

  auto buffer = mlx::core::broadcast_to(
      mlx::core::reshape(init_casted, to_shape(std::vector<int>{})),
      to_shape(padded_shape_vec));

  std::vector<int> scatter_axes(n);
  std::iota(scatter_axes.begin(), scatter_axes.end(), 0);
  auto scattered = mlx::core::scatter_add(buffer, abs_indices, updates, scatter_axes);

  auto orig_shape = tensor_t.shape();
  std::vector<int> slice_starts = low_pad;
  std::vector<int> slice_stops(n);
  for (int i = 0; i < n; ++i)
    slice_stops[i] = low_pad[i] + (int)orig_shape[i];
  std::vector<int> slice_ones(n, 1);
  return mlx::core::slice(scattered, to_shape(slice_starts), to_shape(slice_stops),
                          to_shape(slice_ones));
}

// MLX linalg primitives are CPU-only.  Pin them to the CPU device so they
// compose inside a compiled graph regardless of the graph's default (worker)
// stream — validated to work for both :cpu and :gpu default devices.
static const mlx::core::Device k_linalg_cpu(mlx::core::Device::DeviceType::cpu, 0);

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

    // ── reductions ────────────────────────────────────────────────────────────
    //
    // iattrs = [keep_axes_int, a0, a1, …]
    // keep_axes_int: 0 = false, 1 = true.  Axis list always explicit (Elixir resolves nil).
    // sum, product, reduce_max, reduce_min: emit astype in Elixir lowerer for type changes.
    // all, any: Elixir always emits astype since MLX returns bool_.

    {"sum",
     [](const auto &ops, const auto &attrs) {
       bool keepdims = static_cast<bool>(attrs[0]);
       std::vector<int> axes;
       for (size_t i = 1; i < attrs.size(); i++)
         axes.push_back(static_cast<int>(attrs[i]));
       return mlx::core::sum(ops[0], axes, keepdims);
     }},

    {"product",
     [](const auto &ops, const auto &attrs) {
       bool keepdims = static_cast<bool>(attrs[0]);
       std::vector<int> axes;
       for (size_t i = 1; i < attrs.size(); i++)
         axes.push_back(static_cast<int>(attrs[i]));
       return mlx::core::prod(ops[0], axes, keepdims);
     }},

    {"all",
     [](const auto &ops, const auto &attrs) {
       bool keepdims = static_cast<bool>(attrs[0]);
       std::vector<int> axes;
       for (size_t i = 1; i < attrs.size(); i++)
         axes.push_back(static_cast<int>(attrs[i]));
       return mlx::core::all(ops[0], axes, keepdims);
     }},

    {"any",
     [](const auto &ops, const auto &attrs) {
       bool keepdims = static_cast<bool>(attrs[0]);
       std::vector<int> axes;
       for (size_t i = 1; i < attrs.size(); i++)
         axes.push_back(static_cast<int>(attrs[i]));
       return mlx::core::any(ops[0], axes, keepdims);
     }},

    {"reduce_max",
     [](const auto &ops, const auto &attrs) {
       bool keepdims = static_cast<bool>(attrs[0]);
       std::vector<int> axes;
       for (size_t i = 1; i < attrs.size(); i++)
         axes.push_back(static_cast<int>(attrs[i]));
       return mlx::core::max(ops[0], axes, keepdims);
     }},

    {"reduce_min",
     [](const auto &ops, const auto &attrs) {
       bool keepdims = static_cast<bool>(attrs[0]);
       std::vector<int> axes;
       for (size_t i = 1; i < attrs.size(); i++)
         axes.push_back(static_cast<int>(attrs[i]));
       return mlx::core::min(ops[0], axes, keepdims);
     }},

    // argmax / argmin — iattrs = [axis, keep_axis_int]; axis = -1 means global.
    {"argmax",
     [](const auto &ops, const auto &attrs) {
       int axis = static_cast<int>(attrs[0]);
       bool keepdims = static_cast<bool>(attrs[1]);
       if (axis < 0)
         return mlx::core::argmax(ops[0], keepdims);
       return mlx::core::argmax(ops[0], axis, keepdims);
     }},

    {"argmin",
     [](const auto &ops, const auto &attrs) {
       int axis = static_cast<int>(attrs[0]);
       bool keepdims = static_cast<bool>(attrs[1]);
       if (axis < 0)
         return mlx::core::argmin(ops[0], keepdims);
       return mlx::core::argmin(ops[0], axis, keepdims);
     }},

    // ── dot ───────────────────────────────────────────────────────────────────
    //
    // iattrs = [n_ca, ca…, n_cb, cb…, n_ba, ba…, n_bb, bb…]
    // Elixir lowerer casts both operands to computation_type before the dot op.
    // Non-batched (n_ba=0, n_bb=0): mlx::core::tensordot(left, right, ca, cb).
    // Batched: rebuild einsum spec from shapes + 4 axis lists, call einsum.

    {"dot",
     [](const auto &ops, const auto &attrs) {
       // Parse 4 length-delimited axis lists.
       size_t off = 0;
       auto parse_axis_list = [&]() -> std::vector<int> {
         int n = static_cast<int>(attrs[off++]);
         std::vector<int> v(n);
         for (int i = 0; i < n; i++)
           v[i] = static_cast<int>(attrs[off++]);
         return v;
       };
       auto ca = parse_axis_list();
       auto cb = parse_axis_list();
       auto ba = parse_axis_list();
       auto bb = parse_axis_list();

       if (ba.empty() && bb.empty()) {
         return mlx::core::tensordot(ops[0], ops[1], ca, cb);
       }

       // Batched: build einsum spec by assigning letter labels.
       auto left_shape = ops[0].shape();
       auto right_shape = ops[1].shape();
       int n_left = static_cast<int>(left_shape.size());
       int n_right = static_cast<int>(right_shape.size());

       // Assign a label to every left and right axis.
       const std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
       std::vector<char> ll(n_left), rl(n_right);
       for (int i = 0; i < n_left; i++)
         ll[i] = alphabet[i];
       for (int i = 0; i < n_right; i++)
         rl[i] = alphabet[n_left + i];

       // Share labels for batch and contraction axes.
       auto share = [&](const std::vector<int> &la, const std::vector<int> &ra) {
         for (size_t i = 0; i < la.size(); i++)
           rl[ra[i]] = ll[la[i]];
       };
       share(ba, bb);
       share(ca, cb);

       // Output: batch axes, then free left axes, then free right axes.
       auto contains = [](const std::vector<int> &v, int x) {
         return std::find(v.begin(), v.end(), x) != v.end();
       };
       std::string output;
       for (int b : ba)
         output += ll[b];
       for (int i = 0; i < n_left; i++)
         if (!contains(ca, i) && !contains(ba, i))
           output += ll[i];
       for (int i = 0; i < n_right; i++)
         if (!contains(cb, i) && !contains(bb, i))
           output += rl[i];

       std::string spec = std::string(ll.begin(), ll.end()) + "," +
                          std::string(rl.begin(), rl.end()) + "->" + output;
       return mlx::core::einsum(spec, {ops[0], ops[1]});
     }},

    // ── indexing / selection ─────────────────────────────────────────────────
    //
    // select: operands = [pred, on_true, on_false]; no attrs.
    {"select",
     [](const auto &ops, const auto &) {
       return mlx::core::where(ops[0], ops[1], ops[2]);
     }},

    // clip: operands = [tensor, min, max]; no attrs.
    {"clip",
     [](const auto &ops, const auto &) {
       return mlx::core::clip(ops[0], ops[1], ops[2]);
     }},

    // slice: attrs = [n_dims, dyn_mask, d0..dn-1, l0..ln-1, str0..strn-1, sv0..svn-1]
    // Operands: [tensor, dyn_start_0, dyn_start_1, …] — dynamic starts in axis order.
    //
    // Strategy: one static mlx::core::slice for all static dims (full span for dynamic dims),
    // then mlx::core::take per dynamic dim to apply the dynamic start + strided selection.
    {"slice",
     [](const auto &ops, const auto &attrs) {
       int n_dims = static_cast<int>(attrs[0]);
       int64_t dyn_mask = attrs[1];

       std::vector<int> input_shape(n_dims), lengths(n_dims), strides_v(n_dims),
           sv(n_dims);
       for (int i = 0; i < n_dims; i++)
         input_shape[i] = static_cast<int>(attrs[2 + i]);
       for (int i = 0; i < n_dims; i++)
         lengths[i] = static_cast<int>(attrs[2 + n_dims + i]);
       for (int i = 0; i < n_dims; i++)
         strides_v[i] = static_cast<int>(attrs[2 + 2 * n_dims + i]);
       for (int i = 0; i < n_dims; i++)
         sv[i] = static_cast<int>(attrs[2 + 3 * n_dims + i]);

       // Build the static slice: use clamped static values for static dims,
       // full extent for dynamic dims (handled below via take).
       std::vector<int> starts(n_dims), stops(n_dims), slice_strides(n_dims);
       for (int i = 0; i < n_dims; i++) {
         slice_strides[i] = ((dyn_mask >> i) & 1) ? 1 : strides_v[i];
         if (!((dyn_mask >> i) & 1)) {
           starts[i] = std::max(0, std::min(sv[i], input_shape[i] - lengths[i]));
           stops[i] = starts[i] + lengths[i];
         } else {
           starts[i] = 0;
           stops[i] = input_shape[i];
         }
       }
       auto result =
           mlx::core::slice(ops[0], to_shape(starts), to_shape(stops), to_shape(slice_strides));

       // For each dynamic dim, apply take with arange*stride + clamped_start.
       int dyn_op_idx = 1;
       for (int i = 0; i < n_dims; i++) {
         if (!((dyn_mask >> i) & 1))
           continue;
         int len_i = lengths[i];
         int str_i = strides_v[i];
         // clamped_start = clip(start_arr, 0, dim_i - len_i)
         auto s = mlx::core::astype(ops[dyn_op_idx++], mlx::core::int32);
         auto zero = mlx::core::zeros({}, mlx::core::int32);
         auto max_s = mlx::core::full({}, input_shape[i] - len_i, mlx::core::int32);
         auto clamped = mlx::core::minimum(mlx::core::maximum(s, zero), max_s);
         // idx = arange(len_i) * str_i + clamped
         auto base = mlx::core::arange(0, len_i, 1, mlx::core::int32);
         if (str_i != 1)
           base = mlx::core::multiply(base, mlx::core::full({}, str_i, mlx::core::int32));
         auto idx = mlx::core::add(base, clamped);
         result = mlx::core::take(result, idx, i);
       }
       return result;
     }},

    // put_slice: attrs = [n_dims, dyn_mask, d0..dn-1, l0..ln-1, sv0..svn-1]
    // Operands: [input, slice, dyn_start_0, …]
    //
    // Builds a clamped int32 start array and calls the dynamic slice_update overload.
    {"put_slice",
     [](const auto &ops, const auto &attrs) {
       int n_dims = static_cast<int>(attrs[0]);
       int64_t dyn_mask = attrs[1];

       std::vector<int> input_shape(n_dims), lengths(n_dims), sv(n_dims);
       for (int i = 0; i < n_dims; i++)
         input_shape[i] = static_cast<int>(attrs[2 + i]);
       for (int i = 0; i < n_dims; i++)
         lengths[i] = static_cast<int>(attrs[2 + n_dims + i]);
       for (int i = 0; i < n_dims; i++)
         sv[i] = static_cast<int>(attrs[2 + 2 * n_dims + i]);

       // Build per-dim clamped start components (shape [1]) then concatenate.
       std::vector<mlx::core::array> start_components;
       start_components.reserve(n_dims);
       int dyn_op_idx = 2; // ops[0]=input, ops[1]=slice, ops[2..]=dynamic starts
       for (int i = 0; i < n_dims; i++) {
         int max_start_i = input_shape[i] - lengths[i];
         if ((dyn_mask >> i) & 1) {
           auto s = mlx::core::astype(ops[dyn_op_idx++], mlx::core::int32);
           s = mlx::core::reshape(s, {1});
           auto max_s = mlx::core::full({1}, max_start_i, mlx::core::int32);
           auto zero = mlx::core::zeros({1}, mlx::core::int32);
           start_components.push_back(mlx::core::minimum(mlx::core::maximum(s, zero), max_s));
         } else {
           int clamped_i = std::max(0, std::min(sv[i], max_start_i));
           start_components.push_back(mlx::core::full({1}, clamped_i, mlx::core::int32));
         }
       }
       auto start_arr = mlx::core::concatenate(start_components, 0);

       // All axes: [0, 1, ..., n_dims-1]
       std::vector<int> all_axes(n_dims);
       std::iota(all_axes.begin(), all_axes.end(), 0);
       return mlx::core::slice_update(ops[0], ops[1], start_arr, all_axes);
     }},

    // gather: attrs = [n_gather_axes, a0…, n_tensor_dims, ss0…, n_out_dims, od0…]
    // Operands: [tensor, indices]  — indices shape = [batch…, n_gather_axes]
    {"gather",
     [](const auto &ops, const auto &attrs) {
       int n_gather_axes = static_cast<int>(attrs[0]);
       std::vector<int> axes(n_gather_axes);
       for (int i = 0; i < n_gather_axes; i++)
         axes[i] = static_cast<int>(attrs[1 + i]);

       int n_tensor_dims = static_cast<int>(attrs[1 + n_gather_axes]);
       std::vector<int> slice_sizes(n_tensor_dims);
       for (int i = 0; i < n_tensor_dims; i++)
         slice_sizes[i] = static_cast<int>(attrs[2 + n_gather_axes + i]);

       int n_out_dims = static_cast<int>(attrs[2 + n_gather_axes + n_tensor_dims]);
       std::vector<int> out_shape_v(n_out_dims);
       for (int i = 0; i < n_out_dims; i++)
         out_shape_v[i] = static_cast<int>(attrs[3 + n_gather_axes + n_tensor_dims + i]);

       // Split indices along its last axis: one array per gather axis.
       auto indices = ops[1];
       auto idx_shape = indices.shape();
       int last = static_cast<int>(idx_shape.size()) - 1;

       std::vector<mlx::core::array> indices_list;
       indices_list.reserve(n_gather_axes);
       for (int i = 0; i < n_gather_axes; i++) {
         std::vector<int> s_starts(idx_shape.size(), 0), s_stops(idx_shape.size());
         std::vector<int> s_strides(idx_shape.size(), 1);
         for (size_t j = 0; j < idx_shape.size(); j++)
           s_stops[j] = static_cast<int>(idx_shape[j]);
         s_starts[last] = i;
         s_stops[last] = i + 1;
         auto sl = mlx::core::slice(indices, to_shape(s_starts), to_shape(s_stops),
                                    to_shape(s_strides));
         indices_list.push_back(mlx::core::squeeze(sl, {last}));
       }

       auto result = mlx::core::gather(ops[0], indices_list, axes, to_shape(slice_sizes));
       return mlx::core::reshape(result, to_shape(out_shape_v));
     }},

    // take: operands = [tensor, indices]; attrs = [axis].
    {"take",
     [](const auto &ops, const auto &attrs) {
       int axis = static_cast<int>(attrs[0]);
       return mlx::core::take(ops[0], ops[1], axis);
     }},

    // take_along_axis: operands = [tensor, indices]; attrs = [axis].
    {"take_along_axis",
     [](const auto &ops, const auto &attrs) {
       int axis = static_cast<int>(attrs[0]);
       return mlx::core::take_along_axis(ops[0], ops[1], axis);
     }},

    // indexed_add: attrs = [n_axes, a0…, n_updates_shape, us0…]
    // Operands: [target, indices, updates_pre_reshape]
    // Mirrors EMLX.Backend.indexed_op(:scatter_add, …)
    {"indexed_add",
     [](const auto &ops, const auto &attrs) {
       int n_axes = static_cast<int>(attrs[0]);
       std::vector<int> axes(n_axes);
       for (int i = 0; i < n_axes; i++)
         axes[i] = static_cast<int>(attrs[1 + i]);
       int n_upd = static_cast<int>(attrs[1 + n_axes]);
       std::vector<int> upd_shape(n_upd);
       for (int i = 0; i < n_upd; i++)
         upd_shape[i] = static_cast<int>(attrs[2 + n_axes + i]);

       auto indices = ops[1];
       auto idx_shape = indices.shape();
       int last = static_cast<int>(idx_shape.size()) - 1;

       std::vector<mlx::core::array> indices_list;
       indices_list.reserve(n_axes);
       for (int i = 0; i < n_axes; i++) {
         std::vector<int> s_starts(idx_shape.size(), 0), s_stops(idx_shape.size());
         std::vector<int> s_strides(idx_shape.size(), 1);
         for (size_t j = 0; j < idx_shape.size(); j++)
           s_stops[j] = static_cast<int>(idx_shape[j]);
         s_starts[last] = i;
         s_stops[last] = i + 1;
         auto sl = mlx::core::slice(indices, to_shape(s_starts), to_shape(s_stops),
                                    to_shape(s_strides));
         indices_list.push_back(mlx::core::squeeze(sl, {last}));
       }

       auto updates_reshaped = mlx::core::reshape(ops[2], to_shape(upd_shape));
       return mlx::core::scatter_add(ops[0], indices_list, updates_reshaped, axes);
     }},

    // indexed_put: same structure as indexed_add but uses scatter (replace semantics).
    {"indexed_put",
     [](const auto &ops, const auto &attrs) {
       int n_axes = static_cast<int>(attrs[0]);
       std::vector<int> axes(n_axes);
       for (int i = 0; i < n_axes; i++)
         axes[i] = static_cast<int>(attrs[1 + i]);
       int n_upd = static_cast<int>(attrs[1 + n_axes]);
       std::vector<int> upd_shape(n_upd);
       for (int i = 0; i < n_upd; i++)
         upd_shape[i] = static_cast<int>(attrs[2 + n_axes + i]);

       auto indices = ops[1];
       auto idx_shape = indices.shape();
       int last = static_cast<int>(idx_shape.size()) - 1;

       std::vector<mlx::core::array> indices_list;
       indices_list.reserve(n_axes);
       for (int i = 0; i < n_axes; i++) {
         std::vector<int> s_starts(idx_shape.size(), 0), s_stops(idx_shape.size());
         std::vector<int> s_strides(idx_shape.size(), 1);
         for (size_t j = 0; j < idx_shape.size(); j++)
           s_stops[j] = static_cast<int>(idx_shape[j]);
         s_starts[last] = i;
         s_stops[last] = i + 1;
         auto sl = mlx::core::slice(indices, to_shape(s_starts), to_shape(s_stops),
                                    to_shape(s_strides));
         indices_list.push_back(mlx::core::squeeze(sl, {last}));
       }

       auto updates_reshaped = mlx::core::reshape(ops[2], to_shape(upd_shape));
       return mlx::core::scatter(ops[0], indices_list, updates_reshaped, axes);
     }},

    // ── sort / argsort ─────────────────────────────────────────────────────────
    //
    // iattrs = [axis, asc_int]  (asc_int: 1=ascending, 0=descending)
    //
    // Replicates EMLX.Backend.sort/argsort NaN-aware algorithm:
    //   1. argsort(t, axis)  (negate t first if descending)
    //   2. sorted_values = take_along_axis(t, sort_indices, axis)
    //   3. is_nan = isnan(sorted_values)
    //   4. partition_indices = argsort(is_nan, axis)        (ascending)
    //               OR        argsort(!is_nan, axis)        (descending)
    //      (cast bool→uint8 to avoid Metal metallib gap)
    //   5. return take_along_axis(sorted_values, partition, axis)
    // argsort: same but step 5 applies partition to sort_indices.

    {"sort",
     [](const auto &ops, const auto &attrs) {
       int axis = static_cast<int>(attrs[0]);
       bool asc = (attrs[1] != 0);

       // Step 1: get initial argsort indices.
       auto t = ops[0];
       auto sort_input = asc ? t : mlx::core::negative(t);
       auto sort_idx = mlx::core::argsort(sort_input, axis);

       // Step 2: gather sorted values.
       auto sorted_vals = mlx::core::take_along_axis(t, sort_idx, axis);

       // Step 3: NaN mask.
       auto is_nan = mlx::core::isnan(sorted_vals);

       // Step 4: partition indices (move NaNs to correct end).
       auto is_nan_u8 = mlx::core::astype(is_nan, mlx::core::uint8);
       mlx::core::array partition_input =
           asc ? is_nan_u8
               : mlx::core::astype(mlx::core::logical_not(is_nan), mlx::core::uint8);
       auto partition = mlx::core::argsort(partition_input, axis);

       // Step 5: reorder sorted_values by partition.
       return mlx::core::take_along_axis(sorted_vals, partition, axis);
     }},

    {"argsort",
     [](const auto &ops, const auto &attrs) {
       int axis = static_cast<int>(attrs[0]);
       bool asc = (attrs[1] != 0);

       // Step 1: get initial argsort indices.
       auto t = ops[0];
       auto sort_input = asc ? t : mlx::core::negative(t);
       auto sort_idx = mlx::core::argsort(sort_input, axis);

       // Step 2: gather sorted values to check NaN.
       auto sorted_vals = mlx::core::take_along_axis(t, sort_idx, axis);

       // Step 3: NaN mask.
       auto is_nan = mlx::core::isnan(sorted_vals);

       // Step 4: partition indices.
       auto is_nan_u8 = mlx::core::astype(is_nan, mlx::core::uint8);
       mlx::core::array partition_input =
           asc ? is_nan_u8
               : mlx::core::astype(mlx::core::logical_not(is_nan), mlx::core::uint8);
       auto partition = mlx::core::argsort(partition_input, axis);

       // Step 5: reorder sort_idx by partition.
       return mlx::core::take_along_axis(sort_idx, partition, axis);
     }},

    // ── window reductions ─────────────────────────────────────────────────────
    //
    // Shared helper: build a sliding-window view of a padded tensor.
    // Returns shape [o0,...,on-1, w0,...,wn-1] where oi = (ps[i]-w[i])/s[i]+1.
    // Uses as_strided + slice, matching EMLX.Backend.sliding_window_view.
    //
    // iattrs = [n_dims, op_int, lo0, hi0, …, s0, …, w0, …, wd0, …]
    //   op_int: 0=sum, 1=product, 2=max, 3=min
    //   lo/hi: n_dims pairs of padding (2*n_dims values starting at offset 2)
    //   strides: n_dims values
    //   window: n_dims values
    //   window_dilations: n_dims values

    {"window_sum",
     [](const auto &ops, const auto &attrs) {
       return window_reduce_impl(ops[0], attrs, 0);
     }},
    {"window_product",
     [](const auto &ops, const auto &attrs) {
       return window_reduce_impl(ops[0], attrs, 1);
     }},
    {"window_max",
     [](const auto &ops, const auto &attrs) {
       return window_reduce_impl(ops[0], attrs, 2);
     }},
    {"window_min",
     [](const auto &ops, const auto &attrs) {
       return window_reduce_impl(ops[0], attrs, 3);
     }},

    // ── window_scatter_max / min ───────────────────────────────────────────────
    //
    // iattrs = [n_dims, lo0, hi0, …, s0, …, w0, …]
    // Operands: [tensor_t, source, init_value]
    // Replicates window_scatter_impl from emlx_nif.cpp.

    {"window_scatter_max",
     [](const auto &ops, const auto &attrs) {
       return window_scatter_impl_compiler(ops[0], ops[1], ops[2], attrs, true);
     }},
    {"window_scatter_min",
     [](const auto &ops, const auto &attrs) {
       return window_scatter_impl_compiler(ops[0], ops[1], ops[2], attrs, false);
     }},

    // ── cumulative reductions ─────────────────────────────────────────────────
    //
    // iattrs = [axis, reverse_int]
    // inclusive is always 1 (matches Nx semantics).

    {"cumulative_sum",
     [](const auto &ops, const auto &attrs) {
       int axis = static_cast<int>(attrs[0]);
       bool rev = (attrs[1] != 0);
       return mlx::core::cumsum(ops[0], axis, rev, true);
     }},
    {"cumulative_product",
     [](const auto &ops, const auto &attrs) {
       int axis = static_cast<int>(attrs[0]);
       bool rev = (attrs[1] != 0);
       return mlx::core::cumprod(ops[0], axis, rev, true);
     }},
    {"cumulative_min",
     [](const auto &ops, const auto &attrs) {
       int axis = static_cast<int>(attrs[0]);
       bool rev = (attrs[1] != 0);
       return mlx::core::cummin(ops[0], axis, rev, true);
     }},
    {"cumulative_max",
     [](const auto &ops, const auto &attrs) {
       int axis = static_cast<int>(attrs[0]);
       bool rev = (attrs[1] != 0);
       return mlx::core::cummax(ops[0], axis, rev, true);
     }},

    // ── fft / ifft ────────────────────────────────────────────────────────────
    //
    // iattrs = [axis, n]  where n is the FFT length.

    {"fft",
     [](const auto &ops, const auto &attrs) {
       int axis = static_cast<int>(attrs[0]);
       int n = static_cast<int>(attrs[1]);
       return mlx::core::fft::fft(ops[0], n, axis, mlx::core::fft::FFTNorm::Backward);
     }},
    {"ifft",
     [](const auto &ops, const auto &attrs) {
       int axis = static_cast<int>(attrs[0]);
       int n = static_cast<int>(attrs[1]);
       return mlx::core::fft::ifft(ops[0], n, axis, mlx::core::fft::FFTNorm::Backward);
     }},

    // ── fft2 / ifft2 ──────────────────────────────────────────────────────────
    //
    // iattrs = [ax0, ax1, n0, n1]

    {"fft2",
     [](const auto &ops, const auto &attrs) {
       std::vector<int> axes = {static_cast<int>(attrs[0]), static_cast<int>(attrs[1])};
       std::vector<int> ns = {static_cast<int>(attrs[2]), static_cast<int>(attrs[3])};
       return mlx::core::fft::fft2(ops[0], to_shape(ns), axes,
                                   mlx::core::fft::FFTNorm::Backward);
     }},
    {"ifft2",
     [](const auto &ops, const auto &attrs) {
       std::vector<int> axes = {static_cast<int>(attrs[0]), static_cast<int>(attrs[1])};
       std::vector<int> ns = {static_cast<int>(attrs[2]), static_cast<int>(attrs[3])};
       return mlx::core::fft::ifft2(ops[0], to_shape(ns), axes,
                                    mlx::core::fft::FFTNorm::Backward);
     }},

    // ── creation ops ─────────────────────────────────────────────────────────
    //
    // iota: attrs = [dtype_int, n_dims, axis_int, d0..dn-1]
    // No operands.  axis_int = -1 means flat enumeration (no axis).
    {"iota",
     [](const auto & /*ops*/, const auto &attrs) {
       auto dtype = int_to_dtype(attrs[0]);
       int n = static_cast<int>(attrs[1]);
       int axis = static_cast<int>(attrs[2]);

       std::vector<int> shape(n);
       for (int i = 0; i < n; i++)
         shape[i] = static_cast<int>(attrs[3 + i]);

       if (n == 0) {
         // scalar iota: always 0
         return mlx::core::astype(mlx::core::full({}, 0, mlx::core::int32), dtype);
       }

       if (axis == -1) {
         // flat enumeration: arange(0, product(shape)) reshaped
         int total = 1;
         for (int d : shape)
           total *= d;
         auto flat = mlx::core::arange(0, total, 1, mlx::core::int32);
         auto reshaped = mlx::core::reshape(flat, to_shape(shape));
         return mlx::core::astype(reshaped, dtype);
       } else {
         // axis-specific iota: arange(0, shape[axis]), broadcast to full shape
         int dim = shape[axis];
         auto linear = mlx::core::arange(0, dim, 1, mlx::core::int32);
         std::vector<int> rs(n, 1);
         rs[axis] = dim;
         auto r = mlx::core::reshape(linear, to_shape(rs));
         auto bc = mlx::core::broadcast_to(r, to_shape(shape));
         return mlx::core::astype(bc, dtype);
       }
     }},

    // eye: attrs = [dtype_int, m, n].  No operands.
    {"eye",
     [](const auto & /*ops*/, const auto &attrs) {
       auto dtype = int_to_dtype(attrs[0]);
       int m = static_cast<int>(attrs[1]);
       int n = static_cast<int>(attrs[2]);
       return mlx::core::eye(m, n, 0, dtype);
     }},

    // ── conv_general ─────────────────────────────────────────────────────────
    {"conv_general",
     [](const auto &ops, const auto &attrs) {
       int n_dims = static_cast<int>(attrs[0]);
       int off = 1;

       std::vector<int> strides(n_dims), padding_lo(n_dims), padding_hi(n_dims),
           kernel_dilation(n_dims), input_dilation(n_dims);

       for (int i = 0; i < n_dims; i++)
         strides[i] = static_cast<int>(attrs[off++]);
       for (int i = 0; i < n_dims; i++) {
         padding_lo[i] = static_cast<int>(attrs[off++]);
         padding_hi[i] = static_cast<int>(attrs[off++]);
       }
       for (int i = 0; i < n_dims; i++)
         kernel_dilation[i] = static_cast<int>(attrs[off++]);
       for (int i = 0; i < n_dims; i++)
         input_dilation[i] = static_cast<int>(attrs[off++]);
       int fgs = static_cast<int>(attrs[off]);

      return mlx::core::conv_general(ops[0], ops[1], strides, padding_lo, padding_hi,
                                     kernel_dilation, input_dilation, fgs);
     }},

    // ── linalg (single-output) ────────────────────────────────────────────
    //
    // MLX linalg primitives run on the CPU stream only; we pin them to
    // k_linalg_cpu so they compose inside the compiled graph on any default
    // device.  Multi-output factorizations (qr/eigh/svd/lu) live in
    // multi_op_registry below.

    // cholesky: operands = [a]; no attrs.  Lower-triangular factor (upper=false).
    {"cholesky",
     [](const auto &ops, const auto &) {
       return mlx::core::contiguous(
           mlx::core::linalg::cholesky(ops[0], /*upper=*/false, k_linalg_cpu),
           false, k_linalg_cpu);
     }},

    // solve: operands = [a, b]; no attrs.  Solves A x = b.
    {"solve",
     [](const auto &ops, const auto &) {
       return mlx::core::contiguous(
           mlx::core::linalg::solve(ops[0], ops[1], k_linalg_cpu), false, k_linalg_cpu);
     }},

    // solve_triangular: operands = [a, b]; attrs = [upper_int].
    {"solve_triangular",
     [](const auto &ops, const auto &attrs) {
       bool upper = (attrs[0] != 0);
       return mlx::core::contiguous(
           mlx::core::linalg::solve_triangular(ops[0], ops[1], upper, k_linalg_cpu),
           false, k_linalg_cpu);
     }},

    // ── EMLX.Fast fused kernels (mlx::core::fast) ──────────────────────────
    //
    // Recognized from EMLX.Fast.* runtime_call callbacks (see expr.ex
    // fast_kernel_dispatch/2).  Run on the worker's default stream — these are
    // Metal kernels, so the defn must be compiled/replayed on a GPU worker.
    // Float opts (eps/scale/base) arrive as IEEE-754 bits via attr_to_float.

    // rms_norm: operands = [x, weight]; attrs = [eps_bits].
    {"fast_rms_norm",
     [](const auto &ops, const auto &attrs) {
       return mlx::core::fast::rms_norm(ops[0], ops[1], attr_to_float(attrs[0]));
     }},

    // layer_norm (with bias): operands = [x, weight, bias]; attrs = [eps_bits].
    {"fast_layer_norm",
     [](const auto &ops, const auto &attrs) {
       return mlx::core::fast::layer_norm(ops[0], ops[1], ops[2], attr_to_float(attrs[0]));
     }},

    // layer_norm (weight-only): operands = [x, weight]; attrs = [eps_bits].
    {"fast_layer_norm_no_bias",
     [](const auto &ops, const auto &attrs) {
       return mlx::core::fast::layer_norm(ops[0], ops[1], std::nullopt,
                                          attr_to_float(attrs[0]));
     }},

    // swiglu: operands = [gate, up]; no attrs.  silu(gate) * up.
    {"fast_swiglu",
     [](const auto &ops, const auto &) {
       return mlx::core::multiply(
           mlx::core::multiply(ops[0], mlx::core::sigmoid(ops[0])), ops[1]);
     }},

    // sdpa (no mask): operands = [q, k, v]; attrs = [scale_bits].
    {"fast_sdpa",
     [](const auto &ops, const auto &attrs) {
       return mlx::core::fast::scaled_dot_product_attention(
           ops[0], ops[1], ops[2], attr_to_float(attrs[0]), "", std::nullopt,
           std::nullopt);
     }},

    // sdpa (array mask): operands = [q, k, v, mask]; attrs = [scale_bits].
    {"fast_sdpa_masked",
     [](const auto &ops, const auto &attrs) {
       return mlx::core::fast::scaled_dot_product_attention(
           ops[0], ops[1], ops[2], attr_to_float(attrs[0]), "array", ops[3],
           std::nullopt);
     }},

    // sdpa (causal): operands = [q, k, v]; attrs = [scale_bits].
    {"fast_sdpa_causal",
     [](const auto &ops, const auto &attrs) {
       return mlx::core::fast::scaled_dot_product_attention(
           ops[0], ops[1], ops[2], attr_to_float(attrs[0]), "causal",
           std::nullopt, std::nullopt);
     }},

    // sdpa (causal + key_mask): operands = [q, k, v, key_mask];
    // attrs = [scale_bits, kv_offset].  Unlike the eager NIF, the compiled
    // graph cannot branch on a runtime `all(key_mask)` (that forces an eval),
    // so we always build the combined causal+key_mask additive mask in-graph.
    {"fast_sdpa_causal_key_masked",
     [](const auto &ops, const auto &attrs) {
       const auto &q = ops[0];
       const auto &k = ops[1];
       const auto &v = ops[2];
       const auto &key_mask = ops[3];
       float scale = attr_to_float(attrs[0]);
       int kv_offset = static_cast<int>(attrs[1]);

       auto km = mlx::core::reshape(
           key_mask, {key_mask.shape(0), 1, 1, key_mask.shape(1)});
       int T_q = q.shape(2);
       int T_kv = k.shape(2);

       auto row = mlx::core::reshape(mlx::core::arange(T_q, mlx::core::int32),
                                     {1, 1, T_q, 1});
       auto col = mlx::core::reshape(mlx::core::arange(T_kv, mlx::core::int32),
                                     {1, 1, 1, T_kv});
       auto causal_bool = mlx::core::less_equal(
           col, mlx::core::add(row, mlx::core::array(kv_offset, mlx::core::int32)));
       auto keep = mlx::core::logical_and(km, causal_bool);

       auto mask_dtype = q.dtype();
       auto zero_val = mlx::core::zeros({}, mask_dtype);
       auto neginf_val = mlx::core::full(
           {}, -std::numeric_limits<float>::infinity(), mask_dtype);
       auto additive = mlx::core::where(keep, zero_val, neginf_val);

       return mlx::core::fast::scaled_dot_product_attention(
           q, k, v, scale, "array", additive, std::nullopt);
     }},

    // rope (scalar offset): operands = [a];
    // attrs = [dims, traditional, base_bits, scale_bits, offset].
    {"fast_rope",
     [](const auto &ops, const auto &attrs) {
       return mlx::core::fast::rope(
           ops[0], static_cast<int>(attrs[0]), attrs[1] != 0,
           attr_to_float(attrs[2]), attr_to_float(attrs[3]),
           static_cast<int>(attrs[4]), std::nullopt);
     }},

    // rope (per-batch offset array): operands = [a, position_ids];
    // attrs = [dims, traditional, base_bits, scale_bits].  Offsets are
    // position_ids[:, 0] (matches EMLX.Fast.rope_with_positions_fast_callback).
    {"fast_rope_ids",
     [](const auto &ops, const auto &attrs) {
       const auto &pos = ops[1];
       int B = pos.shape(0);
       auto offsets = mlx::core::reshape(
           mlx::core::slice(pos, to_shape({0, 0}), to_shape({B, 1}),
                            to_shape({1, 1})),
           to_shape({B}));
       return mlx::core::fast::rope(
           ops[0], static_cast<int>(attrs[0]), attrs[1] != 0,
           attr_to_float(attrs[2]), attr_to_float(attrs[3]), offsets,
           std::nullopt);
     }},

    // rope (precomputed freqs): operands = [a, position_ids, freqs];
    // attrs = [dims, traditional, scale_bits].  base = nullopt (freqs supplied).
    {"fast_rope_with_freqs",
     [](const auto &ops, const auto &attrs) {
       const auto &pos = ops[1];
       int B = pos.shape(0);
       auto offsets = mlx::core::reshape(
           mlx::core::slice(pos, to_shape({0, 0}), to_shape({B, 1}),
                            to_shape({1, 1})),
           to_shape({B}));
       return mlx::core::fast::rope(
           ops[0], static_cast<int>(attrs[0]), attrs[1] != 0, std::nullopt,
           attr_to_float(attrs[2]), offsets, ops[2]);
     }},
};

// ── Multi-output op registry ──────────────────────────────────────────────────
//
// Ops that produce more than one result array (linalg factorizations).  Their
// outputs are appended to the flat `results` accumulator in the returned order;
// the Elixir lowerer assigns consecutive result indices to the matching output
// refs (see to_wire/1's list-result handling).  Pinned to the CPU device.
using MultiOpFn = std::function<
    std::vector<mlx::core::array>(const std::vector<mlx::core::array> &ops,
                                  const std::vector<int64_t> &attrs)>;

// Force each linalg output to contiguous layout (on the CPU stream).  MLX's
// factorizations can return strided views; if such a view is a program output
// (or otherwise materialized directly), MLX tries to JIT a strided fused CPU
// kernel that can fail to compile.  A plain contiguous Copy avoids that.
static std::vector<mlx::core::array>
contiguous_all(std::vector<mlx::core::array> arrs) {
  for (auto &a : arrs)
    a = mlx::core::contiguous(a, false, k_linalg_cpu);
  return arrs;
}

static const std::unordered_map<std::string, MultiOpFn> multi_op_registry = {
    // qr (reduced mode): operands = [a]; outputs = [q, r].
    {"qr",
     [](const auto &ops, const auto &) -> std::vector<mlx::core::array> {
       auto [q, r] = mlx::core::linalg::qr(ops[0], k_linalg_cpu);
       return contiguous_all({q, r});
     }},

    // eigh (lower triangle): operands = [a]; outputs = [eigenvalues, eigenvectors].
    {"eigh",
     [](const auto &ops, const auto &) -> std::vector<mlx::core::array> {
       auto [w, v] = mlx::core::linalg::eigh(ops[0], "L", k_linalg_cpu);
       return contiguous_all({w, v});
     }},

    // svd (full matrices): operands = [a]; outputs = [u, s, vt].
    {"svd",
     [](const auto &ops, const auto &) -> std::vector<mlx::core::array> {
       return contiguous_all(mlx::core::linalg::svd(ops[0], /*compute_uv=*/true, k_linalg_cpu));
     }},

    // lu: operands = [a]; outputs = [pivots, l, u] (pivots is a uint32 index
    // vector; the lowerer rebuilds the permutation matrix via eye + take).
    {"lu",
     [](const auto &ops, const auto &) -> std::vector<mlx::core::array> {
       return contiguous_all(mlx::core::linalg::lu(ops[0], k_linalg_cpu));
     }},
};

// ── Global compile-cache mutex ────────────────────────────────────────────────
//
// mlx::core::detail::compile and compile_erase both mutate MLX's process-wide
// compile cache.  compile_program runs on the worker thread; ~Expr runs on
// whichever BEAM scheduler/GC thread drops the last resource reference.  These
// two paths can race, corrupting the cache and causing stale graph replay (e.g.
// a static-indexed put_slice graph replayed for a dynamic-indexed program).
//
// All three cache-touching calls (compile, erase, and the first compiled_fn
// invocation that inserts the traced graph) are serialised through this mutex.
static std::mutex s_mlx_compile_mutex;

// ── Expr destructor ───────────────────────────────────────────────────────────
//
// Evicts the per-Expr entry from MLX's global compile cache so stale compiled
// graphs don't accumulate.  Called by default_dtor<Expr> when the BEAM resource
// is GC'd.

Expr::~Expr() {
  if (compile_id != 0) {
    std::lock_guard<std::mutex> lk(s_mlx_compile_mutex);
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
      if (op_registry.find(name) == op_registry.end() &&
          multi_op_registry.find(name) == multi_op_registry.end())
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
        auto multi_it = multi_op_registry.find(op_names[i]);
        if (multi_it != multi_op_registry.end()) {
          // Multi-output op: append each result in order to the flat accumulator.
          auto outs = multi_it->second(op_inputs, attrs[i]);
          for (auto &o : outs)
            results.push_back(o);
        } else {
          results.push_back(op_registry.at(op_names[i])(op_inputs, attrs[i]));
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

    // Assign a unique ID so MLX's global compile cache has a distinct entry per
    // Expr resource.  All our lambdas share the same C++ type (same capture
    // types), so the public mlx::core::compile() would map them all to the same
    // cache key — causing stale graph reuse across different compiled programs.
    static std::atomic<std::uintptr_t> next_id{1};
    std::uintptr_t unique_id = next_id.fetch_add(1, std::memory_order_relaxed);

    new (ptr) Expr();
    ptr->num_inputs = num_inputs_val;
    ptr->compile_id = unique_id;
    {
      std::lock_guard<std::mutex> lk(s_mlx_compile_mutex);
      ptr->compiled_fn = mlx::core::detail::compile(std::move(fn), unique_id);
    }

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


    // Force-evaluate all inputs on the worker thread before passing them to the
    // compiled function.  MLX tensors created via Elixir (e.g. Nx.tensor/2) may
    // arrive as unevaluated lazy nodes associated with a different computation.
    // If they are consumed inside compiled_fn while still lazy, MLX may
    // propagate stale or garbage data from a previously-evaluated graph that
    // happened to share the same underlying buffer.  Forcing eval here ensures
    // all inputs are materialized scalars/arrays on the worker's stream before
    // the compiled graph is built or replayed.
    mlx::core::eval(inputs);

    std::vector<mlx::core::array> outputs;
    {
      std::lock_guard<std::mutex> lk(s_mlx_compile_mutex);
      outputs = prog->compiled_fn(inputs);
    }

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
