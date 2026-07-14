// emlx_compiler.cpp — implements emlx::native compile/eval NIF logic.
//
// compile_program bakes the program into a capturing lambda and wraps it with
// mlx::core::compile(), so MLX traces the computation graph on first eval and
// replays the cached compiled graph on subsequent calls.  eval_program is a
// thin caller: compiled_fn(inputs) → eval → wrap outputs.
//
// Op dispatch uses a string→function registry instead of an integer opcode
// enum. Adding a new op: register it in `op_registry` below.  No enum, no wire
// integers, no lockstep parity table to maintain.

#include "emlx_compiler.hpp"
#include "emlx_runtime_call_bridge.hpp"
#include "mlx/allocator.h"
#include "mlx/compile_impl.h"
#include "mlx/primitives.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <numeric>
#include <thread>
#include <unordered_map>

namespace emlx {
namespace native {

// ── Op registry
// ───────────────────────────────────────────────────────────────
//
// Each entry maps an op name string (matching the atom used in the Elixir IR)
// to a C++ function: (resolved_operands, integer_attrs) → result array.
// `operands` are already-resolved mlx::core::arrays; `attrs` are the integer
// attribute channel passed verbatim from the IR.
//
// This is the single source of truth for op semantics on the compiler path.
// No explicit device: ops run on the default stream of the current worker
// thread.

using OpFn = std::function<mlx::core::array(
    const std::vector<mlx::core::array> &ops, const std::vector<Attr> &attrs)>;

// Float opts (eps/scale/base) ride the int64 attr channel as their IEEE-754
// double bits (see EMLX.Native.Expr.f64_bits/1).  Reverse the reinterpret here.
static inline float attr_to_float(int64_t bits) {
  double d;
  std::memcpy(&d, &bits, sizeof(double));
  return static_cast<float>(d);
}

// ── Prefill-RoPE helper ──────────────────────────────────────────────────
//
// mlx::fast::rope's offset argument is either a scalar or one starting
// position per batch example (sequential from there) — it cannot express
// arbitrary per-token positions (e.g. left-padded prefill). For that case we
// compose the same split-half rotation from precomputed per-token `angles`
// directly with primitive ops, matching mlx::fast::rope's non-traditional
// (traditional=false) formula bit-for-bit:
//   out[..half] = a[..half]*cos(angles) - a[half..dims]*sin(angles)
//   out[half..] = a[half..dims]*cos(angles) + a[..half]*sin(angles)
// (equivalently: rotate_half = concat(-a[half..dims], a[..half]); out =
// a[..dims]*cos_full + rotate_half*sin_full — the form used here, matching
// emlx_fast.cpp's fast_rope_positions eager NIF body).
// `a` is {B, T, H, D}; `angles` is {B, T, half} (already
// position*inv_freq*scale). No new Metal kernel: this is the existing
// fast_rope_positions composition, exposed as a compiled-graph opcode so
// prefill RoPE stays in one NIF replay.
static mlx::core::array rope_rotate_from_angles(const mlx::core::array &a,
                                                const mlx::core::array &angles,
                                                int dims) {
  using namespace mlx::core;

  int B = a.shape(0);
  int T = a.shape(1);
  int H = a.shape(2);
  int D = a.shape(3);
  int half = dims / 2;

  auto cos_bt1h =
      astype(reshape(cos(angles), to_shape({B, T, 1, half})), a.dtype());
  auto sin_bt1h =
      astype(reshape(sin(angles), to_shape({B, T, 1, half})), a.dtype());

  auto cos_full = concatenate(std::vector<array>{cos_bt1h, cos_bt1h}, 3);
  auto sin_full = concatenate(std::vector<array>{sin_bt1h, sin_bt1h}, 3);

  auto x1 = slice(a, to_shape({0, 0, 0, 0}), to_shape({B, T, H, half}));
  auto x2 = slice(a, to_shape({0, 0, 0, half}), to_shape({B, T, H, dims}));
  auto rotated = concatenate(std::vector<array>{negative(x2), x1}, 3);

  auto a_head = slice(a, to_shape({0, 0, 0, 0}), to_shape({B, T, H, dims}));
  auto rope_head = add(multiply(a_head, cos_full), multiply(rotated, sin_full));

  if (dims == D) {
    return rope_head;
  }
  auto tail = slice(a, to_shape({0, 0, 0, dims}), to_shape({B, T, H, D}));
  return concatenate(std::vector<array>{rope_head, tail}, 3);
}

// ── Window-op helpers ─────────────────────────────────────────────────────
//
// These mirror the Elixir backend's sliding-window algorithm, which uses
// as_strided to build a view then reduces over the window dims.
// No explicit device needed: MLX ops inherit stream from input arrays.

// Build a sliding window view.  padded has shape [...]; window and strides
// have length n=padded.ndim().  Result shape: [o0,...,on-1, w0,...,wn-1].
static mlx::core::array
compiler_sliding_window_view(const mlx::core::array &padded,
                             const std::vector<int> &window,
                             const std::vector<int> &strides) {
  int n = padded.ndim();
  auto ps = padded.shape();

  // Doubled element strides: output dims and window dims share the same
  // strides.
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

  return mlx::core::slice(strided, to_shape(starts), to_shape(stops),
                          to_shape(sl_strides));
}

// Decode attrs and run a window reduction (sum/product/max/min).
// op_code: 0=sum, 1=product, 2=max, 3=min.
// attrs = [n_dims, op_int, lo0, hi0, …, s0, …, w0, …, wd0, …]
static mlx::core::array window_reduce_impl(const mlx::core::array &t,
                                           const std::vector<Attr> &attrs,
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
      // Build dilated axis: start with zeros, then set positions 0, wd, 2*wd,
      // ...
      auto arange_full = mlx::core::arange(0, new_size, 1, mlx::core::int32);
      // Positions occupied by real values: 0, wd, 2*wd, ...
      auto divisible = mlx::core::equal(
          mlx::core::remainder(arange_full,
                               mlx::core::full({}, wd[i], mlx::core::int32)),
          mlx::core::zeros({}, mlx::core::int32));
      // Build dilated from current using take at strided positions.
      // Take the [0, wd, 2*wd, ...] positions from the window_mask axis i.
      auto src_idx = mlx::core::astype(
          mlx::core::divide(arange_full,
                            mlx::core::full({}, wd[i], mlx::core::int32)),
          mlx::core::uint32);
      // For interior positions, take from mask (value 1); replace interior
      // zeros. It's simpler to: take all, then where(divisible, val, 0).
      auto taken = mlx::core::take(window_mask, src_idx, i);
      // Mask out non-original positions.
      // Broadcast divisible to match taken shape.
      std::vector<int> div_shape(window_mask.ndim(), 1);
      div_shape[i] = new_size;
      auto div_nd = mlx::core::reshape(divisible, to_shape(div_shape));
      auto div_bc = mlx::core::broadcast_to(div_nd, taken.shape());
      window_mask = mlx::core::where(div_bc, taken,
                                     mlx::core::zeros({}, mlx::core::bool_));
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
        return mlx::core::full({}, -std::numeric_limits<float>::infinity(),
                               t.dtype());
      } else {
        return mlx::core::full({}, std::numeric_limits<int32_t>::min(),
                               t.dtype());
      }
    case 3: // min: highest representable value
      if (mlx::core::issubdtype(t.dtype(), mlx::core::floating)) {
        return mlx::core::full({}, std::numeric_limits<float>::infinity(),
                               t.dtype());
      } else {
        return mlx::core::full({}, std::numeric_limits<int32_t>::max(),
                               t.dtype());
      }
    default:
      throw std::runtime_error("window_reduce_impl: unknown op_code");
    }
  };
  auto pad_val = make_pad_val();

  auto padded = mlx::core::pad(t, all_axes, to_shape(low_pad),
                               to_shape(high_pad), pad_val, "constant");

  // Sliding window view: shape [o0,...,on-1, w0,...,wn-1].
  auto view = compiler_sliding_window_view(padded, expanded_window, strides);

  // Broadcast window_mask (shape expanded_window) to match view: all batch dims
  // are 1.
  std::vector<int> mask_bc_shape(n, 1);
  for (int w : expanded_window)
    mask_bc_shape.push_back(w);
  auto mask_reshaped = mlx::core::reshape(window_mask, to_shape(mask_bc_shape));
  auto mask_bc = mlx::core::broadcast_to(mask_reshaped, view.shape());

  // Apply mask: replace masked-out positions with pad_val.
  auto masked_view = mlx::core::where(
      mask_bc, view,
      mlx::core::broadcast_to(
          mlx::core::reshape(pad_val, to_shape(std::vector<int>{})),
          view.shape()));

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
static mlx::core::array
window_scatter_impl_compiler(const mlx::core::array &tensor_t,
                             const mlx::core::array &tensor_source,
                             const mlx::core::array &tensor_init_value,
                             const std::vector<Attr> &attrs, bool scatter_max) {
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
  auto padded = mlx::core::pad(tensor_t, all_axes, to_shape(low_pad),
                               to_shape(high_pad), init_casted, "constant");

  auto padded_shape = padded.shape();
  std::vector<int> padded_shape_vec(padded_shape.begin(), padded_shape.end());

  auto window_view = compiler_sliding_window_view(padded, window, strides);

  std::vector<int> out_shape(window_view.shape().begin(),
                             window_view.shape().begin() + n);

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
    auto arange_k =
        mlx::core::astype(mlx::core::arange(0, K, 1), mlx::core::uint32);
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
    auto arange_a = mlx::core::astype(
        mlx::core::arange(0, (int)padded_shape[a], 1), mlx::core::int32);
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
  auto scattered =
      mlx::core::scatter_add(buffer, abs_indices, updates, scatter_axes);

  auto orig_shape = tensor_t.shape();
  std::vector<int> slice_starts = low_pad;
  std::vector<int> slice_stops(n);
  for (int i = 0; i < n; ++i)
    slice_stops[i] = low_pad[i] + (int)orig_shape[i];
  std::vector<int> slice_ones(n, 1);
  return mlx::core::slice(scattered, to_shape(slice_starts),
                          to_shape(slice_stops), to_shape(slice_ones));
}

// MLX linalg primitives are CPU-only.  Pin them to the CPU device so they
// compose inside a compiled graph regardless of the graph's default (worker)
// stream — validated to work for both :cpu and :gpu default devices.
static const mlx::core::Device k_linalg_cpu(mlx::core::Device::DeviceType::cpu,
                                            0);

static const std::unordered_map<std::string, OpFn> op_registry = {
    // ── cast ──────────────────────────────────────────────────────────────
    {"astype",
     [](const auto &ops, const auto &attrs) {
       return mlx::core::astype(ops[0], attrs[0].as_dtype());
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
       return mlx::core::view(ops[0], attrs[0].as_dtype());
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

    // ── quantized_matmul ──────────────────────────────────────────────────────
    //
    // iattrs = [group_size, bits, transpose_int, mode_int, has_bias_int]
    // Operands: [activation, weight, scales, biases?] — biases present only
    // when has_bias_int is 1 (mirrors EMLX.Backend.quantized_dot/4: absent
    // for microscaled modes, since mx::fp_quantize doesn't emit biases).
    // Emitted only by a call-time-specialized program (see
    // EMLX.Native.Expr's "Quantized dot specialization" moduledoc section);
    // `weight` is the untouched physical (packed) parameter ref.
    {"quantized_matmul",
     [](const auto &ops, const auto &attrs) {
       int group_size = static_cast<int>(attrs[0]);
       int bits = static_cast<int>(attrs[1]);
       bool transpose = attrs[2] != 0;
       std::string mode = attrs[3].as_mode();
       bool has_bias = attrs[4] != 0;
       std::optional<mlx::core::array> biases_opt =
           has_bias ? std::make_optional(ops[3]) : std::nullopt;
       return mlx::core::quantized_matmul(ops[0], ops[1], ops[2], biases_opt,
                                          transpose, group_size, bits, mode);
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
       auto dtype = attrs[0].as_dtype();
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
       auto dtype = attrs[0].as_dtype();
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
    // Recognized from `EMLX.Fast.*`'s `:__EMLX__`-tagged `Nx.Defn.Expr.metadata`
    // nodes (see EMLX.Native.Expr's `:metadata` expand_node clause and
    // EMLX.Fast's moduledoc). Run on the worker's default stream — these are
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

    // sdpa (no mask, + sinks): operands = [q, k, v, sinks]; attrs = [scale_bits].
    {"fast_sdpa_sinks",
     [](const auto &ops, const auto &attrs) {
       return mlx::core::fast::scaled_dot_product_attention(
           ops[0], ops[1], ops[2], attr_to_float(attrs[0]), "", std::nullopt,
           ops[3]);
     }},

    // sdpa (array mask): operands = [q, k, v, mask]; attrs = [scale_bits].
    {"fast_sdpa_masked",
     [](const auto &ops, const auto &attrs) {
       return mlx::core::fast::scaled_dot_product_attention(
           ops[0], ops[1], ops[2], attr_to_float(attrs[0]), "array", ops[3],
           std::nullopt);
     }},

    // sdpa (array mask, + sinks): operands = [q, k, v, mask, sinks];
    // attrs = [scale_bits].
    {"fast_sdpa_masked_sinks",
     [](const auto &ops, const auto &attrs) {
       return mlx::core::fast::scaled_dot_product_attention(
           ops[0], ops[1], ops[2], attr_to_float(attrs[0]), "array", ops[3],
           ops[4]);
     }},

    // sdpa (causal): operands = [q, k, v]; attrs = [scale_bits].
    {"fast_sdpa_causal",
     [](const auto &ops, const auto &attrs) {
       return mlx::core::fast::scaled_dot_product_attention(
           ops[0], ops[1], ops[2], attr_to_float(attrs[0]), "causal",
           std::nullopt, std::nullopt);
     }},

    // sdpa (causal, + sinks): operands = [q, k, v, sinks]; attrs = [scale_bits].
    {"fast_sdpa_causal_sinks",
     [](const auto &ops, const auto &attrs) {
       return mlx::core::fast::scaled_dot_product_attention(
           ops[0], ops[1], ops[2], attr_to_float(attrs[0]), "causal",
           std::nullopt, ops[3]);
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

       auto attn_t = mlx::core::fast::scaled_dot_product_attention(
           q, k, v, scale, "array", additive, std::nullopt);

       // Replace NaN with 0 for all-masked rows (softmax(-inf,...,-inf) = NaN
       // in Flash-Attention when seq_len >= Metal tile size, but semantically
       // = 0) -- happens for left-padded prefill query rows that have no
       // causally-visible *and* key_mask-valid key at all.
       return mlx::core::where(mlx::core::isnan(attn_t),
                               mlx::core::zeros_like(attn_t), attn_t);
     }},

    // sdpa (causal + key_mask, + sinks): operands = [q, k, v, key_mask,
    // sinks]; attrs = [scale_bits, kv_offset]. Same in-graph combined mask as
    // "fast_sdpa_causal_key_masked" above, plus the sinks operand.
    {"fast_sdpa_causal_key_masked_sinks",
     [](const auto &ops, const auto &attrs) {
       const auto &q = ops[0];
       const auto &k = ops[1];
       const auto &v = ops[2];
       const auto &key_mask = ops[3];
       const auto &sinks = ops[4];
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

       auto attn_t = mlx::core::fast::scaled_dot_product_attention(
           q, k, v, scale, "array", additive, sinks);

       // See "fast_sdpa_causal_key_masked" above: NaN-guard all-masked rows.
       return mlx::core::where(mlx::core::isnan(attn_t),
                               mlx::core::zeros_like(attn_t), attn_t);
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
    //
    // H>1 (Bumblebee {B,T,H,D} convention) falls back to the hand
    // cos/sin/rotate composition (rope_rotate_from_angles) — see
    // emlx_fast.cpp's multi-head-safe RoPE helpers comment /
    // https://github.com/elixir-nx/emlx/issues/121. H==1 keeps the fused
    // mlx::core::fast::rope path. The full position_ids operand is already
    // available here, so no offset+arange reconstruction is needed.
    {"fast_rope_ids",
     [](const auto &ops, const auto &attrs) {
       const auto &a = ops[0];
       const auto &pos = ops[1];
       int dims = static_cast<int>(attrs[0]);
       bool traditional = attrs[1] != 0;

       if (a.ndim() == 4 && a.shape(2) > 1) {
         if (traditional) {
           throw std::invalid_argument(
               "fast_rope_ids: traditional=true not supported for "
               "multi-head (H>1) input");
         }
         float base = attr_to_float(attrs[2]);
         float scale = attr_to_float(attrs[3]);
         int half = dims / 2;

         std::vector<float> inv_freq_host(half);
         for (int i = 0; i < half; ++i) {
           float expo = (2.0f * static_cast<float>(i)) / static_cast<float>(dims);
           inv_freq_host[i] = 1.0f / std::pow(base, expo);
         }
         auto inv_freq =
             mlx::core::array(inv_freq_host.begin(), {half}, mlx::core::float32);

         int B = pos.shape(0);
         int T = pos.shape(1);
         auto pos_f = mlx::core::astype(pos, mlx::core::float32);
         auto pos_bt1 = mlx::core::reshape(pos_f, to_shape({B, T, 1}));
         auto inv_11h = mlx::core::reshape(inv_freq, to_shape({1, 1, half}));
         auto angles = mlx::core::multiply(
             mlx::core::multiply(pos_bt1, inv_11h),
             mlx::core::array(scale, mlx::core::float32));

         return rope_rotate_from_angles(a, angles, dims);
       }

       int B = pos.shape(0);
       auto offsets = mlx::core::reshape(
           mlx::core::slice(pos, to_shape({0, 0}), to_shape({B, 1}),
                            to_shape({1, 1})),
           to_shape({B}));
       return mlx::core::fast::rope(
           a, dims, traditional,
           attr_to_float(attrs[2]), attr_to_float(attrs[3]), offsets,
           std::nullopt);
     }},

    // rope (precomputed freqs): operands = [a, position_ids, freqs];
    // attrs = [dims, traditional, scale_bits].  base = nullopt (freqs supplied).
    //
    // H>1 falls back to the hand cos/sin/rotate composition — see the
    // "fast_rope_ids" comment above and emlx_fast.cpp's multi-head-safe RoPE
    // helpers comment. H==1 keeps the fused mlx::core::fast::rope path.
    {"fast_rope_with_freqs",
     [](const auto &ops, const auto &attrs) {
       const auto &a = ops[0];
       const auto &pos = ops[1];
       const auto &freqs = ops[2];
       int dims = static_cast<int>(attrs[0]);
       bool traditional = attrs[1] != 0;

       if (a.ndim() == 4 && a.shape(2) > 1) {
         if (traditional) {
           throw std::invalid_argument(
               "fast_rope_with_freqs: traditional=true not supported for "
               "multi-head (H>1) input");
         }
         float scale = attr_to_float(attrs[2]);
         int half = dims / 2;

         int B = pos.shape(0);
         int T = pos.shape(1);
         auto pos_f = mlx::core::astype(pos, mlx::core::float32);
         auto pos_bt1 = mlx::core::reshape(pos_f, to_shape({B, T, 1}));
         auto inv_freq = mlx::core::reciprocal(mlx::core::astype(freqs, mlx::core::float32));
         auto inv_11h = mlx::core::reshape(inv_freq, to_shape({1, 1, half}));
         auto angles = mlx::core::multiply(
             mlx::core::multiply(pos_bt1, inv_11h),
             mlx::core::array(scale, mlx::core::float32));

         return rope_rotate_from_angles(a, angles, dims);
       }

       int B = pos.shape(0);
       auto offsets = mlx::core::reshape(
           mlx::core::slice(pos, to_shape({0, 0}), to_shape({B, 1}),
                            to_shape({1, 1})),
           to_shape({B}));
       return mlx::core::fast::rope(
           a, dims, traditional, std::nullopt,
           attr_to_float(attrs[2]), offsets, freqs);
     }},

    // rope (arbitrary per-token position_ids, base-derived inv_freq):
    // operands = [a, position_ids]; attrs = [dims, traditional, base_bits,
    // scale_bits]. Prefill path (T>1) / high-base decode — mirrors
    // emlx_fast.cpp's fast_rope_positions eager NIF.
    {"fast_rope_positions",
     [](const auto &ops, const auto &attrs) {
       int dims = static_cast<int>(attrs[0]);
       bool traditional = attrs[1] != 0;
       if (traditional) {
         throw std::invalid_argument(
             "fast_rope_positions: traditional=true not supported");
       }
       float base = attr_to_float(attrs[2]);
       float scale = attr_to_float(attrs[3]);
       const auto &a = ops[0];
       const auto &pos = ops[1];
       int half = dims / 2;

       std::vector<float> inv_freq_host(half);
       for (int i = 0; i < half; ++i) {
         float expo = (2.0f * static_cast<float>(i)) / static_cast<float>(dims);
         inv_freq_host[i] = 1.0f / std::pow(base, expo);
       }
       auto inv_freq =
           mlx::core::array(inv_freq_host.begin(), {half}, mlx::core::float32);

       int B = pos.shape(0);
       int T = pos.shape(1);
       auto pos_f = mlx::core::astype(pos, mlx::core::float32);
       auto pos_bt1 = mlx::core::reshape(pos_f, to_shape({B, T, 1}));
       auto inv_11h = mlx::core::reshape(inv_freq, to_shape({1, 1, half}));
       auto angles = mlx::core::multiply(
           mlx::core::multiply(pos_bt1, inv_11h),
           mlx::core::array(scale, mlx::core::float32));

       return rope_rotate_from_angles(a, angles, dims);
     }},

    // rope (arbitrary per-token position_ids, precomputed freqs):
    // operands = [a, position_ids, freqs]; attrs = [dims, traditional,
    // scale_bits]. Prefill path (T>1) for RoPE-scaling strategies that bake a
    // freqs tensor (e.g. :llama3) — mlx::fast::rope's freqs overload takes
    // `inv_freqs = reciprocal(freqs)` (see mlx/fast.cpp default_inv_freqs vs
    // the inputs.size()==3 branch), so we replicate that reciprocal here
    // rather than using `freqs` directly, to stay bit-for-bit with the eager
    // EMLX.Fast.rope_with_freqs_callback path this replaces.
    {"fast_rope_with_freqs_positions",
     [](const auto &ops, const auto &attrs) {
       int dims = static_cast<int>(attrs[0]);
       bool traditional = attrs[1] != 0;
       if (traditional) {
         throw std::invalid_argument(
             "fast_rope_with_freqs_positions: traditional=true not supported");
       }
       float scale = attr_to_float(attrs[2]);
       const auto &a = ops[0];
       const auto &pos = ops[1];
       const auto &freqs = ops[2];
       int half = dims / 2;

       int B = pos.shape(0);
       int T = pos.shape(1);
       auto pos_f = mlx::core::astype(pos, mlx::core::float32);
       auto pos_bt1 = mlx::core::reshape(pos_f, to_shape({B, T, 1}));
       auto inv_freq = mlx::core::reciprocal(mlx::core::astype(freqs, mlx::core::float32));
       auto inv_11h = mlx::core::reshape(inv_freq, to_shape({1, 1, half}));
       auto angles = mlx::core::multiply(
           mlx::core::multiply(pos_bt1, inv_11h),
           mlx::core::array(scale, mlx::core::float32));

       return rope_rotate_from_angles(a, angles, dims);
     }},
};

// ── Multi-output op registry
// ──────────────────────────────────────────────────
//
// Ops that produce more than one result array (linalg factorizations).  Their
// outputs are appended to the flat `results` accumulator in the returned order;
// the Elixir lowerer assigns consecutive result indices to the matching output
// refs (see to_native/1's list-result handling).  Pinned to the CPU device.
using MultiOpFn = std::function<std::vector<mlx::core::array>(
    const std::vector<mlx::core::array> &ops, const std::vector<Attr> &attrs)>;

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

// A genuine `mlx::core::Primitive` for `Nx.runtime_call/4` — see
// EMLX.Native.Expr's moduledoc "Runtime calls" section. Unlike every other
// entry in this file (a plain function called once, during the single
// interpreter-lambda trace `mlx::core::detail::compile()` performs — see
// this file's header comment), a Primitive's `eval_cpu`/`eval_gpu` genuinely
// re-executes on every replay of the cached compiled tape, which is what
// lets the real Elixir callback fire once per `eval_program` call instead of
// once ever. `eval()` blocks the calling worker OS thread inside
// `invoke_runtime_call` (emlx_runtime_call_bridge.hpp) until the Elixir side
// replies via `EMLX.NIF.resolve_runtime_call/3`.
// Forward declaration: EMLXWhile::eval (below) recurses into this to
// interpret its cond/body SubPrograms; the free function's own definition
// (further below) constructs EMLXWhile instances for nested/top-level
// `:while` instructions, hence the mutual forward reference.
static std::vector<mlx::core::array> interpret_instructions(
    const std::vector<Instruction> &instructions,
    const std::vector<Ref> &output_refs,
    const std::shared_ptr<const std::vector<mlx::core::array>> &captures,
    const std::shared_ptr<const std::vector<mlx::core::array>> &constants,
    const std::vector<mlx::core::array> *inputs,
    const std::vector<mlx::core::array> *carry);

// ── EMLXWhile ────────────────────────────────────────────────────────────
//
// Native lowering of a dynamic-trip-count `Nx.Defn.while` loop (see
// EMLX.Native.Expr's `:while` moduledoc section). Interprets the `cond`/
// `body` SubPrograms (emlx_compiler.hpp) directly here in C++, iterating
// until `cond` evaluates false — all inside a single eval_cpu call, so an
// N-iteration loop costs one eval_program NIF round-trip instead of the old
// Elixir-driven split-point path's 2N (`run_while_loop` in emlx.ex).
//
// Pinned to the CPU stream (like MLX's linalg factorizations and
// EMLXRuntimeCall above) so the internal `mlx::core::eval()` calls below —
// needed every iteration to force `cond`'s boolean result and to bound each
// iteration's graph depth (see the loop comment) — never reenter MLX's GPU
// command buffer scheduling from inside eval_gpu. This nested/reentrant
// eval-inside-a-CPU-pinned-primitive's-eval_cpu pattern (itself reached via
// eval_program's outer mlx::core::eval(outputs)) was validated safe by an
// earlier de-risking spike (a hardcoded-iteration-count primitive, since
// removed — see git history on pv-feat/lower-while for the original spike).
//
// `captures_`/`constants_` are shared with the enclosing Program's own
// interpreter (never duplicated per SubProgram — see EMLX.Native.SubProgram
// on the Elixir side); `Nx.Defn.while`'s closure rules mean cond/body only
// ever need the *global* captures/constants tables plus the carry, never
// anything computed mid-loop outside the carry.
class EMLXWhile : public mlx::core::Primitive {
public:
  EMLXWhile(mlx::core::Stream stream, SubProgram cond, SubProgram body,
            std::shared_ptr<const std::vector<mlx::core::array>> captures,
            std::shared_ptr<const std::vector<mlx::core::array>> constants)
      : mlx::core::Primitive(stream), cond_(std::move(cond)),
        body_(std::move(body)), captures_(std::move(captures)),
        constants_(std::move(constants)) {}

  void eval_cpu(const std::vector<mlx::core::array> &inputs,
                std::vector<mlx::core::array> &outputs) override {
    eval(inputs, outputs);
  }
  void eval_gpu(const std::vector<mlx::core::array> &inputs,
                std::vector<mlx::core::array> &outputs) override {
    eval(inputs, outputs);
  }

  // Side-effecting in the general case (a body may itself contain
  // `:runtime_call`) and, regardless, cheap to keep simple — never dedup/CSE.
  bool is_equivalent(const mlx::core::Primitive &) const override {
    return false;
  }

  const char *name() const override { return "EMLXWhile"; }

private:
  bool evaluate_predicate(std::vector<mlx::core::array> &carry) {
    auto outs =
        interpret_instructions(cond_.instructions, cond_.outputs, captures_,
                               constants_, /*inputs=*/nullptr, &carry);
    if (outs.size() != 1)
      throw std::runtime_error(
          "emlx::native: :while condition sub-program must produce exactly "
          "one output, got " +
          std::to_string(outs.size()));

    // Force the condition to a concrete boolean now: this is the only way
    // to make the host-side control-flow decision (loop again or stop).
    mlx::core::eval(outs[0]);

    return outs[0].item<bool>();
  }

  void evaluate_body(std::vector<mlx::core::array> &carry) {
    carry = interpret_instructions(body_.instructions, body_.outputs, captures_,
                                   constants_, /*inputs=*/nullptr, &carry);
    // Force the new carry every iteration rather than chaining lazy graphs
    // across a data-dependent number of iterations: unevaluated chaining
    // would grow the traced graph unboundedly *per eval_program call*
    // (this loop runs after mlx::core::compile()'s one-time trace, so the
    // growth would recur on every single call, not just once).
    mlx::core::eval(carry);
  }

  void eval(const std::vector<mlx::core::array> &inputs,
            std::vector<mlx::core::array> &outputs) {
    std::vector<mlx::core::array> carry = inputs;

    while (evaluate_predicate(carry)) {
      evaluate_body(carry);
    }

    for (size_t i = 0; i < outputs.size(); i++) {
      outputs[i].copy_shared_buffer(carry.at(i));
    }
  }

  SubProgram cond_;
  SubProgram body_;
  std::shared_ptr<const std::vector<mlx::core::array>> captures_;
  std::shared_ptr<const std::vector<mlx::core::array>> constants_;
};

class EMLXRuntimeCall : public mlx::core::Primitive {
public:
  EMLXRuntimeCall(mlx::core::Stream stream, int64_t callback_index)
      : mlx::core::Primitive(stream), callback_index_(callback_index) {}

  void eval_cpu(const std::vector<mlx::core::array> &inputs,
                std::vector<mlx::core::array> &outputs) override {
    eval(inputs, outputs);
  }
  void eval_gpu(const std::vector<mlx::core::array> &inputs,
                std::vector<mlx::core::array> &outputs) override {
    eval(inputs, outputs);
  }

  // Side-effecting (fires an Elixir callback with observable effects) —
  // never dedup/CSE. This is also `Primitive::is_equivalent`'s own default,
  // kept explicit here for clarity rather than relied upon.
  bool is_equivalent(const mlx::core::Primitive &) const override {
    return false;
  }

  const char *name() const override { return "EMLXRuntimeCall"; }

private:
  void eval(const std::vector<mlx::core::array> &inputs,
            std::vector<mlx::core::array> &outputs) {
    emlx::native::invoke_runtime_call(callback_index_, inputs, outputs);
  }

  int64_t callback_index_;
};

static const std::unordered_map<std::string, MultiOpFn> multi_op_registry = {
    // runtime_call: operands = flattened callback-argument leaves, in order.
    // attrs = [callback_index, n_outputs, dtype0, n_dims0, d0.., dtype1,
    // n_dims1, d1.., ...] — see EMLX.Native.Expr's moduledoc iattrs table.
    // outputs = one array per declared output, produced by the
    // EMLXRuntimeCall primitive above (never eagerly computed here).
    {"runtime_call",
     [](const auto &ops, const auto &attrs) -> std::vector<mlx::core::array> {
       size_t off = 0;
       int64_t callback_index = attrs[off++];
       int64_t n_outputs = attrs[off++];

       std::vector<mlx::core::Shape> shapes;
       std::vector<mlx::core::Dtype> dtypes;
       shapes.reserve(static_cast<size_t>(n_outputs));
       dtypes.reserve(static_cast<size_t>(n_outputs));

       for (int64_t i = 0; i < n_outputs; i++) {
         dtypes.push_back(attrs[off++].as_dtype());
         int64_t n_dims = attrs[off++];
         std::vector<int> dims(static_cast<size_t>(n_dims));
         for (int64_t d = 0; d < n_dims; d++) {
           dims[static_cast<size_t>(d)] = static_cast<int>(attrs[off++]);
         }
         shapes.push_back(to_shape(dims));
       }

       // Pinned to the CPU stream (like the linalg factorizations above),
       // regardless of the compiled graph's default device. eval_gpu is
       // never actually reached this way: the primitive does no Metal work
       // of its own (it only blocks the worker thread on the Elixir
       // round-trip and memcpy's the reply into the output buffer), and
       // running it under mlx::core::gpu::eval's Metal command-buffer
       // bookkeeping segfaults — see workdir/native-compiler/32a for
       // details. MLX handles the cross-stream data dependencies (GPU
       // operand arrays feeding a CPU-pinned primitive, and vice versa)
       // the same way it does for the linalg ops.
       auto primitive = std::make_shared<EMLXRuntimeCall>(
           mlx::core::default_stream(k_linalg_cpu), callback_index);
       return mlx::core::array::make_arrays(shapes, dtypes, primitive, ops);
     }},

    // qr (reduced mode): operands = [a]; outputs = [q, r].
    {"qr",
     [](const auto &ops, const auto &) -> std::vector<mlx::core::array> {
       auto [q, r] = mlx::core::linalg::qr(ops[0], k_linalg_cpu);
       return contiguous_all({q, r});
     }},

    // eigh (lower triangle): operands = [a]; outputs = [eigenvalues,
    // eigenvectors].
    {"eigh",
     [](const auto &ops, const auto &) -> std::vector<mlx::core::array> {
       auto [w, v] = mlx::core::linalg::eigh(ops[0], "L", k_linalg_cpu);
       return contiguous_all({w, v});
     }},

    // svd (full matrices): operands = [a]; outputs = [u, s, vt].
    {"svd",
     [](const auto &ops, const auto &) -> std::vector<mlx::core::array> {
       return contiguous_all(
           mlx::core::linalg::svd(ops[0], /*compute_uv=*/true, k_linalg_cpu));
     }},

    // lu: operands = [a]; outputs = [pivots, l, u] (pivots is a uint32 index
    // vector; the lowerer rebuilds the permutation matrix via eye + take).
    {"lu",
     [](const auto &ops, const auto &) -> std::vector<mlx::core::array> {
       return contiguous_all(mlx::core::linalg::lu(ops[0], k_linalg_cpu));
     }},
    // The head-transpose needed for `fast::scaled_dot_product_attention`
    // (which requires {B, N, T, D}) is done internally on q and on the
    // *already-written* k_upd/v_upd -- this is the same single full-cache
    // transpose the unfused path already pays for (its own
    // `build_sdpa_layer` transposes the post-update K/V cache before
    // calling SDPA); fusing does not add a second one, since k_upd/v_upd
    // are returned pre-transpose.
    {"kv_cache_sdpa_update",
     [](const auto &ops, const auto &attrs) -> std::vector<mlx::core::array> {
       const auto &q = ops[0];
       const auto &new_k = ops[1];
       const auto &new_v = ops[2];
       const auto &k_cache = ops[3];
       const auto &v_cache = ops[4];
       const auto &offset = ops[5];
       const auto &key_mask = ops[6];
       float scale = attr_to_float(attrs[0]);
       int kv_offset = static_cast<int>(attrs[1]);

       int T_max = static_cast<int>(k_cache.shape(1));
       int T_q_write = static_cast<int>(new_k.shape(1));

       // Dynamic write-start {0, clamp(offset, 0, T_max-T_q_write), 0, 0} --
       // same clamp-and-concatenate construction as op_registry["put_slice"]
       // (T_max-T_q_write, not T_max-1, so the write itself never runs past
       // the cache's last valid row when T_q_write > 1, e.g. prefill).
       auto s = mlx::core::reshape(mlx::core::astype(offset, mlx::core::int32), {1});
       auto max_s = mlx::core::full({1}, T_max - T_q_write, mlx::core::int32);
       auto zero1 = mlx::core::zeros({1}, mlx::core::int32);
       auto offset_clamped = mlx::core::minimum(mlx::core::maximum(s, zero1), max_s);
       auto start = mlx::core::concatenate({zero1, offset_clamped, zero1, zero1}, 0);

       std::vector<int> all_axes = {0, 1, 2, 3};
       auto k_upd = mlx::core::slice_update(k_cache, new_k, start, all_axes);
       auto v_upd = mlx::core::slice_update(v_cache, new_v, start, all_axes);

       // Head-transpose for SDPA: {B, T, N, D} -> {B, N, T, D}. Mirrors
       // build_sdpa_layer's own Nx.transpose(axes: [0, 2, 1, 3]) exactly.
       std::vector<int> head_transpose = {0, 2, 1, 3};
       auto q_t = mlx::core::transpose(q, head_transpose);
       auto k_t = mlx::core::transpose(k_upd, head_transpose);
       auto v_t = mlx::core::transpose(v_upd, head_transpose);

       // Combined causal + key_mask additive mask -- verbatim copy of
       // "fast_sdpa_causal_key_masked"'s formula above (T_kv = T_max here).
       auto km = mlx::core::reshape(key_mask, {key_mask.shape(0), 1, 1, key_mask.shape(1)});
       int T_q = q_t.shape(2);
       auto row = mlx::core::reshape(mlx::core::arange(T_q, mlx::core::int32), {1, 1, T_q, 1});
       auto col = mlx::core::reshape(mlx::core::arange(T_max, mlx::core::int32), {1, 1, 1, T_max});
       auto causal_bool = mlx::core::less_equal(
           col, mlx::core::add(row, mlx::core::array(kv_offset, mlx::core::int32)));
       auto keep = mlx::core::logical_and(km, causal_bool);
       auto mask_dtype = q_t.dtype();
       auto zero_val = mlx::core::zeros({}, mask_dtype);
       auto neginf_val =
           mlx::core::full({}, -std::numeric_limits<float>::infinity(), mask_dtype);
       auto additive = mlx::core::where(keep, zero_val, neginf_val);

       auto attn_out = mlx::core::fast::scaled_dot_product_attention(
           q_t, k_t, v_t, scale, "array", additive, std::nullopt);

       // Replace NaN with 0 for all-masked rows (softmax(-inf,...,-inf) = NaN
       // in Flash-Attention when seq_len >= Metal tile size, but semantically
       // = 0) -- happens for left-padded prefill query rows that have no
       // causally-visible *and* key_mask-valid key at all. See
       // "fast_sdpa_causal_key_masked" above for the unfused version of this
       // same guard.
       attn_out = mlx::core::where(mlx::core::isnan(attn_out),
                                   mlx::core::zeros_like(attn_out), attn_out);

       return {attn_out, k_upd, v_upd};
     }},
};

// ── interpret_instructions ────────────────────────────────────────────────
//
// Shared flat-instruction-list interpreter used both for a top-level
// Program's own instructions (called from compile_program_impl's traced
// lambda, with `inputs` set and `carry` null) and, recursively, for a
// `:while` instruction's `cond`/`body` SubPrograms (called from
// EMLXWhile::eval, with `carry` set and `inputs` null — SubPrograms never
// contain `{:input, _}` refs, see emlx_compiler.hpp). `results` is always a
// fresh local accumulator: a SubProgram's `{:result, i}` numbering is local
// to that interpretation, distinct from the parent program's own.
static int64_t plugin_attr_int(const Attr &attr, const char *field) {
  if (!attr.is_int())
    throw std::runtime_error(std::string("emlx::native: plugin ") + field +
                             " must be an integer");
  return static_cast<int64_t>(attr);
}

static void resolve_plugin_instruction(Instruction &instr) {
  const auto &attrs = instr.attrs;
  if (attrs.size() < 7)
    throw std::runtime_error("emlx::native: truncated plugin instruction");
  if (plugin_attr_int(attrs[0], "wire version") != 1)
    throw std::runtime_error("emlx::native: unsupported plugin wire version");
  if (!attrs[1].is_binary() || !attrs[2].is_binary())
    throw std::runtime_error("emlx::native: plugin names must be binaries");

  const std::string &plugin_name = attrs[1].as_binary();
  const std::string &callback_name = attrs[2].as_binary();
  if (!emlx_valid_plugin_name(plugin_name) ||
      !emlx_valid_plugin_name(callback_name))
    throw std::runtime_error("emlx::native: plugin names are not canonical");
  auto resolved = emlx_resolve_plugin_callback(plugin_name, callback_name);
  const auto &callback = *resolved.callback;

  const int64_t schema = plugin_attr_int(attrs[3], "schema version");
  const int64_t attr_schema =
      plugin_attr_int(attrs[4], "attribute schema version");
  const int64_t output_count = plugin_attr_int(attrs[5], "output count");
  if (schema <= 0 || schema > UINT32_MAX ||
      static_cast<uint32_t>(schema) != callback.schema_version ||
      attr_schema <= 0 || attr_schema > UINT32_MAX ||
      static_cast<uint32_t>(attr_schema) != callback.attr_schema_version)
    throw std::runtime_error("emlx::native: plugin callback schema mismatch");
  if (output_count < 0 ||
      output_count > EMLX_PLUGIN_OUTPUT_COUNT_MAX_V1)
    throw std::runtime_error("emlx::native: plugin output count exceeds its limit");

  size_t cursor = 6;
  std::vector<PluginOutputTemplate> templates;
  templates.reserve(static_cast<size_t>(output_count));
  for (int64_t output = 0; output < output_count; ++output) {
    if (cursor + 2 > attrs.size() || !attrs[cursor].is_atom())
      throw std::runtime_error("emlx::native: malformed plugin output template");
    auto dtype = attrs[cursor++].as_dtype();
    const int64_t rank = plugin_attr_int(attrs[cursor++], "output rank");
    if (rank < 0 || rank > 16 || cursor + static_cast<size_t>(rank) > attrs.size())
      throw std::runtime_error("emlx::native: malformed plugin output rank");
    mlx::core::Shape shape;
    shape.reserve(static_cast<size_t>(rank));
    for (int64_t axis = 0; axis < rank; ++axis) {
      const int64_t dim = plugin_attr_int(attrs[cursor++], "output dimension");
      if (dim < 0 || dim > INT32_MAX)
        throw std::runtime_error("emlx::native: plugin output dimension is out of range");
      shape.push_back(static_cast<int>(dim));
    }
    templates.push_back({dtype, std::move(shape)});
  }
  if (cursor >= attrs.size())
    throw std::runtime_error("emlx::native: missing plugin callback attribute count");
  const int64_t callback_attr_count =
      plugin_attr_int(attrs[cursor++], "callback attribute count");
  if (callback_attr_count < 0 || callback_attr_count > 16384 ||
      cursor + static_cast<size_t>(callback_attr_count) != attrs.size())
    throw std::runtime_error("emlx::native: malformed plugin callback attributes");
  std::vector<int64_t> callback_attrs;
  callback_attrs.reserve(static_cast<size_t>(callback_attr_count));
  while(cursor < attrs.size()) {
    callback_attrs.push_back(plugin_attr_int(attrs[cursor++], "callback attribute"));
  }

  uint32_t expected_operands = callback.operand_count;
  if (expected_operands == 0) {
    expected_operands = emlx_invoke_plugin_count_policy(
        callback.operand_count_from_attrs,
        {callback_attrs.data(), callback_attrs.size()}, expected_operands,
        "operand", plugin_name, callback_name);
  }
  uint32_t expected_outputs = callback.output_count;
  if (expected_outputs == 0) {
    expected_outputs = emlx_invoke_plugin_count_policy(
        callback.output_count_from_attrs,
        {callback_attrs.data(), callback_attrs.size()}, expected_outputs,
        "output", plugin_name, callback_name);
  }
  if (instr.operands.size() != expected_operands) {
    throw std::runtime_error("emlx::native: plugin operand count mismatch");
  }
  if (expected_operands > EMLX_PLUGIN_OPERAND_COUNT_MAX_V1) {
    throw std::runtime_error("emlx::native: plugin operand count exceeds its limit");
  }
  if (expected_outputs > EMLX_PLUGIN_OUTPUT_COUNT_MAX_V1) {
    throw std::runtime_error("emlx::native: plugin output count exceeds its limit");
  }
  if (templates.size() != expected_outputs) {
    throw std::runtime_error("emlx::native: plugin output count mismatch");
  }

  instr.resolved_plugin = std::move(resolved);
  instr.plugin_attrs = std::move(callback_attrs);
  instr.plugin_outputs = std::move(templates);
}

static std::vector<mlx::core::array> invoke_plugin_instruction(
    const Instruction &instr, const std::vector<mlx::core::array> &operands) {
  if (!instr.resolved_plugin.callback || !emlx::g_current_worker)
    throw std::runtime_error("emlx::native: plugin execution has no current worker");
  const auto &callback = *instr.resolved_plugin.callback;
  const auto device = emlx::g_current_worker->device();
  const uint32_t device_bit = device.type == mlx::core::Device::DeviceType::cpu
                                  ? EMLX_PLUGIN_DEVICE_CPU_V1
                                  : EMLX_PLUGIN_DEVICE_GPU_METAL_V1;
  if ((callback.device_capabilities & device_bit) == 0)
    throw std::runtime_error("emlx::native: plugin callback does not support the worker device");

  const auto stream = emlx::g_current_worker->stream();
  EMLXPluginExecutionContext execution{&device, &stream};
  EMLXPluginCall call{{operands.data(), operands.size()},
                      {instr.plugin_attrs.data(), instr.plugin_attrs.size()},
                      &execution};
  std::vector<mlx::core::array> candidates;
  std::string error;
  bool ok = false;
  try {
    ok = callback.callback(call, candidates, error);
  } catch (const std::bad_alloc &) {
    error = "plugin callback allocation failed";
  } catch (const std::exception &exception) {
    error = exception.what();
  } catch (...) {
    error = "unknown plugin callback exception";
  }
  if (!ok)
    throw std::runtime_error(emlx_plugin_callback_failure_error(
        instr.resolved_plugin.plugin->name, callback.name, error));
  if (candidates.size() != instr.plugin_outputs.size()) {
    throw std::runtime_error(
        "emlx::native: plugin callback returned " +
        std::to_string(candidates.size()) + " outputs, expected " +
        std::to_string(instr.plugin_outputs.size()));
  }
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].shape() != instr.plugin_outputs[i].shape ||
        candidates[i].dtype() != instr.plugin_outputs[i].dtype)
      throw std::runtime_error("emlx::native: plugin callback output template mismatch");
  }
  return candidates;
}

static std::vector<mlx::core::array> interpret_instructions(
    const std::vector<Instruction> &instructions,
    const std::vector<Ref> &output_refs,
    const std::shared_ptr<const std::vector<mlx::core::array>> &captures,
    const std::shared_ptr<const std::vector<mlx::core::array>> &constants,
    const std::vector<mlx::core::array> *inputs,
    const std::vector<mlx::core::array> *carry) {
  std::vector<mlx::core::array> results;
  results.reserve(instructions.size());

  auto resolve = [&](const Ref &ref) -> mlx::core::array {
    switch (ref.kind) {
    case RefKind::Input:
      if (!inputs)
        throw std::runtime_error(
            "emlx::native: {:input, _} ref found inside a :while sub-program");
      return inputs->at(static_cast<size_t>(ref.index));
    case RefKind::Capture:
      return captures->at(static_cast<size_t>(ref.index));
    case RefKind::Const:
      return constants->at(static_cast<size_t>(ref.index));
    case RefKind::Result:
      return results.at(static_cast<size_t>(ref.index));
    case RefKind::Carry:
      if (!carry)
        throw std::runtime_error(
            "emlx::native: {:carry, _} ref found outside a :while sub-program");
      return carry->at(static_cast<size_t>(ref.index));
    }
    throw std::runtime_error("emlx::native: invalid ref kind");
  };

  for (const auto &instr : instructions) {
    std::vector<mlx::core::array> op_inputs;
    op_inputs.reserve(instr.operands.size());
    for (const auto &ref : instr.operands) {
      op_inputs.push_back(resolve(ref));
    }

    const std::string &name = instr.op.to_string();

    // `:while` isn't in op_registry/multi_op_registry (its behavior depends
    // on instr.subprograms, which those registries' OpFn/MultiOpFn
    // signatures have no access to) — dispatched here instead.
    if (name == "while") {
      const SubProgram &cond = instr.subprograms.at(0);
      const SubProgram &body = instr.subprograms.at(1);
      auto primitive =
          std::make_shared<EMLXWhile>(mlx::core::default_stream(k_linalg_cpu),
                                      cond, body, captures, constants);

      std::vector<mlx::core::Shape> shapes;
      std::vector<mlx::core::Dtype> dtypes;
      shapes.reserve(op_inputs.size());
      dtypes.reserve(op_inputs.size());
      for (const auto &a : op_inputs) {
        shapes.push_back(a.shape());
        dtypes.push_back(a.dtype());
      }

      auto outs =
          mlx::core::array::make_arrays(shapes, dtypes, primitive, op_inputs);
      for (auto &o : outs)
        results.push_back(o);
      continue;
    }

    if (name == "plugin") {
      auto outs = invoke_plugin_instruction(instr, op_inputs);
      for (auto &out : outs)
        results.push_back(std::move(out));
      continue;
    }

    auto multi_it = multi_op_registry.find(name);
    if (multi_it != multi_op_registry.end()) {
      // Multi-output op: append each result in order to the flat accumulator.
      auto outs = multi_it->second(op_inputs, instr.attrs);
      for (auto &o : outs)
        results.push_back(o);
    } else {
      results.push_back(op_registry.at(name)(op_inputs, instr.attrs));
    }
  }

  std::vector<mlx::core::array> outputs;
  outputs.reserve(output_refs.size());
  for (const auto &ref : output_refs) {
    outputs.push_back(resolve(ref));
  }
  return outputs;
}

// ── Global compile-cache mutex
// ────────────────────────────────────────────────
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

// ── Expr destructor
// ───────────────────────────────────────────────────────────
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

// ── NIF implementations
// ───────────────────────────────────────────────────────

// compile_program — decodes the wire Program (see EMLX.Native.Program /
// EMLX.Native.Expr.to_native/1, decoded directly by fine::Decoder<Program> in
// emlx_compiler.hpp), builds a capturing interpreter lambda backed by the op
// registry, wraps it with mlx::core::compile(), and stores the result as an
// opaque Expr BEAM resource.
// Validates op names against the registries up front (recursing into
// `:while` SubPrograms) so that any unknown op surfaces at compile time
// rather than inside the interpreter at (first) eval time. `:while` itself
// is valid despite never appearing in op_registry/multi_op_registry — see
// interpret_instructions.
static void validate_instructions(std::vector<Instruction> &instructions,
                                  bool &has_runtime_call) {
  for (auto &instr : instructions) {
    const std::string &name = instr.op.to_string();

    if (name == "while") {
      if (instr.subprograms.size() != 2)
        throw std::runtime_error(
            "emlx::native: :while instruction must have exactly 2 "
            "subprograms (cond, body), got " +
            std::to_string(instr.subprograms.size()));
      validate_instructions(instr.subprograms[0].instructions,
                            has_runtime_call);
      validate_instructions(instr.subprograms[1].instructions,
                            has_runtime_call);
      continue;
    }

    if (name == "plugin") {
      if (!instr.subprograms.empty())
        throw std::runtime_error(
            "emlx::native: :plugin instruction cannot contain subprograms");
      resolve_plugin_instruction(instr);
      continue;
    }

    if (op_registry.find(name) == op_registry.end() &&
        multi_op_registry.find(name) == multi_op_registry.end())
      throw std::runtime_error("emlx::native: unknown op \"" + name + "\"");
    if (name == "runtime_call")
      has_runtime_call = true;
  }
}

fine::Term compile_program_impl(ErlNifEnv *env, Program program) {
  bool has_runtime_call = false;
  validate_instructions(program.instructions, has_runtime_call);

  // Build constant arrays on the current (worker) thread using its default
  // stream.
  std::vector<mlx::core::array> constants;
  constants.reserve(program.constants.size());
  for (const auto &[value, dtype] : program.constants) {
    constants.push_back(mlx::core::full({}, value, dtype));
  }

  int num_inputs_val = program.num_inputs;

  // Build the interpreter lambda capturing all program data, then pass it
  // through mlx::core::compile().  MLX traces the lambda on the first
  // eval_program call (building a compiled computation graph) and replays
  // the cached graph on every subsequent call — no repeated graph construction.
  // captures/constants are shared_ptr-wrapped so a `:while` instruction's
  // EMLXWhile primitive (see interpret_instructions) can hold onto the same
  // tables independently of this lambda's own stack frame/lifetime.
  emlx::function fn =
      [captures = std::make_shared<const std::vector<mlx::core::array>>(
           std::move(program.captures)),
       constants = std::make_shared<const std::vector<mlx::core::array>>(
           std::move(constants)),
       instructions = std::move(program.instructions),
       output_refs = std::move(program.outputs)](
          const std::vector<mlx::core::array> &inputs)
      -> std::vector<mlx::core::array> {
    return interpret_instructions(instructions, output_refs, captures,
                                  constants, &inputs, /*carry=*/nullptr);
  };

  // Allocate the program resource.
  auto *ptr = static_cast<Expr *>(
      enif_alloc_resource(resource_object<Expr>::type, sizeof(Expr)));
  if (!ptr)
    throw std::runtime_error("Failed to allocate Expr resource");

  // Assign a unique ID so MLX's global compile cache has a distinct entry per
  // Expr resource.  All our lambdas share the same C++ type (same capture
  // types), so the public mlx::core::compile() would map them all to the same
  // cache key — causing stale graph reuse across different compiled programs.
  static std::atomic<std::uintptr_t> next_id{1};
  std::uintptr_t unique_id = next_id.fetch_add(1, std::memory_order_relaxed);

  new (ptr) Expr();
  ptr->num_inputs = num_inputs_val;
  ptr->compile_id = unique_id;
  ptr->has_runtime_call = has_runtime_call;
  ptr->num_real_outputs = program.num_real_outputs;
  {
    std::lock_guard<std::mutex> lk(s_mlx_compile_mutex);
    ptr->compiled_fn = mlx::core::detail::compile(std::move(fn), unique_id);
  }

  ERL_NIF_TERM ret = enif_make_resource(env, ptr);
  enif_release_resource(ptr);
  return fine::Term(ret);
}
FINE_ASYNC_NIF(compile_program)

// eval_program — calls the MLX-compiled function against runtime inputs.
// MLX traces on the first call and replays the cached graph on subsequent
// calls. Returns lazy output array refs — materialization is deferred to the
// caller (to_binary / Nx.to_number), matching the Evaluator's deferred-eval
// pattern.
//
// argv[0] : program_ref  (emlx::native::Expr resource)
// argv[1] : input_refs   (list of MLX array resource refs — runtime inputs)
ERL_NIF_TERM eval_program(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
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

    // A program containing an inlined `:runtime_call` node must be forced to
    // materialize now, while the caller pid this NIF call was dispatched
    // with (emlx::g_current_caller_pid, set by emlx::async_dispatch — see
    // emlx_async.hpp) is still in scope: EMLXRuntimeCall::eval_cpu/eval_gpu
    // reads it to know which BEAM process to send the round-trip request
    // to. Deliberately outside `s_mlx_compile_mutex`'s scope above — that
    // lock is process-wide (shared with every other worker's
    // compile_program/eval_program calls), and a runtime_call's blocking
    // wait can run arbitrary, unbounded Elixir code, including calls that
    // themselves need that same lock on another worker thread. Every other
    // program keeps today's lazy/deferred return (no eval() call here).
    if (prog->has_runtime_call) {
      mlx::core::eval(outputs);
      mlx::core::synchronize();
    }

    // Only the real outputs are converted to resources and returned to
    // Elixir — any keepalive tail (see emlx::native::Program::num_real_outputs)
    // was included above solely to force mlx::core::eval to run it for its
    // side effects; the caller has no use for the resulting array itself.
    size_t n = static_cast<size_t>(prog->num_real_outputs);
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
