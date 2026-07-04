#include "qwen3_plugin_abi.hpp"

#include <limits>
#include <optional>
#include <sstream>

// Qwen3 compute plugin — pure MLX graph-building code, no Erlang/erl_nif
// dependency whatsoever. Built as its own shared library (libemlx_qwen3.so)
// and `dlopen`'d by the host NIF library (c_src/emlx_fast/qwen3.cpp) via
// `EMLX.NIF.load_qwen3_plugin/1`. See qwen3_plugin_abi.hpp for the ABI and
// the rationale for this split.

using namespace emlx_qwen3_plugin;

namespace {

mlx::core::Shape to_shape(std::initializer_list<int> v) {
  return mlx::core::Shape(v.begin(), v.end());
}

// ── Shape/argument validation ────────────────────────────────────────────

bool check_rank(const mlx::core::array &tensor, int expected, const char *name,
                 std::string &error) {
  if (tensor.ndim() != expected) {
    std::ostringstream msg;
    msg << name << " expects rank " << expected << ", got rank " << tensor.ndim();
    error = msg.str();
    return false;
  }
  return true;
}

bool check_positive(int value, const char *name, std::string &error) {
  if (value <= 0) {
    error = std::string(name) + " must be positive";
    return false;
  }
  return true;
}

bool check_non_negative(int value, const char *name, std::string &error) {
  if (value < 0) {
    error = std::string(name) + " must be non-negative";
    return false;
  }
  return true;
}

bool check_dim(const mlx::core::array &tensor, int axis, int expected,
                const char *name, const char *dim_name, std::string &error) {
  if (tensor.shape(axis) != expected) {
    std::ostringstream msg;
    msg << name << " " << dim_name << " must be " << expected << ", got "
        << tensor.shape(axis);
    error = msg.str();
    return false;
  }
  return true;
}

bool check_rank_positive(const mlx::core::array &tensor, int rank,
                          const char *name, std::string &error) {
  if (!check_rank(tensor, rank, name, error)) {
    return false;
  }
  for (int axis = 0; axis < rank; ++axis) {
    if (tensor.shape(axis) <= 0) {
      error = std::string(name) + " dimensions must be positive";
      return false;
    }
  }
  return true;
}

bool check_rank4_positive(const mlx::core::array &t, const char *n, std::string &e) {
  return check_rank_positive(t, 4, n, e);
}
bool check_rank3_positive(const mlx::core::array &t, const char *n, std::string &e) {
  return check_rank_positive(t, 3, n, e);
}
bool check_rank2_positive(const mlx::core::array &t, const char *n, std::string &e) {
  return check_rank_positive(t, 2, n, e);
}

bool check_rank1_dim(const mlx::core::array &tensor, int expected,
                      const char *name, std::string &error) {
  if (!check_rank(tensor, 1, name, error)) {
    return false;
  }
  return check_dim(tensor, 0, expected, name, "size", error);
}

bool validate_projection_width(const mlx::core::array &projection, int input_width,
                                int head_dim, const char *name, std::string &error) {
  if (!check_rank2_positive(projection, name, error)) {
    return false;
  }
  if (!check_dim(projection, 0, input_width, name, "input width", error)) {
    return false;
  }
  if ((projection.shape(1) % head_dim) != 0) {
    error = std::string(name) + " output width must be divisible by head_dim";
    return false;
  }
  return true;
}

bool validate_kv_cache_bn(const mlx::core::array &k_cache,
                           const mlx::core::array &v_cache, int batch,
                           int num_kv_heads, int offset, int token_count,
                           int head_dim, std::string &error) {
  if (!check_rank4_positive(k_cache, "k_cache", error) ||
      !check_rank4_positive(v_cache, "v_cache", error)) {
    return false;
  }
  if (!check_dim(k_cache, 0, batch, "k_cache", "batch", error) ||
      !check_dim(v_cache, 0, batch, "v_cache", "batch", error) ||
      !check_dim(k_cache, 1, num_kv_heads, "k_cache", "heads", error) ||
      !check_dim(v_cache, 1, num_kv_heads, "v_cache", "heads", error) ||
      !check_dim(k_cache, 3, head_dim, "k_cache", "head_dim", error) ||
      !check_dim(v_cache, 3, head_dim, "v_cache", "head_dim", error)) {
    return false;
  }
  if (v_cache.shape(2) != k_cache.shape(2)) {
    error = "k_cache and v_cache capacity must match";
    return false;
  }
  int64_t required_len = static_cast<int64_t>(offset) + static_cast<int64_t>(token_count);
  int capacity = k_cache.shape(2);
  if (required_len > capacity) {
    std::ostringstream msg;
    msg << "KV cache capacity " << capacity << " is smaller than required length "
        << required_len;
    error = msg.str();
    return false;
  }
  return true;
}

bool validate_qkv_cache_attention(const mlx::core::array &q,
                                   const mlx::core::array &new_k,
                                   const mlx::core::array &new_v,
                                   const mlx::core::array &k_cache,
                                   const mlx::core::array &v_cache, int offset,
                                   int head_dim, std::string &error) {
  if (!check_rank4_positive(q, "q", error) ||
      !check_rank4_positive(new_k, "new_k", error) ||
      !check_rank4_positive(new_v, "new_v", error) ||
      !check_non_negative(offset, "offset", error) ||
      !check_positive(head_dim, "head_dim", error)) {
    return false;
  }

  int B = q.shape(0);
  int T_new = q.shape(1);
  int N_q = q.shape(2);
  int D = q.shape(3);
  int N_kv = new_k.shape(2);

  if (D != head_dim) {
    error = "q last dimension must match head_dim";
    return false;
  }
  if ((N_q % N_kv) != 0) {
    error = "query heads must be divisible by key/value heads";
    return false;
  }
  if (!check_dim(new_k, 0, B, "new_k", "batch", error) ||
      !check_dim(new_v, 0, B, "new_v", "batch", error) ||
      !check_dim(new_k, 1, T_new, "new_k", "sequence length", error) ||
      !check_dim(new_v, 1, T_new, "new_v", "sequence length", error) ||
      !check_dim(new_v, 2, N_kv, "new_v", "heads", error) ||
      !check_dim(new_k, 3, D, "new_k", "head_dim", error) ||
      !check_dim(new_v, 3, D, "new_v", "head_dim", error)) {
    return false;
  }

  return validate_kv_cache_bn(k_cache, v_cache, B, N_kv, offset, T_new, D, error);
}

// ── Dense/quantized linear projection helpers ────────────────────────────

mlx::core::array linear_in_out(const mlx::core::array &x, const mlx::core::array &weight,
                                const mlx::core::Device &device) {
  if (x.ndim() == 3 && x.shape(1) == 1) {
    auto x_2d = mlx::core::reshape(x, {x.shape(0), x.shape(2)}, device);
    auto out = mlx::core::matmul(x_2d, weight, device);
    return mlx::core::reshape(out, {x.shape(0), 1, weight.shape(1)}, device);
  }
  return mlx::core::matmul(x, weight, device);
}

mlx::core::array linear_out_in(const mlx::core::array &x, const mlx::core::array &weight,
                                const mlx::core::Device &device) {
  return mlx::core::tensordot(x, weight, std::vector<int>{static_cast<int>(x.ndim()) - 1},
                               std::vector<int>{1}, device);
}

mlx::core::array apply_linear(const mlx::core::array &x, const LinearWeight &w,
                               const mlx::core::Device &device) {
  if (w.quantized) {
    std::optional<mlx::core::array> biases_opt =
        w.biases != nullptr ? std::make_optional(*w.biases) : std::nullopt;
    return mlx::core::quantized_matmul(x, *w.weight, *w.scales, biases_opt, w.transpose,
                                        w.group_size, w.bits, w.mode, device);
  }
  return w.transpose ? linear_out_in(x, *w.weight, device)
                      : linear_in_out(x, *w.weight, device);
}

int linear_weight_out_features(const LinearWeight &w) {
  if (w.quantized) {
    return static_cast<int>(w.scales->shape(0));
  }
  return w.transpose ? static_cast<int>(w.weight->shape(0))
                      : static_cast<int>(w.weight->shape(1));
}

int64_t token_to_int64(mlx::core::array &token) {
  mlx::core::eval(token);
  auto dtype = token.dtype();
  if (dtype == mlx::core::uint8) return static_cast<int64_t>(token.item<uint8_t>());
  if (dtype == mlx::core::uint16) return static_cast<int64_t>(token.item<uint16_t>());
  if (dtype == mlx::core::uint32) return static_cast<int64_t>(token.item<uint32_t>());
  if (dtype == mlx::core::uint64) return static_cast<int64_t>(token.item<uint64_t>());
  if (dtype == mlx::core::int8) return static_cast<int64_t>(token.item<int8_t>());
  if (dtype == mlx::core::int16) return static_cast<int64_t>(token.item<int16_t>());
  if (dtype == mlx::core::int32) return static_cast<int64_t>(token.item<int32_t>());
  return token.item<int64_t>();
}

bool check_linear_weight_in(const LinearWeight &w, int expected_in, const char *name,
                             std::string &error) {
  if (w.quantized) {
    if (!check_rank2_positive(*w.weight, name, error) ||
        !check_rank2_positive(*w.scales, name, error)) {
      return false;
    }
    if (w.bits <= 0 || w.bits > 32 || (32 % w.bits) != 0) {
      error = std::string(name) + " has an invalid bits value";
      return false;
    }
    int actual_in = static_cast<int>(w.weight->shape(1)) * (32 / w.bits);
    if (actual_in != expected_in) {
      std::ostringstream msg;
      msg << name << " input width must be " << expected_in << ", got " << actual_in;
      error = msg.str();
      return false;
    }
    return true;
  }

  if (!check_rank2_positive(*w.weight, name, error)) {
    return false;
  }
  int actual_in =
      w.transpose ? static_cast<int>(w.weight->shape(1)) : static_cast<int>(w.weight->shape(0));
  if (actual_in != expected_in) {
    std::ostringstream msg;
    msg << name << " input width must be " << expected_in << ", got " << actual_in;
    error = msg.str();
    return false;
  }
  return true;
}

bool validate_generalized_layer(const mlx::core::array &hidden, const LayerParamsQ &layer,
                                 const KVCache &kv, int offset, int head_dim,
                                 std::string &error) {
  if (!check_rank3_positive(hidden, "hidden", error) ||
      !check_non_negative(offset, "offset", error) ||
      !check_positive(head_dim, "head_dim", error)) {
    return false;
  }

  int B = hidden.shape(0);
  int T_new = hidden.shape(1);
  int H = hidden.shape(2);
  int D = head_dim;

  if (!check_rank1_dim(*layer.norm1, H, "norm1", error) ||
      !check_rank1_dim(*layer.norm2, H, "norm2", error) ||
      !check_rank1_dim(*layer.q_norm, D, "q_norm", error) ||
      !check_rank1_dim(*layer.k_norm, D, "k_norm", error) ||
      !check_linear_weight_in(layer.q_proj, H, "q_proj", error) ||
      !check_linear_weight_in(layer.k_proj, H, "k_proj", error) ||
      !check_linear_weight_in(layer.v_proj, H, "v_proj", error) ||
      !check_linear_weight_in(layer.gate_proj, H, "gate_proj", error) ||
      !check_linear_weight_in(layer.up_proj, H, "up_proj", error)) {
    return false;
  }

  int q_out = linear_weight_out_features(layer.q_proj);
  int k_out = linear_weight_out_features(layer.k_proj);
  int v_out = linear_weight_out_features(layer.v_proj);

  if ((q_out % D) != 0 || (k_out % D) != 0) {
    error = "q_proj/k_proj output width must be divisible by head_dim";
    return false;
  }
  if (v_out != k_out) {
    error = "v_proj output width must match k_proj output width";
    return false;
  }

  int N_q = q_out / D;
  int N_kv = k_out / D;
  if ((N_q % N_kv) != 0) {
    error = "query heads must be divisible by key/value heads";
    return false;
  }

  int attn_width = N_q * D;
  if (!check_linear_weight_in(layer.o_proj, attn_width, "o_proj", error)) {
    return false;
  }
  if (linear_weight_out_features(layer.o_proj) != H) {
    error = "o_proj output width must match hidden width";
    return false;
  }

  int gate_out = linear_weight_out_features(layer.gate_proj);
  int up_out = linear_weight_out_features(layer.up_proj);
  if (up_out != gate_out) {
    error = "up_proj output width must match gate_proj output width";
    return false;
  }

  if (!check_linear_weight_in(layer.down_proj, gate_out, "down_proj", error)) {
    return false;
  }
  if (linear_weight_out_features(layer.down_proj) != H) {
    error = "down_proj output width must match hidden width";
    return false;
  }

  return validate_kv_cache_bn(*kv.k, *kv.v, B, N_kv, offset, T_new, D, error);
}

bool validate_dense_layer(const mlx::core::array &hidden, const LayerParams &layer,
                           const KVCache &kv, int offset, int head_dim, std::string &error) {
  if (!check_rank3_positive(hidden, "hidden", error) ||
      !check_non_negative(offset, "offset", error) ||
      !check_positive(head_dim, "head_dim", error)) {
    return false;
  }

  int B = hidden.shape(0);
  int T_new = hidden.shape(1);
  int H = hidden.shape(2);
  int D = head_dim;

  if (!check_rank1_dim(*layer.norm1, H, "norm1", error) ||
      !check_rank1_dim(*layer.norm2, H, "norm2", error) ||
      !validate_projection_width(*layer.q_proj, H, D, "q_proj", error) ||
      !validate_projection_width(*layer.k_proj, H, D, "k_proj", error) ||
      !validate_projection_width(*layer.v_proj, H, D, "v_proj", error) ||
      !check_dim(*layer.v_proj, 1, layer.k_proj->shape(1), "v_proj", "output width", error) ||
      !check_rank1_dim(*layer.q_norm, D, "q_norm", error) ||
      !check_rank1_dim(*layer.k_norm, D, "k_norm", error)) {
    return false;
  }

  int N_q = layer.q_proj->shape(1) / D;
  int N_kv = layer.k_proj->shape(1) / D;
  int attn_width = N_q * D;

  if ((N_q % N_kv) != 0) {
    error = "query heads must be divisible by key/value heads";
    return false;
  }
  if (!check_rank2_positive(*layer.o_proj, "o_proj", error) ||
      !check_dim(*layer.o_proj, 0, attn_width, "o_proj", "input width", error) ||
      !check_dim(*layer.o_proj, 1, H, "o_proj", "output width", error) ||
      !validate_kv_cache_bn(*kv.k, *kv.v, B, N_kv, offset, T_new, D, error) ||
      !check_rank2_positive(*layer.gate_proj, "gate_proj", error) ||
      !check_rank2_positive(*layer.up_proj, "up_proj", error) ||
      !check_rank2_positive(*layer.down_proj, "down_proj", error) ||
      !check_dim(*layer.gate_proj, 0, H, "gate_proj", "input width", error) ||
      !check_dim(*layer.up_proj, 0, H, "up_proj", "input width", error) ||
      !check_dim(*layer.up_proj, 1, layer.gate_proj->shape(1), "up_proj", "output width", error) ||
      !check_dim(*layer.down_proj, 0, layer.gate_proj->shape(1), "down_proj", "input width",
                 error) ||
      !check_dim(*layer.down_proj, 1, H, "down_proj", "output width", error)) {
    return false;
  }

  return true;
}

// Shared causal/prefill mask builder used by every attention path below.
mlx::core::array build_prefill_mask(const mlx::core::array &q, int T_new, int valid_len,
                                     const mlx::core::Device &device) {
  auto mask_dtype = q.dtype();
  auto zero_val = mlx::core::zeros({}, mask_dtype, device);
  auto neginf_val =
      mlx::core::full({}, -std::numeric_limits<float>::infinity(), mask_dtype, device);
  int kv_offset = valid_len - T_new;
  auto row = mlx::core::reshape(mlx::core::arange(T_new, mlx::core::int32, device),
                                 {1, 1, T_new, 1}, device);
  auto col = mlx::core::reshape(mlx::core::arange(valid_len, mlx::core::int32, device),
                                 {1, 1, 1, valid_len}, device);
  auto causal_bool = mlx::core::less_equal(
      col, mlx::core::add(row, mlx::core::array(kv_offset, mlx::core::int32), device), device);
  return mlx::core::where(causal_bool, zero_val, neginf_val, device);
}

mlx::core::array sdpa(const mlx::core::array &q_rope, const mlx::core::array &k_valid,
                       const mlx::core::array &v_valid, float scale, int T_new,
                       int valid_len, const mlx::core::Device &device) {
  return (T_new == 1)
             ? mlx::core::fast::scaled_dot_product_attention(
                   q_rope, k_valid, v_valid, scale, "", std::nullopt, std::nullopt, device)
             : mlx::core::fast::scaled_dot_product_attention(
                   q_rope, k_valid, v_valid, scale, "array",
                   build_prefill_mask(q_rope, T_new, valid_len, device), std::nullopt, device);
}

// ── Generalized (dense-or-quantized) per-layer compute ───────────────────
// Mirrors `layer_dense_impl` exactly, but threads every projection through
// `apply_linear` instead of assuming a dense weight.
mlx::core::array layer_core_generalized(const mlx::core::array &hidden,
                                         const LayerParamsQ &layer, KVCache &kv, int offset,
                                         float scale, int head_dim, float theta, float eps,
                                         const mlx::core::Device &device,
                                         mlx::core::array *k_out, mlx::core::array *v_out) {
  int B = hidden.shape(0);
  int T_new = hidden.shape(1);
  int D = head_dim;
  int N_q = linear_weight_out_features(layer.q_proj) / D;
  int N_kv = linear_weight_out_features(layer.k_proj) / D;
  int attn_width = N_q * D;
  int valid_len = offset + T_new;

  auto xn = mlx::core::fast::rms_norm(hidden, *layer.norm1, eps, device);
  auto q_flat = apply_linear(xn, layer.q_proj, device);
  auto k_flat = apply_linear(xn, layer.k_proj, device);
  auto v_flat = apply_linear(xn, layer.v_proj, device);

  auto q = mlx::core::reshape(q_flat, {B, T_new, N_q, D}, device);
  auto k = mlx::core::reshape(k_flat, {B, T_new, N_kv, D}, device);
  auto v = mlx::core::reshape(v_flat, {B, T_new, N_kv, D}, device);

  q = mlx::core::fast::rms_norm(q, *layer.q_norm, eps, device);
  k = mlx::core::fast::rms_norm(k, *layer.k_norm, eps, device);

  auto q_bn = mlx::core::transpose(q, {0, 2, 1, 3}, device);
  auto k_bn = mlx::core::transpose(k, {0, 2, 1, 3}, device);
  auto v_bn = mlx::core::transpose(v, {0, 2, 1, 3}, device);

  auto q_rope = mlx::core::fast::rope(q_bn, D, false, theta, 1.0f, offset, std::nullopt, device);
  auto k_rope = mlx::core::fast::rope(k_bn, D, false, theta, 1.0f, offset, std::nullopt, device);

  auto k_cache_owned = std::move(*kv.k);
  auto v_cache_owned = std::move(*kv.v);

  auto k_upd = mlx::core::slice_update(k_cache_owned, k_rope, to_shape({0, 0, offset, 0}),
                                        to_shape({B, N_kv, valid_len, D}), device);
  auto v_upd = mlx::core::slice_update(v_cache_owned, v_bn, to_shape({0, 0, offset, 0}),
                                        to_shape({B, N_kv, valid_len, D}), device);

  auto k_valid = mlx::core::slice(k_upd, to_shape({0, 0, 0, 0}),
                                   to_shape({B, N_kv, valid_len, D}), device);
  auto v_valid = mlx::core::slice(v_upd, to_shape({0, 0, 0, 0}),
                                   to_shape({B, N_kv, valid_len, D}), device);

  auto attn_out_bn = sdpa(q_rope, k_valid, v_valid, scale, T_new, valid_len, device);
  auto attn_out_bthd = mlx::core::transpose(attn_out_bn, {0, 2, 1, 3}, device);
  auto attn_out = mlx::core::reshape(attn_out_bthd, {B, T_new, attn_width}, device);
  auto attn_projected = apply_linear(attn_out, layer.o_proj, device);
  auto attn_hidden = mlx::core::add(hidden, attn_projected, device);

  auto xn2 = mlx::core::fast::rms_norm(attn_hidden, *layer.norm2, eps, device);
  auto gate = apply_linear(xn2, layer.gate_proj, device);
  auto up = apply_linear(xn2, layer.up_proj, device);
  auto mlp = mlx::core::multiply(mlx::core::multiply(gate, mlx::core::sigmoid(gate, device), device),
                                  up, device);
  auto mlp_out = apply_linear(mlp, layer.down_proj, device);

  if (k_out != nullptr) *k_out = k_upd;
  if (v_out != nullptr) *v_out = v_upd;

  return mlx::core::add(attn_hidden, mlp_out, device);
}

// ── Dense per-layer compute ───────────────────────────────────────────────
mlx::core::array layer_dense_impl(const mlx::core::array &hidden, const LayerParams &layer,
                                   KVCache &kv, int offset, float scale, int head_dim,
                                   float theta, float eps, const mlx::core::Device &device,
                                   mlx::core::array *k_out, mlx::core::array *v_out) {
  int B = hidden.shape(0);
  int T_new = hidden.shape(1);
  int D = head_dim;
  int N_q = layer.q_proj->shape(1) / D;
  int N_kv = layer.k_proj->shape(1) / D;
  int attn_width = N_q * D;
  int valid_len = offset + T_new;

  auto xn = mlx::core::fast::rms_norm(hidden, *layer.norm1, eps, device);
  auto q_flat = linear_in_out(xn, *layer.q_proj, device);
  auto k_flat = linear_in_out(xn, *layer.k_proj, device);
  auto v_flat = linear_in_out(xn, *layer.v_proj, device);

  auto q = mlx::core::reshape(q_flat, {B, T_new, N_q, D}, device);
  auto k = mlx::core::reshape(k_flat, {B, T_new, N_kv, D}, device);
  auto v = mlx::core::reshape(v_flat, {B, T_new, N_kv, D}, device);

  q = mlx::core::fast::rms_norm(q, *layer.q_norm, eps, device);
  k = mlx::core::fast::rms_norm(k, *layer.k_norm, eps, device);

  auto q_bn = mlx::core::transpose(q, {0, 2, 1, 3}, device);
  auto k_bn = mlx::core::transpose(k, {0, 2, 1, 3}, device);
  auto v_bn = mlx::core::transpose(v, {0, 2, 1, 3}, device);

  auto q_rope = mlx::core::fast::rope(q_bn, D, false, theta, 1.0f, offset, std::nullopt, device);
  auto k_rope = mlx::core::fast::rope(k_bn, D, false, theta, 1.0f, offset, std::nullopt, device);

  auto k_cache_owned = std::move(*kv.k);
  auto v_cache_owned = std::move(*kv.v);

  auto k_upd = mlx::core::slice_update(k_cache_owned, k_rope, to_shape({0, 0, offset, 0}),
                                        to_shape({B, N_kv, valid_len, D}), device);
  auto v_upd = mlx::core::slice_update(v_cache_owned, v_bn, to_shape({0, 0, offset, 0}),
                                        to_shape({B, N_kv, valid_len, D}), device);

  auto k_valid = mlx::core::slice(k_upd, to_shape({0, 0, 0, 0}),
                                   to_shape({B, N_kv, valid_len, D}), device);
  auto v_valid = mlx::core::slice(v_upd, to_shape({0, 0, 0, 0}),
                                   to_shape({B, N_kv, valid_len, D}), device);

  auto attn_out_bn = sdpa(q_rope, k_valid, v_valid, scale, T_new, valid_len, device);
  auto attn_out_bthd = mlx::core::transpose(attn_out_bn, {0, 2, 1, 3}, device);
  auto attn_out = mlx::core::reshape(attn_out_bthd, {B, T_new, attn_width}, device);
  auto attn_projected = linear_in_out(attn_out, *layer.o_proj, device);
  auto attn_hidden = mlx::core::add(hidden, attn_projected, device);

  auto xn2 = mlx::core::fast::rms_norm(attn_hidden, *layer.norm2, eps, device);
  auto gate = linear_in_out(xn2, *layer.gate_proj, device);
  auto up = linear_in_out(xn2, *layer.up_proj, device);
  auto mlp = mlx::core::multiply(mlx::core::multiply(gate, mlx::core::sigmoid(gate, device), device),
                                  up, device);
  auto mlp_out = linear_in_out(mlp, *layer.down_proj, device);

  if (k_out != nullptr) *k_out = k_upd;
  if (v_out != nullptr) *v_out = v_upd;

  return mlx::core::add(attn_hidden, mlp_out, device);
}

// ── VTable entrypoints ────────────────────────────────────────────────────

bool v_kv_cache_attention(const mlx::core::array &q, const mlx::core::array &new_k,
                           const mlx::core::array &new_v, mlx::core::array &k_cache,
                           mlx::core::array &v_cache, int offset, double scale, int head_dim,
                           double theta, const mlx::core::Device &device, mlx::core::array &out,
                           mlx::core::array &k_upd, mlx::core::array &v_upd,
                           std::string &error) {
  try {
    if (!validate_qkv_cache_attention(q, new_k, new_v, k_cache, v_cache, offset, head_dim,
                                       error)) {
      return false;
    }

    int B = q.shape(0);
    int T_new = q.shape(1);
    int N_q = q.shape(2);
    int D = q.shape(3);
    int N_kv = new_k.shape(2);
    int valid_len = offset + T_new;

    auto q_bn = mlx::core::transpose(q, {0, 2, 1, 3}, device);
    auto k_bn = mlx::core::transpose(new_k, {0, 2, 1, 3}, device);
    auto v_bn = mlx::core::transpose(new_v, {0, 2, 1, 3}, device);

    auto q_rope =
        mlx::core::fast::rope(q_bn, head_dim, false, (float)theta, 1.0f, offset, std::nullopt, device);
    auto k_rope =
        mlx::core::fast::rope(k_bn, head_dim, false, (float)theta, 1.0f, offset, std::nullopt, device);

    auto k_cache_owned = std::move(k_cache);
    auto v_cache_owned = std::move(v_cache);

    k_upd = mlx::core::slice_update(k_cache_owned, k_rope, to_shape({0, 0, offset, 0}),
                                     to_shape({B, N_kv, valid_len, D}), device);
    v_upd = mlx::core::slice_update(v_cache_owned, v_bn, to_shape({0, 0, offset, 0}),
                                     to_shape({B, N_kv, valid_len, D}), device);

    auto k_valid = mlx::core::slice(k_upd, to_shape({0, 0, 0, 0}),
                                     to_shape({B, N_kv, valid_len, D}), device);
    auto v_valid = mlx::core::slice(v_upd, to_shape({0, 0, 0, 0}),
                                     to_shape({B, N_kv, valid_len, D}), device);

    auto attn_out_bn = sdpa(q_rope, k_valid, v_valid, (float)scale, T_new, valid_len, device);
    auto attn_out_bthd = mlx::core::transpose(attn_out_bn, {0, 2, 1, 3}, device);
    out = mlx::core::reshape(attn_out_bthd, {B, T_new, N_q * D}, device);
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  } catch (...) {
    error = "Unknown error in qwen3 plugin kv_cache_attention";
    return false;
  }
}

bool v_mlp(const mlx::core::array &hidden, const mlx::core::array &norm,
           const mlx::core::array &gate_proj, const mlx::core::array &up_proj,
           const mlx::core::array &down_proj, double eps, const mlx::core::Device &device,
           mlx::core::array &out, std::string &error) {
  try {
    if (!check_rank3_positive(hidden, "hidden", error)) {
      return false;
    }
    int H = hidden.shape(2);
    if (!check_rank1_dim(norm, H, "norm", error) ||
        !check_rank2_positive(gate_proj, "gate_proj", error) ||
        !check_rank2_positive(up_proj, "up_proj", error) ||
        !check_rank2_positive(down_proj, "down_proj", error) ||
        !check_dim(gate_proj, 0, H, "gate_proj", "input width", error) ||
        !check_dim(up_proj, 0, H, "up_proj", "input width", error) ||
        !check_dim(up_proj, 1, gate_proj.shape(1), "up_proj", "output width", error) ||
        !check_dim(down_proj, 0, gate_proj.shape(1), "down_proj", "input width", error) ||
        !check_dim(down_proj, 1, H, "down_proj", "output width", error)) {
      return false;
    }

    auto xn = mlx::core::fast::rms_norm(hidden, norm, (float)eps, device);
    auto gate = linear_in_out(xn, gate_proj, device);
    auto up = linear_in_out(xn, up_proj, device);
    auto mlp = mlx::core::multiply(mlx::core::multiply(gate, mlx::core::sigmoid(gate, device), device),
                                    up, device);
    auto proj = linear_in_out(mlp, down_proj, device);
    out = mlx::core::add(hidden, proj, device);
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  } catch (...) {
    error = "Unknown error in qwen3 plugin mlp";
    return false;
  }
}

bool v_attention_residual(const mlx::core::array &hidden, const mlx::core::array &attn_out,
                           const mlx::core::array &o_proj, const mlx::core::Device &device,
                           mlx::core::array &out, std::string &error) {
  try {
    if (!check_rank3_positive(hidden, "hidden", error) ||
        !check_rank3_positive(attn_out, "attn_out", error)) {
      return false;
    }
    int B = hidden.shape(0);
    int T = hidden.shape(1);
    int H = hidden.shape(2);
    if (!check_dim(attn_out, 0, B, "attn_out", "batch", error) ||
        !check_dim(attn_out, 1, T, "attn_out", "sequence length", error) ||
        !check_rank2_positive(o_proj, "o_proj", error) ||
        !check_dim(o_proj, 0, attn_out.shape(2), "o_proj", "input width", error) ||
        !check_dim(o_proj, 1, H, "o_proj", "output width", error)) {
      return false;
    }
    auto projected = linear_in_out(attn_out, o_proj, device);
    out = mlx::core::add(hidden, projected, device);
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  } catch (...) {
    error = "Unknown error in qwen3 plugin attention_residual";
    return false;
  }
}

bool v_attention_block(const mlx::core::array &hidden, const mlx::core::array &norm,
                        const mlx::core::array &q_proj, const mlx::core::array &k_proj,
                        const mlx::core::array &v_proj, const mlx::core::array &o_proj,
                        const mlx::core::array &q_norm, const mlx::core::array &k_norm,
                        mlx::core::array &k_cache, mlx::core::array &v_cache, int offset,
                        double scale, int head_dim, double theta, double eps,
                        const mlx::core::Device &device, mlx::core::array &out,
                        mlx::core::array &k_upd, mlx::core::array &v_upd, std::string &error) {
  try {
    if (!check_rank3_positive(hidden, "hidden", error) ||
        !check_non_negative(offset, "offset", error) ||
        !check_positive(head_dim, "head_dim", error)) {
      return false;
    }

    int B = hidden.shape(0);
    int T_new = hidden.shape(1);
    int H = hidden.shape(2);
    int D = head_dim;
    if (!check_rank1_dim(norm, H, "norm", error) || !check_rank1_dim(q_norm, D, "q_norm", error) ||
        !check_rank1_dim(k_norm, D, "k_norm", error) ||
        !check_rank2_positive(q_proj, "q_proj", error) ||
        !check_rank2_positive(k_proj, "k_proj", error) ||
        !check_rank2_positive(v_proj, "v_proj", error) ||
        !check_dim(q_proj, 0, H, "q_proj", "input width", error) ||
        !check_dim(k_proj, 0, H, "k_proj", "input width", error) ||
        !check_dim(v_proj, 0, H, "v_proj", "input width", error)) {
      return false;
    }

    if ((q_proj.shape(1) % D) != 0 || (k_proj.shape(1) % D) != 0) {
      error = "projection output widths must be divisible by head_dim";
      return false;
    }
    if (v_proj.shape(1) != k_proj.shape(1)) {
      error = "v_proj output width must match k_proj output width";
      return false;
    }

    int N_q = q_proj.shape(1) / D;
    int N_kv = k_proj.shape(1) / D;
    int attn_width = N_q * D;

    if ((N_q % N_kv) != 0) {
      error = "query heads must be divisible by key/value heads";
      return false;
    }
    if (!check_rank2_positive(o_proj, "o_proj", error) ||
        !check_dim(o_proj, 0, attn_width, "o_proj", "input width", error) ||
        !check_dim(o_proj, 1, H, "o_proj", "output width", error) ||
        !validate_kv_cache_bn(k_cache, v_cache, B, N_kv, offset, T_new, D, error)) {
      return false;
    }
    int valid_len = offset + T_new;

    auto xn = mlx::core::fast::rms_norm(hidden, norm, (float)eps, device);
    auto q_flat = linear_in_out(xn, q_proj, device);
    auto k_flat = linear_in_out(xn, k_proj, device);
    auto v_flat = linear_in_out(xn, v_proj, device);

    auto q = mlx::core::reshape(q_flat, {B, T_new, N_q, D}, device);
    auto k = mlx::core::reshape(k_flat, {B, T_new, N_kv, D}, device);
    auto v = mlx::core::reshape(v_flat, {B, T_new, N_kv, D}, device);

    q = mlx::core::fast::rms_norm(q, q_norm, (float)eps, device);
    k = mlx::core::fast::rms_norm(k, k_norm, (float)eps, device);

    auto q_bn = mlx::core::transpose(q, {0, 2, 1, 3}, device);
    auto k_bn = mlx::core::transpose(k, {0, 2, 1, 3}, device);
    auto v_bn = mlx::core::transpose(v, {0, 2, 1, 3}, device);

    auto q_rope =
        mlx::core::fast::rope(q_bn, D, false, (float)theta, 1.0f, offset, std::nullopt, device);
    auto k_rope =
        mlx::core::fast::rope(k_bn, D, false, (float)theta, 1.0f, offset, std::nullopt, device);

    auto k_cache_owned = std::move(k_cache);
    auto v_cache_owned = std::move(v_cache);

    k_upd = mlx::core::slice_update(k_cache_owned, k_rope, to_shape({0, 0, offset, 0}),
                                     to_shape({B, N_kv, valid_len, D}), device);
    v_upd = mlx::core::slice_update(v_cache_owned, v_bn, to_shape({0, 0, offset, 0}),
                                     to_shape({B, N_kv, valid_len, D}), device);

    auto k_valid = mlx::core::slice(k_upd, to_shape({0, 0, 0, 0}),
                                     to_shape({B, N_kv, valid_len, D}), device);
    auto v_valid = mlx::core::slice(v_upd, to_shape({0, 0, 0, 0}),
                                     to_shape({B, N_kv, valid_len, D}), device);

    auto attn_out_bn = sdpa(q_rope, k_valid, v_valid, (float)scale, T_new, valid_len, device);
    auto attn_out_bthd = mlx::core::transpose(attn_out_bn, {0, 2, 1, 3}, device);
    auto attn_out = mlx::core::reshape(attn_out_bthd, {B, T_new, attn_width}, device);
    auto projected = linear_in_out(attn_out, o_proj, device);
    out = mlx::core::add(hidden, projected, device);
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  } catch (...) {
    error = "Unknown error in qwen3 plugin attention_block";
    return false;
  }
}

bool v_layer_dense(const mlx::core::array &hidden, const LayerParams &layer, KVCache &kv,
                    int offset, double scale, int head_dim, double theta, double eps,
                    const mlx::core::Device &device, mlx::core::array &out,
                    mlx::core::array &k_upd, mlx::core::array &v_upd, std::string &error) {
  try {
    if (!validate_dense_layer(hidden, layer, kv, offset, head_dim, error)) {
      return false;
    }
    out = layer_dense_impl(hidden, layer, kv, offset, (float)scale, head_dim, (float)theta,
                            (float)eps, device, &k_upd, &v_upd);
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  } catch (...) {
    error = "Unknown error in qwen3 plugin layer_dense";
    return false;
  }
}

bool v_layer_quantized(const mlx::core::array &hidden, const LayerParamsQ &layer, KVCache &kv,
                        int offset, double scale, int head_dim, double theta, double eps,
                        const mlx::core::Device &device, mlx::core::array &out,
                        mlx::core::array &k_upd, mlx::core::array &v_upd, std::string &error) {
  try {
    if (!validate_generalized_layer(hidden, layer, kv, offset, head_dim, error)) {
      return false;
    }
    out = layer_core_generalized(hidden, layer, kv, offset, (float)scale, head_dim, (float)theta,
                                  (float)eps, device, &k_upd, &v_upd);
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  } catch (...) {
    error = "Unknown error in qwen3 plugin layer_quantized";
    return false;
  }
}

bool v_final_greedy(const mlx::core::array &hidden, const mlx::core::array &norm,
                     const mlx::core::array &lm_head, double eps, const mlx::core::Device &device,
                     mlx::core::array &out, std::string &error) {
  try {
    if (!check_rank3_positive(hidden, "hidden", error)) {
      return false;
    }
    int B = hidden.shape(0);
    int T = hidden.shape(1);
    int H = hidden.shape(2);
    if (!check_rank1_dim(norm, H, "norm", error) ||
        !check_rank2_positive(lm_head, "lm_head", error) ||
        !check_dim(lm_head, 1, H, "lm_head", "hidden width", error)) {
      return false;
    }

    auto last = (T == 1) ? mlx::core::reshape(hidden, {B, H}, device)
                          : mlx::core::reshape(mlx::core::slice(hidden, to_shape({0, T - 1, 0}),
                                                                 to_shape({B, T, H}), device),
                                                {B, H}, device);

    auto normed = mlx::core::fast::rms_norm(last, norm, (float)eps, device);
    auto logits =
        mlx::core::tensordot(normed, lm_head, std::vector<int>{1}, std::vector<int>{1}, device);
    out = mlx::core::argmax(logits, 1, false, device);
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  } catch (...) {
    error = "Unknown error in qwen3 plugin final_greedy";
    return false;
  }
}

bool v_forward_greedy_from_hidden(const mlx::core::array &hidden, std::vector<LayerParams> &layers,
                                   std::vector<KVCache> &kv, const mlx::core::array &norm,
                                   const mlx::core::array &lm_head, int offset, double scale,
                                   int head_dim, double theta, double eps, bool return_token_id,
                                   const mlx::core::Device &device, mlx::core::array &token_out,
                                   int64_t &token_id_out, std::vector<mlx::core::array> &k_out,
                                   std::vector<mlx::core::array> &v_out, std::string &error) {
  try {
    if (layers.size() != kv.size()) {
      error = "Qwen3 greedy forward layers and kv_cache length mismatch";
      return false;
    }
    if (!check_rank3_positive(hidden, "hidden", error) ||
        !check_non_negative(offset, "offset", error) ||
        !check_positive(head_dim, "head_dim", error)) {
      return false;
    }

    auto current = hidden;
    k_out.clear();
    v_out.clear();
    k_out.reserve(layers.size());
    v_out.reserve(layers.size());
    std::vector<mlx::core::array> eval_arrays;
    eval_arrays.reserve((layers.size() * 2) + 1);

    for (size_t i = 0; i < layers.size(); ++i) {
      std::string layer_error;
      if (!validate_dense_layer(current, layers[i], kv[i], offset, head_dim, layer_error)) {
        error = layer_error;
        return false;
      }

      mlx::core::array k_new = *kv[i].k;
      mlx::core::array v_new = *kv[i].v;
      current = layer_dense_impl(current, layers[i], kv[i], offset, (float)scale, head_dim,
                                  (float)theta, (float)eps, device, &k_new, &v_new);

      eval_arrays.push_back(k_new);
      eval_arrays.push_back(v_new);
      k_out.push_back(k_new);
      v_out.push_back(v_new);
    }

    int B = current.shape(0);
    int T = current.shape(1);
    int H = current.shape(2);
    if (return_token_id && B != 1) {
      error = "token_id return paths require batch size 1";
      return false;
    }
    if (!check_rank1_dim(norm, H, "norm", error) ||
        !check_rank2_positive(lm_head, "lm_head", error) ||
        !check_dim(lm_head, 1, H, "lm_head", "hidden width", error)) {
      return false;
    }

    auto last = (T == 1) ? mlx::core::reshape(current, {B, H}, device)
                          : mlx::core::reshape(mlx::core::slice(current, to_shape({0, T - 1, 0}),
                                                                 to_shape({B, T, H}), device),
                                                {B, H}, device);

    auto normed = mlx::core::fast::rms_norm(last, norm, (float)eps, device);
    auto logits = linear_out_in(normed, lm_head, device);
    auto token = mlx::core::argmax(logits, 1, false, device);

    eval_arrays.push_back(token);
    mlx::core::async_eval(eval_arrays);

    if (return_token_id) {
      token_id_out = token_to_int64(token);
    } else {
      token_out = token;
    }
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  } catch (...) {
    error = "Unknown error in qwen3 plugin forward_greedy_from_hidden";
    return false;
  }
}

bool v_forward_greedy_ids_chunk(const mlx::core::array &input_ids,
                                 const mlx::core::array &embed_tokens,
                                 std::vector<LayerParams> &layers,
                                 std::vector<KVCache> &initial_kv, const mlx::core::array &norm,
                                 const mlx::core::array &lm_head, int offset, int count,
                                 double scale, int head_dim, double theta, double eps,
                                 const mlx::core::Device &device,
                                 std::vector<mlx::core::array> &token_out,
                                 std::vector<mlx::core::array> &k_out,
                                 std::vector<mlx::core::array> &v_out, std::string &error) {
  try {
    if (count <= 0) {
      error = "forward_greedy_ids_chunk expects positive count";
      return false;
    }
    if (layers.size() != initial_kv.size()) {
      error = "forward_greedy_ids_chunk layers and kv_cache length mismatch";
      return false;
    }

    size_t layer_count = layers.size();

    if (!check_rank2_positive(input_ids, "input_ids", error) ||
        !check_rank2_positive(embed_tokens, "embed_tokens", error) ||
        !check_non_negative(offset, "offset", error) ||
        !check_positive(head_dim, "head_dim", error) ||
        !check_rank1_dim(norm, embed_tokens.shape(1), "norm", error) ||
        !check_rank2_positive(lm_head, "lm_head", error) ||
        !check_dim(lm_head, 1, embed_tokens.shape(1), "lm_head", "hidden width", error)) {
      return false;
    }
    if (input_ids.shape(0) != 1) {
      error = "forward_greedy_ids_chunk requires batch size 1";
      return false;
    }
    if (input_ids.shape(1) != 1) {
      error = "forward_greedy_ids_chunk requires sequence length 1";
      return false;
    }

    std::vector<mlx::core::array> k_cache;
    std::vector<mlx::core::array> v_cache;
    std::vector<mlx::core::array> next_k_cache;
    std::vector<mlx::core::array> next_v_cache;
    std::vector<mlx::core::array> token_arrays;
    k_cache.reserve(layer_count);
    v_cache.reserve(layer_count);
    next_k_cache.reserve(layer_count);
    next_v_cache.reserve(layer_count);
    token_arrays.reserve(count);

    auto current_ids = input_ids;
    int current_offset = offset;

    for (int step = 0; step < count; ++step) {
      int B = current_ids.shape(0);
      int T = current_ids.shape(1);

      auto ids = mlx::core::reshape(current_ids, {B * T}, device);
      auto current = mlx::core::reshape(mlx::core::take(embed_tokens, ids, 0, device),
                                         {B, T, embed_tokens.shape(1)}, device);

      next_k_cache.clear();
      next_v_cache.clear();

      for (size_t layer_idx = 0; layer_idx < layer_count; ++layer_idx) {
        KVCache kv = (step == 0) ? initial_kv[layer_idx]
                                  : KVCache{&k_cache[layer_idx], &v_cache[layer_idx]};

        std::string layer_error;
        if (!validate_dense_layer(current, layers[layer_idx], kv, current_offset, head_dim,
                                   layer_error)) {
          error = layer_error;
          return false;
        }

        mlx::core::array k_new = *kv.k;
        mlx::core::array v_new = *kv.v;
        current = layer_dense_impl(current, layers[layer_idx], kv, current_offset, (float)scale,
                                    head_dim, (float)theta, (float)eps, device, &k_new, &v_new);

        next_k_cache.push_back(k_new);
        next_v_cache.push_back(v_new);
      }

      int B_out = current.shape(0);
      int T_out = current.shape(1);
      int H_out = current.shape(2);

      auto last =
          (T_out == 1)
              ? mlx::core::reshape(current, {B_out, H_out}, device)
              : mlx::core::reshape(mlx::core::slice(current, to_shape({0, T_out - 1, 0}),
                                                     to_shape({B_out, T_out, H_out}), device),
                                    {B_out, H_out}, device);

      auto normed = mlx::core::fast::rms_norm(last, norm, (float)eps, device);
      auto logits = linear_out_in(normed, lm_head, device);
      auto token = mlx::core::argmax(logits, 1, false, device);

      token_arrays.push_back(token);
      current_ids = mlx::core::reshape(token, {B_out, 1}, device);
      k_cache.swap(next_k_cache);
      v_cache.swap(next_v_cache);
      current_offset += 1;
    }

    std::vector<mlx::core::array> eval_arrays;
    eval_arrays.reserve(token_arrays.size() + (layer_count * 2));
    eval_arrays.insert(eval_arrays.end(), token_arrays.begin(), token_arrays.end());
    for (size_t layer_idx = 0; layer_idx < layer_count; ++layer_idx) {
      eval_arrays.push_back(k_cache[layer_idx]);
      eval_arrays.push_back(v_cache[layer_idx]);
    }
    mlx::core::async_eval(eval_arrays);

    token_out = std::move(token_arrays);
    k_out = std::move(k_cache);
    v_out = std::move(v_cache);
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  } catch (...) {
    error = "Unknown error in qwen3 plugin forward_greedy_ids_chunk";
    return false;
  }
}

bool v_forward_greedy_ids_chunk_quantized(
    const mlx::core::array &input_ids, const mlx::core::array &embed_tokens,
    std::vector<LayerParamsQ> &layers, std::vector<KVCache> &initial_kv,
    const mlx::core::array &norm, const LinearWeight &lm_head, int offset, int count,
    double scale, int head_dim, double theta, double eps, const mlx::core::Device &device,
    std::vector<mlx::core::array> &token_out, std::vector<mlx::core::array> &k_out,
    std::vector<mlx::core::array> &v_out, std::string &error) {
  try {
    if (count <= 0) {
      error = "forward_greedy_ids_chunk_quantized expects positive count";
      return false;
    }
    if (layers.size() != initial_kv.size()) {
      error = "forward_greedy_ids_chunk_quantized layers and kv_cache length mismatch";
      return false;
    }

    size_t layer_count = layers.size();

    if (!check_rank2_positive(input_ids, "input_ids", error) ||
        !check_rank2_positive(embed_tokens, "embed_tokens", error) ||
        !check_non_negative(offset, "offset", error) ||
        !check_positive(head_dim, "head_dim", error) ||
        !check_rank1_dim(norm, embed_tokens.shape(1), "norm", error) ||
        !check_linear_weight_in(lm_head, embed_tokens.shape(1), "lm_head", error)) {
      return false;
    }
    if (input_ids.shape(0) != 1) {
      error = "forward_greedy_ids_chunk_quantized requires batch size 1";
      return false;
    }
    if (input_ids.shape(1) != 1) {
      error = "forward_greedy_ids_chunk_quantized requires sequence length 1";
      return false;
    }

    std::vector<mlx::core::array> k_cache;
    std::vector<mlx::core::array> v_cache;
    std::vector<mlx::core::array> next_k_cache;
    std::vector<mlx::core::array> next_v_cache;
    std::vector<mlx::core::array> token_arrays;
    k_cache.reserve(layer_count);
    v_cache.reserve(layer_count);
    next_k_cache.reserve(layer_count);
    next_v_cache.reserve(layer_count);
    token_arrays.reserve(count);

    auto current_ids = input_ids;
    int current_offset = offset;

    for (int step = 0; step < count; ++step) {
      int B = current_ids.shape(0);
      int T = current_ids.shape(1);

      auto ids = mlx::core::reshape(current_ids, {B * T}, device);
      auto current = mlx::core::reshape(mlx::core::take(embed_tokens, ids, 0, device),
                                         {B, T, embed_tokens.shape(1)}, device);

      next_k_cache.clear();
      next_v_cache.clear();

      for (size_t layer_idx = 0; layer_idx < layer_count; ++layer_idx) {
        KVCache kv = (step == 0) ? initial_kv[layer_idx]
                                  : KVCache{&k_cache[layer_idx], &v_cache[layer_idx]};

        std::string layer_error;
        if (!validate_generalized_layer(current, layers[layer_idx], kv, current_offset, head_dim,
                                         layer_error)) {
          error = layer_error;
          return false;
        }

        mlx::core::array k_new = *kv.k;
        mlx::core::array v_new = *kv.v;
        current = layer_core_generalized(current, layers[layer_idx], kv, current_offset,
                                          (float)scale, head_dim, (float)theta, (float)eps,
                                          device, &k_new, &v_new);

        next_k_cache.push_back(k_new);
        next_v_cache.push_back(v_new);
      }

      int B_out = current.shape(0);
      int T_out = current.shape(1);
      int H_out = current.shape(2);

      auto last =
          (T_out == 1)
              ? mlx::core::reshape(current, {B_out, H_out}, device)
              : mlx::core::reshape(mlx::core::slice(current, to_shape({0, T_out - 1, 0}),
                                                     to_shape({B_out, T_out, H_out}), device),
                                    {B_out, H_out}, device);

      auto normed = mlx::core::fast::rms_norm(last, norm, (float)eps, device);
      auto logits = apply_linear(normed, lm_head, device);
      auto token = mlx::core::argmax(logits, 1, false, device);

      token_arrays.push_back(token);
      current_ids = mlx::core::reshape(token, {B_out, 1}, device);
      k_cache.swap(next_k_cache);
      v_cache.swap(next_v_cache);
      current_offset += 1;
    }

    std::vector<mlx::core::array> eval_arrays;
    eval_arrays.reserve(token_arrays.size() + (layer_count * 2));
    eval_arrays.insert(eval_arrays.end(), token_arrays.begin(), token_arrays.end());
    for (size_t layer_idx = 0; layer_idx < layer_count; ++layer_idx) {
      eval_arrays.push_back(k_cache[layer_idx]);
      eval_arrays.push_back(v_cache[layer_idx]);
    }
    mlx::core::async_eval(eval_arrays);

    token_out = std::move(token_arrays);
    k_out = std::move(k_cache);
    v_out = std::move(v_cache);
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  } catch (...) {
    error = "Unknown error in qwen3 plugin forward_greedy_ids_chunk_quantized";
    return false;
  }
}

const VTable kVTable = {
    v_kv_cache_attention,
    v_mlp,
    v_attention_residual,
    v_attention_block,
    v_layer_dense,
    v_layer_quantized,
    v_final_greedy,
    v_forward_greedy_from_hidden,
    v_forward_greedy_ids_chunk,
    v_forward_greedy_ids_chunk_quantized,
};

} // namespace

extern "C" const VTable *emlx_qwen3_plugin_vtable() { return &kVTable; }
