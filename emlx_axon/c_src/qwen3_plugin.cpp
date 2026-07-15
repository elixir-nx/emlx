#include "emlx/plugin/abi.hpp"

#include <cstring>
#include <climits>
#include <limits>
#include <optional>
#include <sstream>

namespace emlx_qwen3_plugin {

struct LinearWeight {
  bool quantized = false;
  mlx::core::array *weight = nullptr;
  mlx::core::array *scales = nullptr;
  mlx::core::array *biases = nullptr;
  int group_size = 0;
  int bits = 0;
  std::string mode;
  bool transpose = false;
};

struct LayerParams {
  mlx::core::array *norm1;
  mlx::core::array *norm2;
  mlx::core::array *q_norm;
  mlx::core::array *k_norm;
  mlx::core::array *q_proj;
  mlx::core::array *k_proj;
  mlx::core::array *v_proj;
  mlx::core::array *o_proj;
  mlx::core::array *gate_proj;
  mlx::core::array *up_proj;
  mlx::core::array *down_proj;
};

struct LayerParamsQ {
  mlx::core::array *norm1;
  mlx::core::array *norm2;
  mlx::core::array *q_norm;
  mlx::core::array *k_norm;
  LinearWeight q_proj;
  LinearWeight k_proj;
  LinearWeight v_proj;
  LinearWeight o_proj;
  LinearWeight gate_proj;
  LinearWeight up_proj;
  LinearWeight down_proj;
};

struct KVCache {
  mlx::core::array *k;
  mlx::core::array *v;
};

} // namespace emlx_qwen3_plugin

// Qwen3 compute plugin — pure MLX graph-building code, no Erlang/erl_nif
// dependency whatsoever. Built as its own shared library (libemlx_qwen3.so,
// see this project's Makefile) and `dlopen`'d by emlx's host NIF library
// through
// `EMLX.NIF.load_plugin("qwen3", path)`. The generic plugin ABI is owned by
// EMLX; all Qwen3 model schemas remain private to this translation unit.

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

bool validate_tensor_offset(const mlx::core::array &offset, int capacity,
                            int token_count, std::string &error) {
  if (offset.size() != 1 ||
      (offset.dtype() != mlx::core::int32 && offset.dtype() != mlx::core::int64)) {
    error = "offset must be a scalar int32 or int64 tensor";
    return false;
  }
  if (token_count <= 0 || capacity < token_count || capacity > INT_MAX) {
    error = "cache capacity and token count are outside the supported range";
    return false;
  }
  return true;
}

mlx::core::array clamp_offset(const mlx::core::array &offset, int maximum,
                              const mlx::core::Device &device) {
  auto offset_i32 = mlx::core::astype(offset, mlx::core::int32, device);
  auto zero = mlx::core::array(0, mlx::core::int32);
  auto upper = mlx::core::array(maximum, mlx::core::int32);
  return mlx::core::minimum(mlx::core::maximum(offset_i32, zero, device), upper,
                            device);
}

mlx::core::array rope_with_positions(const mlx::core::array &input,
                                     const mlx::core::array &offset,
                                     int dims, float theta,
                                     const mlx::core::Device &device) {
  const int batch = input.shape(0);
  const int tokens = input.shape(1);
  const int heads = input.shape(2);
  const int half = dims / 2;
  auto positions = mlx::core::add(
      mlx::core::arange(tokens, mlx::core::int32, device), offset, device);
  positions = mlx::core::reshape(positions, {1, tokens, 1}, device);
  if (batch != 1)
    positions = mlx::core::broadcast_to(positions, {batch, tokens, 1}, device);
  auto frequency_index = mlx::core::arange(0, dims, 2, mlx::core::float32, device);
  auto exponent = mlx::core::divide(
      frequency_index, mlx::core::array(static_cast<float>(dims)), device);
  auto inverse_frequency = mlx::core::exp(
      mlx::core::multiply(
          exponent, mlx::core::array(-std::log(theta), mlx::core::float32),
          device),
      device);
  auto angles = mlx::core::multiply(
      mlx::core::astype(positions, mlx::core::float32, device),
      mlx::core::reshape(inverse_frequency, {1, 1, half}, device), device);
  auto cosine = mlx::core::astype(
      mlx::core::reshape(mlx::core::cos(angles, device), {batch, tokens, 1, half},
                         device),
      input.dtype(), device);
  auto sine = mlx::core::astype(
      mlx::core::reshape(mlx::core::sin(angles, device), {batch, tokens, 1, half},
                         device),
      input.dtype(), device);
  auto cosine_full =
      mlx::core::concatenate(std::vector<mlx::core::array>{cosine, cosine}, 3,
                             device);
  auto sine_full =
      mlx::core::concatenate(std::vector<mlx::core::array>{sine, sine}, 3,
                             device);
  auto first = mlx::core::slice(input, {0, 0, 0, 0},
                                {batch, tokens, heads, half}, device);
  auto second = mlx::core::slice(input, {0, 0, 0, half},
                                 {batch, tokens, heads, dims}, device);
  auto rotated = mlx::core::concatenate(
      std::vector<mlx::core::array>{mlx::core::negative(second, device), first},
      3, device);
  return mlx::core::add(mlx::core::multiply(input, cosine_full, device),
                        mlx::core::multiply(rotated, sine_full, device), device);
}

bool tensor_offset_attention(
    const mlx::core::array &query, const mlx::core::array &key,
    const mlx::core::array &value, const mlx::core::array &k_cache,
    const mlx::core::array &v_cache, const mlx::core::array &offset,
    float scale, int head_dim, float theta, const mlx::core::Device &device,
    mlx::core::array &attention, mlx::core::array &k_updated,
    mlx::core::array &v_updated, std::string &error) {
  if (!validate_qkv_cache_attention(query, key, value, k_cache, v_cache, 0,
                                    head_dim, error))
    return false;
  const int batch = query.shape(0);
  const int tokens = query.shape(1);
  const int query_heads = query.shape(2);
  const int kv_heads = key.shape(2);
  const int width = query.shape(3);
  const int capacity = k_cache.shape(2);
  if (!validate_tensor_offset(offset, capacity, tokens, error))
    return false;

  auto safe_offset = clamp_offset(offset, capacity - tokens, device);
  auto query_rope = rope_with_positions(query, safe_offset, head_dim, theta, device);
  auto key_rope = rope_with_positions(key, safe_offset, head_dim, theta, device);
  auto query_bn = mlx::core::transpose(query_rope, {0, 2, 1, 3}, device);
  auto key_bn = mlx::core::transpose(key_rope, {0, 2, 1, 3}, device);
  auto value_bn = mlx::core::transpose(value, {0, 2, 1, 3}, device);
  auto start = mlx::core::reshape(safe_offset, {1}, device);
  k_updated = mlx::core::slice_update(k_cache, key_bn, start, {2}, device);
  v_updated = mlx::core::slice_update(v_cache, value_bn, start, {2}, device);

  auto row = mlx::core::reshape(
      mlx::core::add(mlx::core::arange(tokens, mlx::core::int32, device),
                     safe_offset, device),
      {1, 1, tokens, 1}, device);
  auto column = mlx::core::reshape(
      mlx::core::arange(capacity, mlx::core::int32, device),
      {1, 1, 1, capacity}, device);
  auto visible = mlx::core::less_equal(column, row, device);
  auto mask = mlx::core::where(
      visible, mlx::core::zeros({}, query.dtype(), device),
      mlx::core::full({}, -std::numeric_limits<float>::infinity(), query.dtype(),
                      device),
      device);
  auto attended = mlx::core::fast::scaled_dot_product_attention(
      query_bn, k_updated, v_updated, scale, "array", mask, std::nullopt,
      device);
  attention = mlx::core::reshape(
      mlx::core::transpose(attended, {0, 2, 1, 3}, device),
      {batch, tokens, query_heads * width}, device);
  (void)kv_heads;
  return true;
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

    if (return_token_id) {
      eval_arrays.push_back(token);
      mlx::core::async_eval(eval_arrays);
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
                                 bool submit_each_step,
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

      if (submit_each_step) {
        std::vector<mlx::core::array> eval_arrays;
        eval_arrays.reserve(1 + (layer_count * 2));
        eval_arrays.push_back(token);
        for (size_t layer_idx = 0; layer_idx < layer_count; ++layer_idx) {
          eval_arrays.push_back(next_k_cache[layer_idx]);
          eval_arrays.push_back(next_v_cache[layer_idx]);
        }
        mlx::core::async_eval(eval_arrays);
      }

      token_arrays.push_back(token);
      current_ids = mlx::core::reshape(token, {B_out, 1}, device);
      k_cache.swap(next_k_cache);
      v_cache.swap(next_v_cache);
      current_offset += 1;
    }

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

inline constexpr char kPluginName[] = "qwen3";
inline constexpr char kMLPName[] = "mlp";
inline constexpr char kKVCacheAttentionName[] = "kv_cache_attention";
inline constexpr char kKVCacheAttentionTensorName[] =
    "kv_cache_attention_tensor_offset";
inline constexpr char kAttentionResidualName[] = "attention_residual";
inline constexpr char kAttentionBlockName[] = "attention_block";
inline constexpr char kLayerDenseName[] = "layer_dense";
inline constexpr char kLayerGeneralizedName[] = "layer_generalized";
inline constexpr char kFinalGreedyName[] = "final_greedy";
inline constexpr char kForwardDenseName[] = "forward_greedy_dense";
inline constexpr char kChunkDenseName[] = "forward_greedy_chunk_dense";
inline constexpr char kChunkGeneralizedName[] =
    "forward_greedy_chunk_generalized";
inline constexpr emlx::plugin::device_type_t kSupportedDeviceTypes[] = {
    mlx::core::Device::DeviceType::cpu,
    mlx::core::Device::DeviceType::gpu};
inline constexpr emlx::plugin::device_view_t kSupportedDevices{
    kSupportedDeviceTypes,
    sizeof(kSupportedDeviceTypes) / sizeof(kSupportedDeviceTypes[0])};
inline constexpr int64_t kMaxLayerCount = 256;
inline constexpr int64_t kMaxChunkTokenCount = 4096;
inline constexpr uint32_t kMaxDenseChunkOperands =
    4U + static_cast<uint32_t>(kMaxLayerCount) * 13U;
inline constexpr uint32_t kMaxGeneralizedChunkOperands =
    2U + static_cast<uint32_t>(kMaxLayerCount) * (6U + 7U * 3U) + 1U + 3U;
inline constexpr uint32_t kMaxChunkOutputs =
    1U + static_cast<uint32_t>(kMaxLayerCount) * 2U;
static_assert(kMaxDenseChunkOperands <= emlx::plugin::operand_count_max_v1);
static_assert(kMaxGeneralizedChunkOperands <=
              emlx::plugin::operand_count_max_v1);
static_assert(kMaxChunkOutputs <= emlx::plugin::output_count_max_v1);

template <size_t N>
constexpr emlx::plugin::string_view_t string_view(const char (&value)[N]) {
  return {value, N - 1};
}

double f64_from_bits(int64_t bits) {
  uint64_t raw = static_cast<uint64_t>(bits);
  double value;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
}

bool int32_attr(const emlx::plugin::call_t &call, size_t index, const char *name,
                int &value, std::string &error) {
  if (index >= call.attrs.size || call.attrs.data[index] < INT_MIN ||
      call.attrs.data[index] > INT_MAX) {
    error = std::string(name) + " is outside the int32 range";
    return false;
  }
  value = static_cast<int>(call.attrs.data[index]);
  return true;
}

constexpr size_t kLinearDescriptorWidth = 8;

bool quantization_mode(int64_t value, std::string &mode,
                       std::string &error) {
  switch (value) {
  case 0:
    mode = "affine";
    return true;
  case 1:
    mode = "mxfp4";
    return true;
  case 2:
    mode = "mxfp8";
    return true;
  case 3:
    mode = "nvfp4";
    return true;
  default:
    error = "Qwen3 linear descriptor has an unknown quantization mode";
    return false;
  }
}

bool validate_linear_descriptor(emlx::plugin::int64_view_t attrs, size_t &attr_index,
                                uint32_t &operand_index,
                                std::string &error) {
  if (attr_index > attrs.size ||
      attrs.size - attr_index < kLinearDescriptorWidth) {
    error = "Qwen3 linear descriptor is truncated";
    return false;
  }
  const int64_t kind = attrs.data[attr_index];
  const int64_t weight_index = attrs.data[attr_index + 1];
  const int64_t scales_index = attrs.data[attr_index + 2];
  const int64_t biases_index = attrs.data[attr_index + 3];
  const int64_t group_size = attrs.data[attr_index + 4];
  const int64_t bits = attrs.data[attr_index + 5];
  const int64_t mode = attrs.data[attr_index + 6];
  const int64_t transpose = attrs.data[attr_index + 7];
  attr_index += kLinearDescriptorWidth;

  if (weight_index != operand_index || (transpose != 0 && transpose != 1)) {
    error = "Qwen3 linear descriptor has noncanonical operand indexes or transpose";
    return false;
  }
  ++operand_index;

  if (kind == 0) {
    if (scales_index != -1 || biases_index != -1 || group_size != 0 ||
        bits != 0 || mode != 0) {
      error = "Qwen3 dense linear descriptor has quantized fields";
      return false;
    }
    return true;
  }
  if (kind != 1 || scales_index != operand_index || group_size <= 0 ||
      group_size > INT_MAX || bits <= 0 || bits > 32 || 32 % bits != 0 ||
      mode < 0 || mode > 3) {
    error = "Qwen3 quantized linear descriptor is invalid";
    return false;
  }
  ++operand_index;
  if (biases_index == operand_index) {
    ++operand_index;
  } else if (biases_index != -1) {
    error = "Qwen3 quantized linear descriptor has a noncanonical bias index";
    return false;
  }
  return true;
}

bool parse_linear_descriptor(const emlx::plugin::call_t &call, size_t &attr_index,
                             uint32_t &operand_index, LinearWeight &weight,
                             std::string &error) {
  const size_t descriptor_start = attr_index;
  uint32_t next_operand = operand_index;
  if (!validate_linear_descriptor(call.attrs, attr_index, next_operand, error))
    return false;

  const int64_t kind = call.attrs.data[descriptor_start];
  const int64_t weight_index = call.attrs.data[descriptor_start + 1];
  const int64_t scales_index = call.attrs.data[descriptor_start + 2];
  const int64_t biases_index = call.attrs.data[descriptor_start + 3];
  if (next_operand > call.operands.size) {
    error = "Qwen3 linear descriptor references a missing operand";
    return false;
  }

  weight.quantized = kind == 1;
  weight.weight = const_cast<mlx::core::array *>(&call.operands.data[weight_index]);
  weight.scales = kind == 1
                      ? const_cast<mlx::core::array *>(
                            &call.operands.data[scales_index])
                      : nullptr;
  weight.biases = biases_index >= 0
                      ? const_cast<mlx::core::array *>(
                            &call.operands.data[biases_index])
                      : nullptr;
  weight.group_size = static_cast<int>(call.attrs.data[descriptor_start + 4]);
  weight.bits = static_cast<int>(call.attrs.data[descriptor_start + 5]);
  if (!quantization_mode(call.attrs.data[descriptor_start + 6], weight.mode,
                         error))
    return false;
  weight.transpose = call.attrs.data[descriptor_start + 7] == 1;
  operand_index = next_operand;
  return true;
}

bool generalized_layer_operand_count(emlx::plugin::int64_view_t attrs,
                                     uint32_t &count,
                                     std::string &error) {
  if (attrs.size != 7 + 7 * kLinearDescriptorWidth || attrs.data[0] != 1 ||
      attrs.data[6] != 7) {
    error = "generalized layer attributes have an invalid schema";
    return false;
  }
  size_t attr_index = 7;
  uint32_t operand_index = 7;
  for (size_t index = 0; index < 7; ++index) {
    if (!validate_linear_descriptor(attrs, attr_index, operand_index, error))
      return false;
  }
  if (attr_index != attrs.size) {
    error = "generalized layer attributes have trailing fields";
    return false;
  }
  count = operand_index;
  return true;
}

bool generalized_chunk_operand_count(emlx::plugin::int64_view_t attrs,
                                     uint32_t &count,
                                     std::string &error) {
  if (attrs.size < 9 || attrs.data[0] != 1 || attrs.data[1] <= 0 ||
      attrs.data[1] > kMaxLayerCount || attrs.data[8] != attrs.data[1] * 7 + 1 ||
      attrs.data[8] > 1793 ||
      attrs.size != 9 + static_cast<uint64_t>(attrs.data[8]) *
                            kLinearDescriptorWidth) {
    error = "generalized chunk attributes have an invalid schema";
    return false;
  }
  size_t attr_index = 9;
  uint32_t operand_index = 2;
  for (int64_t layer = 0; layer < attrs.data[1]; ++layer) {
    if (operand_index > UINT32_MAX - 6) {
      error = "generalized chunk operand count overflows";
      return false;
    }
    operand_index += 6;
    for (size_t projection = 0; projection < 7; ++projection) {
      if (!validate_linear_descriptor(attrs, attr_index, operand_index, error))
        return false;
    }
  }
  ++operand_index;
  if (!validate_linear_descriptor(attrs, attr_index, operand_index, error) ||
      attr_index != attrs.size)
    return false;
  count = operand_index;
  return true;
}

bool generalized_chunk_output_count(emlx::plugin::int64_view_t attrs,
                                    uint32_t &count,
                                    std::string &error) {
  uint32_t ignored_operands = 0;
  if (!generalized_chunk_operand_count(attrs, ignored_operands, error) ||
      attrs.data[3] <= 0 || attrs.data[3] > kMaxChunkTokenCount) {
    if (error.empty())
      error = "generalized chunk has an invalid token count";
    return false;
  }
  count = 1U + static_cast<uint32_t>(attrs.data[1]) * 2U;
  return true;
}

std::optional<std::string>
plugin_mlp(const emlx::plugin::call_t &call,
           std::vector<mlx::core::array> &outputs) {
  std::string error;
  if (call.operands.size != 5 || call.attrs.size != 1 || !call.execution ||
      !call.execution->device || !call.execution->stream) {
    error = "qwen3/mlp expects five operands, epsilon, and an execution context";
    return error;
  }
  mlx::core::array output = call.operands.data[0];
  if (!v_mlp(call.operands.data[0], call.operands.data[1],
             call.operands.data[2], call.operands.data[3],
             call.operands.data[4], f64_from_bits(call.attrs.data[0]),
             *call.execution->device, output, error))
    return error;
  outputs.push_back(std::move(output));
  return std::nullopt;
}

std::optional<std::string>
plugin_kv_cache_attention(const emlx::plugin::call_t &call,
                          std::vector<mlx::core::array> &outputs) {
  std::string error;
  if (call.operands.size != 5 || call.attrs.size != 4 || !call.execution ||
      !call.execution->device) {
    error = "qwen3/kv_cache_attention has an invalid call contract";
    return error;
  }
  auto k_cache = call.operands.data[3];
  auto v_cache = call.operands.data[4];
  auto output = call.operands.data[0];
  auto k_updated = k_cache;
  auto v_updated = v_cache;
  int offset = 0;
  int head_dim = 0;
  if (!int32_attr(call, 0, "offset", offset, error) ||
      !int32_attr(call, 2, "head_dim", head_dim, error))
    return error;
  if (!v_kv_cache_attention(
          call.operands.data[0], call.operands.data[1], call.operands.data[2],
          k_cache, v_cache, offset, f64_from_bits(call.attrs.data[1]), head_dim,
          f64_from_bits(call.attrs.data[3]), *call.execution->device, output,
          k_updated, v_updated, error))
    return error;
  outputs.push_back(std::move(output));
  outputs.push_back(std::move(k_updated));
  outputs.push_back(std::move(v_updated));
  return std::nullopt;
}

std::optional<std::string>
plugin_kv_cache_attention_tensor(const emlx::plugin::call_t &call,
                                 std::vector<mlx::core::array> &outputs) {
  std::string error;
  if (call.operands.size != 6 || call.attrs.size != 3 || !call.execution ||
      !call.execution->device) {
    error = "qwen3/kv_cache_attention_tensor_offset has an invalid call contract";
    return error;
  }
  int head_dim = 0;
  if (!int32_attr(call, 1, "head_dim", head_dim, error))
    return error;
  auto attention = call.operands.data[0];
  auto k_updated = call.operands.data[3];
  auto v_updated = call.operands.data[4];
  if (!tensor_offset_attention(
          call.operands.data[0], call.operands.data[1], call.operands.data[2],
          call.operands.data[3], call.operands.data[4], call.operands.data[5],
          f64_from_bits(call.attrs.data[0]), head_dim,
          f64_from_bits(call.attrs.data[2]), *call.execution->device, attention,
          k_updated, v_updated, error))
    return error;
  outputs.push_back(std::move(attention));
  outputs.push_back(std::move(k_updated));
  outputs.push_back(std::move(v_updated));
  return std::nullopt;
}

std::optional<std::string>
plugin_attention_residual(const emlx::plugin::call_t &call,
                          std::vector<mlx::core::array> &outputs) {
  std::string error;
  if (call.operands.size != 3 || call.attrs.size != 0 || !call.execution ||
      !call.execution->device) {
    error = "qwen3/attention_residual has an invalid call contract";
    return error;
  }
  auto output = call.operands.data[0];
  if (!v_attention_residual(call.operands.data[0], call.operands.data[1],
                            call.operands.data[2], *call.execution->device,
                            output, error))
    return error;
  outputs.push_back(std::move(output));
  return std::nullopt;
}

std::optional<std::string>
plugin_attention_block(const emlx::plugin::call_t &call,
                       std::vector<mlx::core::array> &outputs) {
  std::string error;
  if (call.operands.size != 10 || call.attrs.size != 5 || !call.execution ||
      !call.execution->device) {
    error = "qwen3/attention_block has an invalid call contract";
    return error;
  }
  auto k_cache = call.operands.data[8];
  auto v_cache = call.operands.data[9];
  auto output = call.operands.data[0];
  auto k_updated = k_cache;
  auto v_updated = v_cache;
  int offset = 0;
  int head_dim = 0;
  if (!int32_attr(call, 0, "offset", offset, error) ||
      !int32_attr(call, 2, "head_dim", head_dim, error))
    return error;
  if (!v_attention_block(
          call.operands.data[0], call.operands.data[1], call.operands.data[2],
          call.operands.data[3], call.operands.data[4], call.operands.data[5],
          call.operands.data[6], call.operands.data[7], k_cache, v_cache,
          offset, f64_from_bits(call.attrs.data[1]), head_dim,
          f64_from_bits(call.attrs.data[3]),
          f64_from_bits(call.attrs.data[4]), *call.execution->device, output,
          k_updated, v_updated, error))
    return error;
  outputs.push_back(std::move(output));
  outputs.push_back(std::move(k_updated));
  outputs.push_back(std::move(v_updated));
  return std::nullopt;
}

std::optional<std::string>
plugin_layer_dense(const emlx::plugin::call_t &call,
                   std::vector<mlx::core::array> &outputs) {
  std::string error;
  if (call.operands.size != 14 || call.attrs.size != 5 || !call.execution ||
      !call.execution->device) {
    error = "qwen3/layer_dense has an invalid call contract";
    return error;
  }
  std::vector<mlx::core::array> operands(call.operands.data,
                                         call.operands.data + call.operands.size);
  LayerParams layer{&operands[1],  &operands[10], &operands[6],
                    &operands[7],  &operands[2],  &operands[3],
                    &operands[4],  &operands[5],  &operands[11],
                    &operands[12], &operands[13]};
  KVCache cache{&operands[8], &operands[9]};
  auto output = operands[0];
  auto k_updated = operands[8];
  auto v_updated = operands[9];
  int offset = 0;
  int head_dim = 0;
  if (!int32_attr(call, 0, "offset", offset, error) ||
      !int32_attr(call, 2, "head_dim", head_dim, error))
    return error;
  if (!v_layer_dense(
          operands[0], layer, cache, offset, f64_from_bits(call.attrs.data[1]),
          head_dim, f64_from_bits(call.attrs.data[3]),
          f64_from_bits(call.attrs.data[4]), *call.execution->device, output,
          k_updated, v_updated, error))
    return error;
  outputs.push_back(std::move(output));
  outputs.push_back(std::move(k_updated));
  outputs.push_back(std::move(v_updated));
  return std::nullopt;
}

std::optional<std::string>
plugin_layer_generalized(const emlx::plugin::call_t &call,
                         std::vector<mlx::core::array> &outputs) {
  std::string error;
  uint32_t expected_operands = 0;
  if (!generalized_layer_operand_count(call.attrs, expected_operands, error) ||
      call.operands.size != expected_operands || !call.execution ||
      !call.execution->device) {
    if (error.empty())
      error = "qwen3/layer_generalized has an invalid call contract";
    return error;
  }
  int offset = 0;
  int head_dim = 0;
  if (!int32_attr(call, 1, "offset", offset, error) ||
      !int32_attr(call, 3, "head_dim", head_dim, error))
    return error;

  LayerParamsQ layer;
  layer.norm1 = const_cast<mlx::core::array *>(&call.operands.data[1]);
  layer.q_norm = const_cast<mlx::core::array *>(&call.operands.data[2]);
  layer.k_norm = const_cast<mlx::core::array *>(&call.operands.data[3]);
  layer.norm2 = const_cast<mlx::core::array *>(&call.operands.data[6]);
  size_t attr_index = 7;
  uint32_t operand_index = 7;
  if (!parse_linear_descriptor(call, attr_index, operand_index, layer.q_proj,
                               error) ||
      !parse_linear_descriptor(call, attr_index, operand_index, layer.k_proj,
                               error) ||
      !parse_linear_descriptor(call, attr_index, operand_index, layer.v_proj,
                               error) ||
      !parse_linear_descriptor(call, attr_index, operand_index, layer.o_proj,
                               error) ||
      !parse_linear_descriptor(call, attr_index, operand_index,
                               layer.gate_proj, error) ||
      !parse_linear_descriptor(call, attr_index, operand_index, layer.up_proj,
                               error) ||
      !parse_linear_descriptor(call, attr_index, operand_index,
                               layer.down_proj, error))
    return error;

  KVCache cache{const_cast<mlx::core::array *>(&call.operands.data[4]),
                const_cast<mlx::core::array *>(&call.operands.data[5])};
  auto output = call.operands.data[0];
  auto k_updated = call.operands.data[4];
  auto v_updated = call.operands.data[5];
  if (!v_layer_quantized(
          call.operands.data[0], layer, cache, offset,
          f64_from_bits(call.attrs.data[2]), head_dim,
          f64_from_bits(call.attrs.data[4]),
          f64_from_bits(call.attrs.data[5]), *call.execution->device, output,
          k_updated, v_updated, error))
    return error;
  outputs.push_back(std::move(output));
  outputs.push_back(std::move(k_updated));
  outputs.push_back(std::move(v_updated));
  return std::nullopt;
}

std::optional<std::string>
plugin_final_greedy(const emlx::plugin::call_t &call,
                    std::vector<mlx::core::array> &outputs) {
  std::string error;
  if (call.operands.size != 3 || call.attrs.size != 1 || !call.execution ||
      !call.execution->device) {
    error = "qwen3/final_greedy has an invalid call contract";
    return error;
  }
  auto output = call.operands.data[0];
  if (!v_final_greedy(call.operands.data[0], call.operands.data[1],
                      call.operands.data[2], f64_from_bits(call.attrs.data[0]),
                      *call.execution->device, output, error))
    return error;
  outputs.push_back(std::move(output));
  return std::nullopt;
}

bool dense_forward_operand_count(emlx::plugin::int64_view_t attrs, uint32_t &count,
                                 std::string &error) {
  if (attrs.size != 6 || attrs.data[0] <= 0 || attrs.data[0] > kMaxLayerCount) {
    error = "dense forward attributes have an invalid layer count";
    return false;
  }
  count = 3U + static_cast<uint32_t>(attrs.data[0]) * 13U;
  return true;
}

bool dense_forward_output_count(emlx::plugin::int64_view_t attrs, uint32_t &count,
                                std::string &error) {
  if (attrs.size != 6 || attrs.data[0] <= 0 || attrs.data[0] > kMaxLayerCount) {
    error = "dense forward attributes have an invalid layer count";
    return false;
  }
  count = 1U + static_cast<uint32_t>(attrs.data[0]) * 2U;
  return true;
}

bool dense_chunk_operand_count(emlx::plugin::int64_view_t attrs, uint32_t &count,
                               std::string &error) {
  if (attrs.size != 8 || attrs.data[0] <= 0 || attrs.data[0] > kMaxLayerCount ||
      (attrs.data[7] != 0 && attrs.data[7] != 1)) {
    error = "dense chunk attributes have an invalid schema";
    return false;
  }
  count = 4U + static_cast<uint32_t>(attrs.data[0]) * 13U;
  return true;
}

bool dense_chunk_output_count(emlx::plugin::int64_view_t attrs, uint32_t &count,
                              std::string &error) {
  if (attrs.size != 8 || attrs.data[0] <= 0 || attrs.data[0] > kMaxLayerCount ||
      attrs.data[2] <= 0 || attrs.data[2] > kMaxChunkTokenCount ||
      (attrs.data[7] != 0 && attrs.data[7] != 1)) {
    error = "dense chunk attributes have an invalid schema";
    return false;
  }
  count = 1U + static_cast<uint32_t>(attrs.data[0]) * 2U;
  return true;
}

void parse_dense_layers(std::vector<mlx::core::array> &operands, size_t start,
                        size_t layer_count, std::vector<LayerParams> &layers,
                        std::vector<KVCache> &caches) {
  layers.reserve(layer_count);
  caches.reserve(layer_count);
  for (size_t layer_index = 0; layer_index < layer_count; ++layer_index) {
    const size_t base = start + layer_index * 13;
    layers.push_back({&operands[base],     &operands[base + 1],
                      &operands[base + 2], &operands[base + 3],
                      &operands[base + 4], &operands[base + 5],
                      &operands[base + 6], &operands[base + 7],
                      &operands[base + 8], &operands[base + 9],
                      &operands[base + 10]});
    caches.push_back({&operands[base + 11], &operands[base + 12]});
  }
}

std::optional<std::string>
plugin_forward_dense(const emlx::plugin::call_t &call,
                     std::vector<mlx::core::array> &outputs) {
  std::string error;
  uint32_t expected_operands = 0;
  uint32_t expected_outputs = 0;
  if (!dense_forward_operand_count(call.attrs, expected_operands, error) ||
      !dense_forward_output_count(call.attrs, expected_outputs, error) ||
      call.operands.size != expected_operands || !call.execution ||
      !call.execution->device)
    return error;
  int offset = 0;
  int head_dim = 0;
  if (!int32_attr(call, 1, "offset", offset, error) ||
      !int32_attr(call, 3, "head_dim", head_dim, error))
    return error;
  const size_t layer_count = static_cast<size_t>(call.attrs.data[0]);
  std::vector<mlx::core::array> operands(call.operands.data,
                                         call.operands.data + call.operands.size);
  std::vector<LayerParams> layers;
  std::vector<KVCache> caches;
  parse_dense_layers(operands, 1, layer_count, layers, caches);
  const size_t tail = 1 + layer_count * 13;
  mlx::core::array token(0);
  int64_t ignored_token_id = 0;
  std::vector<mlx::core::array> keys;
  std::vector<mlx::core::array> values;
  if (!v_forward_greedy_from_hidden(
          operands[0], layers, caches, operands[tail], operands[tail + 1],
          offset, f64_from_bits(call.attrs.data[2]), head_dim,
          f64_from_bits(call.attrs.data[4]), f64_from_bits(call.attrs.data[5]),
          false, *call.execution->device, token, ignored_token_id, keys, values,
          error))
    return error;
  outputs.reserve(expected_outputs);
  outputs.push_back(std::move(token));
  for (size_t index = 0; index < keys.size(); ++index) {
    outputs.push_back(std::move(keys[index]));
    outputs.push_back(std::move(values[index]));
  }
  return std::nullopt;
}

std::optional<std::string>
plugin_chunk_dense(const emlx::plugin::call_t &call,
                   std::vector<mlx::core::array> &outputs) {
  std::string error;
  uint32_t expected_operands = 0;
  uint32_t expected_outputs = 0;
  if (!dense_chunk_operand_count(call.attrs, expected_operands, error) ||
      !dense_chunk_output_count(call.attrs, expected_outputs, error) ||
      call.operands.size != expected_operands || !call.execution ||
      !call.execution->device)
    return error;
  int offset = 0;
  int count = 0;
  int head_dim = 0;
  if (!int32_attr(call, 1, "offset", offset, error) ||
      !int32_attr(call, 2, "count", count, error) ||
      !int32_attr(call, 4, "head_dim", head_dim, error))
    return error;
  const size_t layer_count = static_cast<size_t>(call.attrs.data[0]);
  std::vector<mlx::core::array> operands(call.operands.data,
                                         call.operands.data + call.operands.size);
  std::vector<LayerParams> layers;
  std::vector<KVCache> caches;
  parse_dense_layers(operands, 2, layer_count, layers, caches);
  const size_t tail = 2 + layer_count * 13;
  std::vector<mlx::core::array> tokens;
  std::vector<mlx::core::array> keys;
  std::vector<mlx::core::array> values;
  if (!v_forward_greedy_ids_chunk(
          operands[0], operands[1], layers, caches, operands[tail],
          operands[tail + 1], offset, count, f64_from_bits(call.attrs.data[3]),
          head_dim, f64_from_bits(call.attrs.data[5]),
          f64_from_bits(call.attrs.data[6]), call.attrs.data[7] == 1,
          *call.execution->device, tokens, keys, values, error))
    return error;
  outputs.reserve(expected_outputs);
  outputs.push_back(mlx::core::reshape(
      mlx::core::stack(tokens, 0, *call.execution->device), {count},
      *call.execution->device));
  for (size_t index = 0; index < keys.size(); ++index) {
    outputs.push_back(std::move(keys[index]));
    outputs.push_back(std::move(values[index]));
  }
  return std::nullopt;
}

std::optional<std::string>
plugin_chunk_generalized(const emlx::plugin::call_t &call,
                         std::vector<mlx::core::array> &outputs) {
  std::string error;
  uint32_t expected_operands = 0;
  uint32_t expected_outputs = 0;
  if (!generalized_chunk_operand_count(call.attrs, expected_operands, error) ||
      !generalized_chunk_output_count(call.attrs, expected_outputs, error) ||
      call.operands.size != expected_operands || !call.execution ||
      !call.execution->device) {
    if (error.empty())
      error = "qwen3/forward_greedy_chunk_generalized has an invalid call contract";
    return error;
  }
  int offset = 0;
  int count = 0;
  int head_dim = 0;
  if (!int32_attr(call, 2, "offset", offset, error) ||
      !int32_attr(call, 3, "count", count, error) ||
      !int32_attr(call, 5, "head_dim", head_dim, error))
    return error;

  const size_t layer_count = static_cast<size_t>(call.attrs.data[1]);
  std::vector<LayerParamsQ> layers;
  std::vector<KVCache> caches;
  layers.reserve(layer_count);
  caches.reserve(layer_count);
  size_t attr_index = 9;
  uint32_t operand_index = 2;
  for (size_t layer_index = 0; layer_index < layer_count; ++layer_index) {
    LayerParamsQ layer;
    layer.norm1 = const_cast<mlx::core::array *>(
        &call.operands.data[operand_index++]);
    layer.norm2 = const_cast<mlx::core::array *>(
        &call.operands.data[operand_index++]);
    layer.q_norm = const_cast<mlx::core::array *>(
        &call.operands.data[operand_index++]);
    layer.k_norm = const_cast<mlx::core::array *>(
        &call.operands.data[operand_index++]);
    auto *k_cache = const_cast<mlx::core::array *>(
        &call.operands.data[operand_index++]);
    auto *v_cache = const_cast<mlx::core::array *>(
        &call.operands.data[operand_index++]);
    if (!parse_linear_descriptor(call, attr_index, operand_index,
                                 layer.q_proj, error) ||
        !parse_linear_descriptor(call, attr_index, operand_index,
                                 layer.k_proj, error) ||
        !parse_linear_descriptor(call, attr_index, operand_index,
                                 layer.v_proj, error) ||
        !parse_linear_descriptor(call, attr_index, operand_index,
                                 layer.o_proj, error) ||
        !parse_linear_descriptor(call, attr_index, operand_index,
                                 layer.gate_proj, error) ||
        !parse_linear_descriptor(call, attr_index, operand_index,
                                 layer.up_proj, error) ||
        !parse_linear_descriptor(call, attr_index, operand_index,
                                 layer.down_proj, error))
      return error;
    layers.push_back(std::move(layer));
    caches.push_back({k_cache, v_cache});
  }
  auto &norm = call.operands.data[operand_index++];
  LinearWeight lm_head;
  if (!parse_linear_descriptor(call, attr_index, operand_index, lm_head,
                               error))
    return error;

  std::vector<mlx::core::array> tokens;
  std::vector<mlx::core::array> keys;
  std::vector<mlx::core::array> values;
  if (!v_forward_greedy_ids_chunk_quantized(
          call.operands.data[0], call.operands.data[1], layers, caches, norm,
          lm_head, offset, count, f64_from_bits(call.attrs.data[4]), head_dim,
          f64_from_bits(call.attrs.data[6]),
          f64_from_bits(call.attrs.data[7]), *call.execution->device, tokens,
          keys, values, error))
    return error;
  outputs.reserve(expected_outputs);
  outputs.push_back(mlx::core::reshape(
      mlx::core::stack(tokens, 0, *call.execution->device), {count},
      *call.execution->device));
  for (size_t index = 0; index < keys.size(); ++index) {
    outputs.push_back(std::move(keys[index]));
    outputs.push_back(std::move(values[index]));
  }
  return std::nullopt;
}

constinit const emlx::plugin::callback_descriptor_t kCallbacks[] = {
    {string_view(kMLPName), 1, 1, 5, nullptr, 1, nullptr,
     kSupportedDevices, plugin_mlp},
    {string_view(kKVCacheAttentionName), 1, 1, 5, nullptr, 3, nullptr,
     kSupportedDevices, plugin_kv_cache_attention},
    {string_view(kKVCacheAttentionTensorName), 1, 1, 6, nullptr, 3, nullptr,
     kSupportedDevices, plugin_kv_cache_attention_tensor},
    {string_view(kAttentionResidualName), 1, 1, 3, nullptr, 1, nullptr,
     kSupportedDevices, plugin_attention_residual},
    {string_view(kAttentionBlockName), 1, 1, 10, nullptr, 3, nullptr,
     kSupportedDevices, plugin_attention_block},
    {string_view(kLayerDenseName), 1, 1, 14, nullptr, 3, nullptr,
     kSupportedDevices, plugin_layer_dense},
    {string_view(kLayerGeneralizedName), 1, 1, 0,
     generalized_layer_operand_count, 3, nullptr,
     kSupportedDevices, plugin_layer_generalized},
    {string_view(kFinalGreedyName), 1, 1, 3, nullptr, 1, nullptr,
     kSupportedDevices, plugin_final_greedy},
    {string_view(kForwardDenseName), 1, 1, 0, dense_forward_operand_count, 0,
     dense_forward_output_count, kSupportedDevices,
     plugin_forward_dense},
    {string_view(kChunkDenseName), 1, 1, 0, dense_chunk_operand_count, 0,
     dense_chunk_output_count, kSupportedDevices,
     plugin_chunk_dense},
    {string_view(kChunkGeneralizedName), 1, 1, 0,
     generalized_chunk_operand_count, 0, generalized_chunk_output_count,
     kSupportedDevices, plugin_chunk_generalized},
};

constinit const emlx::plugin::descriptor_t kDescriptor{
    string_view(kPluginName),
    sizeof(emlx::plugin::descriptor_t),
    sizeof(emlx::plugin::callback_descriptor_t),
    static_cast<uint32_t>(sizeof(kCallbacks) / sizeof(kCallbacks[0])),
    kCallbacks};

constinit const emlx::plugin::bootstrap_v1_t kBootstrap{
    emlx::plugin::magic_v1,
    sizeof(emlx::plugin::bootstrap_v1_t),
    emlx::plugin::abi_v1,
    sizeof(emlx::plugin::descriptor_t),
    &kDescriptor};

} // namespace

extern "C" EMLX_PLUGIN_EXPORT const emlx::plugin::bootstrap_v1_t *
emlx_plugin_descriptor_v1() noexcept {
  return &kBootstrap;
}
