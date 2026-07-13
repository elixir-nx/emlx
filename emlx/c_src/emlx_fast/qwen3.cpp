#include "qwen3.hpp"
#include "../emlx_nif_shared.hpp"
#include "../emlx_plugin_registry.hpp"

#include <memory>
#include <cstring>

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

static int64_t qwen3_f64_bits(double value) {
  int64_t bits;
  std::memcpy(&bits, &value, sizeof(value));
  return bits;
}

static int64_t qwen3_quantization_mode(const std::string &mode) {
  if (mode == "affine") return 0;
  if (mode == "mxfp4") return 1;
  if (mode == "mxfp8") return 2;
  if (mode == "nvfp4") return 3;
  throw std::invalid_argument("unsupported Qwen3 quantization mode: " + mode);
}

static void qwen3_append_linear_weight(
    const emlx_qwen3_plugin::LinearWeight &weight,
    std::vector<mlx::core::array> &operands, std::vector<int64_t> &attrs) {
  const int64_t weight_index = static_cast<int64_t>(operands.size());
  operands.push_back(*weight.weight);

  int64_t scales_index = -1;
  int64_t biases_index = -1;
  if (weight.quantized) {
    scales_index = static_cast<int64_t>(operands.size());
    operands.push_back(*weight.scales);
    if (weight.biases) {
      biases_index = static_cast<int64_t>(operands.size());
      operands.push_back(*weight.biases);
    }
  }

  attrs.insert(attrs.end(),
               {weight.quantized ? 1 : 0, weight_index, scales_index,
                biases_index, weight.quantized ? weight.group_size : 0,
                weight.quantized ? weight.bits : 0,
                weight.quantized ? qwen3_quantization_mode(weight.mode) : 0,
                weight.transpose ? 1 : 0});
}

// Qwen3 model accelerators used by emlx_axon. This file is the *host* side
// of the qwen3 NIF/plugin split: it owns everything that touches Erlang
// terms (decoding args, wrapping `mlx::core::array` results back into
// tensor resources) and calls through to the standalone "qwen3" plugin — a
// dynamically loaded shared library with no Erlang dependency at all,
// living in emlx_axon (c_src/qwen3_plugin.cpp there) — for every actual MLX
// computation through the generic EMLX plugin callback registry.
//
// This split exists so the qwen3 compute can live in emlx_axon as its own
// build artifact without dragging erl_nif/resource-type plumbing along
// with it. The plugin is loaded generically via `EMLX.NIF.load_plugin/2`
// (see emlx_plugin_registry.hpp) under the name `"qwen3"`; this file only
// knows how to *decode* Qwen3's compatibility argument shapes, not how a
// plugin gets loaded or represented.

// ── Term decoding helpers ─────────────────────────────────────────────────
// These stay host-side: they read directly off `TENSOR_TYPE`-backed
// resources and the refcounting scheme `enif_alloc_resource` lays out for
// them (see `Qwen3TensorHandle` below), so they cannot move into the
// plugin without also moving Erlang's resource machinery there — which is
// exactly the cross-library resource-type problem this split avoids.

class Qwen3TensorHandle {
public:
  explicit Qwen3TensorHandle(mlx::core::array *ptr) : ptr_(ptr) {
    refcount_ = reinterpret_cast<std::atomic<int> *>(ptr_ + 1);

    if (refcount_->load() == 0) {
      ptr_ = nullptr;
      return;
    }

    ++(*refcount_);
  }

  ~Qwen3TensorHandle() {
    if (is_valid()) {
      if (refcount_->fetch_sub(1) == 0) {
        ptr_->~array();
      }
    }
  }

  bool is_valid() const { return ptr_ != nullptr; }
  mlx::core::array *data() const { return ptr_; }

private:
  mlx::core::array *ptr_;
  std::atomic<int> *refcount_;
};

using Qwen3TensorHandles = std::vector<std::unique_ptr<Qwen3TensorHandle>>;

static bool qwen3_get_tensor(
    ErlNifEnv *env,
    ERL_NIF_TERM term,
    mlx::core::array **out,
    Qwen3TensorHandles &handles,
    ERL_NIF_TERM *error) {
  mlx::core::array *raw = nullptr;

  if (!enif_get_resource(
          env, term, resource_object<mlx::core::array>::type,
          reinterpret_cast<void **>(&raw))) {
    return false;
  }

  auto handle = std::make_unique<Qwen3TensorHandle>(raw);
  if (!handle->is_valid()) {
    *error = nx::nif::error(env, "Tensor has been deallocated");
    return false;
  }

  *out = handle->data();
  handles.push_back(std::move(handle));
  return true;
}

static bool qwen3_get_tensor_or_device_ref(
    ErlNifEnv *env,
    ERL_NIF_TERM term,
    mlx::core::array **out,
    Qwen3TensorHandles &handles,
    ERL_NIF_TERM *error) {
  if (qwen3_get_tensor(env, term, out, handles, error)) {
    return true;
  }
  if (*error != 0) {
    return false;
  }

  int arity = 0;
  const ERL_NIF_TERM *items = nullptr;
  if (!enif_get_tuple(env, term, &arity, &items) || arity != 2) {
    return false;
  }

  return qwen3_get_tensor(env, items[1], out, handles, error);
}

// Decodes a linear weight term — either `{:dense, tensor_ref}` or
// `{:quantized, weight_ref, scales_ref, biases_ref_or_nil, group_size, bits,
// mode, transpose}` (mirrors the tuple built by `EMLX.Native.Qwen3.linear_weight_term/1`
// in emlx.ex) — into an `emlx_qwen3_plugin::LinearWeight`. `dense_transpose`
// selects the dense orientation when the term is `{:dense, ref}` (false for
// the usual {H,out} q/k/v/o/gate/up/down projections, true for the {out,H}
// lm_head convention); quantized terms carry their own explicit `transpose`
// flag.
static bool qwen3_get_linear_weight(
    ErlNifEnv *env,
    ERL_NIF_TERM term,
    bool dense_transpose,
    emlx_qwen3_plugin::LinearWeight &out,
    Qwen3TensorHandles &handles,
    ERL_NIF_TERM *error) {
  int arity = 0;
  const ERL_NIF_TERM *items = nullptr;
  if (!enif_get_tuple(env, term, &arity, &items) || arity < 2) {
    return false;
  }

  std::string tag;
  if (!nx::nif::get_atom(env, items[0], tag)) {
    return false;
  }

  if (tag == "dense" && arity == 2) {
    out.quantized = false;
    out.transpose = dense_transpose;
    return qwen3_get_tensor(env, items[1], &out.weight, handles, error);
  }

  if (tag == "quantized" && arity == 8) {
    out.quantized = true;

    if (!qwen3_get_tensor(env, items[1], &out.weight, handles, error) ||
        !qwen3_get_tensor(env, items[2], &out.scales, handles, error)) {
      return false;
    }

    std::string nil_atom;
    bool biases_nil = nx::nif::get_atom(env, items[3], nil_atom) && nil_atom == "nil";
    if (biases_nil) {
      out.biases = nullptr;
    } else if (!qwen3_get_tensor(env, items[3], &out.biases, handles, error)) {
      return false;
    }

    return nx::nif::get(env, items[4], &out.group_size) &&
           nx::nif::get(env, items[5], &out.bits) &&
           nx::nif::get(env, items[6], out.mode) &&
           nx::nif::get(env, items[7], &out.transpose);
  }

  return false;
}

static bool qwen3_get_layer(
    ErlNifEnv *env,
    ERL_NIF_TERM term,
    emlx_qwen3_plugin::LayerParams &layer,
    Qwen3TensorHandles &handles,
    ERL_NIF_TERM *error) {
  int arity = 0;
  const ERL_NIF_TERM *items = nullptr;
  if (!enif_get_tuple(env, term, &arity, &items) || arity != 11) {
    return false;
  }

  return qwen3_get_tensor(env, items[0], &layer.norm1, handles, error) &&
         qwen3_get_tensor(env, items[1], &layer.norm2, handles, error) &&
         qwen3_get_tensor(env, items[2], &layer.q_norm, handles, error) &&
         qwen3_get_tensor(env, items[3], &layer.k_norm, handles, error) &&
         qwen3_get_tensor(env, items[4], &layer.q_proj, handles, error) &&
         qwen3_get_tensor(env, items[5], &layer.k_proj, handles, error) &&
         qwen3_get_tensor(env, items[6], &layer.v_proj, handles, error) &&
         qwen3_get_tensor(env, items[7], &layer.o_proj, handles, error) &&
         qwen3_get_tensor(env, items[8], &layer.gate_proj, handles, error) &&
         qwen3_get_tensor(env, items[9], &layer.up_proj, handles, error) &&
         qwen3_get_tensor(env, items[10], &layer.down_proj, handles, error);
}

static bool qwen3_get_kv(
    ErlNifEnv *env,
    ERL_NIF_TERM term,
    emlx_qwen3_plugin::KVCache &kv,
    Qwen3TensorHandles &handles,
    ERL_NIF_TERM *error) {
  int arity = 0;
  const ERL_NIF_TERM *items = nullptr;
  if (!enif_get_tuple(env, term, &arity, &items) || arity != 2) {
    return false;
  }

  return qwen3_get_tensor_or_device_ref(env, items[0], &kv.k, handles, error) &&
         qwen3_get_tensor_or_device_ref(env, items[1], &kv.v, handles, error);
}

static bool qwen3_get_layer_generalized(
    ErlNifEnv *env,
    ERL_NIF_TERM term,
    emlx_qwen3_plugin::LayerParamsQ &layer,
    Qwen3TensorHandles &handles,
    ERL_NIF_TERM *error) {
  int arity = 0;
  const ERL_NIF_TERM *items = nullptr;
  if (!enif_get_tuple(env, term, &arity, &items) || arity != 11) {
    return false;
  }

  return qwen3_get_tensor(env, items[0], &layer.norm1, handles, error) &&
         qwen3_get_tensor(env, items[1], &layer.norm2, handles, error) &&
         qwen3_get_tensor(env, items[2], &layer.q_norm, handles, error) &&
         qwen3_get_tensor(env, items[3], &layer.k_norm, handles, error) &&
         qwen3_get_linear_weight(env, items[4], false, layer.q_proj, handles, error) &&
         qwen3_get_linear_weight(env, items[5], false, layer.k_proj, handles, error) &&
         qwen3_get_linear_weight(env, items[6], false, layer.v_proj, handles, error) &&
         qwen3_get_linear_weight(env, items[7], false, layer.o_proj, handles, error) &&
         qwen3_get_linear_weight(env, items[8], false, layer.gate_proj, handles, error) &&
         qwen3_get_linear_weight(env, items[9], false, layer.up_proj, handles, error) &&
         qwen3_get_linear_weight(env, items[10], false, layer.down_proj, handles, error);
}

static ERL_NIF_TERM qwen3_ref_error_or(
    ErlNifEnv *env, ERL_NIF_TERM error, const char *fallback) {
  if (error != 0) {
    return error;
  }

  return nx::nif::error(env, fallback);
}

// Minimal rank/positivity check for `input_ids`/`embed_tokens`, ahead of the
// host-side embedding lookup (`mlx::core::take`) that every
// `qwen3_forward_greedy_*` NIF performs before delegating to the plugin.
// The plugin re-validates everything downstream of the embedded hidden
// state; this only guards the `shape(0)`/`shape(1)` accesses below it.
static bool qwen3_require_rank2_positive(const mlx::core::array &tensor, const char *name,
                                          std::string &error) {
  if (tensor.ndim() != 2) {
    error = std::string(name) + " expects rank 2, got rank " + std::to_string(tensor.ndim());
    return false;
  }
  if (tensor.shape(0) <= 0 || tensor.shape(1) <= 0) {
    error = std::string(name) + " dimensions must be positive";
    return false;
  }
  return true;
}

// qwen3_kv_cache_attention — Qwen3 fused RoPE + KV update + SDPA.
//
// Inputs:
//   q        — {B, T_new, N_q,  D}  Q projection after Q norm
//   new_k    — {B, T_new, N_kv, D}  K projection after K norm
//   new_v    — {B, T_new, N_kv, D}  V projection
//   k_cache  — {B, N_kv, T_max, D}  preallocated key buffer
//   v_cache  — {B, N_kv, T_max, D}  preallocated value buffer
//   offset   — int                  tokens already in cache
//   scale    — float                1/sqrt(head_dim)
//   head_dim — int                  RoPE dimensions
//   theta    — float                RoPE base
//   device   — atom
//
// Returns {attn_out, k_upd, v_upd}.
NIF(qwen3_kv_cache_attention) {
  TENSOR_PARAM(0, q);
  TENSOR_PARAM(1, new_k);
  TENSOR_PARAM(2, new_v);
  TENSOR_PARAM(3, k_cache);
  TENSOR_PARAM(4, v_cache);
  PARAM(5, int, offset);
  PARAM(6, double, scale);
  PARAM(7, int, head_dim);
  PARAM(8, double, theta);
  DEVICE_PARAM(9, device);

  try {
    auto outputs = emlx_invoke_plugin_callback(
        "qwen3", "kv_cache_attention", {*q, *new_k, *new_v, *k_cache, *v_cache},
        {offset, qwen3_f64_bits(scale), head_dim, qwen3_f64_bits(theta)}, device);

    ERL_NIF_TERM result_tuple[3];
    result_tuple[0] = create_tensor_resource(env, std::move(outputs[0]));
    result_tuple[1] = create_tensor_resource(env, std::move(outputs[1]));
    result_tuple[2] = create_tensor_resource(env, std::move(outputs[2]));

    return nx::nif::ok(env, enif_make_tuple3(env, result_tuple[0], result_tuple[1], result_tuple[2]));
  }
  CATCH()
}
ASYNC_NIF(qwen3_kv_cache_attention)

// qwen3_mlp — dense Qwen3 MLP block: RMSNorm + gate/up + SwiGLU + down + residual.
NIF(qwen3_mlp) {
  TENSOR_PARAM(0, hidden);
  TENSOR_PARAM(1, norm);
  TENSOR_PARAM(2, gate_proj);
  TENSOR_PARAM(3, up_proj);
  TENSOR_PARAM(4, down_proj);
  PARAM(5, double, eps);
  DEVICE_PARAM(6, device);

  try {
    auto outputs = emlx_invoke_plugin_callback(
        "qwen3", "mlp", {*hidden, *norm, *gate_proj, *up_proj, *down_proj},
        {qwen3_f64_bits(eps)}, device);
    TENSOR(std::move(outputs[0]));
  }
  CATCH()
}
ASYNC_NIF(qwen3_mlp)

// qwen3_layer — dense Qwen3 transformer layer:
// attention input RMSNorm + dense attention block + RMSNorm after attention
// + dense MLP + residual add.
//
// Returns {hidden_out, k_upd, v_upd}.
NIF(qwen3_layer) {
  TENSOR_PARAM(0, hidden);
  TENSOR_PARAM(1, norm1);
  TENSOR_PARAM(2, q_proj);
  TENSOR_PARAM(3, k_proj);
  TENSOR_PARAM(4, v_proj);
  TENSOR_PARAM(5, o_proj);
  TENSOR_PARAM(6, q_norm);
  TENSOR_PARAM(7, k_norm);
  TENSOR_PARAM(8, k_cache);
  TENSOR_PARAM(9, v_cache);
  TENSOR_PARAM(10, norm2);
  TENSOR_PARAM(11, gate_proj);
  TENSOR_PARAM(12, up_proj);
  TENSOR_PARAM(13, down_proj);
  PARAM(14, int, offset);
  PARAM(15, double, scale);
  PARAM(16, int, head_dim);
  PARAM(17, double, theta);
  PARAM(18, double, eps);
  DEVICE_PARAM(19, device);

  try {
    auto outputs = emlx_invoke_plugin_callback(
        "qwen3", "layer_dense",
        {*hidden, *norm1, *q_proj, *k_proj, *v_proj, *o_proj, *q_norm,
         *k_norm, *k_cache, *v_cache, *norm2, *gate_proj, *up_proj, *down_proj},
        {offset, qwen3_f64_bits(scale), head_dim, qwen3_f64_bits(theta),
         qwen3_f64_bits(eps)},
        device);

    ERL_NIF_TERM result_tuple[3];
    result_tuple[0] = create_tensor_resource(env, std::move(outputs[0]));
    result_tuple[1] = create_tensor_resource(env, std::move(outputs[1]));
    result_tuple[2] = create_tensor_resource(env, std::move(outputs[2]));

    return nx::nif::ok(env, enif_make_tuple3(env, result_tuple[0], result_tuple[1], result_tuple[2]));
  }
  CATCH()
}
ASYNC_NIF(qwen3_layer)

// qwen3_layer_quantized — generalized Qwen3 transformer layer: same fusion as
// `qwen3_layer`, but each of the 7 projections (q/k/v/o/gate/up/down)
// independently accepts a dense-or-quantized weight term (see
// `qwen3_get_linear_weight`).
//
// Returns {hidden_out, k_upd, v_upd}.
NIF(qwen3_layer_quantized) {
  TENSOR_PARAM(0, hidden);

  Qwen3TensorHandles handles;
  ERL_NIF_TERM ref_error = 0;
  emlx_qwen3_plugin::LayerParamsQ layer;
  mlx::core::array *k_cache = nullptr;
  mlx::core::array *v_cache = nullptr;

  if (!qwen3_get_tensor(env, argv[1], &layer.norm1, handles, &ref_error)) {
    return qwen3_ref_error_or(env, ref_error, "qwen3_layer_quantized expects norm1 tensor ref");
  }
  if (!qwen3_get_linear_weight(env, argv[2], false, layer.q_proj, handles, &ref_error)) {
    return qwen3_ref_error_or(env, ref_error, "qwen3_layer_quantized got invalid q_proj weight term");
  }
  if (!qwen3_get_linear_weight(env, argv[3], false, layer.k_proj, handles, &ref_error)) {
    return qwen3_ref_error_or(env, ref_error, "qwen3_layer_quantized got invalid k_proj weight term");
  }
  if (!qwen3_get_linear_weight(env, argv[4], false, layer.v_proj, handles, &ref_error)) {
    return qwen3_ref_error_or(env, ref_error, "qwen3_layer_quantized got invalid v_proj weight term");
  }
  if (!qwen3_get_linear_weight(env, argv[5], false, layer.o_proj, handles, &ref_error)) {
    return qwen3_ref_error_or(env, ref_error, "qwen3_layer_quantized got invalid o_proj weight term");
  }
  if (!qwen3_get_tensor(env, argv[6], &layer.q_norm, handles, &ref_error)) {
    return qwen3_ref_error_or(env, ref_error, "qwen3_layer_quantized expects q_norm tensor ref");
  }
  if (!qwen3_get_tensor(env, argv[7], &layer.k_norm, handles, &ref_error)) {
    return qwen3_ref_error_or(env, ref_error, "qwen3_layer_quantized expects k_norm tensor ref");
  }
  if (!qwen3_get_tensor(env, argv[8], &k_cache, handles, &ref_error)) {
    return qwen3_ref_error_or(env, ref_error, "qwen3_layer_quantized expects k_cache tensor ref");
  }
  if (!qwen3_get_tensor(env, argv[9], &v_cache, handles, &ref_error)) {
    return qwen3_ref_error_or(env, ref_error, "qwen3_layer_quantized expects v_cache tensor ref");
  }
  if (!qwen3_get_tensor(env, argv[10], &layer.norm2, handles, &ref_error)) {
    return qwen3_ref_error_or(env, ref_error, "qwen3_layer_quantized expects norm2 tensor ref");
  }
  if (!qwen3_get_linear_weight(env, argv[11], false, layer.gate_proj, handles, &ref_error)) {
    return qwen3_ref_error_or(env, ref_error, "qwen3_layer_quantized got invalid gate_proj weight term");
  }
  if (!qwen3_get_linear_weight(env, argv[12], false, layer.up_proj, handles, &ref_error)) {
    return qwen3_ref_error_or(env, ref_error, "qwen3_layer_quantized got invalid up_proj weight term");
  }
  if (!qwen3_get_linear_weight(env, argv[13], false, layer.down_proj, handles, &ref_error)) {
    return qwen3_ref_error_or(env, ref_error, "qwen3_layer_quantized got invalid down_proj weight term");
  }

  PARAM(14, int, offset);
  PARAM(15, double, scale);
  PARAM(16, int, head_dim);
  PARAM(17, double, theta);
  PARAM(18, double, eps);
  DEVICE_PARAM(19, device);

  try {
    std::vector<mlx::core::array> operands{
        *hidden, *layer.norm1, *layer.q_norm, *layer.k_norm,
        *k_cache, *v_cache, *layer.norm2};
    std::vector<int64_t> attrs{1, offset, qwen3_f64_bits(scale), head_dim,
                               qwen3_f64_bits(theta), qwen3_f64_bits(eps), 7};
    qwen3_append_linear_weight(layer.q_proj, operands, attrs);
    qwen3_append_linear_weight(layer.k_proj, operands, attrs);
    qwen3_append_linear_weight(layer.v_proj, operands, attrs);
    qwen3_append_linear_weight(layer.o_proj, operands, attrs);
    qwen3_append_linear_weight(layer.gate_proj, operands, attrs);
    qwen3_append_linear_weight(layer.up_proj, operands, attrs);
    qwen3_append_linear_weight(layer.down_proj, operands, attrs);

    auto outputs = emlx_invoke_plugin_callback(
        "qwen3", "layer_generalized", operands, attrs, device);

    ERL_NIF_TERM result_tuple[3];
    result_tuple[0] = create_tensor_resource(env, std::move(outputs[0]));
    result_tuple[1] = create_tensor_resource(env, std::move(outputs[1]));
    result_tuple[2] = create_tensor_resource(env, std::move(outputs[2]));

    return nx::nif::ok(env, enif_make_tuple3(env, result_tuple[0], result_tuple[1], result_tuple[2]));
  }
  CATCH()
}
ASYNC_NIF(qwen3_layer_quantized)

static ERL_NIF_TERM qwen3_wrap_kv_terms(
    ErlNifEnv *env, const std::vector<mlx::core::array> &k, const std::vector<mlx::core::array> &v) {
  std::vector<ERL_NIF_TERM> kv_terms;
  kv_terms.reserve(k.size());
  for (size_t i = 0; i < k.size(); ++i) {
    kv_terms.push_back(
        enif_make_tuple2(env, create_tensor_resource(env, k[i]), create_tensor_resource(env, v[i])));
  }
  return enif_make_list_from_array(env, kv_terms.data(), kv_terms.size());
}

static std::vector<mlx::core::array> qwen3_dense_forward_operands(
    const mlx::core::array &first, const std::vector<emlx_qwen3_plugin::LayerParams> &layers,
    const std::vector<emlx_qwen3_plugin::KVCache> &caches,
    const mlx::core::array &norm, const mlx::core::array &lm_head,
    const mlx::core::array *second = nullptr) {
  std::vector<mlx::core::array> operands;
  operands.reserve((second ? 4 : 3) + layers.size() * 13);
  operands.push_back(first);
  if (second)
    operands.push_back(*second);
  for (size_t index = 0; index < layers.size(); ++index) {
    const auto &layer = layers[index];
    operands.insert(operands.end(),
                    {*layer.norm1, *layer.norm2, *layer.q_norm, *layer.k_norm,
                     *layer.q_proj, *layer.k_proj, *layer.v_proj, *layer.o_proj,
                     *layer.gate_proj, *layer.up_proj, *layer.down_proj,
                     *caches[index].k, *caches[index].v});
  }
  operands.push_back(norm);
  operands.push_back(lm_head);
  return operands;
}

static void qwen3_split_forward_outputs(
    std::vector<mlx::core::array> &outputs, size_t token_count,
    size_t layer_count, std::vector<mlx::core::array> &tokens,
    std::vector<mlx::core::array> &keys, std::vector<mlx::core::array> &values) {
  if (outputs.size() != token_count + layer_count * 2)
    throw std::runtime_error("qwen3 plugin returned the wrong full-forward output count");
  tokens.reserve(token_count);
  keys.reserve(layer_count);
  values.reserve(layer_count);
  for (size_t index = 0; index < token_count; ++index)
    tokens.push_back(std::move(outputs[index]));
  for (size_t index = 0; index < layer_count; ++index) {
    keys.push_back(std::move(outputs[token_count + index * 2]));
    values.push_back(std::move(outputs[token_count + index * 2 + 1]));
  }
}

static void qwen3_split_chunk_outputs(
    std::vector<mlx::core::array> &outputs, size_t token_count,
    size_t layer_count, const mlx::core::Device &device,
    std::vector<mlx::core::array> &tokens,
    std::vector<mlx::core::array> &keys, std::vector<mlx::core::array> &values) {
  if (outputs.size() != 1 + layer_count * 2)
    throw std::runtime_error("qwen3 plugin returned the wrong chunk output count");
  if (outputs[0].ndim() != 1 || outputs[0].shape(0) != token_count)
    throw std::runtime_error("qwen3 plugin returned the wrong chunk token shape");

  auto stacked_tokens = std::move(outputs[0]);
  tokens.reserve(token_count);
  keys.reserve(layer_count);
  values.reserve(layer_count);
  for (size_t index = 0; index < token_count; ++index) {
    tokens.push_back(mlx::core::slice(
        stacked_tokens, {static_cast<int>(index)},
        {static_cast<int>(index + 1)}, device));
  }
  for (size_t index = 0; index < layer_count; ++index) {
    keys.push_back(std::move(outputs[1 + index * 2]));
    values.push_back(std::move(outputs[1 + index * 2 + 1]));
  }
}

static int64_t qwen3_token_to_host(mlx::core::array token,
                                   const mlx::core::Device &device) {
  token = mlx::core::astype(token, mlx::core::int64, device);
  mlx::core::eval(token);
  return token.item<int64_t>();
}

// qwen3_forward_greedy_ids — embedding lookup + dense forward through all layers +
// final greedy token. Returns {token_ids, kv_cache} where token_ids has
// shape {B}.
NIF(qwen3_forward_greedy_ids) {
  TENSOR_PARAM(0, input_ids);
  TENSOR_PARAM(1, embed_tokens);
  PARAM(6, int, offset);
  PARAM(7, double, scale);
  PARAM(8, int, head_dim);
  PARAM(9, double, theta);
  PARAM(10, double, eps);
  DEVICE_PARAM(11, device);

  try {
    Qwen3TensorHandles handles;
    ERL_NIF_TERM ref_error = 0;
    mlx::core::array *norm = nullptr;
    mlx::core::array *lm_head = nullptr;
    if (!qwen3_get_tensor(env, argv[4], &norm, handles, &ref_error)) {
      return qwen3_ref_error_or(
          env, ref_error, "qwen3_forward_greedy_ids expects norm tensor ref");
    }
    if (!qwen3_get_tensor(env, argv[5], &lm_head, handles, &ref_error)) {
      return qwen3_ref_error_or(
          env, ref_error, "qwen3_forward_greedy_ids expects lm_head tensor ref");
    }

    std::string error;
    if (!qwen3_require_rank2_positive(*input_ids, "input_ids", error) ||
        !qwen3_require_rank2_positive(*embed_tokens, "embed_tokens", error)) {
      return nx::nif::error(env, error.c_str());
    }

    unsigned int layer_count = 0;
    unsigned int kv_count = 0;
    if (!enif_get_list_length(env, argv[2], &layer_count)) {
      return nx::nif::error(env, "Qwen3 greedy forward expects layers to be a list");
    }
    if (!enif_get_list_length(env, argv[3], &kv_count)) {
      return nx::nif::error(env, "Qwen3 greedy forward expects kv_cache to be a list");
    }
    if (layer_count != kv_count) {
      return nx::nif::error(env, "Qwen3 greedy forward layers and kv_cache length mismatch");
    }

    std::vector<emlx_qwen3_plugin::LayerParams> layers;
    layers.reserve(layer_count);
    ERL_NIF_TERM layer_head, layer_tail = argv[2];
    while (enif_get_list_cell(env, layer_tail, &layer_head, &layer_tail)) {
      emlx_qwen3_plugin::LayerParams layer;
      if (!qwen3_get_layer(env, layer_head, layer, handles, &ref_error)) {
        return qwen3_ref_error_or(env, ref_error, "Qwen3 greedy forward got invalid layer tuple");
      }
      layers.push_back(layer);
    }

    std::vector<emlx_qwen3_plugin::KVCache> kvs;
    kvs.reserve(kv_count);
    ERL_NIF_TERM kv_head, kv_tail = argv[3];
    while (enif_get_list_cell(env, kv_tail, &kv_head, &kv_tail)) {
      emlx_qwen3_plugin::KVCache kv;
      if (!qwen3_get_kv(env, kv_head, kv, handles, &ref_error)) {
        return qwen3_ref_error_or(env, ref_error, "Qwen3 greedy forward got invalid kv_cache tuple");
      }
      kvs.push_back(kv);
    }

    int B = input_ids->shape(0);
    int T = input_ids->shape(1);
    auto ids = mlx::core::reshape(*input_ids, {B * T}, device);
    auto embedded = mlx::core::reshape(
        mlx::core::take(*embed_tokens, ids, 0, device), {B, T, embed_tokens->shape(1)}, device);

    auto outputs = emlx_invoke_plugin_callback(
        "qwen3", "forward_greedy_dense",
        qwen3_dense_forward_operands(embedded, layers, kvs, *norm, *lm_head),
        {static_cast<int64_t>(layer_count), offset, qwen3_f64_bits(scale),
         head_dim, qwen3_f64_bits(theta), qwen3_f64_bits(eps)},
        device);
    std::vector<mlx::core::array> tokens, k_out, v_out;
    qwen3_split_forward_outputs(outputs, 1, layer_count, tokens, k_out, v_out);

    ERL_NIF_TERM token_term = create_tensor_resource(env, std::move(tokens[0]));
    return nx::nif::ok(env, enif_make_tuple2(env, token_term, qwen3_wrap_kv_terms(env, k_out, v_out)));
  }
  CATCH()
}
ASYNC_NIF(qwen3_forward_greedy_ids)

// qwen3_forward_greedy_ids_chunk — repeatedly decode greedy tokens from a
// single token id tensor without returning to Elixir between decode steps.
// Returns {token_id_refs, kv_cache}.
NIF(qwen3_forward_greedy_ids_chunk) {
  TENSOR_PARAM(0, input_ids);
  TENSOR_PARAM(1, embed_tokens);
  PARAM(6, int, offset);
  PARAM(7, int, count);
  PARAM(8, double, scale);
  PARAM(9, int, head_dim);
  PARAM(10, double, theta);
  PARAM(11, double, eps);
  DEVICE_PARAM(12, device);

  try {
    Qwen3TensorHandles handles;
    ERL_NIF_TERM ref_error = 0;
    mlx::core::array *norm = nullptr;
    mlx::core::array *lm_head = nullptr;
    if (!qwen3_get_tensor(env, argv[4], &norm, handles, &ref_error)) {
      return qwen3_ref_error_or(
          env, ref_error, "qwen3_forward_greedy_ids_chunk expects norm tensor ref");
    }
    if (!qwen3_get_tensor(env, argv[5], &lm_head, handles, &ref_error)) {
      return qwen3_ref_error_or(
          env, ref_error, "qwen3_forward_greedy_ids_chunk expects lm_head tensor ref");
    }

    unsigned int layer_count = 0;
    unsigned int kv_count = 0;
    if (!enif_get_list_length(env, argv[2], &layer_count)) {
      return nx::nif::error(env, "qwen3_forward_greedy_ids_chunk expects layers to be a list");
    }
    if (!enif_get_list_length(env, argv[3], &kv_count)) {
      return nx::nif::error(env, "qwen3_forward_greedy_ids_chunk expects kv_cache to be a list");
    }
    if (layer_count != kv_count) {
      return nx::nif::error(env, "qwen3_forward_greedy_ids_chunk layers and kv_cache length mismatch");
    }

    std::vector<emlx_qwen3_plugin::LayerParams> layers;
    layers.reserve(layer_count);
    ERL_NIF_TERM layer_head, layer_tail = argv[2];
    while (enif_get_list_cell(env, layer_tail, &layer_head, &layer_tail)) {
      emlx_qwen3_plugin::LayerParams layer;
      if (!qwen3_get_layer(env, layer_head, layer, handles, &ref_error)) {
        return qwen3_ref_error_or(
            env, ref_error, "qwen3_forward_greedy_ids_chunk got invalid layer tuple");
      }
      layers.push_back(layer);
    }

    std::vector<emlx_qwen3_plugin::KVCache> initial_kv;
    initial_kv.reserve(kv_count);
    ERL_NIF_TERM kv_head, kv_tail = argv[3];
    while (enif_get_list_cell(env, kv_tail, &kv_head, &kv_tail)) {
      emlx_qwen3_plugin::KVCache kv;
      if (!qwen3_get_kv(env, kv_head, kv, handles, &ref_error)) {
        return qwen3_ref_error_or(
            env, ref_error, "qwen3_forward_greedy_ids_chunk got invalid kv_cache tuple");
      }
      initial_kv.push_back(kv);
    }

    auto outputs = emlx_invoke_plugin_callback(
        "qwen3", "forward_greedy_chunk_dense",
        qwen3_dense_forward_operands(*input_ids, layers, initial_kv, *norm,
                                     *lm_head, embed_tokens),
        {static_cast<int64_t>(layer_count), offset, count,
         qwen3_f64_bits(scale), head_dim, qwen3_f64_bits(theta),
         qwen3_f64_bits(eps)},
        device);
    std::vector<mlx::core::array> token_out, k_out, v_out;
    qwen3_split_chunk_outputs(outputs, static_cast<size_t>(count), layer_count,
                              device, token_out, k_out, v_out);

    std::vector<ERL_NIF_TERM> token_terms;
    token_terms.reserve(token_out.size());
    for (const auto &token : token_out) {
      token_terms.push_back(create_tensor_resource(env, token));
    }

    ERL_NIF_TERM token_list = enif_make_list_from_array(env, token_terms.data(), token_terms.size());
    return nx::nif::ok(env, enif_make_tuple2(env, token_list, qwen3_wrap_kv_terms(env, k_out, v_out)));
  }
  CATCH()
}
ASYNC_NIF(qwen3_forward_greedy_ids_chunk)

// qwen3_forward_greedy_ids_token_id — same as qwen3_forward_greedy_ids, but
// returns the sampled token id as a BEAM integer.
NIF(qwen3_forward_greedy_ids_token_id) {
  TENSOR_PARAM(0, input_ids);
  TENSOR_PARAM(1, embed_tokens);
  PARAM(6, int, offset);
  PARAM(7, double, scale);
  PARAM(8, int, head_dim);
  PARAM(9, double, theta);
  PARAM(10, double, eps);
  DEVICE_PARAM(11, device);

  try {
    Qwen3TensorHandles handles;
    ERL_NIF_TERM ref_error = 0;
    mlx::core::array *norm = nullptr;
    mlx::core::array *lm_head = nullptr;
    if (!qwen3_get_tensor(env, argv[4], &norm, handles, &ref_error)) {
      return qwen3_ref_error_or(
          env, ref_error, "qwen3_forward_greedy_ids_token_id expects norm tensor ref");
    }
    if (!qwen3_get_tensor(env, argv[5], &lm_head, handles, &ref_error)) {
      return qwen3_ref_error_or(
          env, ref_error, "qwen3_forward_greedy_ids_token_id expects lm_head tensor ref");
    }

    unsigned int layer_count = 0;
    unsigned int kv_count = 0;
    if (!enif_get_list_length(env, argv[2], &layer_count)) {
      return nx::nif::error(env, "Qwen3 greedy forward expects layers to be a list");
    }
    if (!enif_get_list_length(env, argv[3], &kv_count)) {
      return nx::nif::error(env, "Qwen3 greedy forward expects kv_cache to be a list");
    }
    if (layer_count != kv_count) {
      return nx::nif::error(env, "Qwen3 greedy forward layers and kv_cache length mismatch");
    }

    std::vector<emlx_qwen3_plugin::LayerParams> layers;
    layers.reserve(layer_count);
    ERL_NIF_TERM layer_head, layer_tail = argv[2];
    while (enif_get_list_cell(env, layer_tail, &layer_head, &layer_tail)) {
      emlx_qwen3_plugin::LayerParams layer;
      if (!qwen3_get_layer(env, layer_head, layer, handles, &ref_error)) {
        return qwen3_ref_error_or(env, ref_error, "Qwen3 greedy forward got invalid layer tuple");
      }
      layers.push_back(layer);
    }

    std::vector<emlx_qwen3_plugin::KVCache> kvs;
    kvs.reserve(kv_count);
    ERL_NIF_TERM kv_head, kv_tail = argv[3];
    while (enif_get_list_cell(env, kv_tail, &kv_head, &kv_tail)) {
      emlx_qwen3_plugin::KVCache kv;
      if (!qwen3_get_kv(env, kv_head, kv, handles, &ref_error)) {
        return qwen3_ref_error_or(env, ref_error, "Qwen3 greedy forward got invalid kv_cache tuple");
      }
      kvs.push_back(kv);
    }

    std::string error;
    if (!qwen3_require_rank2_positive(*input_ids, "input_ids", error) ||
        !qwen3_require_rank2_positive(*embed_tokens, "embed_tokens", error)) {
      return nx::nif::error(env, error.c_str());
    }

    int B = input_ids->shape(0);
    int T = input_ids->shape(1);
    auto ids = mlx::core::reshape(*input_ids, {B * T}, device);
    auto embedded = mlx::core::reshape(
        mlx::core::take(*embed_tokens, ids, 0, device), {B, T, embed_tokens->shape(1)}, device);

    auto outputs = emlx_invoke_plugin_callback(
        "qwen3", "forward_greedy_dense",
        qwen3_dense_forward_operands(embedded, layers, kvs, *norm, *lm_head),
        {static_cast<int64_t>(layer_count), offset, qwen3_f64_bits(scale),
         head_dim, qwen3_f64_bits(theta), qwen3_f64_bits(eps)},
        device);
    std::vector<mlx::core::array> tokens, k_out, v_out;
    qwen3_split_forward_outputs(outputs, 1, layer_count, tokens, k_out, v_out);
    const int64_t token_id_out = qwen3_token_to_host(std::move(tokens[0]), device);

    return nx::nif::ok(
        env, enif_make_tuple2(env, nx::nif::make(env, token_id_out), qwen3_wrap_kv_terms(env, k_out, v_out)));
  }
  CATCH()
}
ASYNC_NIF(qwen3_forward_greedy_ids_token_id)

// qwen3_forward_greedy_token_id — decode variant that accepts the previous
// token as a BEAM integer, avoiding host Nx tensor construction and backend
// transfer for the single token greedy decode hot path.
NIF(qwen3_forward_greedy_token_id) {
  PARAM(0, int64_t, token_id);
  TENSOR_PARAM(1, embed_tokens);
  PARAM(6, int, offset);
  PARAM(7, double, scale);
  PARAM(8, int, head_dim);
  PARAM(9, double, theta);
  PARAM(10, double, eps);
  DEVICE_PARAM(11, device);

  try {
    Qwen3TensorHandles handles;
    ERL_NIF_TERM ref_error = 0;
    mlx::core::array *norm = nullptr;
    mlx::core::array *lm_head = nullptr;
    if (!qwen3_get_tensor(env, argv[4], &norm, handles, &ref_error)) {
      return qwen3_ref_error_or(
          env, ref_error, "qwen3_forward_greedy_token_id expects norm tensor ref");
    }
    if (!qwen3_get_tensor(env, argv[5], &lm_head, handles, &ref_error)) {
      return qwen3_ref_error_or(
          env, ref_error, "qwen3_forward_greedy_token_id expects lm_head tensor ref");
    }
    std::string rank_error;
    if (!qwen3_require_rank2_positive(*embed_tokens, "embed_tokens", rank_error)) {
      return nx::nif::error(env, rank_error.c_str());
    }
    if (token_id < 0 || token_id >= embed_tokens->shape(0)) {
      return nx::nif::error(env, "token_id is outside the embedding vocabulary");
    }

    unsigned int layer_count = 0;
    unsigned int kv_count = 0;
    if (!enif_get_list_length(env, argv[2], &layer_count)) {
      return nx::nif::error(env, "Qwen3 greedy forward expects layers to be a list");
    }
    if (!enif_get_list_length(env, argv[3], &kv_count)) {
      return nx::nif::error(env, "Qwen3 greedy forward expects kv_cache to be a list");
    }
    if (layer_count != kv_count) {
      return nx::nif::error(env, "Qwen3 greedy forward layers and kv_cache length mismatch");
    }

    std::vector<emlx_qwen3_plugin::LayerParams> layers;
    layers.reserve(layer_count);
    ERL_NIF_TERM layer_head, layer_tail = argv[2];
    while (enif_get_list_cell(env, layer_tail, &layer_head, &layer_tail)) {
      emlx_qwen3_plugin::LayerParams layer;
      if (!qwen3_get_layer(env, layer_head, layer, handles, &ref_error)) {
        return qwen3_ref_error_or(env, ref_error, "Qwen3 greedy forward got invalid layer tuple");
      }
      layers.push_back(layer);
    }

    std::vector<emlx_qwen3_plugin::KVCache> kvs;
    kvs.reserve(kv_count);
    ERL_NIF_TERM kv_head, kv_tail = argv[3];
    while (enif_get_list_cell(env, kv_tail, &kv_head, &kv_tail)) {
      emlx_qwen3_plugin::KVCache kv;
      if (!qwen3_get_kv(env, kv_head, kv, handles, &ref_error)) {
        return qwen3_ref_error_or(env, ref_error, "Qwen3 greedy forward got invalid kv_cache tuple");
      }
      kvs.push_back(kv);
    }

    std::string error;
    auto ids = mlx::core::array(token_id, mlx::core::int64);
    auto embedded = mlx::core::reshape(
        mlx::core::take(*embed_tokens, ids, 0, device), {1, 1, embed_tokens->shape(1)}, device);

    auto outputs = emlx_invoke_plugin_callback(
        "qwen3", "forward_greedy_dense",
        qwen3_dense_forward_operands(embedded, layers, kvs, *norm, *lm_head),
        {static_cast<int64_t>(layer_count), offset, qwen3_f64_bits(scale),
         head_dim, qwen3_f64_bits(theta), qwen3_f64_bits(eps)},
        device);
    std::vector<mlx::core::array> tokens, k_out, v_out;
    qwen3_split_forward_outputs(outputs, 1, layer_count, tokens, k_out, v_out);
    const int64_t token_id_out = qwen3_token_to_host(std::move(tokens[0]), device);

    return nx::nif::ok(
        env, enif_make_tuple2(env, nx::nif::make(env, token_id_out), qwen3_wrap_kv_terms(env, k_out, v_out)));
  }
  CATCH()
}
ASYNC_NIF(qwen3_forward_greedy_token_id)

// qwen3_final_greedy — final RMSNorm + dense lm_head + argmax for greedy decode.
NIF(qwen3_final_greedy) {
  TENSOR_PARAM(0, hidden);
  TENSOR_PARAM(1, norm);
  TENSOR_PARAM(2, lm_head);
  PARAM(3, double, eps);
  DEVICE_PARAM(4, device);

  try {
    auto outputs = emlx_invoke_plugin_callback(
        "qwen3", "final_greedy", {*hidden, *norm, *lm_head},
        {qwen3_f64_bits(eps)}, device);
    TENSOR(std::move(outputs[0]));
  }
  CATCH()
}
ASYNC_NIF(qwen3_final_greedy)

// qwen3_attention_residual — dense attention output projection + residual add.
NIF(qwen3_attention_residual) {
  TENSOR_PARAM(0, hidden);
  TENSOR_PARAM(1, attn_out);
  TENSOR_PARAM(2, o_proj);
  DEVICE_PARAM(3, device);

  try {
    auto outputs = emlx_invoke_plugin_callback(
        "qwen3", "attention_residual", {*hidden, *attn_out, *o_proj}, {},
        device);
    TENSOR(std::move(outputs[0]));
  }
  CATCH()
}
ASYNC_NIF(qwen3_attention_residual)

// qwen3_attention_block — dense Qwen3 attention block:
// input RMSNorm + Q/K/V projections + Q/K RMSNorm + RoPE + KV update + SDPA
// + output projection + residual add.
//
// Returns {hidden_out, k_upd, v_upd}.
NIF(qwen3_attention_block) {
  TENSOR_PARAM(0, hidden);
  TENSOR_PARAM(1, norm);
  TENSOR_PARAM(2, q_proj);
  TENSOR_PARAM(3, k_proj);
  TENSOR_PARAM(4, v_proj);
  TENSOR_PARAM(5, o_proj);
  TENSOR_PARAM(6, q_norm);
  TENSOR_PARAM(7, k_norm);
  TENSOR_PARAM(8, k_cache);
  TENSOR_PARAM(9, v_cache);
  PARAM(10, int, offset);
  PARAM(11, double, scale);
  PARAM(12, int, head_dim);
  PARAM(13, double, theta);
  PARAM(14, double, eps);
  DEVICE_PARAM(15, device);

  try {
    auto outputs = emlx_invoke_plugin_callback(
        "qwen3", "attention_block",
        {*hidden, *norm, *q_proj, *k_proj, *v_proj, *o_proj, *q_norm, *k_norm,
         *k_cache, *v_cache},
        {offset, qwen3_f64_bits(scale), head_dim, qwen3_f64_bits(theta),
         qwen3_f64_bits(eps)},
        device);

    ERL_NIF_TERM result_tuple[3];
    result_tuple[0] = create_tensor_resource(env, std::move(outputs[0]));
    result_tuple[1] = create_tensor_resource(env, std::move(outputs[1]));
    result_tuple[2] = create_tensor_resource(env, std::move(outputs[2]));

    return nx::nif::ok(env, enif_make_tuple3(env, result_tuple[0], result_tuple[1], result_tuple[2]));
  }
  CATCH()
}
ASYNC_NIF(qwen3_attention_block)

// qwen3_forward_greedy_ids_chunk_quantized — generalized variant of
// `qwen3_forward_greedy_ids_chunk`: every layer's 7 projections and the
// final `lm_head` each independently accept a dense-or-quantized weight
// term (see `qwen3_get_linear_weight`).
//
// Returns {token_id_refs, kv_cache}.
NIF(qwen3_forward_greedy_ids_chunk_quantized) {
  TENSOR_PARAM(0, input_ids);
  TENSOR_PARAM(1, embed_tokens);
  PARAM(6, int, offset);
  PARAM(7, int, count);
  PARAM(8, double, scale);
  PARAM(9, int, head_dim);
  PARAM(10, double, theta);
  PARAM(11, double, eps);
  DEVICE_PARAM(12, device);

  try {
    Qwen3TensorHandles handles;
    ERL_NIF_TERM ref_error = 0;
    mlx::core::array *norm = nullptr;
    emlx_qwen3_plugin::LinearWeight lm_head;
    if (!qwen3_get_tensor(env, argv[4], &norm, handles, &ref_error)) {
      return qwen3_ref_error_or(
          env, ref_error, "qwen3_forward_greedy_ids_chunk_quantized expects norm tensor ref");
    }
    if (!qwen3_get_linear_weight(env, argv[5], true, lm_head, handles, &ref_error)) {
      return qwen3_ref_error_or(
          env, ref_error,
          "qwen3_forward_greedy_ids_chunk_quantized got invalid lm_head weight term");
    }

    unsigned int layer_count = 0;
    unsigned int kv_count = 0;
    if (!enif_get_list_length(env, argv[2], &layer_count)) {
      return nx::nif::error(
          env, "qwen3_forward_greedy_ids_chunk_quantized expects layers to be a list");
    }
    if (!enif_get_list_length(env, argv[3], &kv_count)) {
      return nx::nif::error(
          env, "qwen3_forward_greedy_ids_chunk_quantized expects kv_cache to be a list");
    }
    if (layer_count != kv_count) {
      return nx::nif::error(
          env, "qwen3_forward_greedy_ids_chunk_quantized layers and kv_cache length mismatch");
    }

    std::vector<emlx_qwen3_plugin::LayerParamsQ> layers;
    layers.reserve(layer_count);
    ERL_NIF_TERM layer_head, layer_tail = argv[2];
    while (enif_get_list_cell(env, layer_tail, &layer_head, &layer_tail)) {
      emlx_qwen3_plugin::LayerParamsQ layer;
      if (!qwen3_get_layer_generalized(env, layer_head, layer, handles, &ref_error)) {
        return qwen3_ref_error_or(
            env, ref_error, "qwen3_forward_greedy_ids_chunk_quantized got invalid layer tuple");
      }
      layers.push_back(layer);
    }

    std::vector<emlx_qwen3_plugin::KVCache> initial_kv;
    initial_kv.reserve(kv_count);
    ERL_NIF_TERM kv_head, kv_tail = argv[3];
    while (enif_get_list_cell(env, kv_tail, &kv_head, &kv_tail)) {
      emlx_qwen3_plugin::KVCache kv;
      if (!qwen3_get_kv(env, kv_head, kv, handles, &ref_error)) {
        return qwen3_ref_error_or(
            env, ref_error, "qwen3_forward_greedy_ids_chunk_quantized got invalid kv_cache tuple");
      }
      initial_kv.push_back(kv);
    }

    std::vector<mlx::core::array> operands{*input_ids, *embed_tokens};
    std::vector<int64_t> attrs{
        1, static_cast<int64_t>(layer_count), offset, count,
        qwen3_f64_bits(scale), head_dim, qwen3_f64_bits(theta),
        qwen3_f64_bits(eps), static_cast<int64_t>(layer_count) * 7 + 1};
    for (size_t index = 0; index < layers.size(); ++index) {
      const auto &layer = layers[index];
      operands.insert(operands.end(), {*layer.norm1, *layer.norm2,
                                       *layer.q_norm, *layer.k_norm,
                                       *initial_kv[index].k,
                                       *initial_kv[index].v});
      qwen3_append_linear_weight(layer.q_proj, operands, attrs);
      qwen3_append_linear_weight(layer.k_proj, operands, attrs);
      qwen3_append_linear_weight(layer.v_proj, operands, attrs);
      qwen3_append_linear_weight(layer.o_proj, operands, attrs);
      qwen3_append_linear_weight(layer.gate_proj, operands, attrs);
      qwen3_append_linear_weight(layer.up_proj, operands, attrs);
      qwen3_append_linear_weight(layer.down_proj, operands, attrs);
    }
    operands.push_back(*norm);
    qwen3_append_linear_weight(lm_head, operands, attrs);

    auto outputs = emlx_invoke_plugin_callback(
        "qwen3", "forward_greedy_chunk_generalized", operands, attrs, device);
    std::vector<mlx::core::array> token_out, k_out, v_out;
    qwen3_split_chunk_outputs(outputs, static_cast<size_t>(count), layer_count,
                              device, token_out, k_out, v_out);

    std::vector<ERL_NIF_TERM> token_terms;
    token_terms.reserve(token_out.size());
    for (const auto &token : token_out) {
      token_terms.push_back(create_tensor_resource(env, token));
    }

    ERL_NIF_TERM token_list = enif_make_list_from_array(env, token_terms.data(), token_terms.size());
    return nx::nif::ok(env, enif_make_tuple2(env, token_list, qwen3_wrap_kv_terms(env, k_out, v_out)));
  }
  CATCH()
}
ASYNC_NIF(qwen3_forward_greedy_ids_chunk_quantized)
