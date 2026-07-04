#include "qwen3.hpp"
#include "../emlx_nif_shared.hpp"
#include "qwen3_plugin_abi.hpp"

#include <dlfcn.h>
#include <memory>

// Qwen3 model accelerators used by emlx_axon. This file is the *host* side
// of the qwen3 NIF/plugin split: it owns everything that touches Erlang
// terms (decoding args, wrapping `mlx::core::array` results back into
// tensor resources) and calls through to `libemlx_qwen3.so` — a standalone,
// dynamically loaded shared library with no Erlang dependency at all — for
// every actual MLX computation. See qwen3_plugin_abi.hpp for the ABI and
// qwen3_plugin.cpp for the compute implementations.
//
// This split exists so the qwen3 compute can eventually move into
// emlx_axon as its own build artifact without dragging erl_nif/resource-type
// plumbing along with it — see `EMLX.NIF.load_qwen3_plugin/1`.

namespace {

const emlx_qwen3_plugin::VTable *g_qwen3_plugin = nullptr;
void *g_qwen3_plugin_handle = nullptr;

} // namespace

// qwen3_require_plugin — every qwen3_* NIF calls this first; the plugin
// must be loaded via `EMLX.NIF.load_qwen3_plugin/1` before any of these can
// run (see EMLX.Application, which loads it eagerly at boot).
static bool qwen3_require_plugin(ErlNifEnv *env, ERL_NIF_TERM *out_error) {
  if (g_qwen3_plugin != nullptr) {
    return true;
  }
  *out_error = nx::nif::error(
      env, "qwen3 plugin not loaded — call EMLX.NIF.load_qwen3_plugin/1 first");
  return false;
}

// load_qwen3_plugin — `dlopen`s the standalone qwen3 compute plugin and
// resolves its vtable. Not worker-routed: `dlopen`/`dlsym` do not touch the
// MLX graph, so this can run directly on the calling (BEAM scheduler)
// thread, same as `command_queue_new`.
NIF(load_qwen3_plugin) {
  std::string path;
  if (!nx::nif::get(env, argv[0], path)) {
    return nx::nif::error(env, "load_qwen3_plugin expects a path string");
  }

  if (g_qwen3_plugin_handle != nullptr) {
    dlclose(g_qwen3_plugin_handle);
    g_qwen3_plugin_handle = nullptr;
    g_qwen3_plugin = nullptr;
  }

  void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    std::ostringstream msg;
    msg << "Failed to load qwen3 plugin at " << path << ": " << dlerror();
    return nx::nif::error(env, msg.str().c_str());
  }

  using VTableFn = const emlx_qwen3_plugin::VTable *(*)();
  auto get_vtable = reinterpret_cast<VTableFn>(dlsym(handle, "emlx_qwen3_plugin_vtable"));
  if (get_vtable == nullptr) {
    dlclose(handle);
    return nx::nif::error(env, "qwen3 plugin is missing the emlx_qwen3_plugin_vtable symbol");
  }

  const emlx_qwen3_plugin::VTable *vtable = get_vtable();
  if (vtable == nullptr) {
    dlclose(handle);
    return nx::nif::error(env, "qwen3 plugin returned a null vtable");
  }

  g_qwen3_plugin_handle = handle;
  g_qwen3_plugin = vtable;
  return nx::nif::ok(env);
}

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
  ERL_NIF_TERM plugin_error;
  if (!qwen3_require_plugin(env, &plugin_error)) {
    return plugin_error;
  }

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
    mlx::core::array out(0), k_upd(0), v_upd(0);
    std::string error;
    if (!g_qwen3_plugin->kv_cache_attention(*q, *new_k, *new_v, *k_cache, *v_cache, offset, scale,
                                             head_dim, theta, device, out, k_upd, v_upd, error)) {
      return nx::nif::error(env, error.c_str());
    }

    ERL_NIF_TERM result_tuple[3];
    result_tuple[0] = create_tensor_resource(env, out);
    result_tuple[1] = create_tensor_resource(env, k_upd);
    result_tuple[2] = create_tensor_resource(env, v_upd);

    return nx::nif::ok(env, enif_make_tuple3(env, result_tuple[0], result_tuple[1], result_tuple[2]));
  }
  CATCH()
}
ASYNC_NIF(qwen3_kv_cache_attention)

// qwen3_mlp — dense Qwen3 MLP block: RMSNorm + gate/up + SwiGLU + down + residual.
NIF(qwen3_mlp) {
  ERL_NIF_TERM plugin_error;
  if (!qwen3_require_plugin(env, &plugin_error)) {
    return plugin_error;
  }

  TENSOR_PARAM(0, hidden);
  TENSOR_PARAM(1, norm);
  TENSOR_PARAM(2, gate_proj);
  TENSOR_PARAM(3, up_proj);
  TENSOR_PARAM(4, down_proj);
  PARAM(5, double, eps);
  DEVICE_PARAM(6, device);

  try {
    mlx::core::array out(0);
    std::string error;
    if (!g_qwen3_plugin->mlp(*hidden, *norm, *gate_proj, *up_proj, *down_proj, eps, device, out,
                              error)) {
      return nx::nif::error(env, error.c_str());
    }

    TENSOR(out);
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
  ERL_NIF_TERM plugin_error;
  if (!qwen3_require_plugin(env, &plugin_error)) {
    return plugin_error;
  }

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
    emlx_qwen3_plugin::LayerParams layer{norm1, norm2, q_norm, k_norm, q_proj,
                                          k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj};
    emlx_qwen3_plugin::KVCache kv{k_cache, v_cache};

    mlx::core::array out(0), k_upd(0), v_upd(0);
    std::string error;
    if (!g_qwen3_plugin->layer_dense(*hidden, layer, kv, offset, scale, head_dim, theta, eps,
                                      device, out, k_upd, v_upd, error)) {
      return nx::nif::error(env, error.c_str());
    }

    ERL_NIF_TERM result_tuple[3];
    result_tuple[0] = create_tensor_resource(env, out);
    result_tuple[1] = create_tensor_resource(env, k_upd);
    result_tuple[2] = create_tensor_resource(env, v_upd);

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
  ERL_NIF_TERM plugin_error;
  if (!qwen3_require_plugin(env, &plugin_error)) {
    return plugin_error;
  }

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
    emlx_qwen3_plugin::KVCache kv{k_cache, v_cache};

    mlx::core::array out(0), k_upd(0), v_upd(0);
    std::string error;
    if (!g_qwen3_plugin->layer_quantized(*hidden, layer, kv, offset, scale, head_dim, theta, eps,
                                          device, out, k_upd, v_upd, error)) {
      return nx::nif::error(env, error.c_str());
    }

    ERL_NIF_TERM result_tuple[3];
    result_tuple[0] = create_tensor_resource(env, out);
    result_tuple[1] = create_tensor_resource(env, k_upd);
    result_tuple[2] = create_tensor_resource(env, v_upd);

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

// qwen3_forward_greedy_ids — embedding lookup + dense forward through all layers +
// final greedy token. Returns {token_ids, kv_cache} where token_ids has
// shape {B}.
NIF(qwen3_forward_greedy_ids) {
  ERL_NIF_TERM plugin_error;
  if (!qwen3_require_plugin(env, &plugin_error)) {
    return plugin_error;
  }

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

    mlx::core::array token_out(0);
    int64_t token_id_out = 0;
    std::vector<mlx::core::array> k_out, v_out;
    if (!g_qwen3_plugin->forward_greedy_from_hidden(
            embedded, layers, kvs, *norm, *lm_head, offset, scale, head_dim, theta, eps, false,
            device, token_out, token_id_out, k_out, v_out, error)) {
      return nx::nif::error(env, error.c_str());
    }

    ERL_NIF_TERM token_term = create_tensor_resource(env, token_out);
    return nx::nif::ok(env, enif_make_tuple2(env, token_term, qwen3_wrap_kv_terms(env, k_out, v_out)));
  }
  CATCH()
}
ASYNC_NIF(qwen3_forward_greedy_ids)

// qwen3_forward_greedy_ids_chunk — repeatedly decode greedy tokens from a
// single token id tensor without returning to Elixir between decode steps.
// Returns {token_id_refs, kv_cache}.
NIF(qwen3_forward_greedy_ids_chunk) {
  ERL_NIF_TERM plugin_error;
  if (!qwen3_require_plugin(env, &plugin_error)) {
    return plugin_error;
  }

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

    std::string error;
    std::vector<mlx::core::array> token_out, k_out, v_out;
    if (!g_qwen3_plugin->forward_greedy_ids_chunk(
            *input_ids, *embed_tokens, layers, initial_kv, *norm, *lm_head, offset, count, scale,
            head_dim, theta, eps, device, token_out, k_out, v_out, error)) {
      return nx::nif::error(env, error.c_str());
    }

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
  ERL_NIF_TERM plugin_error;
  if (!qwen3_require_plugin(env, &plugin_error)) {
    return plugin_error;
  }

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

    mlx::core::array token_out(0);
    int64_t token_id_out = 0;
    std::vector<mlx::core::array> k_out, v_out;
    if (!g_qwen3_plugin->forward_greedy_from_hidden(
            embedded, layers, kvs, *norm, *lm_head, offset, scale, head_dim, theta, eps, true,
            device, token_out, token_id_out, k_out, v_out, error)) {
      return nx::nif::error(env, error.c_str());
    }

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
  ERL_NIF_TERM plugin_error;
  if (!qwen3_require_plugin(env, &plugin_error)) {
    return plugin_error;
  }

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

    mlx::core::array token_out(0);
    int64_t token_id_out = 0;
    std::vector<mlx::core::array> k_out, v_out;
    if (!g_qwen3_plugin->forward_greedy_from_hidden(
            embedded, layers, kvs, *norm, *lm_head, offset, scale, head_dim, theta, eps, true,
            device, token_out, token_id_out, k_out, v_out, error)) {
      return nx::nif::error(env, error.c_str());
    }

    return nx::nif::ok(
        env, enif_make_tuple2(env, nx::nif::make(env, token_id_out), qwen3_wrap_kv_terms(env, k_out, v_out)));
  }
  CATCH()
}
ASYNC_NIF(qwen3_forward_greedy_token_id)

// qwen3_final_greedy — final RMSNorm + dense lm_head + argmax for greedy decode.
NIF(qwen3_final_greedy) {
  ERL_NIF_TERM plugin_error;
  if (!qwen3_require_plugin(env, &plugin_error)) {
    return plugin_error;
  }

  TENSOR_PARAM(0, hidden);
  TENSOR_PARAM(1, norm);
  TENSOR_PARAM(2, lm_head);
  PARAM(3, double, eps);
  DEVICE_PARAM(4, device);

  try {
    mlx::core::array out(0);
    std::string error;
    if (!g_qwen3_plugin->final_greedy(*hidden, *norm, *lm_head, eps, device, out, error)) {
      return nx::nif::error(env, error.c_str());
    }

    TENSOR(out);
  }
  CATCH()
}
ASYNC_NIF(qwen3_final_greedy)

// qwen3_attention_residual — dense attention output projection + residual add.
NIF(qwen3_attention_residual) {
  ERL_NIF_TERM plugin_error;
  if (!qwen3_require_plugin(env, &plugin_error)) {
    return plugin_error;
  }

  TENSOR_PARAM(0, hidden);
  TENSOR_PARAM(1, attn_out);
  TENSOR_PARAM(2, o_proj);
  DEVICE_PARAM(3, device);

  try {
    mlx::core::array out(0);
    std::string error;
    if (!g_qwen3_plugin->attention_residual(*hidden, *attn_out, *o_proj, device, out, error)) {
      return nx::nif::error(env, error.c_str());
    }

    TENSOR(out);
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
  ERL_NIF_TERM plugin_error;
  if (!qwen3_require_plugin(env, &plugin_error)) {
    return plugin_error;
  }

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
    mlx::core::array out(0), k_upd(0), v_upd(0);
    std::string error;
    if (!g_qwen3_plugin->attention_block(*hidden, *norm, *q_proj, *k_proj, *v_proj, *o_proj,
                                          *q_norm, *k_norm, *k_cache, *v_cache, offset, scale,
                                          head_dim, theta, eps, device, out, k_upd, v_upd,
                                          error)) {
      return nx::nif::error(env, error.c_str());
    }

    ERL_NIF_TERM result_tuple[3];
    result_tuple[0] = create_tensor_resource(env, out);
    result_tuple[1] = create_tensor_resource(env, k_upd);
    result_tuple[2] = create_tensor_resource(env, v_upd);

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
  ERL_NIF_TERM plugin_error;
  if (!qwen3_require_plugin(env, &plugin_error)) {
    return plugin_error;
  }

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

    std::string error;
    std::vector<mlx::core::array> token_out, k_out, v_out;
    if (!g_qwen3_plugin->forward_greedy_ids_chunk_quantized(
            *input_ids, *embed_tokens, layers, initial_kv, *norm, lm_head, offset, count, scale,
            head_dim, theta, eps, device, token_out, k_out, v_out, error)) {
      return nx::nif::error(env, error.c_str());
    }

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
