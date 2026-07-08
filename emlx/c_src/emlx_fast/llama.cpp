#include "llama.hpp"
#include "../emlx_nif_shared.hpp"
#include "../emlx_plugin_registry.hpp"
#include "llama_plugin_abi.hpp"

#include <memory>
#include <stdexcept>

static const emlx_llama_plugin::VTable *llama_plugin(ErlNifEnv *env, ERL_NIF_TERM *out_error) {
  const void *vtable = emlx_get_plugin("llama");
  if (vtable != nullptr) {
    return reinterpret_cast<const emlx_llama_plugin::VTable *>(vtable);
  }

  *out_error =
      nx::nif::error(env, "llama plugin not loaded — call EMLX.NIF.load_plugin(\"llama\", path) first");
  return nullptr;
}

class LlamaTensorHandle {
public:
  explicit LlamaTensorHandle(mlx::core::array *ptr) : ptr_(ptr) {
    refcount_ = reinterpret_cast<std::atomic<int> *>(ptr_ + 1);

    if (refcount_->load() == 0) {
      ptr_ = nullptr;
      return;
    }

    ++(*refcount_);
  }

  ~LlamaTensorHandle() {
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

using LlamaTensorHandles = std::vector<std::unique_ptr<LlamaTensorHandle>>;

static bool llama_get_tensor(
    ErlNifEnv *env,
    ERL_NIF_TERM term,
    mlx::core::array **out,
    LlamaTensorHandles &handles,
    ERL_NIF_TERM *error) {
  mlx::core::array *raw = nullptr;

  if (!enif_get_resource(
          env, term, resource_object<mlx::core::array>::type,
          reinterpret_cast<void **>(&raw))) {
    return false;
  }

  auto handle = std::make_unique<LlamaTensorHandle>(raw);
  if (!handle->is_valid()) {
    *error = nx::nif::error(env, "Tensor has been deallocated");
    return false;
  }

  *out = handle->data();
  handles.push_back(std::move(handle));
  return true;
}

static bool llama_get_tensor_or_device_ref(
    ErlNifEnv *env,
    ERL_NIF_TERM term,
    mlx::core::array **out,
    LlamaTensorHandles &handles,
    ERL_NIF_TERM *error) {
  if (llama_get_tensor(env, term, out, handles, error)) {
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

  return llama_get_tensor(env, items[1], out, handles, error);
}

static bool llama_get_layer(
    ErlNifEnv *env,
    ERL_NIF_TERM term,
    emlx_llama_plugin::LayerParams &layer,
    LlamaTensorHandles &handles,
    ERL_NIF_TERM *error) {
  int arity = 0;
  const ERL_NIF_TERM *items = nullptr;
  if (!enif_get_tuple(env, term, &arity, &items) || arity != 9) {
    return false;
  }

  return llama_get_tensor(env, items[0], &layer.norm1, handles, error) &&
         llama_get_tensor(env, items[1], &layer.norm2, handles, error) &&
         llama_get_tensor(env, items[2], &layer.q_proj, handles, error) &&
         llama_get_tensor(env, items[3], &layer.k_proj, handles, error) &&
         llama_get_tensor(env, items[4], &layer.v_proj, handles, error) &&
         llama_get_tensor(env, items[5], &layer.o_proj, handles, error) &&
         llama_get_tensor(env, items[6], &layer.gate_proj, handles, error) &&
         llama_get_tensor(env, items[7], &layer.up_proj, handles, error) &&
         llama_get_tensor(env, items[8], &layer.down_proj, handles, error);
}

static bool llama_get_kv(
    ErlNifEnv *env,
    ERL_NIF_TERM term,
    emlx_llama_plugin::KVCache &kv,
    LlamaTensorHandles &handles,
    ERL_NIF_TERM *error) {
  int arity = 0;
  const ERL_NIF_TERM *items = nullptr;
  if (!enif_get_tuple(env, term, &arity, &items) || arity != 2) {
    return false;
  }

  return llama_get_tensor_or_device_ref(env, items[0], &kv.k, handles, error) &&
         llama_get_tensor_or_device_ref(env, items[1], &kv.v, handles, error);
}

static ERL_NIF_TERM llama_ref_error_or(
    ErlNifEnv *env, ERL_NIF_TERM error, const char *fallback) {
  if (error != 0) {
    return error;
  }

  return nx::nif::error(env, fallback);
}

static bool llama_require_rank2_positive(
    const mlx::core::array &tensor, const char *name, std::string &error) {
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

static bool llama_get_layers_and_kv(
    ErlNifEnv *env,
    ERL_NIF_TERM layers_arg,
    ERL_NIF_TERM kv_arg,
    std::vector<emlx_llama_plugin::LayerParams> &layers,
    std::vector<emlx_llama_plugin::KVCache> &kvs,
    LlamaTensorHandles &handles,
    const char *context) {
  unsigned int layer_count = 0;
  unsigned int kv_count = 0;
  if (!enif_get_list_length(env, layers_arg, &layer_count)) {
    throw std::runtime_error(std::string(context) + " expects layers to be a list");
  }
  if (!enif_get_list_length(env, kv_arg, &kv_count)) {
    throw std::runtime_error(std::string(context) + " expects kv_cache to be a list");
  }
  if (layer_count != kv_count) {
    throw std::runtime_error(std::string(context) + " layers and kv_cache length mismatch");
  }

  ERL_NIF_TERM ref_error = 0;
  layers.reserve(layer_count);
  ERL_NIF_TERM layer_head, layer_tail = layers_arg;
  while (enif_get_list_cell(env, layer_tail, &layer_head, &layer_tail)) {
    emlx_llama_plugin::LayerParams layer;
    if (!llama_get_layer(env, layer_head, layer, handles, &ref_error)) {
      if (ref_error != 0) {
        throw std::runtime_error("Tensor has been deallocated");
      }
      throw std::runtime_error(std::string(context) + " got invalid layer tuple");
    }
    layers.push_back(layer);
  }

  kvs.reserve(kv_count);
  ERL_NIF_TERM kv_head, kv_tail = kv_arg;
  while (enif_get_list_cell(env, kv_tail, &kv_head, &kv_tail)) {
    emlx_llama_plugin::KVCache kv;
    if (!llama_get_kv(env, kv_head, kv, handles, &ref_error)) {
      if (ref_error != 0) {
        throw std::runtime_error("Tensor has been deallocated");
      }
      throw std::runtime_error(std::string(context) + " got invalid kv_cache tuple");
    }
    kvs.push_back(kv);
  }

  return true;
}

static ERL_NIF_TERM llama_wrap_kv_terms(
    ErlNifEnv *env, const std::vector<mlx::core::array> &k, const std::vector<mlx::core::array> &v) {
  std::vector<ERL_NIF_TERM> kv_terms;
  kv_terms.reserve(k.size());
  for (size_t i = 0; i < k.size(); ++i) {
    kv_terms.push_back(
        enif_make_tuple2(env, create_tensor_resource(env, k[i]), create_tensor_resource(env, v[i])));
  }
  return enif_make_list_from_array(env, kv_terms.data(), kv_terms.size());
}

NIF(llama_layer) {
  ERL_NIF_TERM plugin_error;
  const emlx_llama_plugin::VTable *plugin = llama_plugin(env, &plugin_error);
  if (plugin == nullptr) {
    return plugin_error;
  }

  TENSOR_PARAM(0, hidden);
  TENSOR_PARAM(1, norm1);
  TENSOR_PARAM(2, q_proj);
  TENSOR_PARAM(3, k_proj);
  TENSOR_PARAM(4, v_proj);
  TENSOR_PARAM(5, o_proj);
  TENSOR_PARAM(6, k_cache);
  TENSOR_PARAM(7, v_cache);
  TENSOR_PARAM(8, norm2);
  TENSOR_PARAM(9, gate_proj);
  TENSOR_PARAM(10, up_proj);
  TENSOR_PARAM(11, down_proj);
  PARAM(12, int, offset);
  PARAM(13, double, scale);
  PARAM(14, int, head_dim);
  TENSOR_PARAM(15, rope_freqs);
  PARAM(16, double, eps);
  DEVICE_PARAM(17, device);

  try {
    emlx_llama_plugin::LayerParams layer{
        norm1, norm2, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj};
    emlx_llama_plugin::KVCache kv{k_cache, v_cache};

    mlx::core::array out(0), k_upd(0), v_upd(0);
    std::string error;
    if (!plugin->layer_dense(
            *hidden, layer, kv, offset, scale, head_dim, *rope_freqs, eps, device,
            out, k_upd, v_upd, error)) {
      return nx::nif::error(env, error.c_str());
    }

    return nx::nif::ok(
        env,
        enif_make_tuple3(
            env,
            create_tensor_resource(env, out),
            create_tensor_resource(env, k_upd),
            create_tensor_resource(env, v_upd)));
  }
  CATCH()
}
ASYNC_NIF(llama_layer)

NIF(llama_forward_greedy_ids) {
  ERL_NIF_TERM plugin_error;
  const emlx_llama_plugin::VTable *plugin = llama_plugin(env, &plugin_error);
  if (plugin == nullptr) {
    return plugin_error;
  }

  TENSOR_PARAM(0, input_ids);
  TENSOR_PARAM(1, embed_tokens);
  PARAM(6, int, offset);
  PARAM(7, double, scale);
  PARAM(8, int, head_dim);
  TENSOR_PARAM(9, rope_freqs);
  PARAM(10, double, eps);
  DEVICE_PARAM(11, device);

  try {
    LlamaTensorHandles handles;
    ERL_NIF_TERM ref_error = 0;
    mlx::core::array *norm = nullptr;
    mlx::core::array *lm_head = nullptr;
    if (!llama_get_tensor(env, argv[4], &norm, handles, &ref_error)) {
      return llama_ref_error_or(env, ref_error, "llama_forward_greedy_ids expects norm tensor ref");
    }
    if (!llama_get_tensor(env, argv[5], &lm_head, handles, &ref_error)) {
      return llama_ref_error_or(env, ref_error, "llama_forward_greedy_ids expects lm_head tensor ref");
    }

    std::vector<emlx_llama_plugin::LayerParams> layers;
    std::vector<emlx_llama_plugin::KVCache> kvs;
    llama_get_layers_and_kv(env, argv[2], argv[3], layers, kvs, handles, "Llama greedy forward");

    std::string error;
    if (!llama_require_rank2_positive(*input_ids, "input_ids", error) ||
        !llama_require_rank2_positive(*embed_tokens, "embed_tokens", error)) {
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
    if (!plugin->forward_greedy_from_hidden(
            embedded, layers, kvs, *norm, *lm_head, offset, scale, head_dim, *rope_freqs, eps,
            false, device, token_out, token_id_out, k_out, v_out, error)) {
      return nx::nif::error(env, error.c_str());
    }

    return nx::nif::ok(
        env,
        enif_make_tuple2(env, create_tensor_resource(env, token_out), llama_wrap_kv_terms(env, k_out, v_out)));
  }
  CATCH()
}
ASYNC_NIF(llama_forward_greedy_ids)

NIF(llama_forward_greedy_ids_token_id) {
  ERL_NIF_TERM plugin_error;
  const emlx_llama_plugin::VTable *plugin = llama_plugin(env, &plugin_error);
  if (plugin == nullptr) {
    return plugin_error;
  }

  TENSOR_PARAM(0, input_ids);
  TENSOR_PARAM(1, embed_tokens);
  PARAM(6, int, offset);
  PARAM(7, double, scale);
  PARAM(8, int, head_dim);
  TENSOR_PARAM(9, rope_freqs);
  PARAM(10, double, eps);
  DEVICE_PARAM(11, device);

  try {
    LlamaTensorHandles handles;
    ERL_NIF_TERM ref_error = 0;
    mlx::core::array *norm = nullptr;
    mlx::core::array *lm_head = nullptr;
    if (!llama_get_tensor(env, argv[4], &norm, handles, &ref_error)) {
      return llama_ref_error_or(env, ref_error, "llama_forward_greedy_ids_token_id expects norm tensor ref");
    }
    if (!llama_get_tensor(env, argv[5], &lm_head, handles, &ref_error)) {
      return llama_ref_error_or(env, ref_error, "llama_forward_greedy_ids_token_id expects lm_head tensor ref");
    }

    std::vector<emlx_llama_plugin::LayerParams> layers;
    std::vector<emlx_llama_plugin::KVCache> kvs;
    llama_get_layers_and_kv(env, argv[2], argv[3], layers, kvs, handles, "Llama greedy forward");

    std::string error;
    if (!llama_require_rank2_positive(*input_ids, "input_ids", error) ||
        !llama_require_rank2_positive(*embed_tokens, "embed_tokens", error)) {
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
    if (!plugin->forward_greedy_from_hidden(
            embedded, layers, kvs, *norm, *lm_head, offset, scale, head_dim, *rope_freqs, eps,
            true, device, token_out, token_id_out, k_out, v_out, error)) {
      return nx::nif::error(env, error.c_str());
    }

    return nx::nif::ok(
        env,
        enif_make_tuple2(env, nx::nif::make(env, token_id_out), llama_wrap_kv_terms(env, k_out, v_out)));
  }
  CATCH()
}
ASYNC_NIF(llama_forward_greedy_ids_token_id)

NIF(llama_forward_greedy_token_id) {
  ERL_NIF_TERM plugin_error;
  const emlx_llama_plugin::VTable *plugin = llama_plugin(env, &plugin_error);
  if (plugin == nullptr) {
    return plugin_error;
  }

  PARAM(0, int64_t, token_id);
  TENSOR_PARAM(1, embed_tokens);
  PARAM(6, int, offset);
  PARAM(7, double, scale);
  PARAM(8, int, head_dim);
  TENSOR_PARAM(9, rope_freqs);
  PARAM(10, double, eps);
  DEVICE_PARAM(11, device);

  try {
    LlamaTensorHandles handles;
    ERL_NIF_TERM ref_error = 0;
    mlx::core::array *norm = nullptr;
    mlx::core::array *lm_head = nullptr;
    if (!llama_get_tensor(env, argv[4], &norm, handles, &ref_error)) {
      return llama_ref_error_or(env, ref_error, "llama_forward_greedy_token_id expects norm tensor ref");
    }
    if (!llama_get_tensor(env, argv[5], &lm_head, handles, &ref_error)) {
      return llama_ref_error_or(env, ref_error, "llama_forward_greedy_token_id expects lm_head tensor ref");
    }
    std::string error;
    if (!llama_require_rank2_positive(*embed_tokens, "embed_tokens", error)) {
      return nx::nif::error(env, error.c_str());
    }
    if (token_id < 0 || token_id >= embed_tokens->shape(0)) {
      return nx::nif::error(env, "token_id is outside the embedding vocabulary");
    }

    std::vector<emlx_llama_plugin::LayerParams> layers;
    std::vector<emlx_llama_plugin::KVCache> kvs;
    llama_get_layers_and_kv(env, argv[2], argv[3], layers, kvs, handles, "Llama greedy forward");

    auto ids = mlx::core::array(token_id, mlx::core::int64);
    auto embedded = mlx::core::reshape(
        mlx::core::take(*embed_tokens, ids, 0, device), {1, 1, embed_tokens->shape(1)}, device);

    mlx::core::array token_out(0);
    int64_t token_id_out = 0;
    std::vector<mlx::core::array> k_out, v_out;
    if (!plugin->forward_greedy_from_hidden(
            embedded, layers, kvs, *norm, *lm_head, offset, scale, head_dim, *rope_freqs, eps,
            true, device, token_out, token_id_out, k_out, v_out, error)) {
      return nx::nif::error(env, error.c_str());
    }

    return nx::nif::ok(
        env,
        enif_make_tuple2(env, nx::nif::make(env, token_id_out), llama_wrap_kv_terms(env, k_out, v_out)));
  }
  CATCH()
}
ASYNC_NIF(llama_forward_greedy_token_id)

NIF(llama_forward_greedy_ids_chunk) {
  ERL_NIF_TERM plugin_error;
  const emlx_llama_plugin::VTable *plugin = llama_plugin(env, &plugin_error);
  if (plugin == nullptr) {
    return plugin_error;
  }

  TENSOR_PARAM(0, input_ids);
  TENSOR_PARAM(1, embed_tokens);
  PARAM(6, int, offset);
  PARAM(7, int, count);
  PARAM(8, double, scale);
  PARAM(9, int, head_dim);
  TENSOR_PARAM(10, rope_freqs);
  PARAM(11, double, eps);
  DEVICE_PARAM(12, device);

  try {
    LlamaTensorHandles handles;
    ERL_NIF_TERM ref_error = 0;
    mlx::core::array *norm = nullptr;
    mlx::core::array *lm_head = nullptr;
    if (!llama_get_tensor(env, argv[4], &norm, handles, &ref_error)) {
      return llama_ref_error_or(env, ref_error, "llama_forward_greedy_ids_chunk expects norm tensor ref");
    }
    if (!llama_get_tensor(env, argv[5], &lm_head, handles, &ref_error)) {
      return llama_ref_error_or(env, ref_error, "llama_forward_greedy_ids_chunk expects lm_head tensor ref");
    }

    std::vector<emlx_llama_plugin::LayerParams> layers;
    std::vector<emlx_llama_plugin::KVCache> kvs;
    llama_get_layers_and_kv(env, argv[2], argv[3], layers, kvs, handles, "llama_forward_greedy_ids_chunk");

    std::string error;
    std::vector<mlx::core::array> token_out, k_out, v_out;
    if (!plugin->forward_greedy_ids_chunk(
            *input_ids, *embed_tokens, layers, kvs, *norm, *lm_head, offset, count,
            scale, head_dim, *rope_freqs, eps, device, token_out, k_out, v_out, error)) {
      return nx::nif::error(env, error.c_str());
    }

    std::vector<ERL_NIF_TERM> token_terms;
    token_terms.reserve(token_out.size());
    for (const auto &token : token_out) {
      token_terms.push_back(create_tensor_resource(env, token));
    }

    return nx::nif::ok(
        env,
        enif_make_tuple2(
            env,
            enif_make_list_from_array(env, token_terms.data(), token_terms.size()),
            llama_wrap_kv_terms(env, k_out, v_out)));
  }
  CATCH()
}
ASYNC_NIF(llama_forward_greedy_ids_chunk)

NIF(llama_final_greedy) {
  ERL_NIF_TERM plugin_error;
  const emlx_llama_plugin::VTable *plugin = llama_plugin(env, &plugin_error);
  if (plugin == nullptr) {
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
    if (!plugin->final_greedy(*hidden, *norm, *lm_head, eps, device, out, error)) {
      return nx::nif::error(env, error.c_str());
    }

    TENSOR(out);
  }
  CATCH()
}
ASYNC_NIF(llama_final_greedy)
