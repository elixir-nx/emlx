#pragma once

// ABI shared between the host NIF library (emlx's c_src/emlx_fast/qwen3.cpp,
// statically linked into libemlx.so) and the standalone qwen3 compute
// plugin (emlx_axon's c_src/qwen3_plugin.cpp, built as its own shared
// library and loaded at runtime via `EMLX.NIF.load_plugin("qwen3", path)`).
//
// This header lives in emlx (not emlx_axon) because it is the contract the
// *host* decode/encode side depends on — emlx_axon's plugin build adds
// emlx's `c_src/emlx_fast` to its include path to pick it up (see
// emlx_axon/Makefile).
//
// This header intentionally has NO dependency on erl_nif.h: every type here
// is plain C++/MLX so the plugin can be built and `dlopen`'d without ever
// linking against the Erlang runtime. The host decodes Erlang terms into
// these plain structs (see qwen3.cpp's `qwen3_get_*` helpers) before calling
// through `VTable`, and re-encodes the `mlx::core::array` results back into
// tensor resources afterwards. This split works only because the boundary
// carries raw `mlx::core::array`/struct values, never Erlang resource
// terms — a truly independent NIF module can't share tensor resource types
// with the rest of EMLX (Erlang ties resource types to the specific
// module/`.so` that opened them).

#include "mlx/mlx.h"

#include <cstdint>
#include <string>
#include <vector>

namespace emlx_qwen3_plugin {

// A dense-or-quantized projection weight — mirrors `EMLX.Native.Qwen3`'s
// `{:dense, ref}` / `{:quantized, ...}` linear weight term after decoding.
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

// Dense-only per-layer projection set (qwen3_layer / qwen3_forward_greedy_*).
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

// Generalized (dense-or-quantized) per-layer projection set
// (qwen3_layer_quantized / qwen3_forward_greedy_ids_chunk_quantized).
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

// Every entrypoint validates its own inputs and returns `false` + fills
// `error` on failure instead of throwing — C++ exceptions are not relied
// on to cross the `dlopen` boundary.
struct VTable {
  bool (*kv_cache_attention)(
      const mlx::core::array &q, const mlx::core::array &new_k,
      const mlx::core::array &new_v, mlx::core::array &k_cache,
      mlx::core::array &v_cache, int offset, double scale, int head_dim,
      double theta, const mlx::core::Device &device, mlx::core::array &out,
      mlx::core::array &k_upd, mlx::core::array &v_upd, std::string &error);

  bool (*mlp)(const mlx::core::array &hidden, const mlx::core::array &norm,
              const mlx::core::array &gate_proj,
              const mlx::core::array &up_proj,
              const mlx::core::array &down_proj, double eps,
              const mlx::core::Device &device, mlx::core::array &out,
              std::string &error);

  bool (*attention_residual)(
      const mlx::core::array &hidden, const mlx::core::array &attn_out,
      const mlx::core::array &o_proj, const mlx::core::Device &device,
      mlx::core::array &out, std::string &error);

  bool (*attention_block)(
      const mlx::core::array &hidden, const mlx::core::array &norm,
      const mlx::core::array &q_proj, const mlx::core::array &k_proj,
      const mlx::core::array &v_proj, const mlx::core::array &o_proj,
      const mlx::core::array &q_norm, const mlx::core::array &k_norm,
      mlx::core::array &k_cache, mlx::core::array &v_cache, int offset,
      double scale, int head_dim, double theta, double eps,
      const mlx::core::Device &device, mlx::core::array &out,
      mlx::core::array &k_upd, mlx::core::array &v_upd, std::string &error);

  bool (*layer_dense)(const mlx::core::array &hidden,
                       const LayerParams &layer, KVCache &kv, int offset,
                       double scale, int head_dim, double theta, double eps,
                       const mlx::core::Device &device, mlx::core::array &out,
                       mlx::core::array &k_upd, mlx::core::array &v_upd,
                       std::string &error);

  bool (*layer_quantized)(const mlx::core::array &hidden,
                           const LayerParamsQ &layer, KVCache &kv, int offset,
                           double scale, int head_dim, double theta,
                           double eps, const mlx::core::Device &device,
                           mlx::core::array &out, mlx::core::array &k_upd,
                           mlx::core::array &v_upd, std::string &error);

  bool (*final_greedy)(const mlx::core::array &hidden,
                        const mlx::core::array &norm,
                        const mlx::core::array &lm_head, double eps,
                        const mlx::core::Device &device, mlx::core::array &out,
                        std::string &error);

  // Dense greedy forward from an already-embedded hidden state through
  // every layer, then final norm + lm_head + argmax. `layers`/`kv` are
  // parallel per-layer vectors; `kv[i]` is updated in place with the new
  // K/V cache and also returned via `k_out`/`v_out` for the caller to wrap
  // as tensor resources. Exactly one of `token_out`/`token_id_out` is
  // meaningful, selected by `return_token_id`.
  bool (*forward_greedy_from_hidden)(
      const mlx::core::array &hidden, std::vector<LayerParams> &layers,
      std::vector<KVCache> &kv, const mlx::core::array &norm,
      const mlx::core::array &lm_head, int offset, double scale,
      int head_dim, double theta, double eps, bool return_token_id,
      const mlx::core::Device &device, mlx::core::array &token_out,
      int64_t &token_id_out, std::vector<mlx::core::array> &k_out,
      std::vector<mlx::core::array> &v_out, std::string &error);

  // Dense chunked decode: `count` greedy steps starting from `input_ids`
  // ({1,1}), threading the KV cache across steps without returning to
  // Elixir in between.
  bool (*forward_greedy_ids_chunk)(
      const mlx::core::array &input_ids, const mlx::core::array &embed_tokens,
      std::vector<LayerParams> &layers, std::vector<KVCache> &initial_kv,
      const mlx::core::array &norm, const mlx::core::array &lm_head,
      int offset, int count, double scale, int head_dim, double theta,
      double eps, const mlx::core::Device &device,
      std::vector<mlx::core::array> &token_out,
      std::vector<mlx::core::array> &k_out,
      std::vector<mlx::core::array> &v_out, std::string &error);

  // Generalized (dense-or-quantized) chunked decode.
  bool (*forward_greedy_ids_chunk_quantized)(
      const mlx::core::array &input_ids, const mlx::core::array &embed_tokens,
      std::vector<LayerParamsQ> &layers, std::vector<KVCache> &initial_kv,
      const mlx::core::array &norm, const LinearWeight &lm_head, int offset,
      int count, double scale, int head_dim, double theta, double eps,
      const mlx::core::Device &device,
      std::vector<mlx::core::array> &token_out,
      std::vector<mlx::core::array> &k_out,
      std::vector<mlx::core::array> &v_out, std::string &error);
};

} // namespace emlx_qwen3_plugin

// Sole exported entrypoint — `EMLX.NIF.load_plugin("qwen3", path)`
// `dlopen`s the plugin and `dlsym`s the generic `emlx_plugin_vtable` symbol
// (see emlx/c_src/emlx_plugin_registry.hpp) to obtain this vtable.
extern "C" const emlx_qwen3_plugin::VTable *emlx_plugin_vtable();
