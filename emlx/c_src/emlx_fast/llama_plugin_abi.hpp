#pragma once

// ABI shared between emlx's host NIF library and the standalone Llama compute
// plugin built by emlx_axon.

#include "mlx/mlx.h"

#include <cstdint>
#include <string>
#include <vector>

namespace emlx_llama_plugin {

struct LayerParams {
  mlx::core::array *norm1;
  mlx::core::array *norm2;
  mlx::core::array *q_proj;
  mlx::core::array *k_proj;
  mlx::core::array *v_proj;
  mlx::core::array *o_proj;
  mlx::core::array *gate_proj;
  mlx::core::array *up_proj;
  mlx::core::array *down_proj;
};

struct KVCache {
  mlx::core::array *k;
  mlx::core::array *v;
};

struct VTable {
  bool (*layer_dense)(
      const mlx::core::array &hidden, const LayerParams &layer, KVCache &kv,
      int offset, double scale, int head_dim,
      const mlx::core::array &rope_freqs, double eps,
      const mlx::core::Device &device, mlx::core::array &out,
      mlx::core::array &k_upd, mlx::core::array &v_upd,
      std::string &error);

  bool (*forward_greedy_from_hidden)(
      const mlx::core::array &hidden, std::vector<LayerParams> &layers,
      std::vector<KVCache> &kv, const mlx::core::array &norm,
      const mlx::core::array &lm_head, int offset, double scale,
      int head_dim, const mlx::core::array &rope_freqs, double eps,
      bool return_token_id, const mlx::core::Device &device,
      mlx::core::array &token_out, int64_t &token_id_out,
      std::vector<mlx::core::array> &k_out,
      std::vector<mlx::core::array> &v_out, std::string &error);

  bool (*forward_greedy_ids_chunk)(
      const mlx::core::array &input_ids, const mlx::core::array &embed_tokens,
      std::vector<LayerParams> &layers, std::vector<KVCache> &initial_kv,
      const mlx::core::array &norm, const mlx::core::array &lm_head,
      int offset, int count, double scale, int head_dim,
      const mlx::core::array &rope_freqs, double eps,
      const mlx::core::Device &device,
      std::vector<mlx::core::array> &token_out,
      std::vector<mlx::core::array> &k_out,
      std::vector<mlx::core::array> &v_out, std::string &error);

  bool (*final_greedy)(
      const mlx::core::array &hidden, const mlx::core::array &norm,
      const mlx::core::array &lm_head, double eps,
      const mlx::core::Device &device, mlx::core::array &out,
      std::string &error);
};

} // namespace emlx_llama_plugin

extern "C" const emlx_llama_plugin::VTable *emlx_plugin_vtable();
