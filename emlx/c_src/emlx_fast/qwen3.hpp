#pragma once

#include <erl_nif.h>

// Qwen3 native accelerators currently hosted by emlx and called from
// emlx_axon. Keep declarations here so the model native boundary is explicit
// and easier to extract into an emlx_axon extension later.
ERL_NIF_TERM qwen3_kv_cache_attention_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM qwen3_mlp_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM qwen3_layer_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM qwen3_layer_quantized_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM qwen3_forward_greedy_ids_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM qwen3_forward_greedy_ids_chunk_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM qwen3_forward_greedy_ids_chunk_quantized_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM qwen3_forward_greedy_ids_token_id_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM qwen3_forward_greedy_token_id_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM qwen3_final_greedy_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM qwen3_attention_residual_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM qwen3_attention_block_async(ErlNifEnv *, int, const ERL_NIF_TERM []);

// load_qwen3_plugin — `dlopen`s the standalone qwen3 compute plugin
// (libemlx_qwen3.so). Not worker-routed: no `_async` wrapper/argv[0] worker
// ref, same as `command_queue_new`.
ERL_NIF_TERM load_qwen3_plugin(ErlNifEnv *, int, const ERL_NIF_TERM []);
