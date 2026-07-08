#pragma once

#include <erl_nif.h>

ERL_NIF_TERM llama_layer_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM llama_forward_greedy_ids_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM llama_forward_greedy_ids_chunk_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM llama_forward_greedy_ids_token_id_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM llama_forward_greedy_token_id_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM llama_final_greedy_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
