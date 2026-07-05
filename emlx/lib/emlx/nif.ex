defmodule EMLX.NIF do
  @moduledoc """
  Elixir bindings for MLX array operations.
  """

  for {name, arity} <- EMLX.__mlx_functions__() do
    args = Macro.generate_arguments(arity, __MODULE__)

    def unquote(name)(unquote_splicing(args)) do
      :erlang.nif_error(:nif_not_loaded)
    end
  end

  @on_load :load_nifs
  def load_nifs do
    path = :filename.join(:code.priv_dir(:emlx), ~c"libemlx")
    :erlang.load_nif(path, 0)
  end

  # Worker-routed NIF stubs. The first argument is always an
  # EMLX.CommandQueue resource ref; the C++ wrapper (emlx_async.hpp)
  # extracts it and posts the rest to the worker thread, returning a
  # job ref for the caller to `receive` on.
  def to_blob(_worker, _tensor) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def to_blob(_worker, _tensor, _limit) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def eval(_worker, _tensor_ref) do
    :erlang.nif_error(:nif_not_loaded)
  end

  # See the comment on the C++ side (emlx_nif.cpp).
  def eval_many(_worker, _tensor_refs) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def to_device(_worker, _tensor, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def item(_worker, _tensor_ref) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def memory_info do
    :erlang.nif_error(:nif_not_loaded)
  end

  def clear_cache do
    :erlang.nif_error(:nif_not_loaded)
  end

  def reset_peak_memory do
    :erlang.nif_error(:nif_not_loaded)
  end

  def set_memory_limit(_limit) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def set_cache_limit(_limit) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def metal_start_capture(_path) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def metal_stop_capture do
    :erlang.nif_error(:nif_not_loaded)
  end

  def tensor_data_ptr(_tensor) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def array_from_ptr(_addr, _shape, _dtype, _byte_size, _deleter) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def tensor_to_shm(_worker, _tensor, _permissions) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def array_from_shm(_name, _shape, _dtype, _byte_size) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def shm_unlink_handle(_name) do
    :erlang.nif_error(:nif_not_loaded)
  end

  # ── Worker / EMLX.CommandQueue control NIFs ───────────────────────────────
  # `command_queue_post_eval` and `command_queue_post_to_blob` were folded
  # into the generic async dispatch path; their public entry points are
  # `eval/2` and `to_blob/{2,3}` above.

  def command_queue_new(_device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def command_queue_synchronize(_queue_ref) do
    :erlang.nif_error(:nif_not_loaded)
  end

  # ── Native.Expr compiler NIFs ─────────────────────────────────────────────
  # compile_program — builds an opaque ProgramResource from the Native.Expr IR
  # and a set of captured arrays. Worker-routed (argv[0] = worker).
  # `program` is an `EMLX.Native.Wire.Program.t()` (see
  # EMLX.Native.Expr.to_wire/1), decoded directly by `fine` on the C++ side
  # (emlx_compiler.hpp's `Program`/`Instruction` structs) instead of manually
  # parsed positional args.
  # Arity = 1 (worker) + 1 = 2 registered.
  def compile_program(_worker, _program) do
    :erlang.nif_error(:nif_not_loaded)
  end

  # eval_program — replays a compiled ProgramResource against runtime inputs.
  # Worker-routed (argv[0] = worker). Returns a list of output MLX array refs.
  # Arity = 1 (worker) + 2 = 3 registered.
  def eval_program(_worker, _program_ref, _input_refs) do
    :erlang.nif_error(:nif_not_loaded)
  end

  # resolve_runtime_call — delivers the real Elixir callback's reply for one
  # in-flight Nx.runtime_call/4 round trip (see EMLX.await_worker/2 and
  # EMLX.Native.Expr's moduledoc "Runtime calls" section), waking the worker
  # thread blocked inside EMLXRuntimeCall::eval_cpu/eval_gpu. NOT
  # worker-routed — no argv[0] worker ref; runs directly on the calling
  # scheduler thread.
  def resolve_runtime_call(_pending, _status, _result) do
    :erlang.nif_error(:nif_not_loaded)
  end

  # ── Qwen3 fused native NIFs ────────────────────────────────────────────────
  # Elixir wrappers live in `EMLX.Native.Qwen3` (without the `qwen3_` prefix);
  # the NIF/C++ names below keep it (`emlx/c_src/emlx_fast/qwen3.cpp`).
  # Worker-routed (argv[0] = worker), same pattern as the `@mlx_function`
  # macro-generated stubs above.

  def qwen3_kv_cache_attention(
        _worker,
        _q,
        _k,
        _v,
        _k_cache,
        _v_cache,
        _offset,
        _scale,
        _head_dim,
        _theta,
        _device
      ) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def qwen3_mlp(_worker, _hidden, _norm, _gate, _up, _down, _eps, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def qwen3_attention_residual(_worker, _hidden, _attn, _o_proj, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def qwen3_layer(
        _worker,
        _hidden,
        _norm1,
        _q,
        _k,
        _v,
        _o,
        _q_norm,
        _k_norm,
        _k_cache,
        _v_cache,
        _norm2,
        _gate,
        _up,
        _down,
        _offset,
        _scale,
        _head_dim,
        _theta,
        _eps,
        _device
      ) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def qwen3_layer_quantized(
        _worker,
        _hidden,
        _norm1,
        _q,
        _k,
        _v,
        _o,
        _q_norm,
        _k_norm,
        _k_cache,
        _v_cache,
        _norm2,
        _gate,
        _up,
        _down,
        _offset,
        _scale,
        _head_dim,
        _theta,
        _eps,
        _device
      ) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def qwen3_forward_greedy_ids(
        _worker,
        _ids,
        _embed,
        _layers,
        _kv_cache,
        _norm,
        _lm_head,
        _offset,
        _scale,
        _head_dim,
        _theta,
        _eps,
        _device
      ) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def qwen3_forward_greedy_ids_chunk(
        _worker,
        _ids,
        _embed,
        _layers,
        _kv_cache,
        _norm,
        _lm_head,
        _offset,
        _count,
        _scale,
        _head_dim,
        _theta,
        _eps,
        _device
      ) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def qwen3_forward_greedy_ids_chunk_quantized(
        _worker,
        _ids,
        _embed,
        _layers,
        _kv_cache,
        _norm,
        _lm_head,
        _offset,
        _count,
        _scale,
        _head_dim,
        _theta,
        _eps,
        _device
      ) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def qwen3_forward_greedy_ids_token_id(
        _worker,
        _ids,
        _embed,
        _layers,
        _kv_cache,
        _norm,
        _lm_head,
        _offset,
        _scale,
        _head_dim,
        _theta,
        _eps,
        _device
      ) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def qwen3_forward_greedy_token_id(
        _worker,
        _token_id,
        _embed,
        _layers,
        _kv_cache,
        _norm,
        _lm_head,
        _offset,
        _scale,
        _head_dim,
        _theta,
        _eps,
        _device
      ) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def qwen3_final_greedy(_worker, _hidden, _norm, _lm_head, _eps, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def qwen3_attention_block(
        _worker,
        _hidden,
        _norm,
        _q,
        _k,
        _v,
        _o,
        _q_norm,
        _k_norm,
        _k_cache,
        _v_cache,
        _offset,
        _scale,
        _head_dim,
        _theta,
        _eps,
        _device
      ) do
    :erlang.nif_error(:nif_not_loaded)
  end

  # load_plugin — `dlopen`s a standalone, name-keyed native plugin (no
  # erl_nif dependency) and caches its vtable under `name` (see
  # emlx_plugin_registry.hpp). Callers that decode/dispatch into a
  # specific plugin's ABI (e.g. `qwen3_*` NIFs above, which fetch the
  # "qwen3" plugin) error with `{:error, _}` until this has been called
  # successfully for that name — for qwen3, see `EMLXAxon.Application`,
  # which calls it eagerly at boot. Not worker-routed (no argv[0] worker
  # ref): `dlopen` does no MLX graph work.
  def load_plugin(_name, _path) do
    :erlang.nif_error(:nif_not_loaded)
  end
end
