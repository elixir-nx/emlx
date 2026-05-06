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

  # ── Graph capture / replay ─────────────────────────────────────────────────
  # graph_capture/3 — walks the lazy MLX DAG from `outputs` back to `inputs`,
  # builds and simplifies a replayable tape, and returns an opaque compiled_ref.
  # Must be called while the arrays are still lazy (before eval).
  #
  # graph_replay/2 — substitutes `new_inputs` into the stored tape and returns
  # new lazy output arrays without re-dispatching any Nx ops.

  def graph_capture(_inputs, _outputs, _shapeless) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def graph_replay(_compiled_ref, _new_inputs) do
    :erlang.nif_error(:nif_not_loaded)
  end
end
