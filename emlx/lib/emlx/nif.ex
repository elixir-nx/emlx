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
  # op_names is a list of strings matching C++ op_registry keys (e.g. "add").
  # Arity = 1 (worker) + 8 args = 9 registered.
  def compile_program(
        _worker,
        _num_inputs,
        _capture_refs,
        _const_values,
        _const_types,
        _op_names,
        _operands,
        _iattrs,
        _output_refs
      ) do
    :erlang.nif_error(:nif_not_loaded)
  end

  # eval_program — replays a compiled ProgramResource against runtime inputs.
  # Worker-routed (argv[0] = worker). Returns a list of output MLX array refs.
  # Arity = 1 (worker) + 2 = 3 registered.
  def eval_program(_worker, _program_ref, _input_refs) do
    :erlang.nif_error(:nif_not_loaded)
  end

end
