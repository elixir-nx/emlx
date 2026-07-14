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
  @doc """
  Loads the EMLX native image.

  Reloading the Elixir module with the same native image preserves MLX arrays,
  command queues, compiled programs, and plugin callbacks. Replacing the native
  image still requires restarting the BEAM VM because that state lives for the
  lifetime of the process.
  """
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
  # `program` is an `EMLX.Native.Program.t()` (see
  # EMLX.Native.Expr.to_native/1), decoded directly by `fine` on the C++ side
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

  # load_plugin — `dlopen`s a standalone, name-keyed native plugin (no
  # erl_nif dependency), validates its generic descriptor, and publishes its
  # callback map in the process-lifetime registry (see
  # emlx_plugin_registry.hpp). `load_plugin/2` is the compatibility/expert form
  # without an expected build identity. Packaged plugins use `load_plugin/3` so
  # stale artifacts are rejected. Native calls fail until their plugin has been
  # registered — for qwen3, `EMLXAxon.Application` loads it at boot. Loading is
  # not worker-routed because `dlopen` performs no MLX graph work.
  def load_plugin(_name, _path) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc false
  def load_plugin(_name, _path, _expected_plugin_build_id) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc false
  def call_plugin(_worker, _plugin, _callback, _operands, _attrs, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end
end
