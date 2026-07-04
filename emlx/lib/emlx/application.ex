defmodule EMLX.Application do
  @moduledoc """
  OTP Application for EMLX.

  Allocates the application-default `EMLX.CommandQueue` (one per device:
  `:cpu` and `:gpu`) at boot and stashes the NIF resource references in
  `:persistent_term`. These are the workers used by `EMLX.eval/1` and
  `EMLX.to_blob/1` for any process that has not bound its own queue via
  `EMLX.CommandQueue.with_queue/2`.

  Also allocates one *runtime_call worker* per device — see
  `runtime_call_worker/1`'s doc for why a `Nx.runtime_call/4` callback's own
  eager EMLX calls must never route back to the worker that is, right now,
  blocked waiting for that very callback to return.

  See `clean-room-import/01-worker-thread-dispatch.md` for the rationale
  behind `:persistent_term` instead of a `GenServer` + `Registry`.

  ## Idempotency

  `start/2` is safe to call more than once (e.g. from an umbrella where
  `:emlx` is referenced by several apps). The first call to allocate a
  worker for a given device wins; subsequent calls observe the existing
  `:persistent_term` entry and skip. This matters because every
  `:persistent_term.put/2` triggers a global GC scan across every
  process heap on the node — we do exactly two puts (CPU + GPU) over
  the BEAM's lifetime.

  ## GPU absence

  On platforms where MLX cannot allocate a GPU stream (e.g. Linux without
  Metal), `EMLX.NIF.command_queue_new(:gpu)` returns `{:error, _}` and
  this module silently skips the GPU worker. Subsequent
  `EMLX.eval/1` calls on a GPU tensor will raise at use time with the
  underlying `:persistent_term` `ArgumentError`.
  """

  use Application

  @doc false
  def start(_type, _args) do
    EMLX.Profiling.init()
    ensure_default_worker!(:cpu, _gpu_optional? = false)
    ensure_default_worker!(:gpu, _gpu_optional? = true)
    ensure_worker!(:runtime_call_worker, :cpu, _gpu_optional? = false)
    ensure_worker!(:runtime_call_worker, :gpu, _gpu_optional? = true)
    ensure_qwen3_plugin_loaded!()
    Supervisor.start_link([], strategy: :one_for_one, name: __MODULE__)
  end

  @doc false
  def ensure_qwen3_plugin_loaded! do
    path = :filename.join(:code.priv_dir(:emlx), ~c"libemlx_qwen3.so")

    case EMLX.NIF.load_qwen3_plugin(List.to_string(path)) do
      :ok ->
        :ok

      {:error, reason} ->
        raise EMLX.NIFError,
              "EMLX.Application could not load the qwen3 plugin: " <> List.to_string(reason)
    end
  end

  @doc """
  Returns the application-default `EMLX.CommandQueue` NIF resource for
  the given device.

  Raises `ArgumentError` if no worker has been allocated for the device
  (e.g. asking for `:gpu` on a system without Metal).

  **Never** call `:persistent_term.put/2` on the underlying key
  (`{EMLX, :default_worker, device}`) — overwriting a `:persistent_term`
  value triggers a node-wide GC.
  """
  @spec default_worker(:cpu | :gpu) :: reference()
  def default_worker(device) when device in [:cpu, :gpu] do
    :persistent_term.get(persistent_term_key(:default_worker, device))
  end

  @doc """
  Returns the `EMLX.CommandQueue` NIF resource dedicated to running
  `Nx.runtime_call/4` callbacks for the given device.

  `EMLX.handle_runtime_call/5` binds *this* worker (via
  `EMLX.CommandQueue.with_queue/2`'s same `Process.put(:emlx_command_queue,
  _)` mechanism) for the duration of every real callback invocation, so any
  eager EMLX call the callback itself makes (e.g.
  `EMLX.Quantization.dequantize_callback/2` calling `EMLX.dequantize/1`)
  gets worker-routed *here* instead of to the default (or bound) worker that
  is, right now, blocked inside `EMLXRuntimeCall::eval_cpu`/`eval_gpu`
  waiting for that very callback to finish. Reusing that same, still-busy
  worker would either deadlock (its one dedicated OS thread cannot service
  a new job while blocked — see `emlx_worker.hpp`) or, if the blocked wait
  instead pumps that worker's own queue inline, crash (MLX's GPU `eval()`
  is not reentrant on one OS thread — nesting a second `mlx::core::eval`
  call while the first is still on the C++ stack corrupts Metal
  command-buffer/completion-handler state). A second, otherwise-idle
  worker sidesteps both failure modes: it is a distinct OS thread with its
  own `mlx::core::Stream`, so its jobs never contend with (or nest inside)
  the blocked worker's.

  Raises `ArgumentError` if no worker has been allocated for the device
  (e.g. asking for `:gpu` on a system without Metal) — same contract as
  `default_worker/1`.
  """
  @spec runtime_call_worker(:cpu | :gpu) :: reference()
  def runtime_call_worker(device) when device in [:cpu, :gpu] do
    :persistent_term.get(persistent_term_key(:runtime_call_worker, device))
  end

  defp ensure_default_worker!(device, gpu_optional?) do
    ensure_worker!(:default_worker, device, gpu_optional?)
  end

  defp ensure_worker!(kind, device, gpu_optional?) do
    key = persistent_term_key(kind, device)

    case :persistent_term.get(key, :unset) do
      :unset ->
        case EMLX.NIF.command_queue_new(device) do
          {:ok, ref} ->
            :persistent_term.put(key, ref)

          {:error, _reason} when gpu_optional? ->
            :ok

          {:error, reason} ->
            raise EMLX.NIFError,
                  "EMLX.Application could not allocate #{kind} #{device} worker: " <>
                    List.to_string(reason)
        end

      _existing ->
        :ok
    end
  end

  defp persistent_term_key(kind, device), do: {EMLX, kind, device}
end
