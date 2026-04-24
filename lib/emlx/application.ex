defmodule EMLX.Application do
  @moduledoc """
  OTP Application for EMLX.

  Allocates the application-default `EMLX.CommandQueue` (one per device:
  `:cpu` and `:gpu`) at boot and stashes the NIF resource references in
  `:persistent_term`. These are the workers used by `EMLX.eval/1` and
  `EMLX.to_blob/1` for any process that has not bound its own queue via
  `EMLX.CommandQueue.with_queue/2`.

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
    ensure_default_worker!(:cpu, _gpu_optional? = false)
    ensure_default_worker!(:gpu, _gpu_optional? = true)
    Supervisor.start_link([], strategy: :one_for_one, name: __MODULE__)
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
    :persistent_term.get(persistent_term_key(device))
  end

  defp ensure_default_worker!(device, gpu_optional?) do
    key = persistent_term_key(device)

    case :persistent_term.get(key, :unset) do
      :unset ->
        case EMLX.NIF.command_queue_new(device) do
          {:ok, ref} ->
            :persistent_term.put(key, ref)

          {:error, _reason} when gpu_optional? ->
            :ok

          {:error, reason} ->
            raise EMLX.NIFError,
                  "EMLX.Application could not allocate default #{device} worker: " <>
                    List.to_string(reason)
        end

      _existing ->
        :ok
    end
  end

  defp persistent_term_key(device), do: {EMLX, :default_worker, device}
end
