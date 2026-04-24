defmodule EMLX.CommandQueue do
  @moduledoc """
  A handle to an `emlx::Worker` OS thread + `mlx::core::Stream` pair.

  Each `CommandQueue` wraps a NIF resource reference and records the device
  (`:cpu` or `:gpu`) the underlying stream was created for. The device tag is
  necessary because the raw NIF reference carries no device metadata, and
  `EMLX.resolve_worker/1` needs it to decide whether cross-device promotion
  applies.

  ## Usage

      {:ok, q} = EMLX.CommandQueue.new(:gpu)

      EMLX.CommandQueue.with_queue(q, fn ->
        # All EMLX / Nx operations in this block route through `q`
        Nx.add(a, b)
      end)

  ## Process binding

  `with_queue/2` stores `{ref, device}` in the calling process's dictionary
  under `:emlx_command_queue` for the duration of the given function, then
  restores the previous value (supporting nested calls). `EMLX.resolve_worker/1`
  reads this key to bypass the application-default worker.

  ## Cross-device promotion

  When a tensor's device does not match the bound queue's device, the
  behaviour is controlled by application config:

      config :emlx,
        cross_device_promotion: false,  # default: fall back to device-default worker
        warn_cross_device: false        # default: no Logger.warning on mismatch

  See `EMLX.resolve_worker/1` for the promotion logic.
  """

  @enforce_keys [:ref, :device]
  defstruct [:ref, :device]

  @type t :: %__MODULE__{ref: reference(), device: :cpu | :gpu}

  @doc """
  Allocates a new `CommandQueue` for `device`.

  Spawns a dedicated OS thread and a new `mlx::core::Stream` on it. Returns
  `{:error, reason}` if the NIF cannot allocate the stream (e.g. `:gpu` on a
  system without Metal).
  """
  @spec new(:cpu | :gpu) :: {:ok, t()} | {:error, list()}
  def new(device) do
    case EMLX.NIF.command_queue_new(device) do
      {:ok, ref} -> {:ok, %__MODULE__{ref: ref, device: device}}
      {:error, _} = err -> err
    end
  end

  @doc """
  Runs `fun` with `queue` bound to the calling process for the duration.

  Stores `{ref, device}` under `:emlx_command_queue` in the process
  dictionary, then restores the previous value (or deletes the key) in an
  `after` block. Nested calls are safe — each level saves and restores
  independently.
  """
  @spec with_queue(t(), (-> result)) :: result when result: term()
  def with_queue(%__MODULE__{ref: ref, device: device}, fun) when is_function(fun, 0) do
    previous = Process.get(:emlx_command_queue)
    Process.put(:emlx_command_queue, {ref, device})

    try do
      fun.()
    after
      if previous do
        Process.put(:emlx_command_queue, previous)
      else
        Process.delete(:emlx_command_queue)
      end
    end
  end
end
