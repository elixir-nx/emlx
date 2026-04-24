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

  ## Inherited queue and `EMLX.Backend`

  Every EMLX NIF call (including those made from inside `EMLX.Backend`
  callbacks) inherits the bound queue automatically through
  `EMLX.resolve_worker/1`. This means that if your code calls a standard
  `Nx` operation inside a `with_queue/2` block — even indirectly via a
  library that uses `EMLX.Backend` — the work routes through the bound
  queue. This is almost always the desired behaviour (e.g. all ops in a
  Bumblebee forward pass go to the same Metal command queue).

  There is deliberately **no opt-out mechanism**: if you need work to run on
  a different queue, open a nested `with_queue/2` for the inner scope. The
  inner binding shadows the outer one for its duration and is restored on
  exit.

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
  Like `new/1` but raises `EMLX.NIFError` on failure instead of returning
  `{:error, reason}`.

  Useful in one-liner contexts (e.g. `__partitions_options__/1`) where the
  caller has no reasonable recovery path if the device is unavailable.
  """
  @spec new!(:cpu | :gpu) :: t()
  def new!(device) do
    case new(device) do
      {:ok, q} -> q
      {:error, reason} -> raise(EMLX.NIFError, List.to_string(reason))
    end
  end

  @doc """
  Blocks the calling process until all previously enqueued jobs on `queue`
  have finished and MLX has flushed its GPU command buffer.

  Internally posts a barrier job that calls `mlx::core::synchronize(stream)`
  on the worker thread, then blocks in `receive` until the reply arrives.
  """
  @spec synchronize(t()) :: :ok
  def synchronize(%__MODULE__{ref: ref}) do
    case EMLX.NIF.command_queue_synchronize(ref) do
      {:ok, job_ref} ->
        receive do
          {^job_ref, {:ok, _}} -> :ok
          {^job_ref, {:error, reason}} -> raise(EMLX.NIFError, List.to_string(reason))
        end

      {:error, reason} ->
        raise(EMLX.NIFError, List.to_string(reason))
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
