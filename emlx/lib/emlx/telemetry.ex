defmodule EMLX.Telemetry do
  @moduledoc """
  `:telemetry` events emitted by EMLX.

  All span-style events use `:telemetry.span/3` semantics, so attaching to
  `*:start`, `*:stop`, and `*:exception` is sufficient for histograms and
  error tracking.

  ## Events

  ### Evaluation boundaries

  `[:emlx, :eval, :start | :stop | :exception]` — `EMLX.eval/1`. Spans the
  full round-trip through the resolved `EMLX.CommandQueue` worker, including
  the blocking wait for the worker to finish `mlx::core::eval/1` (the real
  latency, not just NIF dispatch). The `:stop` event carries `:duration`
  (monotonic native units).

  `[:emlx, :to_binary, :start | :stop | :exception]` — `EMLX.Backend.to_binary/2`
  (the `Nx.to_binary/1` path). Spans the sync-forcing blob copy. Metadata:
  `:shape`, `:dtype`, `:byte_size` (byte size of the binary actually
  returned to the caller).

  ### Memory stats (poll-driven)

  `[:emlx, :memory, :stats]` — discrete event, not a span. Call
  `EMLX.Telemetry.memory_stats/0` to sample; measurements:

    * `:active_memory` — bytes currently allocated and in use
    * `:peak_memory` — highest active memory since the last
      `EMLX.reset_peak_memory/0`
    * `:cache_memory` — bytes in the allocator cache (freed but not
      returned to the OS)

  Wire this into a periodic task (e.g. `Process.send_after/3` loop) to
  graph memory drift in a long-running serving.

  ## Attaching a handler

      :telemetry.attach(
        "emlx-eval-log",
        [:emlx, :eval, :stop],
        fn _event, measurements, _metadata, _config ->
          IO.inspect(measurements.duration)
        end,
        nil
      )
  """

  @doc false
  def span_eval(fun) do
    :telemetry.span([:emlx, :eval], %{}, fn -> {fun.(), %{}} end)
  end

  @doc false
  def span_to_binary(%Nx.Tensor{shape: shape, type: type}, fun) do
    start_metadata = %{shape: shape, dtype: type}

    :telemetry.span([:emlx, :to_binary], start_metadata, fn ->
      binary = fun.()
      {binary, %{shape: shape, dtype: type, byte_size: byte_size(binary)}}
    end)
  end

  @doc """
  Sample the MLX allocator and emit `[:emlx, :memory, :stats]`.

  Returns the measurements map so callers can also log or plot inline.

  ## Examples

      iex> stats = EMLX.Telemetry.memory_stats()
      iex> Map.keys(stats) |> Enum.sort()
      [:active_memory, :cache_memory, :peak_memory]

  """
  def memory_stats do
    stats = EMLX.memory_info()
    :telemetry.execute([:emlx, :memory, :stats], stats, %{})
    stats
  end
end
