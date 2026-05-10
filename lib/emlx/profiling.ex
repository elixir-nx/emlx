defmodule EMLX.Profiling do
  @moduledoc false

  # Lightweight atomic counters for profiling the MLX eval-dispatch path.
  #
  # Enabled via `config :emlx, :profile_eval, true` in `config/runtime.exs`
  # (or by setting EMLX_PROFILE_EVAL=1, which runtime.exs maps to that key).
  # The counters module guarantees atomic increments with ~10ns overhead per
  # call, which is negligible compared to NIF entry/exit cost (~1–5 µs).
  #
  # Counter indices:
  @eval_idx 1
  @item_idx 2
  @to_blob_idx 3
  @num_counters 3

  @pt_key {EMLX, :profiling_counters}

  @doc """
  Called once from `EMLX.Application.start/2` to allocate the counter slab
  when `config :emlx, :profile_eval, true` is set. No-op otherwise.
  """
  def init do
    if Application.get_env(:emlx, :profile_eval, false) do
      ref = :counters.new(@num_counters, [:atomics])
      :persistent_term.put(@pt_key, ref)
      ref
    end
  end

  @doc "True if profiling counters are active."
  def enabled?, do: :persistent_term.get(@pt_key, nil) != nil

  @doc "Increment the `EMLX.eval/1` counter."
  def inc_eval do
    case :persistent_term.get(@pt_key, nil) do
      nil -> :ok
      ref -> :counters.add(ref, @eval_idx, 1)
    end
  end

  @doc "Increment the `EMLX.item/1` counter."
  def inc_item do
    case :persistent_term.get(@pt_key, nil) do
      nil -> :ok
      ref -> :counters.add(ref, @item_idx, 1)
    end
  end

  @doc "Increment the `EMLX.to_blob/1,2` counter."
  def inc_to_blob do
    case :persistent_term.get(@pt_key, nil) do
      nil -> :ok
      ref -> :counters.add(ref, @to_blob_idx, 1)
    end
  end

  @doc """
  Read current counter snapshot.

      iex> EMLX.Profiling.read()
      {:ok, %{eval: 42, item: 7, to_blob: 7}}
  """
  def read do
    case :persistent_term.get(@pt_key, nil) do
      nil ->
        {:error, :not_initialized}

      ref ->
        {:ok,
         %{
           eval: :counters.get(ref, @eval_idx),
           item: :counters.get(ref, @item_idx),
           to_blob: :counters.get(ref, @to_blob_idx)
         }}
    end
  end

  @doc "Zero all counters. Call between decode steps to get per-step counts."
  def reset do
    case :persistent_term.get(@pt_key, nil) do
      nil ->
        :ok

      ref ->
        :counters.put(ref, @eval_idx, 0)
        :counters.put(ref, @item_idx, 0)
        :counters.put(ref, @to_blob_idx, 0)
    end
  end

  @doc """
  Print a snapshot with an optional step label to stdout.

      EMLX.Profiling.snapshot("decode step 3")
  """
  def snapshot(label \\ "") do
    case read() do
      {:ok, counts} ->
        IO.puts(
          "[EMLX.Profiling#{if label != "", do: " #{label}", else: ""}] " <>
            "eval=#{counts.eval} item=#{counts.item} to_blob=#{counts.to_blob}"
        )

      {:error, :not_initialized} ->
        IO.puts(
          "[EMLX.Profiling] counters not initialized — set `config :emlx, :profile_eval, true` in runtime.exs"
        )
    end
  end
end
