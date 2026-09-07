defmodule EMLX.CompileCache do
  @moduledoc """
  Bounded cache of `native_compile/3` eval closures.

  `EMLX.__compile__/4` stores one closure per `{fun, templates, hooks, device}`
  so repeated `Nx.Defn.jit/2` of the same call site skips retrace. Unbounded
  unless the host sets:

      config :emlx, EMLX.CompileCache,
        max_items: 1024,
        ttl: :timer.minutes(30)

  Both default to `:infinity`. Overflow is FIFO by insert index. TTL is from
  insert time and is swept by this process (`send_after(self(), :expire_entries,
  div(ttl, 2))`), not on cache reads. A sweeper that is already running re-reads
  app env each tick; if TTL is `:infinity` at start, nothing is scheduled until
  the process restarts.

  The structural program table `:emlx_native_dispatch_cache` is shared across
  call sites and is not bounded by these knobs.
  """

  use GenServer

  @doc false
  def start_link(opts \\ []) do
    {name, opts} = Keyword.pop(opts, :name, __MODULE__)
    opts = Keyword.validate!(opts, [:ttl, :max_items])
    gen_opts = if name, do: [name: name], else: []
    GenServer.start_link(__MODULE__, opts, gen_opts)
  end

  @impl true
  def init(opts) do
    state = %{
      table: :ets.new(:emlx_compile_closures, [:set, :private]),
      counter: 0,
      ttl: Keyword.get(opts, :ttl),
      max_items: Keyword.get(opts, :max_items)
    }

    _ = current_max_items(state)
    {:ok, schedule_expire(state)}
  end

  @doc false
  def fetch(key, server \\ __MODULE__) do
    GenServer.call(server, {:fetch, key})
  end

  @doc false
  def put(key, fun, server \\ __MODULE__) do
    GenServer.call(server, {:put, key, fun})
  end

  @impl true
  def handle_call({:fetch, key}, _from, %{table: table} = state) do
    reply =
      case :ets.lookup(table, key) do
        [{^key, fun, _inserted_ms, _index}] -> {:ok, fun}
        [] -> {:error, :cache_miss}
      end

    {:reply, reply, state}
  end

  def handle_call({:put, key, fun}, _from, state) do
    index = state.counter + 1
    :ets.insert(state.table, {key, fun, System.monotonic_time(:millisecond), index})
    evict_overflow(state.table, current_max_items(state), index)
    {:reply, fun, %{state | counter: index}}
  end

  @impl true
  def handle_info(:expire_entries, state) do
    case current_ttl(state) do
      :infinity ->
        {:noreply, state}

      ttl ->
        expire_entries(state.table, ttl)
        {:noreply, schedule_expire(state)}
    end
  end

  defp expire_entries(table, ttl) do
    cutoff = System.monotonic_time(:millisecond) - ttl

    :ets.select_delete(table, [
      {{:"$1", :_, :"$2", :_}, [{:"=<", :"$2", cutoff}], [true]}
    ])
  end

  defp current_ttl(%{ttl: ttl}) when ttl != nil, do: bound!(ttl, :ttl)
  defp current_ttl(_state), do: bound!(config(:ttl), :ttl)

  defp current_max_items(%{max_items: n}) when n != nil, do: bound!(n, :max_items)
  defp current_max_items(_state), do: bound!(config(:max_items), :max_items)

  defp config(key) do
    :emlx
    |> Application.get_env(__MODULE__, [])
    |> Keyword.get(key, :infinity)
  end

  defp schedule_expire(state) do
    case current_ttl(state) do
      :infinity ->
        state

      ttl ->
        Process.send_after(self(), :expire_entries, max(div(ttl, 2), 1))
        state
    end
  end

  defp evict_overflow(_table, :infinity, _index), do: :ok

  defp evict_overflow(table, max_items, index) do
    cutoff = index - max_items

    if cutoff > 0 do
      :ets.select_delete(table, [
        {{:"$1", :_, :_, :"$2"}, [{:"=<", :"$2", cutoff}], [true]}
      ])
    else
      :ok
    end
  end

  defp bound!(:infinity, _key), do: :infinity
  defp bound!(n, _key) when is_integer(n) and n > 0, do: n

  defp bound!(other, key) do
    raise ArgumentError,
          "config :emlx, #{inspect(__MODULE__)}, #{key} must be :infinity or a positive integer, got: #{inspect(other)}"
  end
end
