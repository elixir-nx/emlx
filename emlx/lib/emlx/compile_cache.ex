defmodule EMLX.CompileCache do
  @moduledoc """
  Bounded cache of `native_compile/3` eval closures and compiled programs.

  `EMLX.__compile__/4` stores one closure per `{fun, templates, hooks, device}`
  so repeated `Nx.Defn.jit/2` of the same call site skips retrace. The
  structural program table `:emlx_native_dispatch_cache` and its key table
  `:emlx_dispatch_key_by_id` are shared across call sites. Each compiled
  program holds device memory, for example its captured constants. All three
  tables are unbounded unless the host sets:

      config :emlx, EMLX.CompileCache,
        max_items: 1024,
        ttl: :timer.minutes(30)

  Both default to `:infinity` and apply to each table. Overflow is FIFO by
  insert index. TTL is from insert time and is swept by this process
  (`send_after(self(), :expire_entries, div(ttl, 2))`), not on cache reads. A
  sweeper that is already running re-reads app env each tick; if TTL is
  `:infinity` at start, nothing is scheduled until the process restarts.

  A program stays in memory while a cached closure or a caller still holds it.
  An evicted program compiles again on its next miss.
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
    # The dispatch tables belong to this process, so they live as long as the application.
    EMLX.__create_dispatch_tables__()

    table = :ets.new(:emlx_compile_closures, [:set, :private])

    state = %{
      table: table,
      counters: %{table => 0},
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

  # Inserts into a public table that EMLX reads directly, such as the dispatch
  # cache. Keeps an existing value and returns the value in the table.
  @doc false
  def put_new(table, key, value, server \\ __MODULE__) do
    GenServer.call(server, {:put_new, table, key, value})
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
    {:reply, fun, insert(state, state.table, key, fun)}
  end

  def handle_call({:put_new, table, key, value}, _from, state) do
    case :ets.lookup(table, key) do
      [{_key, existing, _inserted_ms, _index}] -> {:reply, existing, state}
      [] -> {:reply, value, insert(state, table, key, value)}
    end
  end

  @impl true
  def handle_info(:expire_entries, state) do
    case current_ttl(state) do
      :infinity ->
        {:noreply, state}

      ttl ->
        for {table, _index} <- state.counters, do: expire_entries(table, ttl)
        {:noreply, schedule_expire(state)}
    end
  end

  defp insert(state, table, key, value) do
    index = Map.get(state.counters, table, 0) + 1
    :ets.insert(table, {key, value, System.monotonic_time(:millisecond), index})
    evict_overflow(table, current_max_items(state), index)
    put_in(state.counters[table], index)
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
