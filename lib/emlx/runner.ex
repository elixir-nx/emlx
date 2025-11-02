defmodule EMLX.Runner do
  use GenServer

  defstruct [:nif_module, :on_evaluated, :refs, :monitors]

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts[:runner_opts], opts)
  end

  # Direct register to avoid GenServer.call issues during NIF-driven compilation
  def register(name, function) when is_function(function, 1) do
    runner_pid = Process.whereis(name)
    ref = make_ref()
    send(runner_pid, {:register_direct, self(), function, ref})

    receive do
      {:register_reply, ^ref, result} -> result
    after
      30_000 -> raise "Timeout waiting for register reply from #{inspect(name)}"
    end
  end

  def unregister(name, {_pid, ref}) do
    GenServer.call(name, {:unregister, ref}, :infinity)
  end

  def init(opts) do
    opts = Keyword.validate!(opts, [:nif_module, on_evaluated: :nif_call_evaluated])

    {:ok,
     %__MODULE__{
       nif_module: opts[:nif_module],
       on_evaluated: opts[:on_evaluated],
       refs: %{},
       monitors: %{}
     }}
  end

  def handle_call({:unregister, ref}, _from, state) do
    Process.demonitor(ref, [:flush])
    {:reply, :ok, %{state | refs: Map.delete(state.refs, ref)}}
  end

  def handle_info({:DOWN, ref, _, _, _}, state) when is_map_key(state.monitors, ref) do
    {:noreply,
     %{
       state
       | refs: Map.delete(state.refs, state.monitors[ref]),
         monitors: Map.delete(state.monitors, ref)
     }}
  end

  def handle_info({:DOWN, ref, _, _, _}, state) do
    {:noreply, %{state | refs: Map.delete(state.refs, ref)}}
  end

  def handle_info({:execute, resource, ref, args}, state) do
    function = Map.fetch!(state.refs, ref)

    task =
      Task.async(fn ->
        try do
          result = apply(function, List.wrap(args))
          apply(state.nif_module, state.on_evaluated, [resource, {:ok, result}])
        catch
          kind, reason ->
            apply(state.nif_module, state.on_evaluated, [resource, {kind, reason}])
        end
      end)

    {:noreply, %{state | monitors: Map.put(state.monitors, task.ref, ref)}}
  end

  def handle_info({{:eval, resource}, _, _, _, reason}, state) do
    if reason != :normal do
      apply(state.nif_module, :nif_call_evaluated, [resource, {:exit, reason}])
    end

    {:noreply, state}
  end

  def handle_info({ref, _}, state) when is_map_key(state.monitors, ref) do
    {:noreply, state}
  end

  def handle_info({:register_direct, owner, function, reply_ref}, state) do
    monitor_ref = Process.monitor(owner)
    result = {self(), monitor_ref}
    send(owner, {:register_reply, reply_ref, result})
    {:noreply, %{state | refs: Map.put(state.refs, monitor_ref, function)}}
  end

  def handle_info(_msg, state) do
    {:noreply, state}
  end
end
