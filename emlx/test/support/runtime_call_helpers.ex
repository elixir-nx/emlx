defmodule EMLX.RuntimeCallHelpers do
  @moduledoc false
  import ExUnit.Assertions

  def program(kind) do
    {worker, :cpu} = EMLX.resolve_worker(:cpu)
    input = Nx.tensor(7, type: :s32, backend: EMLX.Backend)
    {:cpu, input_ref} = input.data.ref

    fun =
      case kind do
        :io_call -> fn x -> Nx.Defn.Kernel.io_call(x, :value) end
        :runtime_call -> fn x -> Nx.runtime_call(Nx.template({}, :s32), x, fn t, _ -> t end) end
      end

    expr = Nx.Defn.debug_expr_apply(fun, [Nx.template({}, :s32)])
    program = EMLX.Native.Expr.lower(expr, 1, %{}, %{value: fn _ -> :ok end})
    {:ok, job} = EMLX.NIF.compile_program(worker, EMLX.Native.Expr.to_native(program))
    resource = EMLX.await_worker(job)
    reply = if kind == :io_call, do: [], else: [<<7::signed-native-32>>]
    {worker, resource, input_ref, reply}
  end

  def start_call(worker, resource, input_ref) do
    parent = self()

    {caller, monitor} =
      spawn_monitor(fn ->
        assert {:ok, job} = EMLX.NIF.eval_program(worker, resource, [input_ref])
        assert_receive {:emlx_runtime_call, pending, 0, [_]}, 5_000
        send(parent, {:pending, self(), pending})
        assert_receive {^job, result}, 5_000
        send(parent, {:finished, self(), result})
      end)

    assert_receive {:pending, ^caller, pending}, 5_000
    {caller, monitor, pending}
  end

  # Run on an isolated peer: an ExUnit timeout cannot interrupt a native wait.
  def caller_death(kind) do
    {worker, resource, input_ref, reply} = program(kind)

    for _ <- 1..8 do
      {caller, monitor, pending} = start_call(worker, resource, input_ref)
      Process.exit(caller, :kill)
      assert_receive {:DOWN, ^monitor, :process, ^caller, :killed}, 5_000

      # DOWN does not preempt native computation. Allow the worker's 5ms
      # liveness check to abandon the callback before delivering a late reply.
      Process.sleep(100)
      assert {:error, _} = EMLX.NIF.resolve_runtime_call(pending, :ok, reply)
    end

    # Drop the recipient before it can retain the pending message/resource.
    for _ <- 1..8 do
      {caller, monitor} =
        spawn_monitor(fn ->
          assert {:ok, _} = EMLX.NIF.eval_program(worker, resource, [input_ref])
        end)

      assert_receive {:DOWN, ^monitor, :process, ^caller, :normal}, 5_000
    end

    {caller, monitor, pending} = start_call(worker, resource, input_ref)
    assert :ok = EMLX.NIF.resolve_runtime_call(pending, :ok, reply)
    assert_receive {:finished, ^caller, {:ok, [_]}}, 5_000
    assert_receive {:DOWN, ^monitor, :process, ^caller, :normal}, 5_000
    :ok
  end
end
