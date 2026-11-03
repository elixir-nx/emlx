# Run only in a disposable BEAM: a broken native wait or resource lifetime must
# not hang/crash the ExUnit VM. The parent enforces an OS-process timeout.
:os.set_signal(:sigchld, :default)
Application.put_env(:emlx, :default_device, :cpu)
Application.put_env(:nx, :default_backend, EMLX.Backend)
{:ok, _} = Application.ensure_all_started(:emlx)

defmodule RuntimeCallLifecycle do
  def run do
    {worker, :cpu} = EMLX.resolve_worker(:cpu)
    input = Nx.tensor(7, type: :s32, backend: EMLX.Backend)
    {:cpu, input_ref} = input.data.ref

    for kind <- [:hook, :runtime_call] do
      fun =
        case kind do
          :hook ->
            fn x ->
              op = if function_exported?(Nx.Defn.Kernel, :io_call, 2), do: :io_call, else: :hook
              apply(Nx.Defn.Kernel, op, [x, :value])
            end

          :runtime_call ->
            fn x -> Nx.runtime_call(Nx.template({}, :s32), x, fn t, _ -> t end) end
        end

      expr = Nx.Defn.debug_expr_apply(fun, [Nx.template({}, :s32)])
      program = EMLX.Native.Expr.lower(expr, 1, %{}, %{value: fn _ -> :ok end})
      {:ok, compile_job} = EMLX.NIF.compile_program(worker, EMLX.Native.Expr.to_native(program))
      resource = EMLX.await_worker(compile_job)
      reply = if kind == :hook, do: [], else: [<<7::signed-native-32>>]

      # Successful reply, then duplicate reply after the worker released its
      # destination buffers. The resource itself remains reachable here.
      {caller, monitor, pending} = start_call(worker, resource, input_ref)
      :ok = EMLX.NIF.resolve_runtime_call(pending, :ok, reply)

      receive do
        {:finished, ^caller, {:ok, [_]}} -> :ok
      after
        5_000 -> raise "normal callback did not finish"
      end

      await_down(caller, monitor)
      {:error, _} = EMLX.NIF.resolve_runtime_call(pending, :ok, reply)

      # Callback recipient dies after receiving the pending resource. It never
      # sends a reply. The worker's bounded liveness check must abandon it.
      for _ <- 1..8 do
        {caller, monitor, pending} = start_call(worker, resource, input_ref)
        Process.exit(caller, :kill)
        await_down(caller, monitor)
        # Allow the 5ms worker heartbeat to observe death. This is not a claim
        # that a running MLX kernel is preempted on BEAM process death.
        receive do
        after
          100 -> :ok
        end

        {:error, _} = EMLX.NIF.resolve_runtime_call(pending, :ok, reply)
      end

      # Exercise the send-failure/message-discard path: no pending resource is
      # retained by BEAM after recipient death. The worker reference protects it.
      for _ <- 1..8 do
        {caller, monitor} =
          spawn_monitor(fn ->
            {:ok, _} = EMLX.NIF.eval_program(worker, resource, [input_ref])
          end)

        await_down(caller, monitor)
      end

      # The same worker and executable remain usable following abandoned calls.
      {caller, monitor, pending} = start_call(worker, resource, input_ref)
      :ok = EMLX.NIF.resolve_runtime_call(pending, :ok, reply)

      receive do
        {:finished, ^caller, {:ok, [_]}} -> :ok
      after
        5_000 -> raise "worker did not recover after caller death"
      end

      await_down(caller, monitor)
    end

    IO.puts("runtime-call lifecycle OK")
  end

  defp start_call(worker, resource, input_ref) do
    parent = self()

    {caller, monitor} =
      spawn_monitor(fn ->
        {:ok, job} = EMLX.NIF.eval_program(worker, resource, [input_ref])

        receive do
          {:emlx_runtime_call, pending, 0, [_]} ->
            send(parent, {:pending, self(), pending})
        after
          5_000 -> raise "callback request not received"
        end

        receive do
          {^job, result} -> send(parent, {:finished, self(), result})
        after
          5_000 -> raise "callback result not received"
        end
      end)

    receive do
      {:pending, ^caller, pending} -> {caller, monitor, pending}
    after
      5_000 -> raise "pending handle not received"
    end
  end

  defp await_down(caller, monitor) do
    receive do
      {:DOWN, ^monitor, :process, ^caller, _reason} -> :ok
    after
      5_000 -> raise "caller did not exit"
    end
  end
end

RuntimeCallLifecycle.run()
