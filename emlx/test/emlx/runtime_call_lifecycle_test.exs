defmodule EMLX.RuntimeCallLifecycleTest do
  use ExUnit.Case, async: false
  alias EMLX.RuntimeCallHelpers

  for kind <- [:io_call, :runtime_call] do
    test "#{kind} completes and rejects duplicate replies" do
      {worker, resource, input_ref, reply} = RuntimeCallHelpers.program(unquote(kind))
      {caller, monitor, pending} = RuntimeCallHelpers.start_call(worker, resource, input_ref)
      assert :ok = EMLX.NIF.resolve_runtime_call(pending, :ok, reply)
      assert_receive {:finished, ^caller, {:ok, [_]}}, 5_000
      assert_receive {:DOWN, ^monitor, :process, ^caller, :normal}, 5_000
      assert {:error, _} = EMLX.NIF.resolve_runtime_call(pending, :ok, reply)
    end

    @tag :distributed
    @tag timeout: 30_000
    test "#{kind} abandons dead callers and rejects late replies on a peer" do
      {:ok, peer, node} =
        :peer.start(%{
          name: :peer.random_name(~c"emlx_callbacks"),
          connection: :standard_io,
          args: [~c"+S", ~c"2:2"],
          shutdown: :halt
        })

      on_exit(fn -> if Process.alive?(peer), do: :peer.stop(peer) end)
      assert true = :erpc.call(node, :code, :set_path, [:code.get_path()])
      :ok = :erpc.call(node, :os, :set_signal, [:sigchld, :default])
      :ok = :erpc.call(node, Application, :put_env, [:emlx, :default_device, :cpu])
      assert {:ok, _} = :erpc.call(node, Application, :ensure_all_started, [:emlx])
      assert :ok = :erpc.call(node, RuntimeCallHelpers, :caller_death, [unquote(kind)], 15_000)
    end
  end
end
