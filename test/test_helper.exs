backend =
  if String.downcase(System.get_env("EMLX_TEST_DEFAULT_GPU", "false")) in [
       "1",
       "true",
       "yes",
       "t",
       "y"
     ] do
    {EMLX.Backend, device: :gpu}
  else
    EMLX.Backend
  end

Application.put_env(:nx, :default_backend, backend)

# ── Distributed (peer) test setup ────────────────────────────────────────────
# Mirrors EXLA's approach: try to start epmd + a named primary node, then
# spin up one peer node. If any step fails (e.g. epmd unavailable, network
# loopback missing), we exclude :distributed tests gracefully.

try_epmd = fn ->
  case :os.type() do
    {:unix, _} -> {"", 0} == System.cmd("epmd", ["-daemon"])
    _ -> true
  end
end

distributed_exclude =
  cond do
    :distributed in Keyword.get(ExUnit.configuration(), :exclude, []) ->
      [:distributed]

    Code.ensure_loaded?(:peer) and try_epmd.() and
        match?({:ok, _}, Node.start(:"emlx_primary@127.0.0.1", :longnames)) ->
      {:ok, _pid, peer} = :peer.start(%{name: :"emlx_peer@127.0.0.1"})
      true = :erpc.call(peer, :code, :set_path, [:code.get_path()])
      {:ok, _} = :erpc.call(peer, :application, :ensure_all_started, [:nx])
      {:ok, _} = :erpc.call(peer, :application, :ensure_all_started, [:emlx])
      Application.put_env(:emlx, :test_peer_nodes, [peer])
      []

    true ->
      [:distributed]
  end

gpu_exclude =
  case EMLX.NIF.command_queue_new(:gpu) do
    {:ok, _} -> []
    {:error, _} -> [:metal]
  end

ExUnit.start(exclude: distributed_exclude ++ gpu_exclude)
