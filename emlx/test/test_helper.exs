# MLX's CPU backend JIT-compiles fused kernels via `popen`/`pclose`, which
# fails with `ECHILD` under the BEAM's default SIGCHLD=SIG_IGN disposition
# (see `EMLX.Application`'s moduledoc for the full explanation). Library
# code doesn't get to make this call for consumers, but our own test suite
# is exactly the kind of deployment where we do control this trade-off.
:os.set_signal(:sigchld, :default)

use_gpu? =
  String.downcase(System.get_env("EMLX_TEST_DEFAULT_GPU", "false")) in [
    "1",
    "true",
    "yes",
    "t",
    "y"
  ]

{default_device, backend} =
  if use_gpu? do
    {:gpu, {EMLX.Backend, device: :gpu}}
  else
    {:cpu, EMLX.Backend}
  end

Application.put_env(:nx, :default_backend, backend)
Application.put_env(:emlx, :default_device, default_device)

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
      :ok = :erpc.call(peer, :os, :set_signal, [:sigchld, :default])
      :ok = :erpc.call(peer, Application, :put_env, [:emlx, :default_device, default_device])
      :ok = :erpc.call(peer, Application, :put_env, [:nx, :default_backend, backend])
      {:ok, _} = :erpc.call(peer, :application, :ensure_all_started, [:nx])
      {:ok, _} = :erpc.call(peer, :application, :ensure_all_started, [:emlx])
      Application.put_env(:emlx, :test_peer_nodes, [peer])
      []

    true ->
      [:distributed]
  end

gpu_exclude =
  case EMLX.NIF.command_queue_new(:gpu) do
    {:ok, _} -> [:large_memory]
    {:error, _} -> [:metal]
  end

# debug_flags_functional_test.exs needs EMLX compiled with :detect_non_finites
# on (compile_env, baked in at compile time) — excluded unless that build was
# requested, so a normal `mix test` (flags off, per production defaults)
# doesn't fail on it.
debug_flags_exclude =
  if System.get_env("EMLX_DEBUG_FLAGS") == "1", do: [], else: [:debug_flags_functional]

ExUnit.start(exclude: distributed_exclude ++ gpu_exclude ++ debug_flags_exclude)
