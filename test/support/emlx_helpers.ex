defmodule EMLX.Helpers do
  @moduledoc """
  Helper functions for EMLX tests that need to be callable via :erpc on peer
  nodes. Lambdas cannot cross node boundaries, so all peer-side logic lives here
  as named MFAs.
  """

  @doc """
  Returns the peer nodes started by test_helper.exs, or [] when the
  distributed suite was excluded.
  """
  def test_peer_nodes, do: Application.get_env(:emlx, :test_peer_nodes, [])

  @doc """
  Creates an EMLX tensor from `list`, starts an Agent to keep it alive
  (so the underlying MLX array is not GC'd while the test runs), and
  exports an IPC pointer with `permissions: 0o600`.

  Returns `{pointer, type, shape, agent_pid}`. The caller is responsible
  for stopping the agent (`:erpc.call(peer, Process, :exit, [pid, :kill])`).
  """
  def export_ipc_pointer(list) do
    tensor = Nx.tensor(list, backend: EMLX.Backend)
    {:ok, pid} = Agent.start(fn -> tensor end)
    pointer = Nx.to_pointer(tensor, mode: :ipc, permissions: 0o600)
    {pointer, tensor.type, tensor.shape, pid}
  end

  @doc """
  Opens an IPC pointer on this node, reads the tensor as a flat list, and
  returns it. The tensor is not kept alive after this call — the shm is
  unlinked inside array_from_shm so cleanup is automatic.
  """
  def ipc_flat_list(pointer, type, shape) do
    t = Nx.from_pointer(EMLX.Backend, pointer, type, shape)
    Nx.to_flat_list(t)
  end
end
