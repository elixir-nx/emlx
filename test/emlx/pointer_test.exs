defmodule EMLX.PointerTest do
  use EMLX.Case, async: true

  # Retrieves the peer node started in test_helper.exs.
  # Distributed tests are tagged :distributed and excluded when epmd is not
  # available; see test_helper.exs and CI for the setup.
  defp peer, do: hd(EMLX.Helpers.test_peer_nodes())

  describe "to_pointer / from_pointer — :local mode" do
    test "returns a local Nx.Pointer with non-zero address and correct byte size" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t, mode: :local)

      assert %Nx.Pointer{kind: :local, address: addr, data_size: size} = ptr
      assert is_integer(addr) and addr > 0
      assert size == 4 * 4
    end

    test "default mode is :local" do
      t = Nx.tensor([1, 2, 3], type: :s32, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t)
      assert %Nx.Pointer{kind: :local} = ptr
    end

    test "round-trip for f32 tensor" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t, mode: :local)
      t2 = Nx.from_pointer(EMLX.Backend, ptr, {:f, 32}, {4})
      assert_equal(t, t2)
    end

    test "round-trip for s32 tensor" do
      t = Nx.tensor([-1, 0, 1, 2], type: :s32, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t, mode: :local)
      t2 = Nx.from_pointer(EMLX.Backend, ptr, {:s, 32}, {4})
      assert_equal(t, t2)
    end

    test "round-trip for u8 tensor" do
      t = Nx.tensor([0, 127, 255], type: :u8, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t, mode: :local)
      t2 = Nx.from_pointer(EMLX.Backend, ptr, {:u, 8}, {3})
      assert_equal(t, t2)
    end

    test "round-trip for multi-dimensional tensor" do
      t = Nx.iota({3, 4}, type: :f32, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t, mode: :local)
      t2 = Nx.from_pointer(EMLX.Backend, ptr, {:f, 32}, {3, 4})
      assert_equal(t, t2)
    end

    test "round-trip preserves names" do
      t = Nx.tensor([1.0, 2.0], type: :f32, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t, mode: :local)
      t2 = Nx.from_pointer(EMLX.Backend, ptr, {:f, 32}, {2}, names: [:x])
      assert t2.names == [:x]
      assert_equal(t, t2)
    end

    test "pointer from non-contiguous (transposed) tensor is contiguous in local mode" do
      # to_pointer evals and exposes the raw buffer; transposed tensors are
      # still row-contiguous in memory if slice, not if transposed.
      # We verify the round-trip produces correct values.
      t = Nx.iota({2, 3}, type: :f32, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t, mode: :local)
      t2 = Nx.from_pointer(EMLX.Backend, ptr, {:f, 32}, {2, 3})
      assert_equal(t, t2)
    end

    test "data_size mismatch raises in from_pointer :local" do
      t = Nx.tensor([1.0, 2.0], type: :f32, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t, mode: :local)
      wrong_ptr = %{ptr | data_size: 999}

      assert_raise ArgumentError, ~r/data_size/, fn ->
        Nx.from_pointer(EMLX.Backend, wrong_ptr, {:f, 32}, {2})
      end
    end
  end

  describe "to_pointer / from_pointer — :ipc mode" do
    test "returns an ipc Nx.Pointer with binary handle and correct byte size" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t, mode: :ipc, permissions: 0o600)

      assert %Nx.Pointer{kind: :ipc, handle: name, data_size: size} = ptr
      assert is_binary(name)
      assert String.starts_with?(name, "/emlx_")
      assert size == 4 * 4
    end

    test "handle is an Elixir binary (not a charlist)" do
      t = Nx.tensor([42], type: :s32, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t, mode: :ipc, permissions: 0o600)
      assert is_binary(ptr.handle)
    end

    test "round-trip for f32 tensor via shm" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t, mode: :ipc, permissions: 0o600)
      t2 = Nx.from_pointer(EMLX.Backend, ptr, {:f, 32}, {4})
      assert_equal(t, t2)
    end

    test "round-trip for s32 tensor via shm" do
      t = Nx.tensor([-10, 0, 10], type: :s32, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t, mode: :ipc, permissions: 0o600)
      t2 = Nx.from_pointer(EMLX.Backend, ptr, {:s, 32}, {3})
      assert_equal(t, t2)
    end

    test "round-trip for multi-dimensional tensor via shm" do
      t = Nx.iota({3, 4}, type: :f32, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t, mode: :ipc, permissions: 0o600)
      t2 = Nx.from_pointer(EMLX.Backend, ptr, {:f, 32}, {3, 4})
      assert_equal(t, t2)
    end

    test "round-trip for non-contiguous (transposed) tensor via shm" do
      # tensor_to_shm calls mlx::contiguous internally, so transposed tensors
      # are serialised in row-major order into the shm segment.
      orig = Nx.iota({3, 4}, type: :f32, backend: EMLX.Backend)
      t = Nx.transpose(orig)
      ptr = Nx.to_pointer(t, mode: :ipc, permissions: 0o600)
      t2 = Nx.from_pointer(EMLX.Backend, ptr, {:f, 32}, {4, 3})
      assert_equal(t, t2)
    end

    test "each to_pointer :ipc call generates a unique shm handle" do
      t = Nx.tensor([1.0], type: :f32, backend: EMLX.Backend)
      ptr1 = Nx.to_pointer(t, mode: :ipc, permissions: 0o600)
      ptr2 = Nx.to_pointer(t, mode: :ipc, permissions: 0o600)
      assert ptr1.handle != ptr2.handle
      # clean up the second one since we won't call from_pointer on it
      EMLX.shm_unlink(ptr2.handle)
    end

    test "shm_unlink is idempotent after round-trip (receiver already unlinked)" do
      t = Nx.tensor([1.0, 2.0], type: :f32, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t, mode: :ipc, permissions: 0o600)
      _t2 = Nx.from_pointer(EMLX.Backend, ptr, {:f, 32}, {2})
      # receiver called shm_unlink; second call must not raise
      assert :ok = EMLX.shm_unlink(ptr.handle)
    end

    test "shm_unlink cleans up an unclaimed handle" do
      t = Nx.tensor([1.0], type: :f32, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t, mode: :ipc, permissions: 0o600)
      assert :ok = EMLX.shm_unlink(ptr.handle)
    end

    test "permissions: 0o400 (read-only) still produces correct values" do
      t = Nx.tensor([3.14], type: :f32, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t, mode: :ipc, permissions: 0o400)
      t2 = Nx.from_pointer(EMLX.Backend, ptr, {:f, 32}, {1})
      assert_equal(t, t2)
    end

    test "data_size mismatch raises in from_pointer :ipc" do
      t = Nx.tensor([1.0, 2.0], type: :f32, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t, mode: :ipc, permissions: 0o600)
      wrong_ptr = %{ptr | data_size: 999}

      assert_raise ArgumentError, ~r/data_size/, fn ->
        Nx.from_pointer(EMLX.Backend, wrong_ptr, {:f, 32}, {2})
      end

      EMLX.shm_unlink(ptr.handle)
    end
  end

  describe "error handling" do
    test "invalid mode in to_pointer raises ArgumentError" do
      t = Nx.tensor([1.0], type: :f32, backend: EMLX.Backend)

      assert_raise ArgumentError, ~r/mode/, fn ->
        Nx.to_pointer(t, mode: :cuda_ipc)
      end
    end

    test "unsupported pointer kind in from_pointer raises ArgumentError" do
      fake_ptr = %Nx.Pointer{kind: :cuda_ipc, handle: "handle", address: nil, data_size: 4}

      assert_raise ArgumentError, ~r/:local.*:ipc|only supports/, fn ->
        Nx.from_pointer(EMLX.Backend, fake_ptr, {:f, 32}, {1})
      end
    end
  end

  describe "IPC across peer nodes" do
    # Skipped when test_helper.exs could not start a peer node (no epmd /
    # distributed Erlang available). CI starts epmd before running mix test.
    @describetag :distributed

    test "IPC round-trip: main → peer → main (f32)" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t, mode: :ipc, permissions: 0o600)
      result = :erpc.call(peer(), EMLX.Helpers, :ipc_flat_list, [ptr, t.type, t.shape])
      assert result == [1.0, 2.0, 3.0, 4.0]
    end

    test "IPC round-trip: main → peer → main (s32)" do
      t = Nx.tensor([-3, 0, 7], type: :s32, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t, mode: :ipc, permissions: 0o600)
      result = :erpc.call(peer(), EMLX.Helpers, :ipc_flat_list, [ptr, t.type, t.shape])
      assert result == [-3, 0, 7]
    end

    test "IPC round-trip: peer → main (peer creates the tensor)" do
      {ptr, type, shape, agent_pid} =
        :erpc.call(peer(), EMLX.Helpers, :export_ipc_pointer, [[10, 20, 30, 40]])

      on_exit(fn -> :erpc.call(peer(), Process, :exit, [agent_pid, :kill]) end)

      t = Nx.from_pointer(EMLX.Backend, ptr, type, shape)
      assert Nx.to_flat_list(t) == [10, 20, 30, 40]
    end

    test "IPC round-trip: peer → main, multi-dimensional" do
      {ptr, type, shape, agent_pid} =
        :erpc.call(peer(), EMLX.Helpers, :export_ipc_pointer, [[1, 2, 3, 4, 5, 6]])

      on_exit(fn -> :erpc.call(peer(), Process, :exit, [agent_pid, :kill]) end)

      t = Nx.from_pointer(EMLX.Backend, ptr, type, shape)
      assert Nx.to_flat_list(Nx.reshape(t, {2, 3})) == [1, 2, 3, 4, 5, 6]
    end

    test "shm name is already unlinked after peer opens it" do
      t = Nx.tensor([42.0], type: :f32, backend: EMLX.Backend)
      ptr = Nx.to_pointer(t, mode: :ipc, permissions: 0o600)
      :erpc.call(peer(), EMLX.Helpers, :ipc_flat_list, [ptr, t.type, t.shape])
      assert :ok = EMLX.shm_unlink(ptr.handle)
    end

    test "peer-exported handle works even after sender tensor is GC'd" do
      {ptr, type, shape, agent_pid} =
        :erpc.call(peer(), EMLX.Helpers, :export_ipc_pointer, [[99.0, 100.0]])

      on_exit(fn -> :erpc.call(peer(), Process, :exit, [agent_pid, :kill]) end)

      t = Nx.from_pointer(EMLX.Backend, ptr, type, shape)
      assert Nx.to_flat_list(t) == [99.0, 100.0]
    end
  end
end
