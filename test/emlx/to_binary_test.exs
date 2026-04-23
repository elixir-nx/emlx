defmodule EMLX.ToBinaryTest do
  use EMLX.Case, async: true

  test "to_binary for contiguous tensor" do
    t = Nx.iota({4, 3}, type: :f32, backend: EMLX.Backend)
    ref = Nx.iota({4, 3}, type: :f32, backend: Nx.BinaryBackend)
    assert Nx.to_binary(t) == Nx.to_binary(ref)
  end

  test "to_binary for large contiguous tensor" do
    t = Nx.iota({64, 1536}, type: :f32, backend: EMLX.Backend)
    ref = Nx.iota({64, 1536}, type: :f32, backend: Nx.BinaryBackend)
    assert Nx.to_binary(t) == Nx.to_binary(ref)
  end

  test "to_binary for transposed (non-contiguous) tensor" do
    t = Nx.iota({3, 4}, type: :f32, backend: EMLX.Backend) |> Nx.transpose()
    ref = Nx.iota({3, 4}, type: :f32, backend: Nx.BinaryBackend) |> Nx.transpose()
    assert Nx.to_binary(t) == Nx.to_binary(ref)
  end

  test "to_binary with limit for contiguous tensor" do
    t = Nx.iota({100}, type: :f32, backend: EMLX.Backend)
    ref = Nx.iota({100}, type: :f32, backend: Nx.BinaryBackend)
    assert Nx.to_binary(t, limit: 10) == Nx.to_binary(ref, limit: 10)
  end

  # Refcount safety: binary must remain valid after the tensor term is dropped and GC'd.
  # A segfault here means the resource binary does not hold the MLX buffer alive.
  test "binary remains readable after original tensor is GC'd" do
    test_pid = self()
      spawn(fn ->
        t = Nx.iota({4}, type: :f32, backend: EMLX.Backend)

        # Creates the resource binary, which should pin the MLX buffer
        bin =
          t
          |> EMLX.Backend.from_nx()
          |> EMLX.to_blob()

          # t goes out of scope here
        send(test_pid, {:bin, bin})
      end)

    assert_receive {:bin, bin}

    :erlang.garbage_collect()
    assert byte_size(bin) == 4 * 4
    # Force a copy to verify every byte is readable — would segfault on UAF
    assert :binary.copy(bin) == bin
  end

  # Non-contiguous refcount safety: same test via the ct resource path.
  test "non-contiguous binary remains readable after original tensor is GC'd" do
    test_pid = self()
      spawn(fn ->
        t = Nx.iota({3, 4}, type: :f32, backend: EMLX.Backend) |> Nx.transpose()
        bin =
          t
          |> EMLX.Backend.from_nx()
          |> EMLX.to_blob()

        send(test_pid, {:bin, bin})
      end)

    assert_receive {:bin, bin}

    :erlang.garbage_collect()
    assert byte_size(bin) == 3 * 4 * 4
    assert :binary.copy(bin) == bin
  end
end
