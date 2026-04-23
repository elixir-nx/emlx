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
end
