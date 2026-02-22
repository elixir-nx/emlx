defmodule EMLX.ToBinaryTest do
  use ExUnit.Case, async: true

  test "to_binary for contiguous tensor" do
    t = Nx.iota({4, 3}, type: :f32, backend: EMLX.Backend)
    bin = Nx.to_binary(t)
    assert byte_size(bin) == 4 * 3 * 4
    assert Nx.to_flat_list(t) == for(<<f::float-32-native <- bin>>, do: f)
  end

  test "to_binary for large contiguous tensor" do
    t = Nx.iota({64, 1536}, type: :f32, backend: EMLX.Backend)
    bin = Nx.to_binary(t)
    assert byte_size(bin) == 64 * 1536 * 4

    # Verify first and last elements
    <<first::float-32-native, _::binary>> = bin
    assert first == 0.0

    last_offset = (64 * 1536 - 1) * 4
    <<_::binary-size(last_offset), last::float-32-native>> = bin
    assert last == 64 * 1536 - 1
  end

  test "to_binary for transposed (non-contiguous) tensor" do
    t = Nx.iota({3, 4}, type: :f32, backend: EMLX.Backend)
    t_transposed = Nx.transpose(t)
    bin = Nx.to_binary(t_transposed)
    assert byte_size(bin) == 3 * 4 * 4
    assert Nx.to_flat_list(t_transposed) == for(<<f::float-32-native <- bin>>, do: f)
  end

  test "to_binary with limit for contiguous tensor" do
    t = Nx.iota({100}, type: :f32, backend: EMLX.Backend)
    bin = Nx.to_binary(t, limit: 10)
    assert byte_size(bin) == 10 * 4
    floats = for <<f::float-32-native <- bin>>, do: f
    assert floats == Enum.map(0..9, &(&1 * 1.0))
  end
end
