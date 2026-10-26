defmodule EMLX.MemoryTest do
  use EMLX.Case, async: false

  test "memory_info returns active, peak, and cache" do
    info = EMLX.memory_info()
    assert is_map(info)
    assert is_integer(info.active_memory)
    assert is_integer(info.peak_memory)
    assert is_integer(info.cache_memory)
    assert info.active_memory >= 0
    assert info.peak_memory >= 0
    assert info.cache_memory >= 0
  end

  test "allocating a tensor increases active memory" do
    EMLX.clear_cache()
    before = EMLX.memory_info().active_memory
    t = Nx.iota({1024, 1024}, type: :f32, backend: EMLX.Backend)
    EMLX.eval(EMLX.Backend.from_nx(t))
    after_alloc = EMLX.memory_info().active_memory
    assert after_alloc >= before + 1024 * 1024 * 4
  end

  test "clear_cache releases unused memory" do
    t = Nx.iota({1024, 1024}, type: :f32, backend: EMLX.Backend)
    EMLX.eval(EMLX.Backend.from_nx(t))
    Nx.backend_deallocate(t)
    EMLX.clear_cache()
    info = EMLX.memory_info()
    assert is_integer(info.active_memory)
    assert is_integer(info.peak_memory)
    assert info.cache_memory == 0
  end

  # Nx.backend_deallocate/1 frees the MLX array at once. Before, it only marked the tensor,
  # and the memory stayed in use until the BEAM collected the term. The deallocated tensor
  # keeps a scalar of 4 bytes until then.
  test "deallocate frees the memory of a tensor without a garbage collection" do
    EMLX.clear_cache()
    t = Nx.iota({1024, 1024}, type: :f32, backend: EMLX.Backend)
    EMLX.eval(EMLX.Backend.from_nx(t))
    before = EMLX.memory_info().active_memory

    assert Nx.backend_deallocate(t) == :ok
    assert EMLX.memory_info().active_memory <= before - 1024 * 1024 * 4 + 4
  end

  test "a deallocated tensor raises on use, and a second deallocate does nothing" do
    t = Nx.iota({4}, type: :f32, backend: EMLX.Backend)
    assert Nx.backend_deallocate(t) == :ok
    assert Nx.backend_deallocate(t) == :already_deallocated
    assert_raise EMLX.NIFError, ~r/deallocated/, fn -> Nx.to_flat_list(t) end
  end

  # A binary from Nx.to_binary/1 shares the buffer of a contiguous tensor without a copy.
  # The binary keeps its own reference to the buffer, so a deallocate of the tensor does not
  # free the memory that the binary still reads.
  test "a binary keeps the buffer of a deallocated tensor" do
    EMLX.clear_cache()
    t = Nx.iota({1024, 1024}, type: :f32, backend: EMLX.Backend)
    expected = t |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_binary()
    binary = Nx.to_binary(t)
    before = EMLX.memory_info().active_memory

    Nx.backend_deallocate(t)
    assert EMLX.memory_info().active_memory >= before - 4
    assert binary == expected
  end

  test "reset_peak_memory resets the counter" do
    _t = Nx.iota({1024, 1024}, type: :f32, backend: EMLX.Backend)
    EMLX.reset_peak_memory()
    assert EMLX.memory_info().peak_memory == 0
  end

  test "set_memory_limit returns previous limit" do
    prev = EMLX.set_memory_limit(1_000_000_000)
    assert is_integer(prev)
    EMLX.set_memory_limit(prev)
  end

  test "set_cache_limit returns previous limit" do
    prev = EMLX.set_cache_limit(500_000_000)
    assert is_integer(prev)
    EMLX.set_cache_limit(prev)
  end
end
