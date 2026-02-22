defmodule EMLX.MemoryTest do
  use ExUnit.Case, async: false

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
