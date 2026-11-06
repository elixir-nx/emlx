defmodule EMLX.CompileCacheTest do
  use ExUnit.Case, async: true

  alias EMLX.CompileCache

  setup do
    pid = start_supervised!({CompileCache, name: nil, max_items: :infinity, ttl: :infinity})
    %{pid: pid}
  end

  test "unbounded cache keeps every insert", %{pid: pid} do
    CompileCache.put(:a, :fun_a, pid)
    CompileCache.put(:b, :fun_b, pid)

    assert CompileCache.fetch(:a, pid) == {:ok, :fun_a}
    assert CompileCache.fetch(:b, pid) == {:ok, :fun_b}
  end

  test "max_items evicts the oldest insert" do
    pid =
      start_supervised!({CompileCache, name: nil, max_items: 2, ttl: :infinity}, id: :max_items)

    CompileCache.put(:a, :fun_a, pid)
    CompileCache.put(:b, :fun_b, pid)
    CompileCache.put(:c, :fun_c, pid)

    assert CompileCache.fetch(:a, pid) == {:error, :cache_miss}
    assert CompileCache.fetch(:b, pid) == {:ok, :fun_b}
    assert CompileCache.fetch(:c, pid) == {:ok, :fun_c}
  end

  test "put_new keeps the first value and max_items bounds each table" do
    pid = start_supervised!({CompileCache, name: nil, max_items: 2, ttl: :infinity}, id: :put_new)
    table = :ets.new(:programs, [:set, :public])

    assert CompileCache.put_new(table, :a, :first, pid) == :first
    assert CompileCache.put_new(table, :a, :second, pid) == :first

    CompileCache.put(:closure, :fun, pid)
    CompileCache.put_new(table, :b, :b, pid)
    CompileCache.put_new(table, :c, :c, pid)

    assert table |> :ets.tab2list() |> Enum.map(&elem(&1, 0)) |> Enum.sort() == [:b, :c]
    assert CompileCache.fetch(:closure, pid) == {:ok, :fun}
  end

  test "expire tick drops stale entries" do
    pid = start_supervised!({CompileCache, name: nil, max_items: :infinity, ttl: 10}, id: :ttl)
    table = :ets.new(:programs, [:set, :public])

    CompileCache.put(:a, :fun_a, pid)
    CompileCache.put_new(table, :b, :b, pid)
    Process.sleep(15)
    send(pid, :expire_entries)

    assert CompileCache.fetch(:a, pid) == {:error, :cache_miss}
    assert :ets.tab2list(table) == []
  end

  test "re-put of the same key replaces the value", %{pid: pid} do
    CompileCache.put(:a, :fun_old, pid)
    CompileCache.put(:a, :fun_new, pid)

    assert CompileCache.fetch(:a, pid) == {:ok, :fun_new}
  end

  test "invalid max_items raises" do
    assert {:error, {{%ArgumentError{message: message}, _stack}, _child}} =
             start_supervised({CompileCache, name: nil, max_items: 0}, id: :bad_max)

    assert message =~ "EMLX.CompileCache"
  end

  test "invalid ttl raises" do
    assert {:error, {{%ArgumentError{message: message}, _stack}, _child}} =
             start_supervised({CompileCache, name: nil, ttl: 0}, id: :bad_ttl)

    assert message =~ "EMLX.CompileCache"
  end
end
