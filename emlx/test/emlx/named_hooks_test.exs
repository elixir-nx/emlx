defmodule EMLX.NamedHooksTest do
  use ExUnit.Case, async: false
  import Nx.Defn

  defn single(x), do: io_call(x + 1, :value) * 2

  defn defaulted(x) do
    io_call(x + 1, :value, fn t -> send(self(), {:default, Nx.to_number(t)}) end) * 2
  end

  defn nested(x) do
    {x, _} =
      while {x, i = 0}, i < 2 do
        {x, _} =
          while {x, j = 0}, j < 2 do
            {io_call(x + 1, :value), j + 1}
          end

        {x, i + 1}
      end

    x
  end

  defp tensor(x), do: Nx.tensor(x, backend: EMLX.Backend)
  defp callback(tag), do: fn t -> send(self(), {tag, Nx.to_number(t)}) end

  defp cache_sizes do
    closure_table = :sys.get_state(EMLX.CompileCache).table

    Enum.map(
      [:emlx_native_dispatch_cache, :emlx_dispatch_key_by_id, closure_table],
      &:ets.info(&1, :size)
    )
  end

  test "named overrides, defaults, absent and nil overrides agree with Nx" do
    for compiler <- [EMLX, Nx.Defn.Evaluator] do
      assert Nx.to_number(Nx.Defn.jit(&single/1, compiler: compiler).(tensor(3))) == 8

      assert Nx.to_number(Nx.Defn.jit(&defaulted/1, compiler: compiler).(tensor(3))) == 8
      assert_receive {:default, 4}

      for hooks <- [%{value: nil}, %{value: false}, %{unrelated: :unused}] do
        Nx.Defn.jit(&defaulted/1, compiler: compiler, hooks: hooks).(tensor(3))
        assert_receive {:default, 4}
      end

      Nx.Defn.jit(&defaulted/1, compiler: compiler, hooks: %{value: callback(:override)}).(
        tensor(3)
      )

      assert_receive {:override, 4}
      refute_receive {:default, _}, 0
      refute_receive {:override, _}, 0
    end
  end

  test "invalid maps and matching callbacks fail explicitly" do
    assert_raise ArgumentError, ~r/:hooks option must be a map/, fn ->
      Nx.Defn.jit(&single/1, compiler: EMLX, hooks: []).(tensor(1))
    end

    for invalid <- [:bad, fn -> :ok end, fn _, _ -> :ok end] do
      assert_raise ArgumentError, ~r/invalid callback for hook :value/, fn ->
        Nx.Defn.jit(&single/1, compiler: EMLX, hooks: %{value: invalid}).(tensor(1))
      end
    end
  end

  test "overrides inside nested native while fire exactly once per iteration" do
    for compiler <- [EMLX, Nx.Defn.Evaluator] do
      result =
        Nx.Defn.jit(&nested/1, compiler: compiler, hooks: %{value: callback(:iteration)}).(
          tensor(0)
        )

      assert Nx.to_number(result) == 4

      values =
        for _ <- 1..4 do
          assert_receive {:iteration, value}
          value
        end

      assert values == [1, 2, 3, 4]
      refute_receive {:iteration, _}, 0
    end
  end

  test "repeated and concurrent executables retain their own receiver aliases" do
    parent = self()

    executables =
      for n <- 1..12 do
        receiver = :erlang.alias()
        cb = fn t -> send(receiver, {:hook, n, Nx.to_number(t)}) end

        {receiver,
         Nx.Defn.compile(&single/1, [Nx.template({}, :s32)], compiler: EMLX, hooks: %{value: cb})}
      end

    for _ <- 1..2 do
      tasks =
        for {{_alias, executable}, n} <- Enum.with_index(executables, 1) do
          Task.async(fn ->
            result = executable.(tensor(n))
            send(parent, {:result, n, Nx.to_number(result)})
          end)
        end

      Enum.each(tasks, &Task.await(&1, 30_000))

      for n <- 1..12 do
        expected = n + 1
        assert_receive {:hook, ^n, ^expected}
        result = expected * 2
        assert_receive {:result, ^n, ^result}
      end

      refute_receive {:hook, _, _}, 0
    end

    for {receiver, _} <- executables, do: :erlang.unalias(receiver)

    # Also race compilation itself, not just invocation of existing executables.
    tasks =
      for n <- 1..12 do
        receiver = :erlang.alias()

        task =
          Task.async(fn ->
            cb = fn t -> send(receiver, {:compiled_concurrently, n, Nx.to_number(t)}) end

            Nx.Defn.jit(&single/1, compiler: EMLX, hooks: %{value: cb}).(tensor(n))
            |> Nx.to_number()
          end)

        {n, receiver, task}
      end

    for {n, receiver, task} <- tasks do
      expected = n + 1
      assert Task.await(task, 30_000) == expected * 2
      assert_receive {:compiled_concurrently, ^n, ^expected}
      :erlang.unalias(receiver)
    end

    refute_receive {:compiled_concurrently, _, _}, 0
  end

  test "callback-bearing compilation adds no entries to global caches" do
    # Initialize all three tables before measuring; cache bounds are deliberately
    # left at their default infinity so eviction cannot disguise retention.
    Nx.Defn.jit(&single/1, compiler: EMLX).(tensor(0))
    before = cache_sizes()

    for n <- 1..30 do
      receiver = :erlang.alias()
      cb = fn t -> send(receiver, {:retention, n, Nx.to_number(t)}) end
      Nx.Defn.jit(&single/1, compiler: EMLX, hooks: %{value: cb}).(tensor(n))
      expected = n + 1
      assert_receive {:retention, ^n, ^expected}
      :erlang.unalias(receiver)
    end

    assert cache_sizes() == before
  end

  test "inline default callbacks also bypass global executable caches" do
    before = cache_sizes()

    for n <- 1..3 do
      receiver = :erlang.alias()

      fun = fn x ->
        Nx.Defn.Kernel.io_call(x, :inline, fn t ->
          send(receiver, {:inline, n, Nx.to_number(t)})
        end)
      end

      Nx.Defn.jit(fun, compiler: EMLX).(tensor(n))
      assert_receive {:inline, ^n, ^n}
      :erlang.unalias(receiver)
    end

    refute_receive {:inline, _, _}, 0
    assert cache_sizes() == before
  end

  test "raising, throwing and exiting callbacks release the native waiter" do
    for cb <- [
          fn _ -> raise "hook failure" end,
          fn _ -> throw(:hook_failure) end,
          fn _ -> exit(:hook_failure) end
        ] do
      assert_raise EMLX.NIFError, ~r/hook.failure/, fn ->
        Nx.Defn.jit(&single/1, compiler: EMLX, hooks: %{value: cb}).(tensor(1))
      end

      assert Nx.to_number(Nx.Defn.jit(&single/1, compiler: EMLX).(tensor(1))) == 4
    end
  end
end
