defmodule EMLX.NamedHooksTest do
  use ExUnit.Case, async: false
  import Nx.Defn

  # Exercise the same compiler contract on Nx 0.13 (hook) and Nx 1 (io_call).
  deftransform named(value, name) do
    op = if function_exported?(Nx.Defn.Kernel, :io_call, 2), do: :io_call, else: :hook
    apply(Nx.Defn.Kernel, op, [value, name])
  end

  deftransform with_default(value) do
    op = if function_exported?(Nx.Defn.Kernel, :io_call, 3), do: :io_call, else: :hook

    apply(Nx.Defn.Kernel, op, [
      value,
      :value,
      fn t -> send(self(), {:default, Nx.to_number(t)}) end
    ])
  end

  deftransform token_payload(value, passthrough) do
    token = apply(Nx.Defn.Kernel, :create_token, [])
    {token, _} = apply(Nx.Defn.Kernel, :hook_token, [token, value, :token])
    apply(Nx.Defn.Kernel, :attach_token, [token, passthrough])
  end

  defn single(x), do: named(x + 1, :value) * 2
  defn defaulted(x), do: with_default(x + 1) * 2

  defn nested(x) do
    {x, _} =
      while {x, i = 0}, i < 2 do
        {x, _} =
          while {x, j = 0}, j < 2 do
            {named(x + 1, :value), j + 1}
          end

        {x, i + 1}
      end

    x
  end

  defn token_loop(x) do
    {x, _} =
      while {x, i = 0}, i < 3 do
        {token_payload({x + 10, x + 20}, x + 1), i + 1}
      end

    x
  end

  defn token_tuple(x), do: token_payload(x + 10, {x + 1, x + 2})

  defn nested_token(x) do
    {x, _} =
      while {x, i = 0}, i < 2 do
        {x, _} =
          while {x, j = 0}, j < 2 do
            {token_payload(with_default(x + 10), x + 1), j + 1}
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
        op = if function_exported?(Nx.Defn.Kernel, :io_call, 3), do: :io_call, else: :hook

        apply(Nx.Defn.Kernel, op, [
          x,
          :inline,
          fn t -> send(receiver, {:inline, n, Nx.to_number(t)}) end
        ])
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

  @tag timeout: 45_000
  test "caller death and late replies are safe in a disposable BEAM" do
    paths =
      :code.get_path()
      |> Enum.map(&List.to_string/1)
      |> Enum.filter(&(Path.basename(&1) in ["ebin", "consolidated"]))
      |> Enum.reverse()
      |> Enum.flat_map(&["-pa", &1])

    script = Path.expand("../fixtures/runtime_call_lifecycle.exs", __DIR__)
    # Python owns and kills the disposable process on timeout, including if a
    # native callback blocks VM shutdown. No hanging native work enters ExUnit.
    runner = """
    import subprocess, sys
    try:
        p = subprocess.run(sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        sys.stdout.buffer.write(p.stdout)
        sys.exit(p.returncode)
    except subprocess.TimeoutExpired as e:
        sys.stdout.buffer.write(e.stdout or b'')
        print('disposable BEAM timed out')
        sys.exit(124)
    """

    args =
      ["-c", runner, System.find_executable("elixir"), "--erl", "+S 2:2"] ++ paths ++ [script]

    {output, status} = System.cmd("python3", args, stderr_to_stdout: true)
    assert status == 0, output
    assert output =~ "runtime-call lifecycle OK"
  end

  @tag skip: not function_exported?(Nx.Defn.Kernel, :create_token, 0)
  test "Nx 0.13 token hooks in native while alias the trailing outputs" do
    cb = fn {a, b} -> send(self(), {:token, Nx.to_number(a), Nx.to_number(b)}) end

    for compiler <- [EMLX, Nx.Defn.Evaluator] do
      result = Nx.Defn.jit(&token_loop/1, compiler: compiler, hooks: %{token: cb}).(tensor(0))
      assert Nx.to_number(result) == 3

      for i <- 0..2 do
        assert_receive {:token, a, b}
        assert {a, b} == {i + 10, i + 20}
      end

      refute_receive {:token, _, _}, 0
    end

    {a, b} =
      Nx.Defn.jit(&token_tuple/1, compiler: EMLX, hooks: %{token: callback(:tuple_token)}).(
        tensor(5)
      )

    assert {Nx.to_number(a), Nx.to_number(b)} == {6, 7}
    assert_receive {:tuple_token, 15}
    refute_receive {:tuple_token, _}, 0

    for compiler <- [EMLX, Nx.Defn.Evaluator], override? <- [false, true] do
      hooks = %{token: callback(:nested_token)}
      hooks = if override?, do: Map.put(hooks, :value, callback(:barrier)), else: hooks
      tag = if override?, do: :barrier, else: :default
      result = Nx.Defn.jit(&nested_token/1, compiler: compiler, hooks: hooks).(tensor(0))
      assert Nx.to_number(result) == 4

      for i <- 10..13 do
        assert_receive {first_tag, first_value}
        assert {first_tag, first_value} == {tag, i}
        assert_receive {second_tag, second_value}
        assert {second_tag, second_value} == {:nested_token, i}
      end

      refute_receive {^tag, _}, 0
      refute_receive {:nested_token, _}, 0
    end
  end
end
