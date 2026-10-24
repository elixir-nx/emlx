defmodule EMLX.CompilerTest do
  use ExUnit.Case, async: false

  import Nx.Testing

  alias EMLX.CommandQueue

  describe "__to_backend__/1" do
    test "defaults to EMLX.Backend with the configured default device" do
      assert {EMLX.Backend, device: device} = EMLX.__to_backend__([])
      assert device == EMLX.default_device()
    end

    test "honours an explicit :device opt" do
      assert {EMLX.Backend, device: :cpu} = EMLX.__to_backend__(device: :cpu)
    end
  end

  describe "EMLX.Application.configure_default_device/0" do
    setup do
      config = Application.fetch_env(:emlx, :default_device)

      on_exit(fn ->
        case config do
          {:ok, device} -> Application.put_env(:emlx, :default_device, device)
          :error -> Application.delete_env(:emlx, :default_device)
        end
      end)
    end

    # Metal hosts take the :gpu branch and the Linux CI jobs take the :cpu branch.
    test "keeps :gpu without config when MLX reports the GPU as available, else sets :cpu" do
      Application.delete_env(:emlx, :default_device)
      EMLX.Application.configure_default_device()
      expected = if EMLX.NIF.is_available(:gpu) == {:ok, true}, do: :gpu, else: :cpu
      assert EMLX.default_device() == expected
    end

    test "keeps a configured :cpu" do
      Application.put_env(:emlx, :default_device, :cpu)
      EMLX.Application.configure_default_device()
      assert EMLX.default_device() == :cpu
    end

    test "raises on an unknown configured device" do
      Application.put_env(:emlx, :default_device, :tpu)

      assert_raise EMLX.NIFError, ~r/Unknown device: tpu/, fn ->
        EMLX.Application.configure_default_device()
      end
    end
  end

  describe "__partitions_options__/1" do
    @tag :metal
    test "returns a single-element list by default" do
      partitions = EMLX.__partitions_options__([])
      assert length(partitions) == 1
    end

    @tag :metal
    test "default partition has a :device key and a :command_queue struct" do
      [opts] = EMLX.__partitions_options__([])
      assert Keyword.get(opts, :device) == EMLX.default_device()
      assert %CommandQueue{device: device} = Keyword.get(opts, :command_queue)
      assert device == EMLX.default_device()
    end

    test "honours :device opt" do
      [opts] = EMLX.__partitions_options__(device: :cpu)
      assert Keyword.get(opts, :device) == :cpu
      assert %CommandQueue{device: :cpu} = Keyword.get(opts, :command_queue)
    end

    @tag :metal
    test "max_concurrency: 2 returns two partitions" do
      partitions = EMLX.__partitions_options__(max_concurrency: 2)
      assert length(partitions) == 2
    end

    @tag :metal
    test "partitions for max_concurrency: 2 have distinct queue refs" do
      [opts1, opts2] = EMLX.__partitions_options__(max_concurrency: 2)
      q1 = Keyword.get(opts1, :command_queue)
      q2 = Keyword.get(opts2, :command_queue)
      assert %CommandQueue{} = q1
      assert %CommandQueue{} = q2
      assert q1.ref != q2.ref
    end
  end

  describe "option validation" do
    test "unknown opt in __compile__ raises ArgumentError" do
      defmodule IdentFn do
        import Nx.Defn
        defn ident(x), do: x
      end

      assert_raise ArgumentError, ~r/unknown_opt/, fn ->
        Nx.Defn.compile(&IdentFn.ident/1, [Nx.template({}, :f32)],
          compiler: EMLX,
          unknown_opt: true
        )
      end
    end

    @tag :metal
    test "valid opts :device and :max_concurrency do not raise" do
      defmodule IdentFn2 do
        import Nx.Defn
        defn ident(x), do: x
      end

      compiled =
        Nx.Defn.compile(&IdentFn2.ident/1, [Nx.template({}, :f32)],
          compiler: EMLX,
          device: :gpu,
          max_concurrency: 1
        )

      result = compiled.(Nx.tensor(1.0, backend: EMLX.Backend))
      assert_in_delta Nx.to_number(result), 1.0, 1.0e-6
    end
  end

  describe ":cache option" do
    test "cache: false compiles without an entry in EMLX.CompileCache" do
      defmodule CacheFn do
        import Nx.Defn
        defn double(x), do: x * 2
      end

      table = :sys.get_state(EMLX.CompileCache).table
      before = :ets.info(table, :size)
      x = Nx.tensor([1.0, 2.0, 3.0], backend: {EMLX.Backend, device: :cpu})

      compile = fn opts ->
        Nx.Defn.compile(
          &CacheFn.double/1,
          [Nx.template({3}, :f32)],
          [compiler: EMLX, device: :cpu] ++ opts
        )
      end

      assert_all_close(compile.(cache: false).(x), Nx.tensor([2.0, 4.0, 6.0]))
      assert :ets.info(table, :size) == before

      assert_all_close(compile.([]).(x), Nx.tensor([2.0, 4.0, 6.0]))
      assert :ets.info(table, :size) == before + 1
    end
  end

  describe "__jit__/5 queue wrapping" do
    test "without :command_queue, the jit closure is returned as-is" do
      defmodule SumJitFn do
        import Nx.Defn

        defn add(a, b), do: Nx.add(a, b)
      end

      jitted = Nx.Defn.jit(&SumJitFn.add/2, compiler: EMLX)

      result =
        jitted.(Nx.tensor(1.0, backend: EMLX.Backend), Nx.tensor(2.0, backend: EMLX.Backend))

      assert_in_delta Nx.to_number(result), 3.0, 1.0e-6
    end

    @tag :metal
    test "with :command_queue, the jit closure installs the queue during execution" do
      q = CommandQueue.new!(:gpu)

      defmodule AddJitFn do
        import Nx.Defn

        defn add(a, b), do: Nx.add(a, b)
      end

      jitted = Nx.Defn.jit(&AddJitFn.add/2, compiler: EMLX, command_queue: q)

      assert Process.get(:emlx_command_queue) == nil

      result =
        jitted.(Nx.tensor(1.0, backend: EMLX.Backend), Nx.tensor(2.0, backend: EMLX.Backend))

      assert Process.get(:emlx_command_queue) == nil
      assert_in_delta Nx.to_number(result), 3.0, 1.0e-6
    end

    # elixir-nx/emlx#141: two same-device queues share one compiled program.
    # Compile is a portable function; eval uses the bound worker. Materialise
    # inside with_queue — a later to_blob on the default worker is a different
    # hole (lazy result after unbind).
    test "two CPU queues can jit the same graph" do
      q1 = CommandQueue.new!(:cpu)
      q2 = CommandQueue.new!(:cpu)

      defmodule DotExpTwoQueues do
        import Nx.Defn

        defn call(a, b), do: a |> Nx.dot(b) |> Nx.exp()
      end

      jitted = Nx.Defn.jit(&DotExpTwoQueues.call/2, compiler: EMLX, device: :cpu)

      run = fn q ->
        CommandQueue.with_queue(q, fn ->
          a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: {EMLX.Backend, device: :cpu})
          b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]], backend: {EMLX.Backend, device: :cpu})
          jitted.(a, b) |> Nx.sum() |> Nx.to_number()
        end)
      end

      r1 = run.(q1)
      r2 = run.(q2)

      assert_all_close(r1, r2)
    end
  end

  # MLX uses one process-wide default device (the GPU with Metal), so an op without
  # an explicit stream runs on the GPU even inside a :cpu program. The CPU and GPU
  # kernels round differently for these shapes, so equal bits prove the device.
  describe "compiled programs on the :cpu device" do
    for {op, shapes} <- [
          dot: [{256, 512}, {512, 256}],
          sum: [{256, 512}],
          exp_sum: [{256, 512}],
          conv: [{1, 128, 64, 64}, {128, 128, 3, 3}]
        ] do
      @tag :metal
      test "#{op} gives the eager CPU result" do
        fun = device_op(unquote(op))
        key = Nx.Random.key(7)

        {args, _key} =
          Enum.map_reduce(unquote(Macro.escape(shapes)), key, fn shape, key ->
            Nx.Random.normal(key, shape: shape, type: :f32)
          end)

        args = Enum.map(args, &Nx.backend_transfer(&1, {EMLX.Backend, device: :cpu}))
        eager = apply(fun, args) |> Nx.backend_transfer(Nx.BinaryBackend)

        compiled =
          Nx.Defn.jit(fun, compiler: EMLX, device: :cpu)
          |> apply(args)
          |> Nx.backend_transfer(Nx.BinaryBackend)

        assert Nx.to_binary(compiled) == Nx.to_binary(eager)
      end
    end
  end

  defp device_op(:dot), do: &Nx.dot/2
  defp device_op(:sum), do: &Nx.sum(&1, axes: [1])
  defp device_op(:exp_sum), do: &(&1 |> Nx.exp() |> Nx.sum(axes: [0]))
  defp device_op(:conv), do: &Nx.conv(&1, &2, padding: :same)

  # The loop condition is a carry that the graph before the loop computes. MLX runs
  # the :while primitive on the worker thread, possibly before the stream thread
  # computes that carry. Without a barrier, the loop can read the condition before
  # it is computed and then not run.
  describe "compiled :while" do
    defmodule CountDown do
      import Nx.Defn

      defn count_down(x) do
        {x, _} =
          while {x, positive = Nx.greater(Nx.sum(x), 0)}, positive do
            x = Nx.subtract(x, 1)
            {x, Nx.greater(Nx.sum(x), 0)}
          end

        x
      end
    end

    test "reads a condition that the graph before the loop computes" do
      fun = Nx.Defn.jit(&CountDown.count_down/1, compiler: EMLX, device: :cpu)

      results =
        for _ <- 1..100 do
          fun.(Nx.tensor([3.0, 2.0], backend: {EMLX.Backend, device: :cpu}))
          |> Nx.to_flat_list()
        end

      assert Enum.uniq(results) == [[0.0, -1.0]]
    end
  end

  describe "__compile__/4 queue wrapping" do
    test "without :command_queue, the compiled closure is returned as-is" do
      # Compile a minimal defn function and verify the closure can be called.
      defmodule SumFn do
        import Nx.Defn

        defn add(a, b), do: Nx.add(a, b)
      end

      compiled =
        Nx.Defn.compile(&SumFn.add/2, [Nx.template({}, :f32), Nx.template({}, :f32)],
          compiler: EMLX
        )

      result =
        compiled.(Nx.tensor(1.0, backend: EMLX.Backend), Nx.tensor(2.0, backend: EMLX.Backend))

      assert_in_delta Nx.to_number(result), 3.0, 1.0e-6
    end

    @tag :metal
    test "with :command_queue, the compiled closure installs the queue during execution" do
      q = CommandQueue.new!(:gpu)

      defmodule AddFn do
        import Nx.Defn

        defn add(a, b), do: Nx.add(a, b)
      end

      compiled =
        Nx.Defn.compile(&AddFn.add/2, [Nx.template({}, :f32), Nx.template({}, :f32)],
          compiler: EMLX,
          command_queue: q
        )

      # Call the wrapped closure; the queue should be active during execution.
      assert Process.get(:emlx_command_queue) == nil

      result =
        compiled.(Nx.tensor(1.0, backend: EMLX.Backend), Nx.tensor(2.0, backend: EMLX.Backend))

      assert Process.get(:emlx_command_queue) == nil
      assert_in_delta Nx.to_number(result), 3.0, 1.0e-6
    end
  end

  describe "dispatch cache tables" do
    test "survive the exit of the first process that compiles" do
      add_one = fn x -> Nx.add(x, 1) end

      task =
        Task.async(fn ->
          input = Nx.tensor([0.0, 1.0], backend: {EMLX.Backend, device: :cpu})
          Nx.Defn.jit(add_one, compiler: EMLX, device: :cpu).(input) |> Nx.to_flat_list()
        end)

      assert Task.await(task) == [1.0, 2.0]

      for table <- [:emlx_native_dispatch_cache, :emlx_dispatch_key_by_id] do
        assert :ets.info(table, :owner) == Process.whereis(EMLX.CompileCache)
      end
    end
  end

  # MLX compiles with recursion as deep as the graph. A 512 KB thread stack
  # overflows on this chain and the VM stops with SIGBUS.
  test "compiles a chain of 20,000 ops" do
    chain = fn x ->
      Enum.reduce(1..10_000, x, fn _, acc -> acc |> Nx.sin() |> Nx.add(0.001) end)
    end

    result = Nx.Defn.jit(chain, compiler: EMLX).(Nx.tensor(0.5, backend: EMLX.Backend))
    assert_all_close(result, chain.(Nx.tensor(0.5, backend: Nx.BinaryBackend)))
  end

  describe "non-finite constants in compiled graphs" do
    # Nx.Constants.neg_infinity/infinity/nan return constant tensors whose
    # value is stored as an atom (:neg_infinity, :infinity, :nan) in the
    # compiled program's constants table.  Before the fix, to_native/1 called
    # `v * 1.0` on every constant value, which raises ArithmeticError for atoms.

    for type <- [:f32, :f16, :bf16] do
      @type_under_test type

      test "neg_infinity constant compiles and produces -inf (#{type})" do
        t = @type_under_test

        x = Nx.tensor([1.0, -2.0], type: t, backend: EMLX.Backend)

        result =
          Nx.Defn.jit(
            fn x -> Nx.select(Nx.greater(x, 0), Nx.Constants.neg_infinity(t), x) end,
            compiler: EMLX
          ).(x)

        assert_all_close(result, Nx.tensor([:neg_infinity, -2.0], type: t))
      end

      test "infinity constant compiles and produces +inf (#{type})" do
        t = @type_under_test

        x = Nx.tensor([1.0, -2.0], type: t, backend: EMLX.Backend)

        result =
          Nx.Defn.jit(
            fn x -> Nx.select(Nx.greater(x, 0), Nx.Constants.infinity(t), x) end,
            compiler: EMLX
          ).(x)

        assert_all_close(result, Nx.tensor([:infinity, -2.0], type: t))
      end

      test "nan constant compiles and NaN propagates through arithmetic (#{type})" do
        t = @type_under_test

        x = Nx.tensor([1.0, -2.0], type: t, backend: EMLX.Backend)

        result =
          Nx.Defn.jit(
            fn x -> Nx.select(Nx.greater(x, 0), Nx.Constants.nan(t), x) end,
            compiler: EMLX
          ).(x)

        # The first element should be NaN; check via is_nan.
        assert Nx.to_number(Nx.is_nan(result[0])) == 1
        assert_all_close(result[1], Nx.tensor(-2.0, type: t))
      end
    end
  end
end
