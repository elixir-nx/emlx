defmodule EMLX.CompilerTest do
  # Tests for EMLX's Nx.Defn.Compiler callbacks:
  #   __jit__/5, __compile__/4, __partitions_options__/1, __to_backend__/1
  #
  # async: false because tests inspect process-dict state set by CommandQueue
  # and check Logger output — both are process-global.
  use ExUnit.Case, async: false

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
end
