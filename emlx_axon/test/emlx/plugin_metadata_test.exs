defmodule EMLXAxon.PluginMetadataTest.Proof do
  @moduledoc false

  import Nx.Defn

  alias EMLXAxon.Native.Plugin

  deftransform scale_add(x, scale, bias) do
    attrs = [Plugin.f64_bits(scale), Plugin.f64_bits(bias)]

    if Plugin.traced?(x) do
      template = Nx.to_template(x)

      fused =
        Plugin.metadata(
          Nx.runtime_call(
            template,
            {x},
            [scale: scale, bias: bias],
            &scale_add_callback/2
          ),
          "proof",
          "scale_add",
          [x],
          attrs,
          template
        )

      Nx.Defn.Kernel.custom_grad(fused, [x], fn _gradient ->
        raise ArgumentError,
              "native plugin operation proof/scale_add does not support gradients"
      end)
    else
      scale_add_callback({x}, scale: scale, bias: bias)
    end
  end

  @doc false
  def scale_add_callback({x}, opts) do
    [output] =
      Plugin.call(
        "proof",
        "scale_add",
        [x],
        [Plugin.f64_bits(opts[:scale]), Plugin.f64_bits(opts[:bias])],
        Nx.to_template(x)
      )

    output
  end

  defn compiled(x), do: scale_add(x, 2.0, 1.0)
  defn wrong_shape(x), do: operation(x, "wrong_shape")
  defn differentiated(x), do: Nx.Defn.grad(x, &scale_add(&1, 2.0, 1.0))

  deftransform operation(x, callback) do
    attrs = [Plugin.f64_bits(2.0), Plugin.f64_bits(1.0)]
    template = Nx.to_template(x)

    if Plugin.traced?(x) do
      Plugin.metadata(
        Nx.runtime_call(
          template,
          {x},
          [callback: callback],
          &operation_callback/2
        ),
        "proof",
        callback,
        [x],
        attrs,
        template
      )
    else
      operation_callback({x}, callback: callback)
    end
  end

  @doc false
  def operation_callback({x}, opts) do
    [output] =
      Plugin.call(
        "proof",
        opts[:callback],
        [x],
        [Plugin.f64_bits(2.0), Plugin.f64_bits(1.0)],
        Nx.to_template(x)
      )

    output
  end
end

defmodule EMLXAxon.PluginMetadataTest do
  use ExUnit.Case, async: false

  alias EMLX.Native.Expr
  alias EMLXAxon.PluginMetadataTest.Proof

  setup_all do
    temporary =
      Path.join(System.tmp_dir!(), "emlx-plugin-proof-#{System.unique_integer([:positive])}")

    File.mkdir_p!(temporary)
    path = compile_proof_plugin!(temporary)
    assert :ok = EMLX.NIF.load_plugin("proof", path)
    on_exit(fn -> File.rm_rf!(temporary) end)
    :ok
  end

  test "private proof callback works eagerly and under the EMLX compiler" do
    input = Nx.tensor([1.0, 2.0], backend: EMLX.Backend)

    eager = Proof.scale_add(input, 2.0, 1.0)
    compiled = Nx.Defn.jit(&Proof.compiled/1, compiler: EMLX).(input)

    assert_all_close(eager, Nx.tensor([3.0, 5.0]))
    assert_all_close(compiled, Nx.tensor([3.0, 5.0]))
  end

  test "proof metadata lowers to one plugin instruction without a runtime call" do
    expr = Nx.Defn.debug_expr_apply(&Proof.compiled/1, [Nx.template({2}, :f32)])
    wire = expr |> Expr.lower() |> Expr.to_native()

    assert [%EMLX.Native.Instruction{op: :plugin, attrs: attrs}] = wire.instructions
    assert Enum.at(attrs, 1) == "proof"
    assert Enum.at(attrs, 2) == "scale_add"
    refute Enum.any?(wire.instructions, &(&1.op == :runtime_call))
  end

  test "plugin wire rejects malformed and noncanonical fields" do
    wire = proof_wire()
    [instruction] = wire.instructions

    invalid_attrs = [
      List.replace_at(instruction.attrs, 0, 2),
      List.replace_at(instruction.attrs, 1, :qwen3),
      List.replace_at(instruction.attrs, 3, 2),
      Enum.drop(instruction.attrs, -1),
      instruction.attrs ++ [0]
    ]

    Enum.each(invalid_attrs, fn attrs ->
      invalid = %{wire | instructions: [%{instruction | attrs: attrs}]}
      assert_raise EMLX.NIFError, fn -> compile_wire!(invalid) end
    end)
  end

  test "packaged loader rejects stale build identities and remains idempotent" do
    path = Application.app_dir(:emlx_axon, "priv/libemlx_qwen3.so")
    build_id = EMLXAxon.Native.Qwen3PluginBuild.build_id()

    assert {:error, _} = EMLX.NIF.load_plugin("qwen3", path, String.duplicate("1", 64))
    assert :ok = EMLX.NIF.load_plugin("qwen3", path, build_id)
  end

  test "fallback remains correct under Nx.Defn.Evaluator" do
    input = Nx.tensor([1.0, 2.0], backend: EMLX.Backend)
    result = Nx.Defn.jit(&Proof.compiled/1, compiler: Nx.Defn.Evaluator).(input)
    assert_all_close(result, Nx.tensor([3.0, 5.0]))
  end

  test "callback failures publish no partial output and later calls remain clean" do
    input = Nx.tensor([1.0, 2.0], backend: EMLX.Backend)

    assert_raise EMLX.NIFError, ~r/intentional partial failure/, fn ->
      Proof.operation(input, "partial_failure")
    end

    assert_all_close(Proof.scale_add(input, 2.0, 1.0), Nx.tensor([3.0, 5.0]))
  end

  test "compiled dispatch rejects an output that violates its traced template" do
    input = Nx.tensor([1.0, 2.0], backend: EMLX.Backend)

    assert_raise EMLX.NIFError, ~r/output template mismatch/, fn ->
      Nx.Defn.jit(&Proof.wrong_shape/1, compiler: EMLX).(input)
    end
  end

  test "device capabilities are checked before callback dispatch" do
    input = Nx.tensor([1.0, 2.0], backend: {EMLX.Backend, device: :cpu})

    assert_raise EMLX.NIFError, ~r/does not support the worker device/, fn ->
      Proof.operation(input, "gpu_only_scale_add")
    end

    assert_all_close(Proof.operation(input, "cpu_only_scale_add"), Nx.tensor([3.0, 5.0]))
  end

  test "compiled execution works on an explicitly selected command queue" do
    queue = EMLX.CommandQueue.new!(:cpu)
    input = Nx.tensor([1.0, 2.0], backend: {EMLX.Backend, device: :cpu})

    result =
      EMLX.CommandQueue.with_queue(queue, fn ->
        Nx.Defn.jit(&Proof.compiled/1, compiler: EMLX).(input)
      end)

    assert_all_close(result, Nx.tensor([3.0, 5.0]))
  end

  test "gradients fail explicitly" do
    input = Nx.tensor([1.0, 2.0], backend: EMLX.Backend)

    assert_raise ArgumentError, ~r/proof\/scale_add does not support gradients/, fn ->
      Nx.Defn.jit(&Proof.differentiated/1, compiler: EMLX).(input)
    end

    assert_raise ArgumentError, ~r/proof\/scale_add does not support gradients/, fn ->
      Nx.Defn.jit(&Proof.differentiated/1, compiler: Nx.Defn.Evaluator).(input)
    end
  end

  defp assert_all_close(left, right) do
    assert Nx.all_close(left, right, atol: 1.0e-5, rtol: 1.0e-5) |> Nx.to_number() == 1
  end

  defp proof_wire do
    Nx.Defn.debug_expr_apply(&Proof.compiled/1, [Nx.template({2}, :f32)])
    |> Expr.lower()
    |> Expr.to_native()
  end

  defp compile_wire!(wire) do
    {worker, _device} = EMLX.resolve_worker(EMLX.default_device())

    EMLX.NIF.compile_program(worker, wire)
    |> EMLX.unwrap!()
    |> EMLX.await_worker()
  end

  defp compile_proof_plugin!(temporary) do
    emlx_priv = Application.app_dir(:emlx, "priv")
    source = Path.expand("../support/native_plugin_fixture.cpp", __DIR__)
    output = Path.join(temporary, "libemlx_plugin_proof.so")

    compiler =
      System.find_executable(System.get_env("CXX") || "c++") || raise "missing C++ compiler"

    mlx_lib = Path.join(emlx_priv, "mlx/lib")

    args = [
      "-std=c++20",
      "-O2",
      "-fPIC",
      "-fvisibility=hidden",
      "-I#{Path.join(emlx_priv, "include")}",
      "-isystem",
      Path.join(emlx_priv, "mlx/include"),
      source,
      "-L#{mlx_lib}",
      "-lmlx",
      "-shared",
      "-Wl,-rpath,#{mlx_lib}",
      "-o",
      output
    ]

    case System.cmd(compiler, args, stderr_to_stdout: true) do
      {_, 0} -> output
      {message, status} -> raise "proof plugin compile failed (#{status}):\n#{message}"
    end
  end
end
