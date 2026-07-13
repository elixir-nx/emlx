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

defmodule EMLXAxon.PluginMetadataTest.LoaderEnvFixture do
  @moduledoc false

  def load(path), do: :erlang.load_nif(String.to_charlist(path), 0)
  def put(_name, _value), do: :erlang.nif_error(:not_loaded)
  def delete(_name), do: :erlang.nif_error(:not_loaded)
end

defmodule EMLXAxon.PluginMetadataTest do
  use ExUnit.Case, async: false

  alias EMLX.Native.Expr
  alias EMLXAxon.PluginMetadataTest.LoaderEnvFixture
  alias EMLXAxon.PluginMetadataTest.Proof

  setup_all do
    temporary =
      Path.join(System.tmp_dir!(), "emlx-plugin-proof-#{System.unique_integer([:positive])}")

    File.mkdir_p!(temporary)
    path = compile_proof_plugin!(temporary)
    loader_env = compile_loader_env_nif!(temporary)
    assert :ok = EMLX.NIF.load_plugin("proof", path)
    assert :ok = LoaderEnvFixture.load(loader_env)
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

    [1, "proof", "scale_add", 1, 1, 1, :float32, 1, 2, 2, _scale, _bias] =
      instruction.attrs

    invalid_attrs =
      [
        List.replace_at(instruction.attrs, 0, 0),
        List.replace_at(instruction.attrs, 0, 2),
        List.replace_at(instruction.attrs, 0, "1"),
        List.replace_at(instruction.attrs, 1, :proof),
        List.replace_at(instruction.attrs, 1, ""),
        List.replace_at(instruction.attrs, 1, String.duplicate("a", 129)),
        List.replace_at(instruction.attrs, 1, "invalid/name"),
        List.replace_at(instruction.attrs, 2, :scale_add),
        List.replace_at(instruction.attrs, 2, ""),
        List.replace_at(instruction.attrs, 2, String.duplicate("a", 129)),
        List.replace_at(instruction.attrs, 2, "invalid/name"),
        List.replace_at(instruction.attrs, 3, 0),
        List.replace_at(instruction.attrs, 3, 2),
        List.replace_at(instruction.attrs, 3, "1"),
        List.replace_at(instruction.attrs, 4, 0),
        List.replace_at(instruction.attrs, 4, 2),
        List.replace_at(instruction.attrs, 4, "1"),
        List.replace_at(instruction.attrs, 5, -1),
        List.replace_at(instruction.attrs, 5, 1025),
        List.replace_at(instruction.attrs, 5, "1"),
        List.replace_at(instruction.attrs, 6, :not_a_dtype),
        List.replace_at(instruction.attrs, 6, "f32"),
        List.replace_at(instruction.attrs, 7, -1),
        List.replace_at(instruction.attrs, 7, 17),
        List.replace_at(instruction.attrs, 7, "1"),
        List.replace_at(instruction.attrs, 8, -1),
        List.replace_at(instruction.attrs, 8, 2_147_483_648),
        List.replace_at(instruction.attrs, 8, "2"),
        List.replace_at(instruction.attrs, 9, -1),
        List.replace_at(instruction.attrs, 9, 16_385),
        List.replace_at(instruction.attrs, 9, "2"),
        Enum.take(instruction.attrs, 6),
        Enum.take(instruction.attrs, 7),
        Enum.take(instruction.attrs, 8),
        Enum.take(instruction.attrs, 9),
        Enum.drop(instruction.attrs, -1),
        instruction.attrs ++ [0]
      ] ++
        Enum.map(0..(length(instruction.attrs) - 1), &Enum.take(instruction.attrs, &1))

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

  test "loaders reject malformed identities and descriptors before publication" do
    path = Application.app_dir(:emlx_axon, "priv/libemlx_qwen3.so")

    for malformed <- ["", "ABC", String.duplicate("A", 64), String.duplicate("0", 63)] do
      assert {:error, reason} = EMLX.NIF.load_plugin("qwen3", path, malformed)
      assert List.to_string(reason) =~ "64 lowercase hexadecimal"
    end

    temporary =
      Path.join(System.tmp_dir!(), "emlx-plugin-malformed-#{System.unique_integer([:positive])}")

    File.mkdir_p!(temporary)
    on_exit(fn -> File.rm_rf!(temporary) end)

    fixtures = [
      {"bad_magic", :bad_magic, "bootstrap is incompatible"},
      {"bad_bootstrap_size", :bad_bootstrap_size, "bootstrap is incompatible"},
      {"bad_bootstrap_abi", :bad_bootstrap_abi, "bootstrap is incompatible"},
      {"bad_header_hash", :bad_header_hash, "bootstrap is incompatible"},
      {"bad_layout", :bad_layout, "bootstrap is incompatible"},
      {"bad_pointer_width", :bad_pointer_width, "bootstrap is incompatible"},
      {"bad_endianness", :bad_endianness, "bootstrap is incompatible"},
      {"bad_descriptor_size", :bad_descriptor_size, "bootstrap is incompatible"},
      {"null_descriptor", :null_descriptor, "bootstrap is incompatible"},
      {"misaligned_descriptor", :misaligned_descriptor, "descriptor pointer is not aligned"},
      {"bad_descriptor_abi", :bad_descriptor_abi, "descriptor is incompatible"},
      {"bad_descriptor_header_abi", :bad_descriptor_header_abi, "descriptor is incompatible"},
      {"bad_descriptor_header_hash", :bad_descriptor_header_hash, "descriptor is incompatible"},
      {"bad_callback_descriptor_size", :bad_callback_descriptor_size,
       "descriptor is incompatible"},
      {"null_plugin_name", :null_plugin_name, "field name is missing"},
      {"null_callbacks", :null_callbacks, "callback table is invalid"},
      {"too_many_callbacks", :too_many_callbacks, "callback table is invalid"},
      {"misaligned_callbacks", :misaligned_callbacks, "callback table pointer is not aligned"},
      {"null_callback", :null_callback, "callback descriptor is invalid"},
      {"bad_callback_schema", :bad_callback_schema, "callback descriptor is invalid"},
      {"bad_attr_schema", :bad_attr_schema, "callback descriptor is invalid"},
      {"bad_callback_name", :bad_callback_name, "callback descriptor is invalid"},
      {"bad_debug_utf8", :bad_debug_utf8, "debug_name is not valid UTF-8"},
      {"bad_device", :bad_device, "callback descriptor is invalid"},
      {"bad_operand_policy", :bad_operand_policy, "operand policy is invalid"},
      {"bad_output_policy", :bad_output_policy, "output policy is invalid"},
      {"duplicate_callback", :duplicate_callback, "callback names must be unique"}
    ]

    Enum.each(fixtures, fn {name, fault, expected} ->
      fixture = compile_proof_plugin!(temporary, name: name, fault: fault)
      assert {:error, reason} = EMLX.NIF.load_plugin(name, fixture)
      assert List.to_string(reason) =~ expected
    end)
  end

  test "packaged loader rejects every runtime loader override before path lookup" do
    variables =
      if match?({:unix, :darwin}, :os.type()) do
        ~w(DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH DYLD_INSERT_LIBRARIES
           DYLD_FRAMEWORK_PATH DYLD_FALLBACK_FRAMEWORK_PATH DYLD_ROOT_PATH)
      else
        ~w(LD_LIBRARY_PATH LD_PRELOAD LD_AUDIT)
      end

    for variable <- variables, value <- ["", "/untrusted"] do
      variable_chars = String.to_charlist(variable)
      assert :ok = LoaderEnvFixture.put(variable_chars, String.to_charlist(value))

      try do
        assert {:error, reason} =
                 EMLX.NIF.load_plugin(
                   "override_test",
                   "/path/that/does/not/exist",
                   String.duplicate("0", 64)
                 )

        message = List.to_string(reason)
        assert message =~ "runtime loader override #{variable}"
        refute message =~ "plugin path does not resolve"
      after
        assert :ok = LoaderEnvFixture.delete(variable_chars)
      end
    end
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

  test "eager dispatch contains exceptions from dynamic count policies" do
    input = Nx.tensor([1.0, 2.0], backend: EMLX.Backend)

    assert_raise EMLX.NIFError, ~r/intentional operand policy exception/, fn ->
      Proof.operation(input, "throwing_operand_policy")
    end

    assert_raise EMLX.NIFError, ~r/intentional output policy exception/, fn ->
      Proof.operation(input, "throwing_output_policy")
    end

    assert_all_close(Proof.scale_add(input, 2.0, 1.0), Nx.tensor([3.0, 5.0]))
  end

  test "callback errors are bounded UTF-8 and identify empty failures" do
    input = Nx.tensor([1.0, 2.0], backend: EMLX.Backend)

    oversized =
      assert_raise EMLX.NIFError, fn ->
        Proof.operation(input, "oversized_error")
      end

    assert byte_size(oversized.message) == 4096
    assert String.valid?(oversized.message)
    assert String.ends_with?(oversized.message, "... [truncated] in NIF.call_plugin/5")

    assert_raise EMLX.NIFError, ~r/plugin returned invalid UTF-8 error detail/, fn ->
      Proof.operation(input, "invalid_utf8_error")
    end

    assert_raise EMLX.NIFError,
                 ~r/plugin callback "proof\/empty_error" failed without error detail/,
                 fn -> Proof.operation(input, "empty_error") end

    assert_all_close(Proof.scale_add(input, 2.0, 1.0), Nx.tensor([3.0, 5.0]))
  end

  test "callback exceptions and wrong output counts publish no candidate outputs" do
    input = Nx.tensor([1.0, 2.0], backend: EMLX.Backend)

    failures = [
      {"throw_after_output", ~r/intentional callback exception/},
      {"unknown_throw_after_output", ~r/unknown plugin callback exception/},
      {"wrong_output_count", ~r/wrong output count/}
    ]

    Enum.each(failures, fn {callback, message} ->
      assert_raise EMLX.NIFError, message, fn -> Proof.operation(input, callback) end
      assert_all_close(input, Nx.tensor([1.0, 2.0]))
      assert_all_close(Proof.scale_add(input, 2.0, 1.0), Nx.tensor([3.0, 5.0]))
    end)
  end

  test "compiled dispatch uses the same bounded transactional failure boundary" do
    input = Nx.tensor([1.0, 2.0], backend: EMLX.Backend)

    oversized =
      assert_raise EMLX.NIFError, fn ->
        Nx.Defn.jit(
          fn value -> Proof.operation(value, "oversized_error") end,
          compiler: EMLX
        ).(input)
      end

    assert byte_size(oversized.message) <= 4096 + 32
    assert String.valid?(oversized.message)
    assert oversized.message =~ "... [truncated]"

    for callback <- [
          "throw_after_output",
          "unknown_throw_after_output",
          "wrong_output_count"
        ] do
      assert_raise EMLX.NIFError, fn ->
        Nx.Defn.jit(fn value -> Proof.operation(value, callback) end, compiler: EMLX).(input)
      end

      assert_all_close(input, Nx.tensor([1.0, 2.0]))
      assert_all_close(Proof.scale_add(input, 2.0, 1.0), Nx.tensor([3.0, 5.0]))
    end
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

  defp compile_proof_plugin!(temporary, opts \\ []) do
    emlx_priv = Application.app_dir(:emlx, "priv")
    source = Path.expand("../support/native_plugin_fixture.cpp", __DIR__)
    name = Keyword.get(opts, :name, "proof")
    output = Path.join(temporary, "libemlx_plugin_#{name}.so")

    compiler =
      System.find_executable(System.get_env("CXX") || "c++") || raise "missing C++ compiler"

    mlx_lib = Path.join(emlx_priv, "mlx/lib")

    defines =
      [~s(-DEMLX_FIXTURE_PLUGIN_NAME="#{name}")] ++
        case Keyword.get(opts, :fault) do
          nil -> []
          :bad_magic -> ["-DEMLX_FIXTURE_BAD_MAGIC"]
          :bad_bootstrap_size -> ["-DEMLX_FIXTURE_BAD_BOOTSTRAP_SIZE"]
          :bad_bootstrap_abi -> ["-DEMLX_FIXTURE_BAD_BOOTSTRAP_ABI"]
          :bad_header_hash -> ["-DEMLX_FIXTURE_BAD_HEADER_HASH"]
          :bad_layout -> ["-DEMLX_FIXTURE_BAD_LAYOUT"]
          :bad_pointer_width -> ["-DEMLX_FIXTURE_BAD_POINTER_WIDTH"]
          :bad_endianness -> ["-DEMLX_FIXTURE_BAD_ENDIANNESS"]
          :bad_descriptor_size -> ["-DEMLX_FIXTURE_BAD_DESCRIPTOR_SIZE"]
          :null_descriptor -> ["-DEMLX_FIXTURE_NULL_DESCRIPTOR"]
          :misaligned_descriptor -> ["-DEMLX_FIXTURE_MISALIGNED_DESCRIPTOR"]
          :bad_descriptor_abi -> ["-DEMLX_FIXTURE_BAD_DESCRIPTOR_ABI"]
          :bad_descriptor_header_abi -> ["-DEMLX_FIXTURE_BAD_DESCRIPTOR_HEADER_ABI"]
          :bad_descriptor_header_hash -> ["-DEMLX_FIXTURE_BAD_DESCRIPTOR_HEADER_HASH"]
          :bad_callback_descriptor_size -> ["-DEMLX_FIXTURE_BAD_CALLBACK_DESCRIPTOR_SIZE"]
          :null_plugin_name -> ["-DEMLX_FIXTURE_NULL_PLUGIN_NAME"]
          :null_callbacks -> ["-DEMLX_FIXTURE_NULL_CALLBACKS"]
          :too_many_callbacks -> ["-DEMLX_FIXTURE_TOO_MANY_CALLBACKS"]
          :misaligned_callbacks -> ["-DEMLX_FIXTURE_MISALIGNED_CALLBACKS"]
          :null_callback -> ["-DEMLX_FIXTURE_NULL_CALLBACK"]
          :bad_callback_schema -> ["-DEMLX_FIXTURE_BAD_CALLBACK_SCHEMA"]
          :bad_attr_schema -> ["-DEMLX_FIXTURE_BAD_ATTR_SCHEMA"]
          :bad_callback_name -> ["-DEMLX_FIXTURE_BAD_CALLBACK_NAME"]
          :bad_debug_utf8 -> ["-DEMLX_FIXTURE_BAD_DEBUG_UTF8"]
          :bad_device -> ["-DEMLX_FIXTURE_BAD_DEVICE"]
          :bad_operand_policy -> ["-DEMLX_FIXTURE_BAD_OPERAND_POLICY"]
          :bad_output_policy -> ["-DEMLX_FIXTURE_BAD_OUTPUT_POLICY"]
          :duplicate_callback -> ["-DEMLX_FIXTURE_DUPLICATE_CALLBACK"]
        end

    args =
      defines ++
        [
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

  defp compile_loader_env_nif!(temporary) do
    source = Path.expand("../support/loader_env_nif.c", __DIR__)
    load_path = Path.join(temporary, "loader_env_nif")
    output = load_path <> ".so"
    compiler = System.find_executable(System.get_env("CC") || "cc") || raise "missing C compiler"

    erts_include =
      :code.root_dir()
      |> List.to_string()
      |> Path.join("erts-#{:erlang.system_info(:version)}/include")

    platform_args =
      if match?({:unix, :darwin}, :os.type()), do: ["-undefined", "dynamic_lookup"], else: []

    args =
      ["-fPIC", "-shared", "-I#{erts_include}", source] ++
        platform_args ++ ["-o", output]

    case System.cmd(compiler, args, stderr_to_stdout: true) do
      {_, 0} -> load_path
      {message, status} -> raise "loader environment NIF compile failed (#{status}):\n#{message}"
    end
  end
end
