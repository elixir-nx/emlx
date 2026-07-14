defmodule EMLX.NativeLifecycleTest do
  use ExUnit.Case, async: true

  setup_all do
    root = Path.expand("../..", __DIR__)

    temporary =
      Path.join(System.tmp_dir!(), "emlx-native-lifecycle-#{System.unique_integer([:positive])}")

    File.mkdir_p!(temporary)
    fixture = compile_fixture!(root, temporary)
    plugin = compile_plugin!(root, temporary)
    on_exit(fn -> File.rm_rf!(temporary) end)
    %{fixture: fixture, plugin: plugin}
  end

  test "normal load publishes once and duplicate publication does not recapture", context do
    {output, 0} = run_subprocess(context.fixture, context.plugin, :normal)

    assert output =~ "{:load, :ok}"
    assert output =~ "{:status, {:ready, 1, 1}}"
    assert output =~ "{:registration, :ok}"
    assert output =~ "{:status_registered, {:ready, 1, 1}}"
    assert output =~ "{:duplicate, :already_initialized}"
    assert output =~ "{:status_after, {:ready, 1, 1}}"
  end

  test "NIF upgrade preserves existing native state", context do
    {output, 0} = run_upgrade_subprocess(context.fixture, context.fixture, context.plugin)

    assert output =~ "{:load, :ok}"
    assert output =~ "{:registration, :ok}"
    refute output =~ "Library upgrade-call unsuccessful"
    assert output =~ "{:upgrade_modules, [EMLX.NativeLifecycleFixture]}"
    assert output =~ "{:status_after, {:ready, 1, 1}}"
    assert output =~ "{:registration_after, :ok}"
  end

  test "registry-only mode leaves host state uninitialized", context do
    {output, 0} = run_subprocess(context.fixture, context.plugin, :registry_only)

    assert output =~ "{:load, :ok}"
    assert output =~ "{:status, {:uninitialized, 0, 0}}"
    assert output =~ "EMLX host MLX identity is unavailable"
    assert output =~ "{:status_registered, {:uninitialized, 0, 0}}"
  end

  test "resource and build failures return standard load failures", context do
    cases = [
      resource_failure: "resource_open_failed",
      build_mismatch: "build_mismatch"
    ]

    for {mode, reason} <- cases do
      {output, 0} = run_subprocess(context.fixture, context.plugin, mode)
      assert output =~ "EMLX NIF load failed: #{reason}"
      assert output =~ "{:load, {:error, {:load,"
    end
  end

  test "allocation, standard, and unknown exceptions cannot escape NIF load", context do
    cases = [
      bad_alloc: "allocation_failed",
      standard_exception: "internal_error",
      unknown_exception: "internal_error"
    ]

    for {mode, reason} <- cases do
      {output, 0} = run_subprocess(context.fixture, context.plugin, mode)
      assert output =~ "EMLX NIF load failed: #{reason}"
      assert output =~ "{:load, {:error, {:load,"
    end
  end

  test "release NIF contains no lifecycle fixture selectors or counters" do
    image = Application.app_dir(:emlx, "priv/libemlx.so")
    {symbols, 0} = System.cmd("nm", [image], stderr_to_stdout: true)

    for forbidden <- [
          "registry_only",
          "resource_failure",
          "emlx_native_image_test_capture_count",
          "emlx_native_image_test_identity_count"
        ] do
      refute symbols =~ forbidden
    end
  end

  defp run_subprocess(fixture, plugin, mode) do
    script = """
    defmodule EMLX.NativeLifecycleFixture do
      def load(path, mode), do: :erlang.load_nif(String.to_charlist(path), mode)
      def status(), do: :erlang.nif_error(:not_loaded)
      def duplicate_publication(), do: :erlang.nif_error(:not_loaded)
      def register_plugin(_name, _path), do: :erlang.nif_error(:not_loaded)
    end

    result = EMLX.NativeLifecycleFixture.load(#{inspect(fixture)}, #{inspect(mode)})
    IO.inspect({:load, result})

    if result == :ok do
      IO.inspect({:status, EMLX.NativeLifecycleFixture.status()})

      registration =
        EMLX.NativeLifecycleFixture.register_plugin(
          "lifecycle-proof",
          #{inspect(plugin)}
        )

      IO.inspect({:registration, registration})
      IO.inspect({:status_registered, EMLX.NativeLifecycleFixture.status()})

      if #{inspect(mode)} == :normal do
        IO.inspect({:duplicate, EMLX.NativeLifecycleFixture.duplicate_publication()})
        IO.inspect({:status_after, EMLX.NativeLifecycleFixture.status()})
      end
    end
    """

    System.cmd(
      System.find_executable("elixir") || raise("missing Elixir executable"),
      ["-e", script],
      stderr_to_stdout: true
    )
  end

  defp run_upgrade_subprocess(fixture, upgrade_fixture, plugin) do
    script = """
    defmodule EMLX.NativeLifecycleFixture do
      def load(path, mode), do: :erlang.load_nif(String.to_charlist(path), mode)
      def status(), do: :erlang.nif_error(:not_loaded)
      def duplicate_publication(), do: :erlang.nif_error(:not_loaded)
      def register_plugin(_name, _path), do: :erlang.nif_error(:not_loaded)
    end

    IO.inspect({:load, EMLX.NativeLifecycleFixture.load(#{inspect(fixture)}, :normal)})
    IO.inspect({:registration, EMLX.NativeLifecycleFixture.register_plugin("lifecycle-proof", #{inspect(plugin)})})
    upgrade_source = ~S'''
    defmodule EMLX.NativeLifecycleFixture do
      @on_load :load_upgrade
      def load_upgrade(), do: :erlang.load_nif(#{inspect(upgrade_fixture)}, :normal)
      def status(), do: :erlang.nif_error(:not_loaded)
      def duplicate_publication(), do: :erlang.nif_error(:not_loaded)
      def register_plugin(_name, _path), do: :erlang.nif_error(:not_loaded)
    end
    '''

    upgrade_modules =
      upgrade_source
      |> Code.compile_string()
      |> Enum.map(&elem(&1, 0))

    IO.inspect({:upgrade_modules, upgrade_modules})
    IO.inspect({:status_after, EMLX.NativeLifecycleFixture.status()})
    IO.inspect({:registration_after, EMLX.NativeLifecycleFixture.register_plugin("lifecycle-proof", #{inspect(plugin)})})
    """

    System.cmd(
      System.find_executable("elixir") || raise("missing Elixir executable"),
      ["-e", script],
      stderr_to_stdout: true
    )
  end

  defp compile_fixture!(root, temporary) do
    emlx_priv = Application.app_dir(:emlx, "priv")
    mlx_lib = Path.join(emlx_priv, "mlx/lib")
    load_path = Path.join(temporary, "native_lifecycle_fixture")
    output = load_path <> ".so"

    erts_include =
      :code.root_dir()
      |> List.to_string()
      |> Path.join("erts-#{:erlang.system_info(:version)}/include")

    platform_args =
      if match?({:unix, :darwin}, :os.type()) do
        ["-undefined", "dynamic_lookup", "-flat_namespace", "-Wl,-rpath,#{mlx_lib}"]
      else
        ["-Wl,-rpath,#{mlx_lib}", "-ldl"]
      end

    args = [
      "-std=c++20",
      "-O2",
      "-fPIC",
      "-shared",
      "-DEMLX_NATIVE_IMAGE_TESTING",
      "-DEMLX_PLUGIN_REGISTRY_TESTING",
      "-I#{Path.join(root, "c_src")}",
      "-I#{Path.join(emlx_priv, "include")}",
      "-I#{erts_include}",
      "-isystem",
      Path.join(emlx_priv, "mlx/include"),
      Path.join(root, "test/support/native_lifecycle_fixture.cpp"),
      Path.join(root, "c_src/emlx_nif_lifecycle.cpp"),
      Path.join(root, "c_src/emlx_native_image.cpp"),
      Path.join(root, "c_src/emlx_plugin_registry.cpp"),
      Path.join(root, "c_src/emlx_sha256.cpp"),
      "-I#{Path.join(root, "deps/fine/c_include")}",
      "-L#{mlx_lib}",
      "-lmlx"
    ]

    compile!(compiler(), args ++ platform_args ++ ["-o", output])
    load_path
  end

  defp compile_plugin!(root, temporary) do
    emlx_priv = Application.app_dir(:emlx, "priv")
    mlx_lib = Path.join(emlx_priv, "mlx/lib")
    output = Path.join(temporary, "libemlx_lifecycle_proof.so")

    platform_args =
      if match?({:unix, :darwin}, :os.type()) do
        ["-undefined", "dynamic_lookup", "-flat_namespace", "-Wl,-rpath,#{mlx_lib}"]
      else
        ["-Wl,-rpath,#{mlx_lib}"]
      end

    args = [
      "-std=c++20",
      "-O2",
      "-fPIC",
      "-shared",
      "-fvisibility=hidden",
      "-I#{Path.join(emlx_priv, "include")}",
      "-isystem",
      Path.join(emlx_priv, "mlx/include"),
      Path.join(root, "test/support/native_registry_plugin.cpp"),
      "-L#{mlx_lib}",
      "-lmlx"
    ]

    compile!(compiler(), args ++ platform_args ++ ["-o", output])
    output
  end

  defp compile!(compiler, args) do
    case System.cmd(compiler, args, stderr_to_stdout: true) do
      {_, 0} -> :ok
      {message, status} -> raise "fixture compilation failed (#{status}):\n#{message}"
    end
  end

  defp compiler do
    System.find_executable(System.get_env("CXX") || "c++") || raise "missing C++ compiler"
  end
end
