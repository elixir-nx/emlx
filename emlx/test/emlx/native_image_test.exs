defmodule EMLX.NativeImageTest do
  use ExUnit.Case, async: true

  setup_all do
    root = Path.expand("../..", __DIR__)

    temporary =
      Path.join(System.tmp_dir!(), "emlx-native-image-#{System.unique_integer([:positive])}")

    File.mkdir_p!(temporary)
    anchor = compile_anchor!(temporary)
    fixture = compile_fixture!(root, temporary)
    on_exit(fn -> File.rm_rf!(temporary) end)
    %{anchor: anchor, fixture: fixture, temporary: temporary}
  end

  test "captures an unchanged native image", context do
    path = copy_anchor!(context, "unchanged")
    assert run!(context.fixture, ["unchanged", path]) == "unchanged"
  end

  test "rejects deterministic same-inode mutation between file snapshots", context do
    path = copy_anchor!(context, "mutate")
    assert run!(context.fixture, ["mutate", path]) == "changed_while_hashing"
  end

  test "atomic replacement keeps the identity pinned to the opened inode", context do
    path = copy_anchor!(context, "replace")
    assert run!(context.fixture, ["replace", path]) == "opened_original"
  end

  test "release NIF contains no native-image test seam" do
    image = Application.app_dir(:emlx, "priv/libemlx.so")
    {symbols, 0} = System.cmd("nm", [image], stderr_to_stdout: true)

    refute symbols =~ "emlx_native_image_set_test_barrier"
    refute symbols =~ "g_test_barrier"
    refute symbols =~ "registry_only"
  end

  defp copy_anchor!(context, name) do
    extension = if match?({:unix, :darwin}, :os.type()), do: "dylib", else: "so"
    destination = Path.join(context.temporary, "#{name}.#{extension}")
    File.cp!(context.anchor, destination)
    destination
  end

  defp compile_anchor!(temporary) do
    source = Path.join(temporary, "anchor.cpp")
    extension = if match?({:unix, :darwin}, :os.type()), do: "dylib", else: "so"
    output = Path.join(temporary, "anchor.#{extension}")

    File.write!(source, """
    extern "C" __attribute__((visibility("default")))
    const char *emlx_test_anchor() { return "anchor"; }
    """)

    shared_args = if match?({:unix, :darwin}, :os.type()), do: ["-dynamiclib"], else: ["-shared"]
    compile!(compiler(), ["-std=c++20", "-fPIC", source] ++ shared_args ++ ["-o", output])
    output
  end

  defp compile_fixture!(root, temporary) do
    emlx_priv = Application.app_dir(:emlx, "priv")
    mlx_lib = Path.join(emlx_priv, "mlx/lib")
    output = Path.join(temporary, "native_image_fixture")

    platform_args =
      if match?({:unix, :darwin}, :os.type()) do
        ["-Wl,-rpath,#{mlx_lib}"]
      else
        ["-Wl,-rpath,#{mlx_lib}", "-ldl"]
      end

    args = [
      "-std=c++20",
      "-O2",
      "-DEMLX_NATIVE_IMAGE_TESTING",
      "-I#{Path.join(root, "c_src")}",
      "-isystem",
      Path.join(emlx_priv, "mlx/include"),
      Path.join(root, "test/support/native_image_fixture.cpp"),
      Path.join(root, "c_src/emlx_native_image.cpp"),
      Path.join(root, "c_src/emlx_sha256.cpp"),
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

  defp run!(executable, args) do
    case System.cmd(executable, args, stderr_to_stdout: true) do
      {output, 0} -> String.trim(output)
      {message, status} -> raise "fixture failed (#{status}):\n#{message}"
    end
  end

  defp compiler do
    System.find_executable(System.get_env("CXX") || "c++") || raise "missing C++ compiler"
  end
end
