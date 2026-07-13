defmodule EMLXAxon.PluginArtifactTest do
  use ExUnit.Case, async: true

  setup do
    temporary =
      Path.join(System.tmp_dir!(), "emlx-plugin-artifact-#{System.unique_integer([:positive])}")

    File.mkdir_p!(temporary)
    on_exit(fn -> File.rm_rf!(temporary) end)
    %{temporary: temporary}
  end

  test "packaged Qwen3 object and image satisfy the native artifact policy" do
    priv = Application.app_dir(:emlx_axon, "priv")

    assert inspect_artifact("object", Path.join(priv, "objs/qwen3_plugin.o")) == {"", 0}
    assert inspect_artifact("image", Path.join(priv, "libemlx_qwen3.so"), policy()) == {"", 0}
  end

  test "object inspection rejects dynamic initialization and destruction", context do
    source = Path.join(context.temporary, "dynamic.cpp")
    object = Path.join(context.temporary, "dynamic.o")

    File.write!(source, """
    volatile int sink = 0;
    struct Dynamic {
      Dynamic() { sink = 1; }
      ~Dynamic() { sink = 0; }
    };
    Dynamic dynamic;
    """)

    compile!(compiler(), ["-std=c++20", "-fPIC", "-c", source, "-o", object])

    assert {message, 1} = inspect_artifact("object", object)
    assert message =~ "dynamic initialization or exit-time destruction"
  end

  test "image inspection rejects additional exported symbols", context do
    source = Path.join(context.temporary, "unexpected.cpp")
    image = Path.join(context.temporary, "unexpected.so")

    File.write!(source, """
    extern "C" __attribute__((visibility("default")))
    void unexpected_export() {}
    """)

    platform_args =
      if match?({:unix, :darwin}, :os.type()), do: ["-undefined", "dynamic_lookup"], else: []

    compile!(
      compiler(),
      ["-std=c++20", "-fPIC", "-shared", source] ++ platform_args ++ ["-o", image]
    )

    assert {message, 1} = inspect_artifact("image", image, policy())
    assert message =~ "must export only emlx_plugin_descriptor_v1"
  end

  test "image inspection enforces the committed dependency policy", context do
    restricted_policy = Path.join(context.temporary, "restricted-policy.txt")

    denied_rule =
      if match?({:unix, :darwin}, :os.type()) do
        "darwin.mlx=@rpath/libmlx.dylib\n"
      else
        "linux.mlx=libmlx.so\n"
      end

    policy()
    |> File.read!()
    |> String.replace(denied_rule, "")
    |> then(&File.write!(restricted_policy, &1))

    image = Application.app_dir(:emlx_axon, "priv/libemlx_qwen3.so")
    assert {message, 1} = inspect_artifact("image", image, restricted_policy)
    assert message =~ "unexpected"
  end

  defp inspect_artifact(mode, artifact, policy \\ nil) do
    args = if policy, do: [mode, artifact, policy], else: [mode, artifact]
    System.cmd(inspector(), args, stderr_to_stdout: true)
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

  defp inspector do
    Path.expand("../../c_src/inspect_plugin.sh", __DIR__)
  end

  defp policy do
    Path.expand("../../c_src/plugin_dependency_policy_v1.txt", __DIR__)
  end
end
