defmodule EMLXAxon.CompilerProbeTest do
  use ExUnit.Case, async: true

  alias EMLXAxon.Mix.CompilerProbe

  test "detects Apple Clang before its GNU compatibility macros" do
    macros = """
    #define __GNUC__ 4
    #define __apple_build_version__ 17000013
    #define __clang__ 1
    """

    assert CompilerProbe.family_from_macros(macros) == {:ok, "clang"}
  end

  test "detects upstream Clang" do
    macros = """
    #define __GNUC__ 4
    #define __clang__ 1
    #define __clang_major__ 20
    """

    assert CompilerProbe.family_from_macros(macros) == {:ok, "clang"}
  end

  test "detects GCC independently of version banner wording" do
    macros = """
    #define __GNUC__ 13
    #define __GNUC_MINOR__ 3
    """

    assert CompilerProbe.family_from_macros(macros) == {:ok, "gcc"}
  end

  test "rejects unsupported predefined macros" do
    assert CompilerProbe.family_from_macros("#define _MSC_VER 1940\n") == :error
  end

  test "reports unsupported and failed compiler probes", context do
    unsupported = executable!(context.test, "unsupported", "echo '#define _MSC_VER 1940'")
    failing = executable!(context.test, "failing", "echo 'probe failed' >&2\nexit 23")

    assert_raise Mix.Error, ~r/unsupported C\+\+ compiler/, fn ->
      CompilerProbe.detect!(unsupported)
    end

    assert_raise Mix.Error, ~r/failed to probe.*exit status 23.*probe failed/s, fn ->
      CompilerProbe.detect!(failing)
    end
  end

  test "requires every packaged EMLX plugin build artifact", context do
    priv = Path.join(System.tmp_dir!(), "emlx-probe-#{context.test}")
    on_exit(fn -> File.rm_rf!(priv) end)

    files = [
      "include/emlx/plugin/abi.hpp",
      "include/emlx/plugin/toolchain.hpp",
      "include/emlx/plugin/build_compat.hpp",
      "build_support/emlx_plugin_tool_wrapper",
      "build_support/emlx_build_identity"
    ]

    Enum.each(files, fn relative ->
      path = Path.join(priv, relative)
      File.mkdir_p!(Path.dirname(path))
      File.write!(path, "fixture")
    end)

    assert :ok = CompilerProbe.ensure_plugin_build_support!(priv)
    File.rm!(Path.join(priv, "include/emlx/plugin/abi.hpp"))

    assert_raise Mix.Error, ~r/does not provide native plugin build support/, fn ->
      CompilerProbe.ensure_plugin_build_support!(priv)
    end
  end

  defp executable!(test_name, name, body) do
    directory = Path.join(System.tmp_dir!(), "emlx-compiler-probe-#{test_name}")
    File.mkdir_p!(directory)
    path = Path.join(directory, name)
    File.write!(path, "#!/bin/sh\n#{body}\n")
    File.chmod!(path, 0o755)
    ExUnit.Callbacks.on_exit(fn -> File.rm_rf!(directory) end)
    path
  end
end
