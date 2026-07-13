defmodule EMLX.PluginToolWrapperTest do
  use ExUnit.Case, async: true

  setup_all do
    root = Path.expand("../..", __DIR__)

    temporary =
      Path.join(System.tmp_dir!(), "emlx-tool-wrapper-#{System.unique_integer([:positive])}")

    File.mkdir_p!(temporary)
    wrapper = Path.join(temporary, "emlx_plugin_tool_wrapper")
    compiler = System.find_executable("c++") || raise "c++ compiler is required"

    {output, status} =
      System.cmd(
        compiler,
        [
          "-std=c++20",
          "-O2",
          Path.join(root, "c_src/tools/emlx_plugin_tool_wrapper.cpp"),
          "-o",
          wrapper
        ],
        stderr_to_stdout: true
      )

    if status != 0, do: raise("failed to compile tool wrapper:\n#{output}")

    fake_tool = Path.join(temporary, "fake tool")

    File.write!(
      fake_tool,
      "#!/bin/sh\n" <>
        "args_file=\"$1\"\n" <>
        "env_file=\"$2\"\n" <>
        "shift 2\n" <>
        "printf '%s\\n' \"$@\" > \"$args_file\"\n" <>
        "/usr/bin/env > \"$env_file\"\n"
    )

    File.chmod!(fake_tool, 0o755)
    on_exit(fn -> File.rm_rf!(temporary) end)
    %{wrapper: wrapper, fake_tool: fake_tool, temporary: temporary}
  end

  test "preserves argv boundaries and passes only the explicit environment", context do
    args_file = Path.join(context.temporary, "arguments")
    env_file = Path.join(context.temporary, "environment")

    {output, status} =
      System.cmd(
        context.wrapper,
        wrapper_args("compiler", "compile", context.fake_tool, [
          args_file,
          env_file,
          "argument with spaces",
          ~s(quote"inside),
          "back\\slash"
        ]),
        env: [
          {"CPATH", "/injected/include"},
          {"CPLUS_INCLUDE_PATH", "/injected/cpp"},
          {"LIBRARY_PATH", "/injected/lib"},
          {"CXXFLAGS", "-flto"},
          {"LD_PRELOAD", "/injected/library"},
          {"UNLISTED_PLUGIN_TEST", "secret"}
        ],
        stderr_to_stdout: true
      )

    assert status == 0, output
    assert File.read!(args_file) == "argument with spaces\nquote\"inside\nback\\slash\n"
    environment = File.read!(env_file)
    assert environment =~ "LC_ALL=C\n"
    assert environment =~ "LANG=C\n"
    assert environment =~ ~r{TMPDIR=/tmp/emlx-plugin-env\.[^/]+/tmp\n}
    assert environment =~ ~r{HOME=/tmp/emlx-plugin-env\.[^/]+/home\n}
    refute environment =~ "PATH="
    refute environment =~ "CPATH="
    refute environment =~ "CPLUS_INCLUDE_PATH="
    refute environment =~ "LIBRARY_PATH="
    refute environment =~ "CXXFLAGS="
    refute environment =~ "LD_PRELOAD="
    refute environment =~ "UNLISTED_PLUGIN_TEST="
    assert_isolated_environment_removed(environment)
  end

  test "removes the isolated environment after a tool failure", context do
    args_file = Path.join(context.temporary, "failure-arguments")
    env_file = Path.join(context.temporary, "failure-environment")
    tool = fake_tool!(context.temporary, "failing tool", "exit 23")

    {_output, status} =
      System.cmd(
        context.wrapper,
        wrapper_args("compiler", "compile", tool, [args_file, env_file]),
        stderr_to_stdout: true
      )

    assert status == 23
    env = File.read!(env_file)
    assert_isolated_environment_removed(env)
  end

  test "removes the isolated environment before propagating a tool signal", context do
    args_file = Path.join(context.temporary, "signal-arguments")
    env_file = Path.join(context.temporary, "signal-environment")
    tool = fake_tool!(context.temporary, "signaled tool", "kill -TERM $$")

    {_output, status} =
      System.cmd(
        context.wrapper,
        wrapper_args("compiler", "compile", tool, [args_file, env_file]),
        stderr_to_stdout: true
      )

    assert status == 128 + 15
    env = File.read!(env_file)
    assert_isolated_environment_removed(env)
  end

  test "rejects response files and LTO before invoking the real tool", context do
    marker = Path.join(context.temporary, "must-not-exist")

    for forbidden <- [
          ["@arguments"],
          ["-Wl,@arguments"],
          ["-Xlinker", "@arguments"],
          ["-flto"],
          ["-flto=thin"]
        ] do
      {_output, status} =
        System.cmd(
          context.wrapper,
          wrapper_args("compiler", "compile", context.fake_tool, [marker, marker | forbidden]),
          stderr_to_stdout: true
        )

      assert status == 2
      refute File.exists?(marker)
    end
  end

  test "rejects role mismatches and wrapper recursion", context do
    {_output, status} =
      System.cmd(
        context.wrapper,
        wrapper_args("linker", "compile", context.fake_tool, ["unused"]),
        stderr_to_stdout: true
      )

    assert status == 2

    {_output, status} =
      System.cmd(
        context.wrapper,
        wrapper_args("compiler", "compile", context.wrapper, ["unused"]),
        stderr_to_stdout: true
      )

    assert status == 2
  end

  defp wrapper_args(role, mode, real, arguments) do
    ["--role", role, "--mode", mode, "--real", real, "--" | arguments]
  end

  defp fake_tool!(temporary, name, final_command) do
    path = Path.join(temporary, name)

    File.write!(
      path,
      "#!/bin/sh\n" <>
        "args_file=\"$1\"\n" <>
        "env_file=\"$2\"\n" <>
        "/usr/bin/env > \"$env_file\"\n" <>
        final_command <> "\n"
    )

    File.chmod!(path, 0o755)
    path
  end

  defp assert_isolated_environment_removed(environment) do
    values =
      environment
      |> String.split("\n", trim: true)
      |> Map.new(fn entry ->
        [name, value] = String.split(entry, "=", parts: 2)
        {name, value}
      end)

    refute File.exists?(values["TMPDIR"])
    refute File.exists?(values["HOME"])
    refute File.exists?(Path.dirname(values["TMPDIR"]))
  end
end
