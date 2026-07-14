defmodule EMLXAxon.Mix.CompilerProbe do
  @moduledoc false

  @probe_args ["-dM", "-E", "-x", "c++", "/dev/null"]

  def detect!(compiler) do
    case System.cmd(compiler, @probe_args, stderr_to_stdout: true) do
      {macros, 0} ->
        case family_from_macros(macros) do
          {:ok, family} -> family
          :error -> Mix.raise("unsupported C++ compiler for the native plugin")
        end

      {output, status} ->
        detail = output |> String.trim() |> String.slice(0, 512)

        Mix.raise(
          "failed to probe the C++ compiler (exit status #{status})" <>
            if(detail == "", do: "", else: ": #{detail}")
        )
    end
  end

  def family_from_macros(macros) when is_binary(macros) do
    cond do
      Regex.match?(~r/^#define __clang__(?:\s|$)/m, macros) -> {:ok, "clang"}
      Regex.match?(~r/^#define __GNUC__(?:\s|$)/m, macros) -> {:ok, "gcc"}
      true -> :error
    end
  end

  def ensure_plugin_build_support!(emlx_priv_dir) do
    required = [
      "include/emlx/plugin/abi.hpp",
      "include/emlx/plugin/toolchain.hpp",
      "include/emlx/plugin/build_compat.hpp",
      "build_support/emlx_plugin_tool_wrapper",
      "build_support/emlx_build_identity"
    ]

    case Enum.find(required, &(not File.regular?(Path.join(emlx_priv_dir, &1)))) do
      nil ->
        :ok

      missing ->
        Mix.raise(
          "the resolved EMLX dependency does not provide native plugin build support " <>
            "(missing priv/#{missing}); use the sibling EMLX checkout during joint development " <>
            "and publish EMLX before publishing EMLXAxon"
        )
    end
  end
end
