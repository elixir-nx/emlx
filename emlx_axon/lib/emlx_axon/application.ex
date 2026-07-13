defmodule EMLXAxon.Application do
  @moduledoc """
  OTP Application for EMLXAxon.

  Eagerly loads the standalone qwen3 compute plugin (`c_src/qwen3_plugin.cpp`,
  built as `priv/libemlx_qwen3.so`) into emlx's generic native plugin
  registry via `EMLX.NIF.load_plugin/3`. The operations in
  `EMLXAxon.Qwen3.Native` and the direct `EMLX.Native.Qwen3` compatibility
  wrappers cannot dispatch until this has run. Since `:emlx` is a dependency
  of `:emlx_axon`, OTP starts it first, so `EMLX.NIF.load_plugin/3` is
  available by the time this runs.
  """

  use Application

  @doc false
  def start(_type, _args) do
    ensure_qwen3_plugin_loaded!()
    Supervisor.start_link([], strategy: :one_for_one, name: __MODULE__)
  end

  @doc false
  def ensure_qwen3_plugin_loaded! do
    path = :filename.join(:code.priv_dir(:emlx_axon), ~c"libemlx_qwen3.so")

    case EMLX.NIF.load_plugin(
           "qwen3",
           List.to_string(path),
           EMLXAxon.Native.Qwen3PluginBuild.build_id()
         ) do
      :ok ->
        :ok

      {:error, reason} ->
        raise_plugin_load!(reason)
    end
  end

  defp raise_plugin_load!(reason) do
    raise EMLX.NIFError,
          "EMLXAxon.Application could not load the qwen3 plugin: " <> List.to_string(reason)
  end
end
