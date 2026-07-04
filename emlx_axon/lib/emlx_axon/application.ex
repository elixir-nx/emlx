defmodule EMLXAxon.Application do
  @moduledoc """
  OTP Application for EMLXAxon.

  Eagerly loads the standalone qwen3 compute plugin (`c_src/qwen3_plugin.cpp`,
  built as `priv/libemlx_qwen3.so`) into emlx's generic native plugin
  registry via `EMLX.NIF.load_plugin/2`. Every `EMLX.Native.Qwen3.*` NIF
  errors with `{:error, _}` until this has run — since `:emlx` is a
  dependency of `:emlx_axon`, OTP starts it first, so `EMLX.NIF.load_plugin/2`
  is always available by the time this runs.
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

    case EMLX.NIF.load_plugin("qwen3", List.to_string(path)) do
      :ok ->
        :ok

      {:error, reason} ->
        raise EMLX.NIFError,
              "EMLXAxon.Application could not load the qwen3 plugin: " <> List.to_string(reason)
    end
  end
end
