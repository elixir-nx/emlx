defmodule EMLXAxon.Application do
  @moduledoc """
  OTP Application for EMLXAxon.

  Eagerly loads standalone model compute plugins into emlx's generic native
  plugin registry via `EMLX.NIF.load_plugin/2`. Every `EMLX.Native.*` model NIF
  errors with `{:error, _}` until this has run — since `:emlx` is a
  dependency of `:emlx_axon`, OTP starts it first, so `EMLX.NIF.load_plugin/2`
  is always available by the time this runs.
  """

  use Application

  @doc false
  def start(_type, _args) do
    ensure_qwen3_plugin_loaded!()
    ensure_llama_plugin_loaded!()
    Supervisor.start_link([], strategy: :one_for_one, name: __MODULE__)
  end

  @doc false
  def ensure_qwen3_plugin_loaded! do
    ensure_plugin_loaded!("qwen3", ~c"libemlx_qwen3.so")
  end

  @doc false
  def ensure_llama_plugin_loaded! do
    ensure_plugin_loaded!("llama", ~c"libemlx_llama.so")
  end

  defp ensure_plugin_loaded!(name, lib_name) do
    path = :filename.join(:code.priv_dir(:emlx_axon), lib_name)

    case EMLX.NIF.load_plugin(name, List.to_string(path)) do
      :ok ->
        :ok

      {:error, reason} ->
        raise EMLX.NIFError,
              "EMLXAxon.Application could not load the #{name} plugin: " <>
                List.to_string(reason)
    end
  end
end
