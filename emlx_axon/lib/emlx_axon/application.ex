defmodule EMLXAxon.Application do
  @moduledoc """
  OTP Application for EMLXAxon.

  Eagerly loads the standalone model compute plugins into emlx's generic
  native plugin registry. Since `:emlx` is a dependency of `:emlx_axon`, OTP
  starts it first, so `EMLX.NIF.load_plugin/2` is available here.
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
    ensure_plugin_loaded!("qwen3", "libemlx_qwen3.so")
  end

  @doc false
  def ensure_llama_plugin_loaded! do
    ensure_plugin_loaded!("llama", "libemlx_llama.so")
  end

  defp ensure_plugin_loaded!(name, filename) do
    path = :filename.join(:code.priv_dir(:emlx_axon), String.to_charlist(filename))

    case EMLX.NIF.load_plugin(name, List.to_string(path)) do
      :ok ->
        :ok

      {:error, reason} ->
        raise_plugin_load!(name, reason)
    end
  end

  defp raise_plugin_load!(name, reason) do
    raise EMLX.NIFError,
          "EMLXAxon.Application could not load the #{name} plugin: " <> List.to_string(reason)
  end
end
