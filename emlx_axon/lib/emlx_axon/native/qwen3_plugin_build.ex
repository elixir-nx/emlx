defmodule EMLXAxon.Native.Qwen3PluginBuild do
  @moduledoc false

  @build_id_path Application.app_dir(:emlx_axon, "priv/emlx_qwen3.build_id")
  @external_resource @build_id_path
  @build_id @build_id_path |> File.read!() |> String.trim()

  unless String.match?(@build_id, ~r/\A[0-9a-f]{64}\z/) do
    raise "invalid generated Qwen3 plugin build identity"
  end

  @doc false
  def build_id, do: @build_id
end
