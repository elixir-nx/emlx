defmodule EMLXAxon.MixProject do
  use Mix.Project

  @version "0.4.0"
  @source_url "https://github.com/elixir-nx/emlx"

  def project do
    [
      app: :emlx_axon,
      version: @version,
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs(),
      aliases: aliases(),
      description: "Axon model rewrites to swap supported nodes for EMLX.Fast Metal shaders",
      package: package(),
      # elixir_make builds the standalone Qwen3 compute plugin
      # (c_src/qwen3_plugin.cpp), loaded at runtime by EMLX.NIF.load_plugin/2.
      # A
      # function ref so `Application.app_dir(:emlx, ...)` (needs the
      # :emlx dep already compiled) isn't called while this project/0
      # map is being built — mirrors emlx's own `Fine.include_dir()`
      # deferral in its mix.exs.
      make_env: fn ->
        emlx_priv_dir = Application.app_dir(:emlx, "priv")

        %{
          "MLX_INCLUDE_DIR" => Path.join(emlx_priv_dir, "mlx/include"),
          "MLX_LIB_DIR" => Path.join(emlx_priv_dir, "mlx/lib"),
          "EMLX_PLUGIN_INCLUDE_DIR" => Path.join(emlx_priv_dir, "include")
        }
      end,
      compilers: [:elixir_make] ++ Mix.compilers()
    ]
  end

  def application do
    [extra_applications: [:logger], mod: {EMLXAxon.Application, []}]
  end

  defp deps do
    [
      {:elixir_make, "~> 0.6"},
      emlx_dep(),
      {:axon, "~> 0.7"},
      {:bumblebee, "~> 0.7"},
      {:ex_doc, "~> 0.34", only: :docs}
    ]
  end

  defp emlx_dep do
    if System.get_env("EMLX_AXON_LOCAL_EMLX") == "true" do
      {:emlx, path: "../emlx", override: true}
    else
      # Release sequencing: this source tree needs the generic plugin ABI added
      # after v0.4.0. Keep the Hex requirement publishable until the maintainer
      # assigns and releases that EMLX version, then raise this lower bound before
      # publishing the matching EMLXAxon release.
      {:emlx, "~> 0.4.0"}
    end
  end

  def cli do
    [preferred_envs: [docs: :docs, "hex.publish": :docs]]
  end

  defp docs do
    [
      main: "EMLXAxon",
      source_url: @source_url,
      source_url_pattern: "#{@source_url}/blob/v#{@version}/emlx_axon/%{path}#L%{line}",
      extras: ["README.md"]
    ]
  end

  defp aliases do
    [
      test: ["test --exclude bumblebee"]
    ]
  end

  defp package do
    [
      files: ~w(lib c_src .formatter.exs mix.exs README.md LICENSE Makefile),
      links: %{"GitHub" => @source_url},
      licenses: ["Apache-2.0"],
      maintainers: ["Paulo Valente"]
    ]
  end
end
