defmodule EMLXAxon.MixProject do
  use Mix.Project

  @version "0.3.0"
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
      # elixir_make — builds the standalone qwen3 compute plugin
      # (c_src/qwen3_plugin.cpp), `dlopen`'d at runtime by emlx's
      # `EMLX.NIF.load_plugin/2` (see lib/emlx_axon/application.ex). A
      # function ref so `Application.app_dir(:emlx, ...)` (needs the
      # :emlx dep already compiled) isn't called while this project/0
      # map is being built — mirrors emlx's own `Fine.include_dir()`
      # deferral in its mix.exs.
      make_env: fn ->
        emlx_priv_dir = Application.app_dir(:emlx, "priv")

        %{
          "MLX_INCLUDE_DIR" => Path.join(emlx_priv_dir, "mlx/include"),
          "MLX_LIB_DIR" => Path.join(emlx_priv_dir, "mlx/lib"),
          "QWEN3_ABI_INCLUDE_DIR" => Path.join(Mix.Project.deps_paths()[:emlx], "c_src/emlx_fast")
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
      {:emlx, path: "../emlx"},
      {:nx, github: "elixir-nx/nx", branch: "main", sparse: "nx", override: true},
      {:axon, "~> 0.7"},
      {:bumblebee, "~> 0.7"},
      {:ex_doc, "~> 0.34", only: :docs}
    ]
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
      links: %{"GitHub" => @source_url},
      licenses: ["Apache-2.0"],
      maintainers: ["Paulo Valente"]
    ]
  end
end
