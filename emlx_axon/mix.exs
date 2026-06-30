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
      package: package()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [

      {:emlx, path: "../emlx"},
      # {:emlx, "~> 0.3"},
      {:nx, path: "../../nx/nx", override: true},
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
