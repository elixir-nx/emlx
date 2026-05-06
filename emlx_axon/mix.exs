defmodule EMLX.Axon.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/elixir-nx/emlx"

  def project do
    [
      app: :emlx_axon,
      version: @version,
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
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
      {:emlx, path: ".."},
      {:axon, "~> 0.7"},
      # Inherit emlx's Nx git pin to avoid constraint conflicts.
      {:nx, github: "elixir-nx/nx", sparse: "nx", override: true},
      {:bumblebee, github: "elixir-nx/bumblebee", override: true, only: [:dev, :test]},
      # Native serving uses validation/ forward pass for the 28-layer generate loop.
      {:emlx_validation, path: "../validation", only: [:dev, :test]}
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
      licenses: ["Apache-2.0"]
    ]
  end
end
