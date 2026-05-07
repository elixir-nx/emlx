defmodule EMLX.Validation.MixProject do
  use Mix.Project

  def project do
    [
      app: :emlx_validation,
      version: "0.1.0",
      elixir: "~> 1.17",
      deps: deps(),
      test_paths: ["test"],
      elixirc_paths: ["lib", "test/support"]
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:emlx, path: ".."},
      # Inherit emlx's Nx git pin so axon's Hex constraint does not conflict.
      {:nx, github: "elixir-nx/nx", sparse: "nx", override: true},
      {:bumblebee, github: "elixir-nx/bumblebee", override: true},
      {:axon, "~> 0.7"},
      {:benchee, "~> 1.3", only: :dev}
      # exla omitted — only needed for golden regeneration, added manually when required
    ]
  end
end
