# Provide aliases for running common tasks
defmodule EMLXRoot do
  use Mix.Project

  def project do
    [
      app: :emlx_root,
      version: "0.1.0",
      deps: [{:emlx, path: "emlx"}],
      aliases: [
        setup: cmd("deps.get"),
        compile: cmd("compile"),
        test: cmd("test"),
        format: cmd("format")
      ]
    ]
  end

  defp cmd(command) do
    ansi = IO.ANSI.enabled?()
    base = ["--erl", "-elixir ansi_enabled #{ansi}", "-S", "mix", command]

    for app <- ~w(emlx emlx_axon) do
      fn args ->
        {_, res} = System.cmd("elixir", base ++ args, into: IO.binstream(:stdio, :line), cd: app)

        if res > 0 do
          System.at_exit(fn _ -> exit({:shutdown, 1}) end)
        end
      end
    end
  end
end
