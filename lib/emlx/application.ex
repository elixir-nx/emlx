defmodule EMLX.Application do
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      {NifCall.Runner,
       runner_opts: [nif_module: EMLX.NIF, on_evaluated: :nif_call_evaluated], name: EMLX.Runner}
    ]

    opts = [strategy: :one_for_one, name: EMLX.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
