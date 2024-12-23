defmodule EMLX.Application do
  use Application

  def start(_type, _args) do
    children = [
      {NifCall.Runner, runner_opts: [nif_module: EMLX.NIF], name: EMLX.Runner}
    ]

    Supervisor.start_link(children, strategy: :one_for_one, name: EMLX.Supervisor)
  end
end
