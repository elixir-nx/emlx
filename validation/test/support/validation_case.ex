defmodule EMLX.ValidationCase do
  use ExUnit.CaseTemplate

  using do
    quote do
      import Nx.Testing
    end
  end

  setup do
    EMLX.GPUPool.checkout()
    on_exit(fn -> EMLX.GPUPool.checkin() end)
    Nx.default_backend({EMLX.Backend, device: :gpu})
    :ok
  end
end
