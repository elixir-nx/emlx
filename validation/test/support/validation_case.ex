defmodule EMLX.ValidationCase do
  use ExUnit.CaseTemplate

  using do
    quote do
      import Nx.Testing
    end
  end

  setup do
    Nx.default_backend({EMLX.Backend, device: :gpu})
    :ok
  end
end
