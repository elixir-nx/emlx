defmodule EMLX.Case do
  @moduledoc """
  Test case for tensor assertions
  """

  use ExUnit.CaseTemplate

  using do
    quote do
      import EMLX.Case
      import Nx.Testing
    end
  end

  setup do
    Nx.default_backend(EMLX.Backend)
    :ok
  end
end
