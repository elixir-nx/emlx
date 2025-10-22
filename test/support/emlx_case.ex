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
end
