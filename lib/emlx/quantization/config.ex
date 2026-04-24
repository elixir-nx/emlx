defmodule EMLX.Quantization.Config do
  @moduledoc false

  @enforce_keys [:scales, :biases, :group_size, :bits]
  defstruct [:scales, :biases, :group_size, :bits]

  @type t :: %__MODULE__{
          scales: Nx.Tensor.t(),
          biases: Nx.Tensor.t(),
          group_size: pos_integer(),
          bits: 2 | 4 | 8
        }
end
