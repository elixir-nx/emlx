defmodule EMLX.Quantization.Config do
  @moduledoc false

  @enforce_keys [:scales, :biases, :group_size, :bits]
  defstruct [:scales, :biases, :group_size, :bits, transpose: nil]

  @type t :: %__MODULE__{
          scales: Nx.Tensor.t(),
          biases: Nx.Tensor.t(),
          group_size: pos_integer(),
          bits: 2 | 4 | 8,
          # When set, overrides the auto-detection of the transpose flag in
          # quantized_dot (based on right_axes). nil = auto-detect.
          transpose: boolean() | nil
        }
end
