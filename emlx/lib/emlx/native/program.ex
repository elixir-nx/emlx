
defmodule EMLX.Native.Program do
  @moduledoc false
  @enforce_keys [:num_inputs, :captures, :constants, :instructions, :outputs]
  defstruct [:num_inputs, :captures, :constants, :instructions, :outputs]

  @type constant :: {float(), atom()}

  @type t :: %__MODULE__{
          num_inputs: non_neg_integer(),
          captures: [reference()],
          constants: [constant()],
          instructions: [EMLX.Native.Instruction.t()],
          outputs: [EMLX.Native.Instruction.ref()]
        }
end
