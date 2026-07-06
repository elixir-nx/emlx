defmodule EMLX.Native.Program do
  @moduledoc false
  @enforce_keys [:num_inputs, :captures, :constants, :instructions, :outputs, :num_real_outputs]
  defstruct [:num_inputs, :captures, :constants, :instructions, :outputs, :num_real_outputs]

  @type constant :: {float(), atom()}

  @type t :: %__MODULE__{
          num_inputs: non_neg_integer(),
          captures: [reference()],
          constants: [constant()],
          instructions: [EMLX.Native.Instruction.t()],
          # Full wire output list: real outputs followed by any keepalive
          # tail (see EMLX.Native.Expr.t/0's `keepalive_refs` doc) — every
          # entry here is force-evaluated by `eval_program` when the program
          # has a runtime call, but only the first `num_real_outputs` are
          # converted to resource terms and returned to Elixir.
          outputs: [EMLX.Native.Instruction.ref()],
          num_real_outputs: non_neg_integer()
        }
end
