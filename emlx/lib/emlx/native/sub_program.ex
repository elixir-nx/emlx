defmodule EMLX.Native.SubProgram do
  @moduledoc false

  # A `:while` instruction's condition or body, lowered as its own
  # self-contained flat instruction list instead of being inlined into the
  # parent program's own `instructions` — see `EMLX.Native.Expr`'s `:while`
  # moduledoc section. Decoded directly on the C++ side by Fine (see
  # emlx/compiler.hpp's `SubProgram` struct + `fine::Decoder<SubProgram>`).
  #
  # `instructions`' own `{:result, i}` refs are local to this sub-program (a
  # fresh flat accumulator per interpretation, distinct from the parent
  # program's own `{:result, i}` numbering). A `{:carry, i}` ref resolves
  # against whichever loop-carry vector the interpreting `EMLXWhile`
  # primitive passes in for the current iteration. `{:capture, i}` /
  # `{:const, i}` resolve against the *same* shared captures/constants
  # tables as the parent program — no separate tables per sub-program.
  # `{:input, i}` never appears here.

  @enforce_keys [:instructions, :outputs]
  defstruct [:instructions, :outputs]

  @type t :: %__MODULE__{
          instructions: [EMLX.Native.Instruction.t()],
          outputs: [EMLX.Native.Instruction.ref()]
        }
end
