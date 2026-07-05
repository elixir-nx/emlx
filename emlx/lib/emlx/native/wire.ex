defmodule EMLX.Native.Wire.Instruction do
  @moduledoc false

  # One compiled-program instruction, as passed to the `compile_program` NIF.
  # Decoded directly on the C++ side by Fine (see emlx_compiler.hpp's
  # `Instruction` struct + `fine::Decoder<Instruction>`), replacing the old
  # to_wire/1 format's parallel `op_names`/`operands`/`attrs` lists.

  @enforce_keys [:op, :operands, :attrs]
  defstruct [:op, :operands, :attrs]

  # A reference to an already-produced value, resolved on the C++ side against
  # the runtime inputs / closed-over captures / closed-over constants / the
  # flat per-eval results accumulator (in that order).
  @type ref :: {:input, non_neg_integer()}
              | {:capture, non_neg_integer()}
              | {:const, non_neg_integer()}
              | {:result, non_neg_integer()}

  # Most attrs are plain integers (shapes, axes, flags, f64_bits-encoded
  # floats); a few are MLX dtype atoms or quantized_matmul mode atoms (see
  # emlx_compiler.hpp's `Attr` type) — no int<->meaning lookup table needs to
  # be kept in sync between Elixir and C++ for those.
  @type attr :: integer() | atom()

  @type t :: %__MODULE__{op: atom(), operands: [ref()], attrs: [attr()]}
end

defmodule EMLX.Native.Wire.Program do
  @moduledoc false

  # The full wire payload for the `compile_program` NIF (see
  # EMLX.Native.Expr.to_wire/1), decoded directly by Fine on the C++ side
  # (emlx_compiler.hpp's `Program` struct) instead of 8 positional NIF args.

  @enforce_keys [:num_inputs, :captures, :constants, :instructions, :outputs]
  defstruct [:num_inputs, :captures, :constants, :instructions, :outputs]

  @type constant :: {float(), atom()}

  @type t :: %__MODULE__{
          num_inputs: non_neg_integer(),
          captures: [reference()],
          constants: [constant()],
          instructions: [EMLX.Native.Wire.Instruction.t()],
          outputs: [EMLX.Native.Wire.Instruction.ref()]
        }
end
