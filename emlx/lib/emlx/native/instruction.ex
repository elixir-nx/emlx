defmodule EMLX.Native.Instruction do
  @moduledoc false

  # One compiled-program instruction, as passed to the `compile_program` NIF.
  # Decoded directly on the C++ side by Fine (see emlx_compiler.hpp's
  # `Instruction` struct + `fine::Decoder<Instruction>`), replacing the old
  # to_native/1 format's parallel `op_names`/`operands`/`attrs` lists.

  @enforce_keys [:op, :operands, :attrs]
  defstruct [:op, :operands, :attrs, subprograms: []]

  # A reference to an already-produced value, resolved on the C++ side against
  # the runtime inputs / closed-over captures / closed-over constants / the
  # flat per-eval results accumulator / (inside a `:while` sub-program only)
  # the current loop-carry slot (in that order).
  @type ref ::
          {:input, non_neg_integer()}
          | {:capture, non_neg_integer()}
          | {:const, non_neg_integer()}
          | {:result, non_neg_integer()}
          | {:carry, non_neg_integer()}

  # Most attrs are plain integers (shapes, axes, flags, f64_bits-encoded
  # floats); a few are MLX dtype atoms or quantized_matmul mode atoms (see
  # emlx_compiler.hpp's `Attr` type) — no int<->meaning lookup table needs to
  # be kept in sync between Elixir and C++ for those.
  @type attr :: integer() | atom()

  @type t :: %__MODULE__{
          op: atom(),
          operands: [ref()],
          attrs: [attr()],
          subprograms: [EMLX.Native.SubProgram.t()]
        }
end
