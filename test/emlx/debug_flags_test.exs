defmodule EMLX.DebugFlagsTest do
  use ExUnit.Case, async: true

  @moduledoc """
  Verifies that debug flags default to false and that call-site branches are
  dead-code-eliminated in the compiled BEAM when the flags are off.

  The private helpers (`assert_in_bounds!`, `assert_no_nan_inf!`) are defined
  unconditionally so the compiler can resolve them at the abstract-code stage.
  Call sites live inside `if @flag do … end` blocks; the Erlang compiler
  eliminates the unreachable branch from the final BEAM opcodes.

  When `EMLX.Fast` is implemented (task 05), add a parallel opcode test for
  each fast-kernel `*_impl/2` function here.
  """

  @enable_bounds_check Application.compile_env(:emlx, :enable_bounds_check, false)
  @detect_non_finites Application.compile_env(:emlx, :detect_non_finites, false)

  test "debug flags default to false" do
    assert @enable_bounds_check == false
    assert @detect_non_finites == false
  end

  test "assert_in_bounds! call is absent from BEAM opcodes of gather/4 when flag is off" do
    # The Erlang compiler eliminates the `case false of true -> call() end` branch
    # entirely from the BEAM instruction stream. Verify by walking the disassembled
    # opcodes for gather/4 and confirming the private helper is not called.
    beam_path = :code.which(EMLX.Backend)
    {:beam_file, _mod, _exports, _attrs, _info, functions} = :beam_disasm.file(beam_path)

    gather_fn =
      Enum.find(functions, fn
        {:function, :gather, 4, _, _} -> true
        _ -> false
      end)

    refute gather_fn == nil, "gather/4 must exist in EMLX.Backend"

    {:function, :gather, 4, _, instructions} = gather_fn
    called_fns = local_calls(instructions)

    refute :assert_in_bounds! in called_fns,
           "assert_in_bounds! must not appear in gather/4 opcodes when :enable_bounds_check is false"
  end

  test "assert_no_nan_inf! call is absent from BEAM opcodes of dot/7 when flag is off" do
    beam_path = :code.which(EMLX.Backend)
    {:beam_file, _mod, _exports, _attrs, _info, functions} = :beam_disasm.file(beam_path)

    dot_fn =
      Enum.find(functions, fn
        {:function, :dot, 7, _, _} -> true
        _ -> false
      end)

    refute dot_fn == nil, "dot/7 must exist in EMLX.Backend"

    {:function, :dot, 7, _, instructions} = dot_fn
    called_fns = local_calls(instructions)

    refute :assert_no_nan_inf! in called_fns,
           "assert_no_nan_inf! must not appear in dot/7 opcodes when :detect_non_finites is false"
  end

  # Collect all local (same-module) function names referenced in a BEAM opcode list.
  defp local_calls(instructions) do
    for instr <- instructions,
        match?({:call, _arity, {EMLX.Backend, _name, _}}, instr) or
          match?({:call_only, _arity, {EMLX.Backend, _name, _}}, instr),
        {_op, _arity, {EMLX.Backend, name, _}} = instr do
      name
    end
  end
end
