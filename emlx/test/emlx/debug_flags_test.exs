defmodule EMLX.DebugFlagsTest do
  use ExUnit.Case, async: true

  @moduledoc """
  Verifies that debug flags default to false and that call-site branches are
  dead-code-eliminated in the compiled BEAM when the flags are off.

  The assertion macros (`EMLX.Backend`'s private `assert_in_bounds!`, and
  `EMLX.Debug`'s public `assert_no_nan_inf!`, shared with `EMLX.Fast`) are
  defined unconditionally so the compiler can resolve them at the
  abstract-code stage. Each one branches on its flag *inside the macro body*
  at the macro's own compile time, so a call site either inlines the real
  assertion or inlines nothing at all — never a runtime `if`. The
  `dot`/`gather` tests below check this the way the macro's own definition
  would suggest (no call named after the macro ever appears, on or off,
  since macros don't compile to calls); the `assert_no_nan_inf!` extension
  tests (`conv`, `EMLX.Fast`) check the more meaningful thing instead: that
  the *expanded* NaN/Inf-check calls (`EMLX.is_nan/1`, `EMLX.is_infinity/1`)
  are themselves absent from the compiled function when the flag is off.
  """

  @enable_bounds_check Application.compile_env(:emlx, :enable_bounds_check, false)
  @detect_non_finites Application.compile_env(:emlx, :detect_non_finites, false)
  @compiler_debug Application.compile_env(:emlx, :compiler_debug, false)

  test "debug flags default to false" do
    assert @enable_bounds_check == false
    assert @detect_non_finites == false
    assert @compiler_debug == false
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

  test "EMLX.is_nan/EMLX.is_infinity calls are absent from EMLX.Backend's conv/4 opcodes when flag is off" do
    refute finite_check_calls(EMLX.Backend, :conv, 4) != [],
           "EMLX.is_nan/EMLX.is_infinity must not appear in conv/4 opcodes when :detect_non_finites is false"
  end

  test "EMLX.is_nan/EMLX.is_infinity calls are absent from EMLX.Fast's rms_norm/layer_norm/sdpa callbacks when flag is off" do
    for {fun, arity} <- [
          {:rms_norm_callback, 2},
          {:layer_norm_callback, 2},
          {:layer_norm_no_bias_callback, 2},
          {:sdpa_callback, 2},
          {:sdpa_masked_callback, 2},
          {:sdpa_causal_callback, 2},
          {:sdpa_causal_key_masked_callback, 2}
        ] do
      refute finite_check_calls(EMLX.Fast, fun, arity) != [],
             "EMLX.is_nan/EMLX.is_infinity must not appear in #{fun}/#{arity} opcodes " <>
               "when :detect_non_finites is false"
    end
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

  # `assert_no_nan_inf!` (EMLX.Debug) is a macro: it never compiles to a call
  # named after itself, on or off (see moduledoc). What actually distinguishes
  # "compiled in" from "dead-code-eliminated" is whether the *expanded* body's
  # own calls — `EMLX.is_nan/1` and `EMLX.is_infinity/1` — appear in the
  # target function's opcode list at all.
  defp finite_check_calls(module, fun, arity) do
    beam_path = :code.which(module)
    {:beam_file, _mod, _exports, _attrs, _info, functions} = :beam_disasm.file(beam_path)

    target_fn =
      Enum.find(functions, fn
        {:function, ^fun, ^arity, _, _} -> true
        _ -> false
      end)

    refute target_fn == nil, "#{inspect(module)}.#{fun}/#{arity} must exist"

    {:function, ^fun, ^arity, _, instructions} = target_fn

    for instr <- instructions,
        match?({:call_ext, _arity, {:extfunc, EMLX, :is_nan, _}}, instr) or
          match?({:call_ext, _arity, {:extfunc, EMLX, :is_infinity, _}}, instr) do
      instr
    end
  end
end
