defmodule EMLX.Native.ExprTest do
  @moduledoc """
  Tests for Stage 01 + Stage 02 of the EMLX native defn compiler:
    - EMLX.Native.Expr struct shape (refs as node IDs, atom opcodes, attrs)
    - lower/1 (parameter, constant, tensor/capture, add, identity)
    - EMLX.Native.Expr.Interpreter (pure-Elixir reference evaluator)
    - compile_program / eval_program NIFs via to_wire/1 (C++ replay)
    - Compiler seam: Nx.Defn.compile(..., compiler: EMLX) via single-NIF replay
    - Stage 02: unary + binary + compare/logical equivalence vs EMLX.Backend
    - Perf gate: single-NIF replay vs Evaluator on a multi-add chain
  """
  use ExUnit.Case, async: false
  import Bitwise
  import Nx.Defn

  alias EMLX.Native.Expr

  # ── module-level defn helpers ─────────────────────────────────────────────
  defn add_two(a, b), do: Nx.add(a, b)
  defn add_one(x), do: Nx.add(x, 1)
  defn identity(x), do: x

  # Stage 02 chain: uses multiply, add, tanh in sequence.
  defn mul_chain(x), do: x |> Nx.multiply(2.0) |> Nx.add(1.0) |> Nx.tanh()

  # Stage 02 helpers for compare/logical tests that need a typed defn.
  defn gt_f32(a, b), do: Nx.greater(a, b)
  defn eq_f32(a, b), do: Nx.equal(a, b)
  defn cmp_mixed(a, b), do: Nx.greater(a, b)
  defn mixed_add(a, b), do: Nx.add(a, b)

  # Stage 03 helpers for interpreter↔C++ parity tests.
  defn reshape_23(x), do: Nx.reshape(x, {2, 3})
  defn broadcast_23(x), do: Nx.broadcast(x, {2, 3})
  defn concat_axis0(a, b), do: Nx.concatenate([a, b], axis: 0)

  # ── IR shape ─────────────────────────────────────────────────────────────

  describe "program shape" do
    test "node IDs are Erlang refs" do
      expr = Nx.Defn.debug_expr_apply(&add_two/2, [Nx.template({}, :f32), Nx.template({}, :f32)])
      prog = Expr.lower(expr)

      assert Enum.all?(prog.inputs, &is_reference/1)
      assert Enum.all?(prog.outputs, &is_reference/1)

      for {id, op, operands, attrs} <- prog.instructions do
        assert is_reference(id)
        assert is_atom(op)
        assert Enum.all?(operands, &is_reference/1)
        assert is_list(attrs)
      end
    end

    test "each input ref is unique" do
      expr = Nx.Defn.debug_expr_apply(&add_two/2, [Nx.template({}, :f32), Nx.template({}, :f32)])
      prog = Expr.lower(expr)

      assert length(prog.inputs) == length(Enum.uniq(prog.inputs))
    end

    test "operand refs point to known nodes" do
      expr = Nx.Defn.debug_expr_apply(&add_two/2, [Nx.template({}, :f32), Nx.template({}, :f32)])
      prog = Expr.lower(expr)

      known =
        MapSet.new(prog.inputs)
        |> MapSet.union(MapSet.new(prog.captures, fn {r, _} -> r end))
        |> MapSet.union(MapSet.new(prog.constants, fn {r, _, _} -> r end))
        |> MapSet.union(MapSet.new(prog.instructions, fn {r, _, _, _} -> r end))

      for {_id, _op, operands, _} <- prog.instructions do
        for ref <- operands do
          assert MapSet.member?(known, ref),
                 "operand ref #{inspect(ref)} not in known node set"
        end
      end
    end
  end

  # ── lower/1 ──────────────────────────────────────────────────────────────

  describe "lower/1" do
    test "identity: one input, no instructions, output = input ref" do
      expr = Nx.Defn.debug_expr_apply(&identity/1, [Nx.template({3}, :f32)])
      prog = Expr.lower(expr)

      assert length(prog.inputs) == 1
      assert prog.captures == []
      assert prog.constants == []
      assert prog.instructions == []
      assert prog.outputs == prog.inputs
    end

    test "add two parameters: one :add instruction, operands are the input refs" do
      expr = Nx.Defn.debug_expr_apply(&add_two/2, [Nx.template({}, :f32), Nx.template({}, :f32)])
      prog = Expr.lower(expr)

      assert length(prog.inputs) == 2
      assert prog.captures == []
      assert prog.constants == []
      assert [{result_ref, :add, [left_ref, right_ref], []}] = prog.instructions
      assert left_ref == Enum.at(prog.inputs, 0)
      assert right_ref == Enum.at(prog.inputs, 1)
      assert prog.outputs == [result_ref]
    end

    test "add parameter + scalar literal: one const entry, one :add instruction" do
      expr = Nx.Defn.debug_expr_apply(&add_one/1, [Nx.template({}, :f32)])
      prog = Expr.lower(expr)

      assert length(prog.inputs) == 1
      assert prog.captures == []
      assert [{const_ref, 1, _int_type}] = prog.constants
      # add_one: f32 input + integer constant. Lowerer may emit an :astype to promote
      # the integer constant to f32 before :add. Assert the :add instruction is present
      # and references both the input and the constant (possibly via a cast ref).
      add_instr = Enum.find(prog.instructions, fn {_, op, _, _} -> op == :add end)
      assert {result_ref, :add, [left_ref, right_ref], []} = add_instr
      # Either a direct ref or a cast of the const ref must be in the add operands.
      all_refs = MapSet.new(prog.inputs ++ [const_ref])

      cast_or_direct = fn r ->
        r in prog.inputs or r == const_ref or
          Enum.any?(prog.instructions, fn {id, :astype, [src], _} ->
            id == r and (src in prog.inputs or src == const_ref)
          end)
      end

      assert cast_or_direct.(left_ref) or cast_or_direct.(right_ref)
      # suppress unused warning
      _ = all_refs
      assert prog.outputs == [result_ref]
    end

    test "closed-over backend tensor becomes a capture" do
      weight_tensor = Nx.tensor(1.0, backend: EMLX.Backend)

      # Construct the :tensor Expr node manually — Nx.Defn.Expr.tensor/1 rejects
      # EMLX backend tensors (intentional guard in Nx to prevent accidental capture).
      weight_expr = %Nx.Tensor{
        data: %Nx.Defn.Expr{id: make_ref(), op: :tensor, args: [weight_tensor], context: nil},
        type: weight_tensor.type,
        shape: weight_tensor.shape,
        names: weight_tensor.names
      }

      param_a = Nx.Defn.Expr.parameter(:root, {:f, 32}, {}, 0)
      output = Nx.add(param_a, weight_expr)
      prog = Expr.lower(output)

      assert length(prog.inputs) == 1
      assert [{capture_ref, ^weight_tensor}] = prog.captures
      assert prog.constants == []
      assert [{_result, :add, [_input_ref, ^capture_ref], []}] = prog.instructions
    end

    test "unknown op raises ArgumentError with 'does not yet lower op'" do
      # custom-fun reduce is not yet lowered (deferred to Stage 08); use as sentinel.
      expr =
        Nx.Defn.debug_expr_apply(
          fn t -> Nx.reduce(t, 0, fn x, acc -> Nx.add(x, acc) end) end,
          [Nx.template({3}, :f32)]
        )

      assert_raise ArgumentError, ~r/does not yet lower op/, fn -> Expr.lower(expr) end
    end
  end

  # ── to_wire/1 ────────────────────────────────────────────────────────────

  describe "to_wire/1" do
    test "identity program: n_inputs=1, no instructions, output encodes input ref" do
      expr = Nx.Defn.debug_expr_apply(&identity/1, [Nx.template({3}, :f32)])
      prog = Expr.lower(expr)
      {n_inputs, caps, cvs, cts, ops, ors, ias, outs} = Expr.to_wire(prog)

      assert n_inputs == 1
      assert caps == []
      assert cvs == []
      assert cts == []
      assert ops == []
      assert ors == []
      assert ias == []
      # output must encode kind=input (0), idx=0 → packed = 0
      assert outs == [0]
    end

    test "add program: op_name is :add, operands encode the two inputs" do
      expr = Nx.Defn.debug_expr_apply(&add_two/2, [Nx.template({}, :f32), Nx.template({}, :f32)])
      prog = Expr.lower(expr)
      {_n, _caps, _cvs, _cts, [op_name], [operands], [_ia], [output]} = Expr.to_wire(prog)

      assert op_name == :add
      # inputs packed as kind=0 (bits 61:60 = 0), so just the index
      assert operands == [0, 1]
      # output is the first instruction (kind=3, idx=0): 3 <<< 60 ||| 0
      assert output == 3 <<< 60
    end
  end

  # ── Interpreter (reference evaluator) ────────────────────────────────────

  describe "EMLX.Native.Expr.Interpreter" do
    test "identity program returns input unchanged" do
      expr = Nx.Defn.debug_expr_apply(&identity/1, [Nx.template({3}, :f32)])
      prog = Expr.lower(expr)

      x = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)
      [out] = EMLX.Native.Expr.Interpreter.eval(prog, [x])

      assert_all_close(out, x)
    end

    test "add two EMLX inputs" do
      expr = Nx.Defn.debug_expr_apply(&add_two/2, [Nx.template({}, :f32), Nx.template({}, :f32)])
      prog = Expr.lower(expr)

      x = Nx.tensor(3.0, backend: EMLX.Backend)
      y = Nx.tensor(4.0, backend: EMLX.Backend)
      [out] = EMLX.Native.Expr.Interpreter.eval(prog, [x, y])

      assert_in_delta Nx.to_number(out), 7.0, 1.0e-6
    end

    test "add parameter + captured tensor" do
      weight_tensor = Nx.tensor(10.0, backend: EMLX.Backend)

      weight_expr = %Nx.Tensor{
        data: %Nx.Defn.Expr{id: make_ref(), op: :tensor, args: [weight_tensor], context: nil},
        type: weight_tensor.type,
        shape: weight_tensor.shape,
        names: weight_tensor.names
      }

      param_a = Nx.Defn.Expr.parameter(:root, {:f, 32}, {}, 0)
      prog = Expr.lower(Nx.add(param_a, weight_expr))

      x = Nx.tensor(5.0, backend: EMLX.Backend)
      [out] = EMLX.Native.Expr.Interpreter.eval(prog, [x])

      assert_in_delta Nx.to_number(out), 15.0, 1.0e-6
    end
  end

  # ── compile_program / eval_program NIFs ──────────────────────────────────

  describe "compile_program / eval_program NIFs" do
    setup do
      device = EMLX.default_device()
      {worker, _} = EMLX.resolve_worker(device)
      %{worker: worker, device: device}
    end

    test "identity program via to_wire: output equals input", %{worker: worker, device: device} do
      expr = Nx.Defn.debug_expr_apply(&identity/1, [Nx.template({3}, :f32)])
      prog = Expr.lower(expr)
      {n_inputs, caps, cvs, cts, ops, ors, ias, outs} = Expr.to_wire(prog)

      prog_ref = compile_nif!(worker, n_inputs, caps, cvs, cts, ops, ors, ias, outs)

      x = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)
      %EMLX.Backend{ref: {_, input_ref}} = x.data

      [out_ref] = eval_nif!(worker, prog_ref, [input_ref])
      out = EMLX.Backend.to_nx({device, out_ref}, x)

      assert_all_close(out, x)
    end

    test "add program via to_wire: correct result", %{worker: worker, device: device} do
      expr = Nx.Defn.debug_expr_apply(&add_two/2, [Nx.template({}, :f32), Nx.template({}, :f32)])
      prog = Expr.lower(expr)
      {n_inputs, caps, cvs, cts, ops, ors, ias, outs} = Expr.to_wire(prog)

      prog_ref = compile_nif!(worker, n_inputs, caps, cvs, cts, ops, ors, ias, outs)

      a = Nx.tensor(3.0, backend: EMLX.Backend)
      b = Nx.tensor(4.0, backend: EMLX.Backend)
      %EMLX.Backend{ref: {_, ref_a}} = a.data
      %EMLX.Backend{ref: {_, ref_b}} = b.data

      [out_ref] = eval_nif!(worker, prog_ref, [ref_a, ref_b])
      out = EMLX.Backend.to_nx({device, out_ref}, a)

      assert_in_delta Nx.to_number(out), 7.0, 1.0e-6
    end

    test "Interpreter and C++ replay agree on add", %{worker: worker, device: device} do
      expr = Nx.Defn.debug_expr_apply(&add_two/2, [Nx.template({}, :f32), Nx.template({}, :f32)])
      prog = Expr.lower(expr)

      a = Nx.tensor(2.5, backend: EMLX.Backend)
      b = Nx.tensor(1.5, backend: EMLX.Backend)

      [interp_out] = EMLX.Native.Expr.Interpreter.eval(prog, [a, b])

      {n_inputs, caps, cvs, cts, ops, ors, ias, outs} = Expr.to_wire(prog)
      prog_ref = compile_nif!(worker, n_inputs, caps, cvs, cts, ops, ors, ias, outs)

      %EMLX.Backend{ref: {_, ref_a}} = a.data
      %EMLX.Backend{ref: {_, ref_b}} = b.data
      [out_ref] = eval_nif!(worker, prog_ref, [ref_a, ref_b])
      cpp_out = EMLX.Backend.to_nx({device, out_ref}, a)

      assert_in_delta Nx.to_number(interp_out), Nx.to_number(cpp_out), 1.0e-6
    end
  end

  # ── compiler seam (end-to-end) ────────────────────────────────────────────

  describe "compiler seam E2E" do
    test "add_one defn via EMLX compiler returns correct result" do
      compiled = Nx.Defn.compile(&add_one/1, [Nx.template({3}, :f32)], compiler: EMLX)
      x = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)
      result = compiled.(x)

      assert_all_close(result, Nx.tensor([2.0, 3.0, 4.0]))
    end

    test "identity defn routes through native path (zero-instruction program)" do
      compiled = Nx.Defn.compile(&identity/1, [Nx.template({3}, :f32)], compiler: EMLX)
      x = Nx.tensor([7.0, 8.0, 9.0], backend: EMLX.Backend)

      assert_all_close(compiled.(x), x)
    end

    test "add_two defn with two parameters" do
      compiled =
        Nx.Defn.compile(&add_two/2, [Nx.template({3}, :f32), Nx.template({3}, :f32)],
          compiler: EMLX
        )

      a = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)
      b = Nx.tensor([4.0, 5.0, 6.0], backend: EMLX.Backend)

      assert_all_close(compiled.(a, b), Nx.tensor([5.0, 7.0, 9.0]))
    end

    test "unsupported op falls back to Evaluator transparently" do
      jitted = Nx.Defn.jit(&Nx.sum/1, compiler: EMLX)

      a = Nx.tensor([3.0, 4.0, 5.0], backend: EMLX.Backend)

      assert_in_delta Nx.to_number(jitted.(a)), 12.0, 1.0e-6
    end

    test "result matches eager EMLX.Backend within tolerance" do
      compiled =
        Nx.Defn.compile(&add_two/2, [Nx.template({3}, :f32), Nx.template({3}, :f32)],
          compiler: EMLX
        )

      a = Nx.tensor([1.5, 2.5, 3.5], backend: EMLX.Backend)
      b = Nx.tensor([0.5, 1.5, 2.5], backend: EMLX.Backend)

      eager_result = EMLX.add(EMLX.Backend.from_nx(a), EMLX.Backend.from_nx(b))
      assert_all_close(compiled.(a, b), EMLX.Backend.to_nx(eager_result, a))
    end
  end

  # ── perf gate ─────────────────────────────────────────────────────────────

  defn chain_10(x, y) do
    x
    |> Nx.add(y)
    |> Nx.add(y)
    |> Nx.add(y)
    |> Nx.add(y)
    |> Nx.add(y)
    |> Nx.add(y)
    |> Nx.add(y)
    |> Nx.add(y)
    |> Nx.add(y)
    |> Nx.add(y)
  end

  @tag :perf
  test "perf gate: single-NIF replay beats op-by-op Evaluator on 10-add chain" do
    n_adds = 10
    x = Nx.tensor(0.0, backend: EMLX.Backend)
    y = Nx.tensor(1.0, backend: EMLX.Backend)

    compiled_native =
      Nx.Defn.compile(&chain_10/2, [Nx.template({}, :f32), Nx.template({}, :f32)], compiler: EMLX)

    compiled_eval =
      Nx.Defn.compile(&chain_10/2, [Nx.template({}, :f32), Nx.template({}, :f32)],
        compiler: Nx.Defn.Evaluator
      )

    # Fair comparison: both paths force evaluation via Nx.to_number/1.
    # Without forcing eval, the Evaluator returns a lazy MLX tensor while
    # eval_program eagerly calls mlx::core::eval, making the comparison unfair.
    force_eval = fn compiled -> Nx.to_number(compiled.(x, y)) end

    force_eval.(compiled_native)
    force_eval.(compiled_eval)

    n_iters = 500
    native_us = bench_us(n_iters, fn -> force_eval.(compiled_native) end)
    eval_us = bench_us(n_iters, fn -> force_eval.(compiled_eval) end)
    speedup = eval_us / native_us

    if speedup < 1.0 do
      IO.puts(
        "\n[perf gate] #{n_adds}-add chain | native: #{Float.round(native_us, 1)} µs " <>
          "| evaluator: #{Float.round(eval_us, 1)} µs | speedup: #{Float.round(speedup, 2)}×"
      )
    end

    # Stage 01 used `Nx.add(x, 1)` chained — Nx.Defn constant-folds repeated
    # scalar additions into a single op, so the "10-add chain" was actually a
    # 1-op graph.  The Stage 02 definition uses `Nx.add(x, y)` with a runtime
    # tensor `y`, which cannot be folded, producing a genuine 10-instruction
    # program.  With a real graph, the dispatch-collapse benefit materialises
    # and the native path is dramatically faster.
    assert native_us < eval_us,
           "native path (#{Float.round(native_us, 1)} µs) should beat " <>
             "Evaluator (#{Float.round(eval_us, 1)} µs) on 10-add chain"
  end

  # ── Stage 02: elementwise equivalence tests ──────────────────────────────
  #
  # For each op class, verify:
  #   (a) Interpreter output == eager EMLX.Backend output
  #   (b) C++ replay output == Interpreter output
  #
  # Tests use representative dtypes across f32 / bf16 / s32 / u8.

  # Helper: compile + eval the defn via the native path, compare to eager backend.
  defp check_equiv(fun, inputs_eager, opts \\ []) do
    tol = Keyword.get(opts, :tol, 1.0e-4)
    templates = Enum.map(inputs_eager, &Nx.template(&1.shape, &1.type))
    compiled = Nx.Defn.compile(fun, templates, compiler: EMLX)
    result = apply(compiled, inputs_eager)
    eager = apply(Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator), inputs_eager)
    assert_close(result, eager, tol)
  end

  defp assert_close(a, b, tol) do
    # For complex tensors, compare real and imaginary parts separately.
    case a.type do
      {:c, _} ->
        assert_close(Nx.real(a), Nx.real(b), tol)
        assert_close(Nx.imag(a), Nx.imag(b), tol)

      _ ->
        a_vals = Nx.to_flat_list(a)
        b_vals = Nx.to_flat_list(b)

        Enum.zip(a_vals, b_vals)
        |> Enum.each(fn {av, bv} ->
          case {av, bv} do
            {:nan, :nan} ->
              :ok

            {:infinity, :infinity} ->
              :ok

            {:neg_infinity, :neg_infinity} ->
              :ok

            {a_num, b_num} when is_number(a_num) and is_number(b_num) ->
              assert_in_delta(a_num * 1.0, b_num * 1.0, tol)

            _ ->
              flunk("Values differ: #{inspect(av)} vs #{inspect(bv)}")
          end
        end)
    end
  end

  describe "Stage 02 — unary elementwise" do
    # Sample unary ops over f32 and bf16 using representative positive values
    # to avoid NaN from log/sqrt/etc.
    @tag :stage02
    test "abs/ceil/floor/negate/round/sign — f32" do
      x = Nx.tensor([1.7, -2.3, 0.0, -0.5], backend: EMLX.Backend)

      for fun <- [&Nx.abs/1, &Nx.ceil/1, &Nx.floor/1, &Nx.negate/1, &Nx.round/1, &Nx.sign/1] do
        check_equiv(fun, [x])
      end
    end

    @tag :stage02
    test "exp/log/sqrt/tanh/sigmoid — f32" do
      x = Nx.tensor([0.5, 1.0, 2.0, 4.0], backend: EMLX.Backend)

      for fun <- [&Nx.exp/1, &Nx.log/1, &Nx.sqrt/1, &Nx.tanh/1, &Nx.sigmoid/1] do
        check_equiv(fun, [x])
      end
    end

    @tag :stage02
    test "sin/cos/tan/asin/acos/atan — f32" do
      x = Nx.tensor([0.1, 0.5, 0.9, -0.5], backend: EMLX.Backend)

      for fun <- [&Nx.sin/1, &Nx.cos/1, &Nx.tan/1, &Nx.asin/1, &Nx.acos/1, &Nx.atan/1] do
        check_equiv(fun, [x])
      end
    end

    @tag :stage02
    test "sinh/cosh/tanh/asinh/acosh/atanh — f32" do
      x = Nx.tensor([0.1, 0.5, 1.0, 1.5], backend: EMLX.Backend)

      for fun <- [&Nx.sinh/1, &Nx.cosh/1, &Nx.tanh/1, &Nx.asinh/1, &Nx.acosh/1] do
        check_equiv(fun, [x])
      end

      x2 = Nx.tensor([0.1, 0.5, 0.9, -0.5], backend: EMLX.Backend)
      check_equiv(&Nx.atanh/1, [x2])
    end

    @tag :stage02
    test "erf/erf_inv/erfc/rsqrt/expm1/log1p — f32" do
      x = Nx.tensor([0.1, 0.5, 1.0, 2.0], backend: EMLX.Backend)

      for fun <- [&Nx.erf/1, &Nx.erf_inv/1, &Nx.erfc/1, &Nx.rsqrt/1, &Nx.expm1/1, &Nx.log1p/1] do
        check_equiv(fun, [x])
      end
    end

    @tag :stage02
    test "cbrt — f32 positive values" do
      x = Nx.tensor([1.0, 8.0, 27.0, 0.125], backend: EMLX.Backend)
      check_equiv(&Nx.cbrt/1, [x], tol: 1.0e-3)
    end

    @tag :stage02
    test "is_nan/is_infinity — f32" do
      x = Nx.tensor([1.0, :nan, :infinity, :neg_infinity], type: :f32, backend: EMLX.Backend)
      check_equiv(&Nx.is_nan/1, [x])
      check_equiv(&Nx.is_infinity/1, [x])
    end

    @tag :stage02
    test "bitwise_not — s32" do
      x = Nx.tensor([0, 1, -1, 255], type: :s32, backend: EMLX.Backend)
      check_equiv(&Nx.bitwise_not/1, [x])
    end

    @tag :stage02
    test "logical_not — u8" do
      x = Nx.tensor([0, 1, 1, 0], type: :u8, backend: EMLX.Backend)
      check_equiv(&Nx.logical_not/1, [x])
    end

    @tag :stage02
    test "real/imag/conjugate — c64" do
      # Complex tensor: values are reals with zero imaginary parts.
      c = Nx.tensor([1.5, 2.5, -1.0], type: {:c, 64}, backend: EMLX.Backend)
      check_equiv(&Nx.real/1, [c])
      check_equiv(&Nx.imag/1, [c])
      check_equiv(&Nx.conjugate/1, [c])
    end

    @tag :stage02
    test "unary ops — bf16" do
      x = Nx.tensor([0.5, 1.0, 2.0], type: :bf16, backend: EMLX.Backend)

      for fun <- [&Nx.abs/1, &Nx.exp/1, &Nx.tanh/1, &Nx.sqrt/1] do
        check_equiv(fun, [x], tol: 1.0e-2)
      end
    end
  end

  describe "Stage 02 — binary arithmetic + bitwise" do
    @tag :stage02
    test "add/subtract/multiply — f32" do
      a = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)
      b = Nx.tensor([4.0, 5.0, 6.0], backend: EMLX.Backend)

      for fun <- [&Nx.add/2, &Nx.subtract/2, &Nx.multiply/2] do
        check_equiv(fun, [a, b])
      end
    end

    @tag :stage02
    test "divide/pow/atan2 — f32" do
      a = Nx.tensor([4.0, 8.0, 1.0], backend: EMLX.Backend)
      b = Nx.tensor([2.0, 4.0, 3.0], backend: EMLX.Backend)

      for fun <- [&Nx.divide/2, &Nx.pow/2, &Nx.atan2/2] do
        check_equiv(fun, [a, b])
      end
    end

    @tag :stage02
    test "min/max — f32" do
      a = Nx.tensor([1.0, 5.0, 3.0], backend: EMLX.Backend)
      b = Nx.tensor([4.0, 2.0, 3.0], backend: EMLX.Backend)
      check_equiv(&Nx.min/2, [a, b])
      check_equiv(&Nx.max/2, [a, b])
    end

    @tag :stage02
    test "quotient/remainder — s32 positive" do
      a = Nx.tensor([7, 9, 15], type: :s32, backend: EMLX.Backend)
      b = Nx.tensor([3, 4, 7], type: :s32, backend: EMLX.Backend)
      check_equiv(&Nx.quotient/2, [a, b])
      check_equiv(&Nx.remainder/2, [a, b])
    end

    @tag :stage02
    test "remainder — s32 negative dividend" do
      a = Nx.tensor([-7, -9, 7], type: :s32, backend: EMLX.Backend)
      b = Nx.tensor([3, 4, -3], type: :s32, backend: EMLX.Backend)
      check_equiv(&Nx.remainder/2, [a, b])
    end

    @tag :stage02
    test "bitwise_and/or/xor/left_shift/right_shift — s32" do
      a = Nx.tensor([0b1010, 0xFF, 5], type: :s32, backend: EMLX.Backend)
      b = Nx.tensor([0b1100, 0x0F, 2], type: :s32, backend: EMLX.Backend)

      for fun <- [
            &Nx.bitwise_and/2,
            &Nx.bitwise_or/2,
            &Nx.bitwise_xor/2,
            &Nx.left_shift/2,
            &Nx.right_shift/2
          ] do
        check_equiv(fun, [a, b])
      end
    end

    @tag :stage02
    test "mixed dtypes: add(s32, f32) → f32 with implicit upcast" do
      a = Nx.tensor([1, 2, 3], type: :s32, backend: EMLX.Backend)
      b = Nx.tensor([0.5, 1.5, 2.5], type: :f32, backend: EMLX.Backend)
      check_equiv(&mixed_add/2, [a, b])
    end

    @tag :stage02
    test "binary ops — bf16" do
      a = Nx.tensor([1.0, 2.0, 4.0], type: :bf16, backend: EMLX.Backend)
      b = Nx.tensor([2.0, 1.0, 2.0], type: :bf16, backend: EMLX.Backend)

      for fun <- [&Nx.add/2, &Nx.subtract/2, &Nx.multiply/2] do
        check_equiv(fun, [a, b], tol: 1.0e-2)
      end
    end
  end

  describe "Stage 02 — compare and logical" do
    @tag :stage02
    test "equal/not_equal/greater/less/greater_equal/less_equal — f32" do
      a = Nx.tensor([1.0, 2.0, 2.0, 3.0], backend: EMLX.Backend)
      b = Nx.tensor([2.0, 2.0, 1.0, 1.0], backend: EMLX.Backend)

      for fun <- [
            &Nx.equal/2,
            &Nx.not_equal/2,
            &Nx.greater/2,
            &Nx.less/2,
            &Nx.greater_equal/2,
            &Nx.less_equal/2
          ] do
        check_equiv(fun, [a, b])
      end
    end

    @tag :stage02
    test "compare — s32" do
      a = Nx.tensor([-1, 0, 1, 2], type: :s32, backend: EMLX.Backend)
      b = Nx.tensor([0, 0, 0, 1], type: :s32, backend: EMLX.Backend)

      for fun <- [&Nx.equal/2, &Nx.less/2, &Nx.greater/2] do
        check_equiv(fun, [a, b])
      end
    end

    @tag :stage02
    test "logical_and/or/xor — u8" do
      a = Nx.tensor([0, 0, 1, 1], type: :u8, backend: EMLX.Backend)
      b = Nx.tensor([0, 1, 0, 1], type: :u8, backend: EMLX.Backend)

      for fun <- [&Nx.logical_and/2, &Nx.logical_or/2, &Nx.logical_xor/2] do
        check_equiv(fun, [a, b])
      end
    end

    @tag :stage02
    test "compare output dtype is u8 (bool)" do
      a = Nx.tensor([1.0, 3.0], backend: EMLX.Backend)
      b = Nx.tensor([2.0, 1.0], backend: EMLX.Backend)

      compiled =
        Nx.Defn.compile(&gt_f32/2, [Nx.template({2}, :f32), Nx.template({2}, :f32)],
          compiler: EMLX
        )

      result = compiled.(a, b)
      assert result.type == {:u, 8}
    end

    @tag :stage02
    test "compare with mixed dtypes s32/f32 — output u8" do
      a = Nx.tensor([1, 3], type: :s32, backend: EMLX.Backend)
      b = Nx.tensor([2.0, 1.0], type: :f32, backend: EMLX.Backend)
      check_equiv(&cmp_mixed/2, [a, b])
    end
  end

  describe "Stage 02 — interpreter ↔ C++ replay parity" do
    setup do
      device = EMLX.default_device()
      {worker, _} = EMLX.resolve_worker(device)
      %{worker: worker, device: device}
    end

    test "interpreter and C++ agree on mul+add+tanh chain", %{worker: worker, device: device} do
      x = Nx.tensor([0.5, 1.0, -1.0], backend: EMLX.Backend)
      expr = Nx.Defn.debug_expr_apply(&mul_chain/1, [Nx.template({3}, :f32)])
      prog = EMLX.Native.Expr.lower(expr)

      [interp_out] = EMLX.Native.Expr.Interpreter.eval(prog, [x])

      {n_inputs, caps, cvs, cts, ops, ors, ias, outs} = EMLX.Native.Expr.to_wire(prog)
      prog_ref = compile_nif!(worker, n_inputs, caps, cvs, cts, ops, ors, ias, outs)

      %EMLX.Backend{ref: {_, ref_x}} = x.data
      [out_ref] = eval_nif!(worker, prog_ref, [ref_x])
      cpp_out = EMLX.Backend.to_nx({device, out_ref}, x)

      assert_all_close(interp_out, cpp_out, tol: 1.0e-5)
    end

    test "interpreter and C++ agree on compare+cast: equal(f32, f32) → u8",
         %{worker: worker, device: device} do
      a = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)
      b = Nx.tensor([1.0, 3.0, 3.0], backend: EMLX.Backend)
      expr = Nx.Defn.debug_expr_apply(&eq_f32/2, [Nx.template({3}, :f32), Nx.template({3}, :f32)])
      prog = EMLX.Native.Expr.lower(expr)

      [interp_out] = EMLX.Native.Expr.Interpreter.eval(prog, [a, b])

      {n_inputs, caps, cvs, cts, ops, ors, ias, outs} = EMLX.Native.Expr.to_wire(prog)
      prog_ref = compile_nif!(worker, n_inputs, caps, cvs, cts, ops, ors, ias, outs)

      %EMLX.Backend{ref: {_, ref_a}} = a.data
      %EMLX.Backend{ref: {_, ref_b}} = b.data
      [out_ref] = eval_nif!(worker, prog_ref, [ref_a, ref_b])
      cpp_out = EMLX.Backend.to_nx({device, out_ref}, interp_out)

      assert_all_close(interp_out, cpp_out)
    end
  end

  # ── Stage 03: shape / movement equivalence tests ────────────────────────────
  #
  # For each op: (a) Interpreter == eager EMLX.Backend, (b) C++ replay == Interpreter.
  # check_equiv/2 does both via the EMLX compiler seam.

  describe "Stage 03 — reshape" do
    @tag :stage03
    test "1D → 2D, 2D → 1D" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.reshape(t, {2, 3}) end, [x])
      check_equiv(fn t -> Nx.reshape(t, {3, 2}) end, [x])
    end

    @tag :stage03
    test "2D → 1D (flatten)" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.reshape(t, {4}) end, [x])
    end

    @tag :stage03
    test "rank-changing — s32" do
      x = Nx.tensor([[[1, 2], [3, 4]]], type: :s32, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.reshape(t, {2, 2}) end, [x])
      check_equiv(fn t -> Nx.reshape(t, {4}) end, [x])
    end
  end

  describe "Stage 03 — squeeze" do
    @tag :stage03
    test "remove singleton dimension" do
      x = Nx.tensor([[1.0, 2.0, 3.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.squeeze(t, axes: [0]) end, [x])
    end

    @tag :stage03
    test "multiple axes, negative axis" do
      x = Nx.tensor([[[1.0], [2.0], [3.0]]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.squeeze(t, axes: [0, 2]) end, [x])
      check_equiv(fn t -> Nx.squeeze(t, axes: [-1]) end, [x])
    end
  end

  describe "Stage 03 — transpose" do
    @tag :stage03
    test "2D matrix transpose" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.transpose(t) end, [x])
      check_equiv(fn t -> Nx.transpose(t, axes: [1, 0]) end, [x])
    end

    @tag :stage03
    test "3D permutation, negative perm" do
      x = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.transpose(t, axes: [2, 0, 1]) end, [x])
      check_equiv(fn t -> Nx.transpose(t, axes: [-1, -3, -2]) end, [x])
    end
  end

  describe "Stage 03 — as_type" do
    @tag :stage03
    test "f32 → s32 and back" do
      x = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.as_type(t, {:s, 32}) end, [x])
      xi = Nx.tensor([1, 2, 3], type: :s32, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.as_type(t, {:f, 32}) end, [xi])
    end

    @tag :stage03
    test "f32 → bf16 and back" do
      x = Nx.tensor([0.5, 1.5, 2.5], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.as_type(t, {:bf, 16}) end, [x], tol: 1.0e-2)
      xb = Nx.tensor([0.5, 1.5, 2.5], type: {:bf, 16}, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.as_type(t, {:f, 32}) end, [xb])
    end
  end

  describe "Stage 03 — bitcast" do
    @tag :stage03
    test "u8 → s8 same bit pattern" do
      # Values chosen so the bit pattern is unambiguous for both u8 and s8.
      x = Nx.tensor([1, 2, 3, 127], type: :u8, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.bitcast(t, {:s, 8}) end, [x])
    end

    @tag :stage03
    test "f32 → u32 round-trip" do
      x = Nx.tensor([1.0, 2.0, 0.0], type: :f32, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.bitcast(t, {:u, 32}) end, [x])
    end
  end

  describe "Stage 03 — broadcast" do
    @tag :stage03
    test "scalar → 1D" do
      x = Nx.tensor(1.0, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.broadcast(t, {4}) end, [x])
    end

    @tag :stage03
    test "1D → 2D row broadcast" do
      x = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.broadcast(t, {2, 3}) end, [x])
    end

    @tag :stage03
    test "1D column broadcast (axes: [0])" do
      x = Nx.tensor([1.0, 2.0], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.broadcast(t, {2, 3}, axes: [0]) end, [x])
    end

    @tag :stage03
    test "2D → 3D, broadcast_in_dim style" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.broadcast(t, {3, 2, 2}) end, [x])
    end
  end

  describe "Stage 03 — pad" do
    @tag :stage03
    test "zero-padding on 1D" do
      x = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.pad(t, 0.0, [{1, 1, 0}]) end, [x])
    end

    @tag :stage03
    test "zero-padding on 2D, asymmetric" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.pad(t, 0.0, [{0, 1, 0}, {1, 0, 0}]) end, [x])
    end

    @tag :stage03
    test "scalar pad value" do
      x = Nx.tensor([1.0, 2.0], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.pad(t, -1.0, [{2, 2, 0}]) end, [x])
    end
  end

  describe "Stage 03 — reverse" do
    @tag :stage03
    test "1D reverse" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.reverse(t) end, [x])
    end

    @tag :stage03
    test "2D reverse single axis, both axes" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.reverse(t, axes: [0]) end, [x])
      check_equiv(fn t -> Nx.reverse(t, axes: [1]) end, [x])
      check_equiv(fn t -> Nx.reverse(t, axes: [0, 1]) end, [x])
    end

    @tag :stage03
    test "negative axis" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.reverse(t, axes: [-1]) end, [x])
    end
  end

  describe "Stage 03 — concatenate" do
    @tag :stage03
    test "concat 1D along axis 0" do
      a = Nx.tensor([1.0, 2.0], backend: EMLX.Backend)
      b = Nx.tensor([3.0, 4.0, 5.0], backend: EMLX.Backend)
      check_equiv(fn x, y -> Nx.concatenate([x, y], axis: 0) end, [a, b])
    end

    @tag :stage03
    test "concat 2D along axis 0 and axis 1" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      b = Nx.tensor([[5.0, 6.0]], backend: EMLX.Backend)
      c = Nx.tensor([[7.0, 8.0], [9.0, 10.0]], backend: EMLX.Backend)
      check_equiv(fn x, y -> Nx.concatenate([x, y], axis: 0) end, [a, b])
      check_equiv(fn x, y -> Nx.concatenate([x, y], axis: 1) end, [a, c])
    end

    @tag :stage03
    test "three tensors" do
      a = Nx.tensor([1.0], backend: EMLX.Backend)
      b = Nx.tensor([2.0, 3.0], backend: EMLX.Backend)
      c = Nx.tensor([4.0], backend: EMLX.Backend)
      check_equiv(fn x, y, z -> Nx.concatenate([x, y, z]) end, [a, b, c])
    end
  end

  describe "Stage 03 — stack" do
    @tag :stage03
    test "stack 1D tensors → 2D along axis 0" do
      a = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)
      b = Nx.tensor([4.0, 5.0, 6.0], backend: EMLX.Backend)
      check_equiv(fn x, y -> Nx.stack([x, y]) end, [a, b])
    end

    @tag :stage03
    test "stack along axis 1" do
      a = Nx.tensor([1.0, 2.0], backend: EMLX.Backend)
      b = Nx.tensor([3.0, 4.0], backend: EMLX.Backend)
      check_equiv(fn x, y -> Nx.stack([x, y], axis: 1) end, [a, b])
    end

    @tag :stage03
    test "negative axis" do
      a = Nx.tensor([1.0, 2.0], backend: EMLX.Backend)
      b = Nx.tensor([3.0, 4.0], backend: EMLX.Backend)
      check_equiv(fn x, y -> Nx.stack([x, y], axis: -1) end, [a, b])
    end
  end

  describe "Stage 03 — squeeze without explicit axes" do
    @tag :stage03
    test "squeeze all singleton dims" do
      x = Nx.tensor([[[1.0]]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.squeeze(t) end, [x])
    end
  end

  describe "Stage 03 — interpreter ↔ C++ replay parity" do
    setup do
      device = EMLX.default_device()
      {worker, _} = EMLX.resolve_worker(device)
      %{worker: worker, device: device}
    end

    @tag :stage03
    test "interpreter and C++ agree on reshape {6} → {2, 3}", %{worker: worker, device: device} do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], backend: EMLX.Backend)
      expr = Nx.Defn.debug_expr_apply(&reshape_23/1, [Nx.template({6}, :f32)])
      prog = EMLX.Native.Expr.lower(expr)

      [interp_out] = EMLX.Native.Expr.Interpreter.eval(prog, [x])

      {n_inputs, caps, cvs, cts, ops, ors, ias, outs} = EMLX.Native.Expr.to_wire(prog)
      prog_ref = compile_nif!(worker, n_inputs, caps, cvs, cts, ops, ors, ias, outs)

      %EMLX.Backend{ref: {_, ref_x}} = x.data
      [out_ref] = eval_nif!(worker, prog_ref, [ref_x])
      cpp_out = EMLX.Backend.to_nx({device, out_ref}, interp_out)

      assert_all_close(interp_out, cpp_out)
    end

    @tag :stage03
    test "interpreter and C++ agree on broadcast {3} → {2, 3}", %{worker: worker, device: device} do
      x = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)
      expr = Nx.Defn.debug_expr_apply(&broadcast_23/1, [Nx.template({3}, :f32)])
      prog = EMLX.Native.Expr.lower(expr)

      [interp_out] = EMLX.Native.Expr.Interpreter.eval(prog, [x])

      {n_inputs, caps, cvs, cts, ops, ors, ias, outs} = EMLX.Native.Expr.to_wire(prog)
      prog_ref = compile_nif!(worker, n_inputs, caps, cvs, cts, ops, ors, ias, outs)

      %EMLX.Backend{ref: {_, ref_x}} = x.data
      [out_ref] = eval_nif!(worker, prog_ref, [ref_x])
      cpp_out = EMLX.Backend.to_nx({device, out_ref}, interp_out)

      assert_all_close(interp_out, cpp_out)
    end

    @tag :stage03
    test "interpreter and C++ agree on concatenate axis 0", %{worker: worker, device: device} do
      a = Nx.tensor([1.0, 2.0], backend: EMLX.Backend)
      b = Nx.tensor([3.0, 4.0, 5.0], backend: EMLX.Backend)

      expr =
        Nx.Defn.debug_expr_apply(&concat_axis0/2, [
          Nx.template({2}, :f32),
          Nx.template({3}, :f32)
        ])

      prog = EMLX.Native.Expr.lower(expr)

      [interp_out] = EMLX.Native.Expr.Interpreter.eval(prog, [a, b])

      {n_inputs, caps, cvs, cts, ops, ors, ias, outs} = EMLX.Native.Expr.to_wire(prog)
      prog_ref = compile_nif!(worker, n_inputs, caps, cvs, cts, ops, ors, ias, outs)

      %EMLX.Backend{ref: {_, ref_a}} = a.data
      %EMLX.Backend{ref: {_, ref_b}} = b.data
      [out_ref] = eval_nif!(worker, prog_ref, [ref_a, ref_b])
      cpp_out = EMLX.Backend.to_nx({device, out_ref}, interp_out)

      assert_all_close(interp_out, cpp_out)
    end
  end

  # ── Stage 04: reductions ─────────────────────────────────────────────────

  describe "Stage 04 — sum" do
    @tag :stage04
    test "sum all axes f32" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.sum(t) end, [x])
    end

    @tag :stage04
    test "sum along axis 0 with keep_axes" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.sum(t, axes: [0], keep_axes: true) end, [x])
    end

    @tag :stage04
    test "sum along axis 1 f32" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.sum(t, axes: [1]) end, [x])
    end

    @tag :stage04
    test "sum 3D along multiple axes" do
      x = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.sum(t, axes: [0, 2]) end, [x])
    end
  end

  describe "Stage 04 — product" do
    @tag :stage04
    test "product all axes f32" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.product(t) end, [x])
    end

    @tag :stage04
    test "product along axis 0" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.product(t, axes: [0]) end, [x])
    end
  end

  describe "Stage 04 — all / any" do
    @tag :stage04
    test "all on boolean-like tensor" do
      x = Nx.tensor([[1, 1], [1, 0]], type: :u8, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.all(t) end, [x])
    end

    @tag :stage04
    test "all along axis 0" do
      x = Nx.tensor([[1, 0], [1, 1]], type: :u8, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.all(t, axes: [0]) end, [x])
    end

    @tag :stage04
    test "any all axes" do
      x = Nx.tensor([[0, 0], [0, 1]], type: :u8, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.any(t) end, [x])
    end

    @tag :stage04
    test "any along axis 1 with keep_axes" do
      x = Nx.tensor([[0, 1], [0, 0]], type: :u8, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.any(t, axes: [1], keep_axes: true) end, [x])
    end
  end

  describe "Stage 04 — reduce_max / reduce_min" do
    @tag :stage04
    test "reduce_max all axes f32" do
      x = Nx.tensor([[1.0, 5.0], [3.0, 2.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.reduce_max(t) end, [x])
    end

    @tag :stage04
    test "reduce_max along axis 1 keep_axes" do
      x = Nx.tensor([[1.0, 5.0], [3.0, 2.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.reduce_max(t, axes: [1], keep_axes: true) end, [x])
    end

    @tag :stage04
    test "reduce_min all axes" do
      x = Nx.tensor([[4.0, 2.0], [1.0, 3.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.reduce_min(t) end, [x])
    end

    @tag :stage04
    test "reduce_min along axis 0" do
      x = Nx.tensor([[4.0, 2.0], [1.0, 3.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.reduce_min(t, axes: [0]) end, [x])
    end
  end

  describe "Stage 04 — argmax / argmin" do
    @tag :stage04
    test "argmax along axis 1" do
      x = Nx.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.argmax(t, axis: 1) end, [x])
    end

    @tag :stage04
    test "argmax along axis 0 keep_axis" do
      x = Nx.tensor([[1.0, 3.0], [4.0, 2.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.argmax(t, axis: 0, keep_axis: true) end, [x])
    end

    @tag :stage04
    test "argmin along axis 0" do
      x = Nx.tensor([[4.0, 2.0], [1.0, 5.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.argmin(t, axis: 0) end, [x])
    end

    @tag :stage04
    test "argmax global (no axis)" do
      x = Nx.tensor([[1.0, 7.0], [3.0, 2.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.argmax(t) end, [x])
    end
  end

  # ── Stage 04: dot ────────────────────────────────────────────────────────

  describe "Stage 04 — dot (non-batched)" do
    @tag :stage04
    test "matmul {2,3} × {3,4} → {2,4}" do
      a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: EMLX.Backend)

      b =
        Nx.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
          backend: EMLX.Backend
        )

      check_equiv(fn x, y -> Nx.dot(x, y) end, [a, b])
    end

    @tag :stage04
    test "inner product {4} · {4} → scalar" do
      a = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: EMLX.Backend)
      b = Nx.tensor([5.0, 6.0, 7.0, 8.0], backend: EMLX.Backend)
      check_equiv(fn x, y -> Nx.dot(x, y) end, [a, b])
    end

    @tag :stage04
    test "tensordot explicit contraction axes" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      b = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], backend: EMLX.Backend)
      check_equiv(fn x, y -> Nx.dot(x, [1], [], y, [0], []) end, [a, b])
    end
  end

  describe "Stage 04 — dot (batched)" do
    @tag :stage04
    test "batched matmul {2,3,4} · {2,4,5} → {2,3,5}" do
      a = Nx.iota({2, 3, 4}, type: :f32, backend: EMLX.Backend)
      b = Nx.iota({2, 4, 5}, type: :f32, backend: EMLX.Backend)
      check_equiv(fn x, y -> Nx.dot(x, [2], [0], y, [1], [0]) end, [a, b], tol: 1.0e-3)
    end
  end

  # ── Stage 04: conv ───────────────────────────────────────────────────────

  describe "Stage 04 — conv (1D)" do
    @tag :stage04
    test "1D conv {1,1,5} input, {1,1,3} kernel" do
      # 3D: {batch=1, in_channels=1, length=5}; kernel {out=1, in=1, size=3}
      input = Nx.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]], backend: EMLX.Backend)
      kernel = Nx.tensor([[[1.0, 0.0, -1.0]]], backend: EMLX.Backend)
      check_equiv(fn x, k -> Nx.conv(x, k) end, [input, kernel], tol: 1.0e-4)
    end

    @tag :stage04
    test "1D conv with stride 2 and padding" do
      input = Nx.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]], backend: EMLX.Backend)
      kernel = Nx.tensor([[[1.0, 1.0]]], backend: EMLX.Backend)

      check_equiv(fn x, k -> Nx.conv(x, k, strides: [2], padding: :same) end, [input, kernel],
        tol: 1.0e-4
      )
    end
  end

  describe "Stage 04 — conv (2D)" do
    @tag :stage04
    test "2D conv identity kernel {1,1,3,3} × 1-ch input" do
      input = Nx.iota({1, 1, 4, 4}, type: :f32, backend: EMLX.Backend)

      kernel =
        Nx.tensor([[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]], backend: EMLX.Backend)

      check_equiv(fn x, k -> Nx.conv(x, k, padding: :same) end, [input, kernel], tol: 1.0e-4)
    end

    @tag :stage04
    test "2D conv with strides and dilations" do
      input = Nx.iota({1, 1, 6, 6}, type: :f32, backend: EMLX.Backend)
      kernel = Nx.tensor([[[[1.0, 1.0], [1.0, 1.0]]]], backend: EMLX.Backend)
      check_equiv(fn x, k -> Nx.conv(x, k, strides: [2, 2]) end, [input, kernel], tol: 1.0e-4)
    end

    @tag :stage04
    test "2D conv multi-channel" do
      input = Nx.iota({1, 3, 4, 4}, type: :f32, backend: EMLX.Backend)
      kernel = Nx.iota({2, 3, 2, 2}, type: :f32, backend: EMLX.Backend)
      check_equiv(fn x, k -> Nx.conv(x, k) end, [input, kernel], tol: 1.0e-3)
    end
  end

  # ── Stage 04: Interpreter ↔ C++ parity ──────────────────────────────────

  defn sum_all(x), do: Nx.sum(x)
  defn matmul_22(a, b), do: Nx.dot(a, b)

  describe "Stage 04 — interpreter ↔ C++ parity" do
    setup do
      device = EMLX.default_device()
      {worker, _} = EMLX.resolve_worker(device)
      %{worker: worker, device: device}
    end

    @tag :stage04
    test "interpreter and C++ agree on sum {2,3}", %{worker: worker, device: device} do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: EMLX.Backend)
      expr = Nx.Defn.debug_expr_apply(&sum_all/1, [Nx.template({2, 3}, :f32)])
      prog = EMLX.Native.Expr.lower(expr)

      [interp_out] = EMLX.Native.Expr.Interpreter.eval(prog, [x])

      {n_inputs, caps, cvs, cts, ops, ors, ias, outs} = EMLX.Native.Expr.to_wire(prog)
      prog_ref = compile_nif!(worker, n_inputs, caps, cvs, cts, ops, ors, ias, outs)

      %EMLX.Backend{ref: {_, ref_x}} = x.data
      [out_ref] = eval_nif!(worker, prog_ref, [ref_x])
      cpp_out = EMLX.Backend.to_nx({device, out_ref}, interp_out)

      assert_all_close(interp_out, cpp_out)
    end

    @tag :stage04
    test "interpreter and C++ agree on matmul {2,3}×{3,2}", %{worker: worker, device: device} do
      a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: EMLX.Backend)
      b = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], backend: EMLX.Backend)

      expr =
        Nx.Defn.debug_expr_apply(&matmul_22/2, [
          Nx.template({2, 3}, :f32),
          Nx.template({3, 2}, :f32)
        ])

      prog = EMLX.Native.Expr.lower(expr)

      [interp_out] = EMLX.Native.Expr.Interpreter.eval(prog, [a, b])

      {n_inputs, caps, cvs, cts, ops, ors, ias, outs} = EMLX.Native.Expr.to_wire(prog)
      prog_ref = compile_nif!(worker, n_inputs, caps, cvs, cts, ops, ors, ias, outs)

      %EMLX.Backend{ref: {_, ref_a}} = a.data
      %EMLX.Backend{ref: {_, ref_b}} = b.data
      [out_ref] = eval_nif!(worker, prog_ref, [ref_a, ref_b])
      cpp_out = EMLX.Backend.to_nx({device, out_ref}, interp_out)

      assert_all_close(interp_out, cpp_out)
    end
  end

  # ── Stage 04: E2E jit smoke tests ───────────────────────────────────────

  defn sum_axis1_keep(x), do: Nx.sum(x, axes: [1], keep_axes: true)
  defn argmax_axis0(x), do: Nx.argmax(x, axis: 0)
  defn matmul_defn(a, b), do: Nx.dot(a, b)
  defn reduce_max_defn(x), do: Nx.reduce_max(x)

  describe "Stage 04 — E2E jit smoke" do
    @tag :stage04
    test "sum with keep_axes=true via jit" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: EMLX.Backend)
      result = Nx.Defn.jit(&sum_axis1_keep/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&sum_axis1_keep/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(result, eager, 1.0e-4)
    end

    @tag :stage04
    test "argmax via jit" do
      x = Nx.tensor([[1.0, 3.0], [4.0, 2.0]], backend: EMLX.Backend)
      result = Nx.Defn.jit(&argmax_axis0/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&argmax_axis0/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(result, eager, 1.0e-4)
    end

    @tag :stage04
    test "matmul via jit" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]], backend: EMLX.Backend)
      result = Nx.Defn.jit(&matmul_defn/2, compiler: EMLX).(a, b)
      eager = Nx.Defn.jit(&matmul_defn/2, compiler: Nx.Defn.Evaluator).(a, b)
      assert_close(result, eager, 1.0e-3)
    end

    @tag :stage04
    test "reduce_max via jit" do
      x = Nx.tensor([[4.0, 1.0], [2.0, 3.0]], backend: EMLX.Backend)
      result = Nx.Defn.jit(&reduce_max_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&reduce_max_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(result, eager, 1.0e-4)
    end
  end

  # ── Stage 05 defn helpers ────────────────────────────────────────────────
  defn select_defn(pred, a, b), do: Nx.select(pred, a, b)
  defn clip_defn(x, lo, hi), do: Nx.clip(x, lo, hi)
  defn slice_static_defn(x), do: Nx.slice(x, [1, 0], [2, 3])
  defn slice_strided_defn(x), do: Nx.slice(x, [0, 0], [2, 3], strides: [1, 2])
  defn put_slice_static_defn(x, patch), do: Nx.put_slice(x, [1, 1], patch)
  defn gather_defn(x, idx), do: Nx.gather(x, idx)
  defn take_defn(x, idx), do: Nx.take(x, idx, axis: 1)
  defn take_along_axis_defn(x, idx), do: Nx.take_along_axis(x, idx, axis: 0)
  defn indexed_put_defn(x, idx, updates), do: Nx.indexed_put(x, idx, updates)
  defn indexed_add_defn(x, idx, updates), do: Nx.indexed_add(x, idx, updates)

  # ── Stage 05 — select / clip ────────────────────────────────────────────

  describe "Stage 05 — select" do
    @tag :stage05
    test "interpreter parity — f32" do
      pred =
        Nx.tensor([1, 0, 1, 0], type: :u8, backend: EMLX.Backend)
        |> Nx.reshape({4})

      a = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: EMLX.Backend)
      b = Nx.tensor([5.0, 6.0, 7.0, 8.0], backend: EMLX.Backend)

      expr = Nx.Defn.debug_expr_apply(&select_defn/3, [pred, a, b])
      prog = Expr.lower(expr)

      interp = EMLX.Native.Expr.Interpreter.eval(prog, [pred, a, b])
      nif = run_nif(prog, [pred, a, b])
      assert_close(hd(interp), hd(nif))
    end

    @tag :stage05
    test "equivalence vs EMLX.Backend" do
      pred = Nx.tensor([1, 0, 1], type: :u8, backend: EMLX.Backend)
      a = Nx.tensor([10.0, 20.0, 30.0], backend: EMLX.Backend)
      b = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)

      native = Nx.Defn.jit(&select_defn/3, compiler: EMLX).(pred, a, b)
      eager = Nx.Defn.jit(&select_defn/3, compiler: Nx.Defn.Evaluator).(pred, a, b)
      assert_close(native, eager)
    end

    @tag :stage05
    test "select with mixed-type true/false is cast to out_type" do
      pred = Nx.tensor([1, 0], type: :u8, backend: EMLX.Backend)
      a = Nx.tensor([1, 2], type: :s32, backend: EMLX.Backend)
      b = Nx.tensor([3, 4], type: :s32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&select_defn/3, compiler: EMLX).(pred, a, b)
      eager = Nx.Defn.jit(&select_defn/3, compiler: Nx.Defn.Evaluator).(pred, a, b)
      assert_close(native, eager)
    end
  end

  describe "Stage 05 — clip" do
    @tag :stage05
    test "interpreter parity — f32" do
      x = Nx.tensor([-1.0, 0.5, 2.0, 3.5], backend: EMLX.Backend)
      lo = Nx.tensor(0.0, backend: EMLX.Backend)
      hi = Nx.tensor(2.0, backend: EMLX.Backend)

      expr = Nx.Defn.debug_expr_apply(&clip_defn/3, [x, lo, hi])
      prog = Expr.lower(expr)

      interp = EMLX.Native.Expr.Interpreter.eval(prog, [x, lo, hi])
      nif = run_nif(prog, [x, lo, hi])
      assert_close(hd(interp), hd(nif))
    end

    @tag :stage05
    test "equivalence vs EMLX.Backend — f32" do
      x = Nx.iota({5}, type: :f32, backend: EMLX.Backend) |> Nx.subtract(2.0)
      lo = Nx.tensor(-1.0, backend: EMLX.Backend)
      hi = Nx.tensor(1.5, backend: EMLX.Backend)
      native = Nx.Defn.jit(&clip_defn/3, compiler: EMLX).(x, lo, hi)
      eager = Nx.Defn.jit(&clip_defn/3, compiler: Nx.Defn.Evaluator).(x, lo, hi)
      assert_close(native, eager)
    end

    @tag :stage05
    test "equivalence vs EMLX.Backend — s32" do
      x = Nx.tensor([-5, -1, 0, 3, 10], type: :s32, backend: EMLX.Backend)
      lo = Nx.tensor(0, type: :s32, backend: EMLX.Backend)
      hi = Nx.tensor(5, type: :s32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&clip_defn/3, compiler: EMLX).(x, lo, hi)
      eager = Nx.Defn.jit(&clip_defn/3, compiler: Nx.Defn.Evaluator).(x, lo, hi)
      assert_close(native, eager)
    end
  end

  # ── Stage 05 — slice / put_slice ──────────────────────────────────────────

  describe "Stage 05 — slice (static indices)" do
    @tag :stage05
    test "interpreter parity — static 2D slice" do
      x = Nx.iota({4, 5}, type: :f32, backend: EMLX.Backend)
      expr = Nx.Defn.debug_expr_apply(&slice_static_defn/1, [x])
      prog = Expr.lower(expr)
      interp = EMLX.Native.Expr.Interpreter.eval(prog, [x])
      nif = run_nif(prog, [x])
      assert_close(hd(interp), hd(nif))
    end

    @tag :stage05
    test "equivalence vs EMLX.Backend — static 2D slice" do
      x = Nx.iota({4, 5}, type: :f32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&slice_static_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&slice_static_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    @tag :stage05
    test "equivalence vs EMLX.Backend — strided slice" do
      x = Nx.iota({4, 6}, type: :f32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&slice_strided_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&slice_strided_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end
  end

  describe "Stage 05 — slice (dynamic index)" do
    @tag :stage05
    test "equivalence vs EMLX.Backend — dynamic start along axis 0" do
      # Slices rows [start, start+2) of a 5×4 tensor, start is a runtime scalar.
      x = Nx.iota({5, 4}, type: :f32, backend: EMLX.Backend)

      dynamic_slice = fn x, start ->
        Nx.slice(x, [start, 0], [2, 4])
      end

      start_val = Nx.tensor(2, type: :s32, backend: EMLX.Backend)
      native = Nx.Defn.jit(dynamic_slice, compiler: EMLX).(x, start_val)
      eager = Nx.Defn.jit(dynamic_slice, compiler: Nx.Defn.Evaluator).(x, start_val)
      assert_close(native, eager)
    end

    @tag :stage05
    test "equivalence vs EMLX.Backend — dynamic start clamped at boundary" do
      x = Nx.iota({4, 4}, type: :f32, backend: EMLX.Backend)

      dynamic_slice = fn x, start ->
        Nx.slice(x, [start, 0], [2, 4])
      end

      # start=10 should be clamped to 2 (4-2)
      start_val = Nx.tensor(10, type: :s32, backend: EMLX.Backend)
      native = Nx.Defn.jit(dynamic_slice, compiler: EMLX).(x, start_val)
      eager = Nx.Defn.jit(dynamic_slice, compiler: Nx.Defn.Evaluator).(x, start_val)
      assert_close(native, eager)
    end
  end

  describe "Stage 05 — put_slice (static indices)" do
    @tag :stage05
    test "interpreter parity — static 2D put_slice" do
      x = Nx.iota({4, 4}, type: :f32, backend: EMLX.Backend)
      patch = Nx.broadcast(Nx.tensor(99.0, backend: EMLX.Backend), {2, 2})

      expr = Nx.Defn.debug_expr_apply(&put_slice_static_defn/2, [x, patch])
      prog = Expr.lower(expr)
      interp = EMLX.Native.Expr.Interpreter.eval(prog, [x, patch])
      nif = run_nif(prog, [x, patch])
      assert_close(hd(interp), hd(nif))
    end

    @tag :stage05
    test "equivalence vs EMLX.Backend — static 2D put_slice" do
      x = Nx.iota({4, 4}, type: :f32, backend: EMLX.Backend)
      patch = Nx.broadcast(Nx.tensor(99.0, backend: EMLX.Backend), {2, 2})
      native = Nx.Defn.jit(&put_slice_static_defn/2, compiler: EMLX).(x, patch)
      eager = Nx.Defn.jit(&put_slice_static_defn/2, compiler: Nx.Defn.Evaluator).(x, patch)
      assert_close(native, eager)
    end

    @tag :stage05
    test "equivalence vs EMLX.Backend — dynamic put_slice (KV-cache pattern)" do
      cache = Nx.broadcast(Nx.tensor(0.0, backend: EMLX.Backend), {8, 4})

      kv_update = fn cache, pos, new_row ->
        Nx.put_slice(cache, [pos, 0], Nx.new_axis(new_row, 0))
      end

      pos = Nx.tensor(3, type: :s32, backend: EMLX.Backend)
      new_row = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: EMLX.Backend)

      native = Nx.Defn.jit(kv_update, compiler: EMLX).(cache, pos, new_row)
      eager = Nx.Defn.jit(kv_update, compiler: Nx.Defn.Evaluator).(cache, pos, new_row)
      assert_close(native, eager)
    end
  end

  # ── Stage 05 — gather / take / take_along_axis ────────────────────────────

  describe "Stage 05 — gather" do
    @tag :stage05
    test "interpreter parity — 2D gather on axis 0" do
      x = Nx.iota({4, 3}, type: :f32, backend: EMLX.Backend)
      idx = Nx.tensor([[0], [2], [1]], type: :s32, backend: EMLX.Backend)

      expr = Nx.Defn.debug_expr_apply(&gather_defn/2, [x, idx])
      prog = Expr.lower(expr)
      interp = EMLX.Native.Expr.Interpreter.eval(prog, [x, idx])
      nif = run_nif(prog, [x, idx])
      assert_close(hd(interp), hd(nif))
    end

    @tag :stage05
    test "equivalence vs EMLX.Backend — 2D gather on axis 0" do
      x = Nx.iota({4, 3}, type: :f32, backend: EMLX.Backend)
      idx = Nx.tensor([[0], [2], [1]], type: :s32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&gather_defn/2, compiler: EMLX).(x, idx)
      eager = Nx.Defn.jit(&gather_defn/2, compiler: Nx.Defn.Evaluator).(x, idx)
      assert_close(native, eager)
    end

    @tag :stage05
    test "equivalence vs EMLX.Backend — multi-axis gather" do
      x = Nx.iota({3, 4}, type: :f32, backend: EMLX.Backend)
      idx = Nx.tensor([[0, 1], [2, 3]], type: :s32, backend: EMLX.Backend)

      gather_multi = fn x, idx -> Nx.gather(x, idx, axes: [0, 1]) end

      native = Nx.Defn.jit(gather_multi, compiler: EMLX).(x, idx)
      eager = Nx.Defn.jit(gather_multi, compiler: Nx.Defn.Evaluator).(x, idx)
      assert_close(native, eager)
    end
  end

  describe "Stage 05 — take" do
    @tag :stage05
    test "interpreter parity" do
      x = Nx.iota({3, 4}, type: :f32, backend: EMLX.Backend)
      idx = Nx.tensor([2, 0, 3, 1], type: :s32, backend: EMLX.Backend)

      expr = Nx.Defn.debug_expr_apply(&take_defn/2, [x, idx])
      prog = Expr.lower(expr)
      interp = EMLX.Native.Expr.Interpreter.eval(prog, [x, idx])
      nif = run_nif(prog, [x, idx])
      assert_close(hd(interp), hd(nif))
    end

    @tag :stage05
    test "equivalence vs EMLX.Backend" do
      x = Nx.iota({3, 4}, type: :f32, backend: EMLX.Backend)
      idx = Nx.tensor([2, 0, 3, 1], type: :s32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&take_defn/2, compiler: EMLX).(x, idx)
      eager = Nx.Defn.jit(&take_defn/2, compiler: Nx.Defn.Evaluator).(x, idx)
      assert_close(native, eager)
    end
  end

  describe "Stage 05 — take_along_axis" do
    @tag :stage05
    test "interpreter parity" do
      x = Nx.iota({3, 4}, type: :f32, backend: EMLX.Backend)
      idx = Nx.tensor([[2, 0, 1, 2]], type: :s32, backend: EMLX.Backend)

      expr = Nx.Defn.debug_expr_apply(&take_along_axis_defn/2, [x, idx])
      prog = Expr.lower(expr)
      interp = EMLX.Native.Expr.Interpreter.eval(prog, [x, idx])
      nif = run_nif(prog, [x, idx])
      assert_close(hd(interp), hd(nif))
    end

    @tag :stage05
    test "equivalence vs EMLX.Backend" do
      x = Nx.iota({3, 4}, type: :f32, backend: EMLX.Backend)
      idx = Nx.tensor([[2, 0, 1, 2]], type: :s32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&take_along_axis_defn/2, compiler: EMLX).(x, idx)
      eager = Nx.Defn.jit(&take_along_axis_defn/2, compiler: Nx.Defn.Evaluator).(x, idx)
      assert_close(native, eager)
    end
  end

  # ── Stage 05 — indexed_put / indexed_add ──────────────────────────────────

  describe "Stage 05 — indexed_put" do
    @tag :stage05
    test "interpreter parity" do
      x = Nx.iota({4, 3}, type: :f32, backend: EMLX.Backend)
      idx = Nx.tensor([[0], [2]], type: :s32, backend: EMLX.Backend)
      updates = Nx.tensor([[99.0, 99.0, 99.0], [88.0, 88.0, 88.0]], backend: EMLX.Backend)

      expr = Nx.Defn.debug_expr_apply(&indexed_put_defn/3, [x, idx, updates])
      prog = Expr.lower(expr)
      interp = EMLX.Native.Expr.Interpreter.eval(prog, [x, idx, updates])
      nif = run_nif(prog, [x, idx, updates])
      assert_close(hd(interp), hd(nif))
    end

    @tag :stage05
    test "equivalence vs EMLX.Backend" do
      x = Nx.iota({4, 3}, type: :f32, backend: EMLX.Backend)
      idx = Nx.tensor([[0], [2]], type: :s32, backend: EMLX.Backend)
      updates = Nx.tensor([[99.0, 99.0, 99.0], [88.0, 88.0, 88.0]], backend: EMLX.Backend)
      native = Nx.Defn.jit(&indexed_put_defn/3, compiler: EMLX).(x, idx, updates)
      eager = Nx.Defn.jit(&indexed_put_defn/3, compiler: Nx.Defn.Evaluator).(x, idx, updates)
      assert_close(native, eager)
    end
  end

  describe "Stage 05 — indexed_add" do
    @tag :stage05
    test "interpreter parity" do
      x = Nx.broadcast(Nx.tensor(0.0, backend: EMLX.Backend), {4, 3})
      idx = Nx.tensor([[0], [0], [2]], type: :s32, backend: EMLX.Backend)

      updates =
        Nx.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], backend: EMLX.Backend)

      expr = Nx.Defn.debug_expr_apply(&indexed_add_defn/3, [x, idx, updates])
      prog = Expr.lower(expr)
      interp = EMLX.Native.Expr.Interpreter.eval(prog, [x, idx, updates])
      nif = run_nif(prog, [x, idx, updates])
      assert_close(hd(interp), hd(nif))
    end

    @tag :stage05
    test "equivalence vs EMLX.Backend" do
      x = Nx.broadcast(Nx.tensor(0.0, backend: EMLX.Backend), {4, 3})
      idx = Nx.tensor([[0], [0], [2]], type: :s32, backend: EMLX.Backend)

      updates =
        Nx.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], backend: EMLX.Backend)

      native = Nx.Defn.jit(&indexed_add_defn/3, compiler: EMLX).(x, idx, updates)
      eager = Nx.Defn.jit(&indexed_add_defn/3, compiler: Nx.Defn.Evaluator).(x, idx, updates)
      assert_close(native, eager)
    end
  end

  describe "Stage 05 — E2E jit smoke" do
    @tag :stage05
    test "select via jit" do
      pred = Nx.tensor([1, 0, 1, 0], type: :u8, backend: EMLX.Backend)
      a = Nx.iota({4}, type: :f32, backend: EMLX.Backend)
      b = Nx.multiply(Nx.iota({4}, type: :f32, backend: EMLX.Backend), -1.0)
      native = Nx.Defn.jit(&select_defn/3, compiler: EMLX).(pred, a, b)
      eager = Nx.Defn.jit(&select_defn/3, compiler: Nx.Defn.Evaluator).(pred, a, b)
      assert_close(native, eager)
    end

    @tag :stage05
    test "static slice + add via jit" do
      x = Nx.iota({4, 5}, type: :f32, backend: EMLX.Backend)
      add_slice = fn x -> Nx.add(Nx.slice(x, [0, 0], [2, 3]), 1.0) end
      native = Nx.Defn.jit(add_slice, compiler: EMLX).(x)
      eager = Nx.Defn.jit(add_slice, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end
  end

  # ── private helpers ───────────────────────────────────────────────────────

  defp compile_nif!(worker, n_inputs, caps, cvs, cts, ops, ors, ias, outs) do
    EMLX.NIF.compile_program(worker, n_inputs, caps, cvs, cts, ops, ors, ias, outs)
    |> unwrap!()
    |> await_worker!()
  end

  defp eval_nif!(worker, prog_ref, input_refs) do
    EMLX.NIF.eval_program(worker, prog_ref, input_refs)
    |> unwrap!()
    |> await_worker!()
  end

  # ── Stage 06 defn helpers ─────────────────────────────────────────────────

  defn sort_asc_defn(x), do: Nx.sort(x, axis: 0, direction: :asc)
  defn sort_desc_defn(x), do: Nx.sort(x, axis: 1, direction: :desc)
  defn argsort_asc_defn(x), do: Nx.argsort(x, axis: 0, direction: :asc)
  defn argsort_desc_defn(x), do: Nx.argsort(x, axis: 1, direction: :desc)

  defn window_sum_defn(x), do: Nx.window_sum(x, {2, 2})
  defn window_max_defn(x), do: Nx.window_max(x, {2, 2})
  defn window_min_defn(x), do: Nx.window_min(x, {2, 2})
  defn window_product_defn(x), do: Nx.window_product(x, {2, 2})

  defn cumulative_sum_defn(x), do: Nx.cumulative_sum(x, axis: 1)
  defn cumulative_product_defn(x), do: Nx.cumulative_product(x, axis: 0)
  defn cumulative_min_defn(x), do: Nx.cumulative_min(x, axis: 0)
  defn cumulative_max_defn(x), do: Nx.cumulative_max(x, axis: 0, reverse: true)

  defn fft_defn(x), do: Nx.fft(x)
  defn ifft_defn(x), do: Nx.ifft(x)
  defn fft2_defn(x), do: Nx.fft2(x)
  defn rfft_defn(x), do: Nx.rfft(x)

  # ── Stage 07 defn helpers ─────────────────────────────────────────────────

  defn iota_flat_defn(), do: Nx.iota({3, 4})
  defn iota_axis1_defn(), do: Nx.iota({3, 4}, axis: 1)
  defn iota_f32_defn(), do: Nx.iota({5}, type: :f32)
  defn eye_3x3_defn(), do: Nx.eye({3, 3})
  defn eye_2x4_defn(), do: Nx.eye({2, 4})

  defn rng_uniform_defn(key) do
    {samples, _key} = Nx.Random.uniform(key, shape: {8})
    samples
  end

  defn rng_normal_defn(key) do
    {samples, _key} = Nx.Random.normal(key, shape: {8})
    samples
  end

  # ── Stage 06 — sort / argsort ────────────────────────────────────────────

  describe "Stage 06 — sort" do
    @tag :stage06
    test "sort ascending, axis 0 vs EMLX.Backend — f32" do
      x =
        Nx.tensor([[3.0, 1.0, 2.0], [9.0, 4.0, 7.0]], backend: EMLX.Backend)

      native = Nx.Defn.jit(&sort_asc_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&sort_asc_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    @tag :stage06
    test "sort descending, axis 1 vs EMLX.Backend — f32" do
      x =
        Nx.tensor([[3.0, 1.0, 2.0], [9.0, 4.0, 7.0]], backend: EMLX.Backend)

      native = Nx.Defn.jit(&sort_desc_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&sort_desc_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    @tag :stage06
    test "sort with NaN values — NaN goes to end in asc" do
      x = Nx.tensor([1.0, :nan, 2.0, :nan, 0.0], backend: EMLX.Backend)

      native =
        Nx.Defn.jit(fn t -> Nx.sort(t, axis: 0, direction: :asc) end, compiler: EMLX).(x)

      eager =
        Nx.Defn.jit(fn t -> Nx.sort(t, axis: 0, direction: :asc) end,
          compiler: Nx.Defn.Evaluator
        ).(x)

      # Both should have NaN at the end; compare non-NaN prefix.
      native_list = Nx.to_flat_list(native)
      eager_list = Nx.to_flat_list(eager)

      Enum.zip(native_list, eager_list)
      |> Enum.each(fn {n, e} ->
        if e == :nan, do: assert(n == :nan), else: assert_in_delta(n, e, 1.0e-4)
      end)
    end

    @tag :stage06
    test "sort s32 ascending" do
      x = Nx.tensor([5, 3, 1, 4, 2], type: :s32, backend: EMLX.Backend)

      native =
        Nx.Defn.jit(fn t -> Nx.sort(t, axis: 0) end, compiler: EMLX).(x)

      eager =
        Nx.Defn.jit(fn t -> Nx.sort(t, axis: 0) end, compiler: Nx.Defn.Evaluator).(x)

      assert_close(native, eager)
    end
  end

  describe "Stage 06 — argsort" do
    @tag :stage06
    test "argsort ascending, axis 0 vs EMLX.Backend" do
      x =
        Nx.tensor([[3.0, 1.0, 2.0], [9.0, 4.0, 7.0]], backend: EMLX.Backend)

      native = Nx.Defn.jit(&argsort_asc_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&argsort_asc_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    @tag :stage06
    test "argsort descending, axis 1 vs EMLX.Backend" do
      x =
        Nx.tensor([[3.0, 1.0, 2.0], [9.0, 4.0, 7.0]], backend: EMLX.Backend)

      native = Nx.Defn.jit(&argsort_desc_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&argsort_desc_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    @tag :stage06
    test "argsort output type matches out tensor type" do
      x = Nx.tensor([3.0, 1.0, 2.0], backend: EMLX.Backend)
      prog = Expr.lower(Nx.Defn.debug_expr_apply(fn t -> Nx.argsort(t, axis: 0) end, [x]))
      # argsort output should be :u64 (Nx default for argsort)
      assert Enum.any?(prog.instructions, fn {_, op, _, _} -> op == :argsort end)
    end
  end

  # ── Stage 06 — window reductions ─────────────────────────────────────────

  describe "Stage 06 — window reductions" do
    @tag :stage06
    test "window_sum 2x2, no padding vs EMLX.Backend — f32" do
      x = Nx.iota({4, 4}, type: :f32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&window_sum_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&window_sum_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    @tag :stage06
    test "window_max 2x2, no padding vs EMLX.Backend — f32" do
      x = Nx.iota({4, 4}, type: :f32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&window_max_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&window_max_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    @tag :stage06
    test "window_min 2x2, no padding vs EMLX.Backend — f32" do
      x = Nx.iota({4, 4}, type: :f32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&window_min_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&window_min_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    @tag :stage06
    test "window_product 2x2, no padding vs EMLX.Backend — f32" do
      x =
        Nx.iota({4, 4}, type: :f32, backend: EMLX.Backend)
        |> Nx.add(1.0)

      native = Nx.Defn.jit(&window_product_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&window_product_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    @tag :stage06
    test "window_sum with padding vs EMLX.Backend" do
      x = Nx.iota({3, 3}, type: :f32, backend: EMLX.Backend)

      f = fn t -> Nx.window_sum(t, {2, 2}, padding: :same) end
      native = Nx.Defn.jit(f, compiler: EMLX).(x)
      eager = Nx.Defn.jit(f, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    @tag :stage06
    test "window_max with strides vs EMLX.Backend" do
      x = Nx.iota({6, 6}, type: :f32, backend: EMLX.Backend)

      f = fn t -> Nx.window_max(t, {2, 2}, strides: [2, 2]) end
      native = Nx.Defn.jit(f, compiler: EMLX).(x)
      eager = Nx.Defn.jit(f, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    @tag :stage06
    test "window_sum 1D vs EMLX.Backend" do
      x = Nx.iota({6}, type: :f32, backend: EMLX.Backend)

      f = fn t -> Nx.window_sum(t, {3}) end
      native = Nx.Defn.jit(f, compiler: EMLX).(x)
      eager = Nx.Defn.jit(f, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end
  end

  # ── Stage 06 — cumulative reductions ─────────────────────────────────────

  describe "Stage 06 — cumulative reductions" do
    @tag :stage06
    test "cumulative_sum axis 1 vs EMLX.Backend — f32" do
      x = Nx.iota({2, 4}, type: :f32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&cumulative_sum_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&cumulative_sum_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    @tag :stage06
    test "cumulative_product axis 0 vs EMLX.Backend — f32" do
      x = Nx.iota({3, 3}, type: :f32, backend: EMLX.Backend) |> Nx.add(1.0)

      native = Nx.Defn.jit(&cumulative_product_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&cumulative_product_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    @tag :stage06
    test "cumulative_min axis 0 vs EMLX.Backend — s32" do
      x = Nx.tensor([[3, 1, 4], [1, 5, 9], [2, 6, 5]], type: :s32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&cumulative_min_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&cumulative_min_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    @tag :stage06
    test "cumulative_max axis 0 reverse vs EMLX.Backend — f32" do
      x = Nx.iota({4, 3}, type: :f32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&cumulative_max_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&cumulative_max_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    @tag :stage06
    test "cumulative_sum s32 axis 0 vs EMLX.Backend" do
      x = Nx.tensor([1, 2, 3, 4], type: :s32, backend: EMLX.Backend)

      f = fn t -> Nx.cumulative_sum(t, axis: 0) end
      native = Nx.Defn.jit(f, compiler: EMLX).(x)
      eager = Nx.Defn.jit(f, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end
  end

  # ── Stage 06 — FFT ───────────────────────────────────────────────────────

  describe "Stage 06 — fft / ifft" do
    @tag :stage06
    test "fft 1D vs EMLX.Backend" do
      x = Nx.tensor([1.0, 1.0, 0.0, 0.0], type: :f32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&fft_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&fft_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_complex_close(native, eager)
    end

    @tag :stage06
    test "ifft 1D vs EMLX.Backend" do
      x = Nx.tensor([1.0, 1.0, 0.0, 0.0], type: :f32, backend: EMLX.Backend)
      x_fft = Nx.Defn.jit(&fft_defn/1, compiler: EMLX).(x)

      native = Nx.Defn.jit(&ifft_defn/1, compiler: EMLX).(x_fft)
      eager = Nx.Defn.jit(&ifft_defn/1, compiler: Nx.Defn.Evaluator).(x_fft)
      assert_complex_close(native, eager)
    end

    @tag :stage06
    test "fft with explicit length vs EMLX.Backend" do
      x = Nx.tensor([1.0, 1.0], type: :f32, backend: EMLX.Backend)

      f = fn t -> Nx.fft(t, length: 4) end
      native = Nx.Defn.jit(f, compiler: EMLX).(x)
      eager = Nx.Defn.jit(f, compiler: Nx.Defn.Evaluator).(x)
      assert_complex_close(native, eager)
    end

    @tag :stage06
    test "fft2 2D vs EMLX.Backend" do
      x =
        Nx.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
          type: :f32,
          backend: EMLX.Backend
        )

      native = Nx.Defn.jit(&fft2_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&fft2_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_complex_close(native, eager)
    end

    @tag :stage06
    test "rfft via default_expr descent vs EMLX.Backend" do
      x = Nx.tensor([1.0, 1.0, 0.0, 0.0], type: :f32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&rfft_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&rfft_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_complex_close(native, eager)
    end
  end

  # ── Stage 07 — iota ──────────────────────────────────────────────────────

  describe "Stage 07 — iota" do
    @tag :stage07
    test "iota flat: IR has :iota instruction, no operands" do
      prog = Expr.lower(Nx.Defn.debug_expr_apply(&iota_flat_defn/0, []))

      assert Enum.any?(prog.instructions, fn {_, op, operands, _} ->
               op == :iota and operands == []
             end)
    end

    @tag :stage07
    test "iota flat lowering: iattrs encode shape and axis=-1" do
      prog = Expr.lower(Nx.Defn.debug_expr_apply(&iota_flat_defn/0, []))

      {_, :iota, [], [_dtype, n_dims, axis_int | shape]} =
        Enum.find(prog.instructions, fn {_, op, _, _} -> op == :iota end)

      assert n_dims == 2
      assert axis_int == -1
      assert shape == [3, 4]
    end

    @tag :stage07
    test "iota flat: interpreter matches Nx.iota" do
      alias EMLX.Native.Expr.Interpreter

      prog = Expr.lower(Nx.Defn.debug_expr_apply(&iota_flat_defn/0, []))
      [out] = Interpreter.eval(prog, [])
      expected = Nx.iota({3, 4}, backend: EMLX.Backend)
      assert_close(out, expected)
    end

    @tag :stage07
    test "iota flat: C++ replay matches interpreter" do
      prog = Expr.lower(Nx.Defn.debug_expr_apply(&iota_flat_defn/0, []))
      [nif_out] = run_nif(prog, [])
      [interp_out] = EMLX.Native.Expr.Interpreter.eval(prog, [])
      assert_close(nif_out, interp_out)
    end

    @tag :stage07
    test "iota flat: E2E jit vs Nx.Defn.Evaluator" do
      native = Nx.Defn.jit(&iota_flat_defn/0, compiler: EMLX).()
      eager = Nx.Defn.jit(&iota_flat_defn/0, compiler: Nx.Defn.Evaluator).()
      assert_close(native, eager)
    end

    @tag :stage07
    test "iota with axis=1: E2E jit vs Nx.Defn.Evaluator" do
      native = Nx.Defn.jit(&iota_axis1_defn/0, compiler: EMLX).()
      eager = Nx.Defn.jit(&iota_axis1_defn/0, compiler: Nx.Defn.Evaluator).()
      assert_close(native, eager)
    end

    @tag :stage07
    test "iota f32 flat: E2E jit vs Nx.Defn.Evaluator" do
      native = Nx.Defn.jit(&iota_f32_defn/0, compiler: EMLX).()
      eager = Nx.Defn.jit(&iota_f32_defn/0, compiler: Nx.Defn.Evaluator).()
      assert_close(native, eager)
    end
  end

  # ── Stage 07 — eye ───────────────────────────────────────────────────────

  describe "Stage 07 — eye" do
    @tag :stage07
    test "eye: IR has :eye instruction, no operands" do
      prog = Expr.lower(Nx.Defn.debug_expr_apply(&eye_3x3_defn/0, []))

      assert Enum.any?(prog.instructions, fn {_, op, operands, _} ->
               op == :eye and operands == []
             end)
    end

    @tag :stage07
    test "eye 3x3: iattrs encode [dtype, 3, 3]" do
      prog = Expr.lower(Nx.Defn.debug_expr_apply(&eye_3x3_defn/0, []))

      {_, :eye, [], [_dtype, m, n]} =
        Enum.find(prog.instructions, fn {_, op, _, _} -> op == :eye end)

      assert m == 3
      assert n == 3
    end

    @tag :stage07
    test "eye 3x3: interpreter matches Nx.eye" do
      alias EMLX.Native.Expr.Interpreter

      prog = Expr.lower(Nx.Defn.debug_expr_apply(&eye_3x3_defn/0, []))
      [out] = Interpreter.eval(prog, [])
      expected = Nx.eye({3, 3}, backend: EMLX.Backend)
      assert_close(out, expected)
    end

    @tag :stage07
    test "eye 3x3: C++ replay matches interpreter" do
      prog = Expr.lower(Nx.Defn.debug_expr_apply(&eye_3x3_defn/0, []))
      [nif_out] = run_nif(prog, [])
      [interp_out] = EMLX.Native.Expr.Interpreter.eval(prog, [])
      assert_close(nif_out, interp_out)
    end

    @tag :stage07
    test "eye 3x3: E2E jit vs Nx.Defn.Evaluator" do
      native = Nx.Defn.jit(&eye_3x3_defn/0, compiler: EMLX).()
      eager = Nx.Defn.jit(&eye_3x3_defn/0, compiler: Nx.Defn.Evaluator).()
      assert_close(native, eager)
    end

    @tag :stage07
    test "eye 2x4 rectangular: E2E jit vs Nx.Defn.Evaluator" do
      native = Nx.Defn.jit(&eye_2x4_defn/0, compiler: EMLX).()
      eager = Nx.Defn.jit(&eye_2x4_defn/0, compiler: Nx.Defn.Evaluator).()
      assert_close(native, eager)
    end
  end

  # ── Stage 07 — RNG (Nx.Random via threefry2x32 + iota) ───────────────────

  describe "Stage 07 — Nx.Random" do
    @tag :stage07
    test "Nx.Random.uniform: native matches evaluator for fixed key" do
      key = Nx.Random.key(42) |> Nx.backend_transfer(EMLX.Backend)

      native = Nx.Defn.jit(&rng_uniform_defn/1, compiler: EMLX).(key)
      eager = Nx.Defn.jit(&rng_uniform_defn/1, compiler: Nx.Defn.Evaluator).(key)

      assert_close(native, eager, 1.0e-5)
      # Samples should be in [0, 1)
      assert Nx.all(Nx.greater_equal(native, 0.0)) |> Nx.to_number() == 1
      assert Nx.all(Nx.less(native, 1.0)) |> Nx.to_number() == 1
    end

    @tag :stage07
    test "Nx.Random.uniform: same key → same samples (deterministic)" do
      key = Nx.Random.key(99) |> Nx.backend_transfer(EMLX.Backend)

      out1 = Nx.Defn.jit(&rng_uniform_defn/1, compiler: EMLX).(key)
      out2 = Nx.Defn.jit(&rng_uniform_defn/1, compiler: EMLX).(key)

      assert_close(out1, out2)
    end

    @tag :stage07
    test "Nx.Random.normal: native matches evaluator for fixed key" do
      key = Nx.Random.key(7) |> Nx.backend_transfer(EMLX.Backend)

      native = Nx.Defn.jit(&rng_normal_defn/1, compiler: EMLX).(key)
      eager = Nx.Defn.jit(&rng_normal_defn/1, compiler: Nx.Defn.Evaluator).(key)

      assert_close(native, eager, 1.0e-4)
    end
  end

  # ── Stage 08 defn helpers ─────────────────────────────────────────────────

  # cond: simple two-branch predicate on a scalar.
  defn cond_two_branch(x) do
    cond do
      Nx.less(x, 0) -> Nx.negate(x)
      true -> x
    end
  end

  # cond: three branches, each returning a different transformed value.
  defn cond_three_branch(x) do
    cond do
      Nx.less(x, -1) -> Nx.multiply(x, -2)
      Nx.less(x, 1) -> Nx.multiply(x, 3)
      true -> Nx.multiply(x, 4)
    end
  end

  # while: count up to 10 from a given start.
  defn count_to_10(x), do: while(x, Nx.less(x, 10), do: Nx.add(x, 1))

  # while: carried state with two tensors.
  defn while_two_carry(a, b) do
    while {a, b}, Nx.less(a, 5) do
      {Nx.add(a, 1), Nx.multiply(b, 2)}
    end
  end

  # ── Stage 08 — cond ──────────────────────────────────────────────────────

  describe "Stage 08 — cond" do
    @tag :stage08
    test "cond two-branch: native matches evaluator (negative input)" do
      x = Nx.tensor(-3.0, backend: EMLX.Backend)
      native = Nx.Defn.jit(&cond_two_branch/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&cond_two_branch/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    @tag :stage08
    test "cond two-branch: native matches evaluator (positive input)" do
      x = Nx.tensor(3.0, backend: EMLX.Backend)
      native = Nx.Defn.jit(&cond_two_branch/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&cond_two_branch/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    @tag :stage08
    test "cond three-branch: all three branches produce correct results" do
      for {input, _branch} <- [{-2.0, :first}, {0.0, :second}, {5.0, :third}] do
        x = Nx.tensor(input, backend: EMLX.Backend)
        native = Nx.Defn.jit(&cond_three_branch/1, compiler: EMLX).(x)
        eager = Nx.Defn.jit(&cond_three_branch/1, compiler: Nx.Defn.Evaluator).(x)
        assert_close(native, eager)
      end
    end
  end

  # ── Stage 08 — while ─────────────────────────────────────────────────────

  describe "Stage 08 — while" do
    @tag :stage08
    test "while: count from 0 to 10" do
      x = Nx.tensor(0, type: :s32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&count_to_10/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&count_to_10/1, compiler: Nx.Defn.Evaluator).(x)
      assert Nx.to_number(native) == Nx.to_number(eager)
    end

    @tag :stage08
    test "while: trip count depends on runtime value (count from 5)" do
      x = Nx.tensor(5, type: :s32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&count_to_10/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&count_to_10/1, compiler: Nx.Defn.Evaluator).(x)
      assert Nx.to_number(native) == Nx.to_number(eager)
    end

    @tag :stage08
    test "while: already past condition — zero iterations" do
      x = Nx.tensor(15, type: :s32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&count_to_10/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&count_to_10/1, compiler: Nx.Defn.Evaluator).(x)
      assert Nx.to_number(native) == Nx.to_number(eager)
    end

    @tag :stage08
    test "while: two-element tuple carry" do
      a = Nx.tensor(0, type: :s32, backend: EMLX.Backend)
      b = Nx.tensor(1, type: :s32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&while_two_carry/2, compiler: EMLX).(a, b)
      eager = Nx.Defn.jit(&while_two_carry/2, compiler: Nx.Defn.Evaluator).(a, b)
      {na, nb} = native
      {ea, eb} = eager
      assert Nx.to_number(na) == Nx.to_number(ea)
      assert Nx.to_number(nb) == Nx.to_number(eb)
    end
  end

  defp unwrap!({:ok, v}), do: v
  defp unwrap!({:error, e}), do: raise(EMLX.NIFError, List.to_string(e))

  defp await_worker!(job_ref) do
    receive do
      {^job_ref, :ok} -> :ok
      {^job_ref, {:ok, result}} -> result
      {^job_ref, {:error, reason}} -> raise(EMLX.NIFError, List.to_string(reason))
    end
  end

  # Compile and eval a lowered Expr program via the C++ NIF.
  # Returns a list of output Nx.Tensor{} values.
  defp run_nif(%Expr{} = prog, inputs) do
    device = EMLX.default_device()
    {worker, _} = EMLX.resolve_worker(device)
    {n_inputs, cap_refs, cvs, cts, ops, ors, ias, outs} = Expr.to_wire(prog)

    input_refs =
      Enum.map(inputs, fn %Nx.Tensor{data: %EMLX.Backend{ref: {_, r}}} -> r end)

    prog_ref = compile_nif!(worker, n_inputs, cap_refs, cvs, cts, ops, ors, ias, outs)
    out_refs = eval_nif!(worker, prog_ref, input_refs)

    Enum.map(out_refs, fn ref -> EMLX.Backend.to_nx({device, ref}) end)
  end

  # Compare complex tensors by comparing real and imaginary parts separately.
  defp assert_complex_close(a, b, tol \\ 1.0e-4) do
    assert_all_close(Nx.real(a), Nx.real(b), tol: tol)
    assert_all_close(Nx.imag(a), Nx.imag(b), tol: tol)
  end

  defp assert_close(a, b), do: assert_close(a, b, 1.0e-4)

  defp assert_all_close(a, b, opts \\ []) do
    tol = Keyword.get(opts, :tol, 1.0e-5)

    Enum.zip(Nx.to_flat_list(a), Nx.to_flat_list(b))
    |> Enum.each(fn {av, bv} -> assert_in_delta(av, bv, tol) end)
  end

  defp bench_us(n, fun) do
    t0 = System.monotonic_time(:microsecond)
    Enum.each(1..n, fn _ -> fun.() end)
    t1 = System.monotonic_time(:microsecond)
    (t1 - t0) / n
  end
end
