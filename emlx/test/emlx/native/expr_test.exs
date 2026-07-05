defmodule EMLX.Native.ExprTest.NestedWhileAst do
  @moduledoc false
  # Builds a `depth`-levels-deep `while`-inside-`while` `defn` body (AST),
  # used by `EMLX.Native.ExprTest` to generate `deeply_nested_while_*` defns
  # -- exercising nesting depths well beyond what's practical to hand-write,
  # to pin both "deep nesting still works" and "the defensive depth cap in
  # `EMLX.Native.Expr`'s `expand_while_native/6` actually fires". Must live
  # in its own module: a `defp` helper can't be called from `ExprTest`'s own
  # top-level `for`/`Code.eval_quoted` (its own functions aren't callable
  # until the module finishes compiling).
  def build(0, leaf), do: leaf

  def build(depth, leaf) do
    inner = build(depth - 1, quote(do: acc))

    quote do
      {acc, _} =
        while {acc = unquote(leaf), i = 0}, Nx.less(i, 1) do
          {unquote(inner), Nx.add(i, 1)}
        end

      acc
    end
  end
end

defmodule EMLX.Native.ExprTest do
  @moduledoc """
  Tests for the EMLX native defn compiler:
    - EMLX.Native.Expr struct shape (refs as node IDs, atom opcodes, attrs)
    - lower/1 (parameter, constant, tensor/capture, add, identity)
    - compile_program / eval_program NIFs via to_native/1 (C++ replay)
    - Compiler seam: Nx.Defn.compile(..., compiler: EMLX) via single-NIF replay
    - Unary + binary + compare/logical equivalence vs EMLX.Backend
  """
  use ExUnit.Case, async: false
  import Nx.Defn

  alias EMLX.Native.Expr

  # ── module-level defn helpers ─────────────────────────────────────────────
  defn add_two(a, b), do: Nx.add(a, b)
  defn add_one(x), do: Nx.add(x, 1)
  defn identity(x), do: x

  defn gt_f32(a, b), do: Nx.greater(a, b)
  defn cmp_mixed(a, b), do: Nx.greater(a, b)
  defn mixed_add(a, b), do: Nx.add(a, b)

  defn two_quantized_dots(x, w1, w2) do
    Nx.concatenate([Nx.dot(x, w1), Nx.dot(x, w2)], axis: 1)
  end

  defn dequant_surrounded(x, qw) do
    dense = EMLX.Quantization.dequantize(qw)
    Nx.add(x, dense) |> Nx.multiply(2.0)
  end

  defn dequant_only(qw), do: EMLX.Quantization.dequantize(qw)

  defn two_runtime_calls(x, qw1, qw2) do
    Nx.add(EMLX.Quantization.dequantize(qw1), EMLX.Quantization.dequantize(qw2)) |> Nx.add(x)
  end

  defn runtime_call_inside_while(x0, n, qw) do
    {result, _, _, _} =
      while {acc = x0, i = 0, n, qw}, Nx.less(i, n) do
        {Nx.add(acc, EMLX.Quantization.dequantize(qw)), i + 1, n, qw}
      end

    result
  end

  defn quantized_matmul_surrounded(x, qw) do
    EMLX.Quantization.quantized_matmul(x, qw) |> Nx.add(1.0) |> Nx.multiply(2.0)
  end

  defn dequant_surrounded_other_site(x, qw) do
    dense = EMLX.Quantization.dequantize(qw)
    Nx.add(x, dense) |> Nx.multiply(2.0)
  end

  defn hook_top_level(a, b) do
    hooked = hook(Nx.multiply(Nx.add(a, b), 2), :mid, fn t -> send(self(), {:mid, t}) end)
    Nx.subtract(hooked, 1)
  end

  defn hook_unused_value(a, b) do
    _ = hook(Nx.multiply(a, b), :dbg, fn t -> send(self(), {:dbg, t}) end)
    Nx.add(a, b)
  end

  defn hook_name_only(a, b) do
    _ = hook(Nx.multiply(a, b), :no_fn)
    Nx.add(a, b)
  end

  defn hook_tuple_payload(a, b) do
    hook({Nx.add(a, b), Nx.subtract(a, b)}, :pair, fn {s, d} -> send(self(), {:pair, s, d}) end)
  end

  defn hook_in_cond_branch(a, b) do
    pred = Nx.any(Nx.greater(a, 0))

    cond do
      pred -> hook(Nx.add(a, b), :branch_true, fn t -> send(self(), {:branch_true, t}) end)
      true -> Nx.subtract(a, b)
    end
  end

  defn hook_in_while_body(a) do
    {result, _} =
      while {acc = Nx.tensor(0), i = a}, Nx.less(0, i) do
        acc = hook(Nx.add(acc, i), :iter, fn t -> send(self(), {:iter, t}) end)
        {acc, i - 1}
      end

    result
  end

  # Generates `deeply_nested_while_20`/`deeply_nested_while_over_cap` defns
  # from `EMLX.Native.ExprTest.NestedWhileAst.build/2` (see its moduledoc).
  for {name, depth} <- [deeply_nested_while_20: 20, deeply_nested_while_over_cap: 70] do
    body = EMLX.Native.ExprTest.NestedWhileAst.build(depth, quote(do: x))

    Code.eval_quoted(
      quote do
        defn unquote(name)(x), do: unquote(body)
      end,
      [],
      __ENV__
    )
  end

  # A `while` whose body contains another `while` — natively lowerable: the
  # inner `while` lowers to its own `EMLXWhile` primitive, nested inside the
  # outer body's sub-program (see `EMLX.Native.Expr.native_while_eligible?/2`).
  defn nested_while_top_level(x) do
    {out, _i} =
      while {out = x, i = 0}, Nx.less(i, 2) do
        {inner, _j} =
          while {inner = out, j = 0}, Nx.less(j, 2) do
            {Nx.add(inner, 1.0), j + 1}
          end

        {inner, i + 1}
      end

    out
  end

  # Exercises the Nx.Defn.Graph.split-chain path (hook before AND after a
  # non-bare `while`, i.e. the while has surrounding work on both sides) --
  # this is the shape that surfaced the `Nx.Defn.Graph` `:token` rewrite gap
  # (see Results).
  defn hook_around_while(a) do
    seed = hook(Nx.multiply(a, 2), :seed, fn t -> send(self(), {:seed, t}) end)

    {result, _} =
      while {acc = Nx.tensor(0), i = seed}, Nx.less(0, i) do
        {Nx.add(acc, i), i - 1}
      end

    hook(Nx.add(result, 1), :final, fn t -> send(self(), {:final, t}) end)
  end

  defn hook_in_reduce_body(t) do
    Nx.reduce(t, Nx.tensor(0), fn x, acc ->
      hook(Nx.add(acc, x), :step, fn v -> send(self(), {:step, v}) end)
    end)
  end

  # A hook genuinely nested inside a `cond` that is itself inside a reduce
  # body must still raise -- the always-executes exemption above must not
  # blanket-exempt a real cond-branch hook one level deeper.
  defn hook_in_cond_in_reduce_body(t) do
    Nx.reduce(t, Nx.tensor(0), fn x, acc ->
      cond do
        Nx.any(Nx.greater(x, 0)) ->
          hook(Nx.add(acc, x), :pos, fn v -> send(self(), {:pos, v}) end)

        true ->
          acc
      end
    end)
  end

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

    test "closed-over tensor executes correctly (capture, not just lowering shape)" do
      # Same manually-built capture Expr as the lowering-shape test above --
      # `Nx.Defn.jit`/`check_equiv` can't exercise this naturally: Nx itself
      # guards against a real `defn` closure embedding an EMLX.Backend tensor
      # (see the comment above), so this can only be reached by hand-building
      # the Expr tree. Runs it through the real to_native/NIF path (not just
      # inspecting `prog`) to prove the capture actually evaluates correctly.
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
      [out] = run_nif(prog, [x])

      assert_in_delta Nx.to_number(out), 15.0, 1.0e-6
    end

    test "unknown op raises ArgumentError with 'does not yet lower op'" do
      expr = %Nx.Tensor{
        data: %Nx.Defn.Expr{id: make_ref(), op: :this_op_does_not_exist, args: [], context: nil},
        type: {:f, 32},
        shape: {},
        names: []
      }

      assert_raise ArgumentError, ~r/does not yet lower op/, fn -> Expr.lower(expr) end
    end
  end

  # ── to_native/1 ──────────────────────────────────────────────────────────

  describe "to_native/1" do
    test "identity program: n_inputs=1, no instructions, output encodes input ref" do
      expr = Nx.Defn.debug_expr_apply(&identity/1, [Nx.template({3}, :f32)])
      prog = Expr.lower(expr)
      program = Expr.to_native(prog)

      assert %EMLX.Native.Program{} = program
      assert program.num_inputs == 1
      assert program.captures == []
      assert program.constants == []
      assert program.instructions == []
      assert program.outputs == [{:input, 0}]
    end

    test "add program: op_name is :add, operands encode the two inputs" do
      expr = Nx.Defn.debug_expr_apply(&add_two/2, [Nx.template({}, :f32), Nx.template({}, :f32)])
      prog = Expr.lower(expr)
      %EMLX.Native.Program{instructions: [instr], outputs: [output]} = Expr.to_native(prog)

      assert instr.op == :add
      assert instr.operands == [{:input, 0}, {:input, 1}]
      assert output == {:result, 0}
    end
  end

  # ── compile_program / eval_program NIFs ──────────────────────────────────

  describe "compile_program / eval_program NIFs" do
    setup do
      device = EMLX.default_device()
      {worker, _} = EMLX.resolve_worker(device)
      %{worker: worker, device: device}
    end

    test "identity program via to_native: output equals input", %{worker: worker, device: device} do
      expr = Nx.Defn.debug_expr_apply(&identity/1, [Nx.template({3}, :f32)])
      prog = Expr.lower(expr)
      wire = Expr.to_native(prog)

      prog_ref = compile_nif!(worker, wire)

      x = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)
      %EMLX.Backend{ref: {_, input_ref}} = x.data

      [out_ref] = eval_nif!(worker, prog_ref, [input_ref])
      out = EMLX.Backend.to_nx({device, out_ref}, x)

      assert_all_close(out, x)
    end

    test "add program via to_native: correct result", %{worker: worker, device: device} do
      expr = Nx.Defn.debug_expr_apply(&add_two/2, [Nx.template({}, :f32), Nx.template({}, :f32)])
      prog = Expr.lower(expr)
      wire = Expr.to_native(prog)

      prog_ref = compile_nif!(worker, wire)

      a = Nx.tensor(3.0, backend: EMLX.Backend)
      b = Nx.tensor(4.0, backend: EMLX.Backend)
      %EMLX.Backend{ref: {_, ref_a}} = a.data
      %EMLX.Backend{ref: {_, ref_b}} = b.data

      [out_ref] = eval_nif!(worker, prog_ref, [ref_a, ref_b])
      out = EMLX.Backend.to_nx({device, out_ref}, a)

      assert_in_delta Nx.to_number(out), 7.0, 1.0e-6
    end

    # Native `:while` lowering (checkpoint c — see EMLX.Native.Expr's
    # `:while` TODO / EMLXWhile in emlx_compiler.cpp): a hand-built wire
    # `:while` instruction (no Elixir lowering path emits this yet — that's
    # checkpoint d) with `RefKind::Carry` refs inside its cond/body
    # `EMLX.Native.SubProgram`s. Exercises the whole native interpreter loop
    # end to end: `EMLXWhile::eval` calls `interpret_instructions`
    # recursively for cond (`carry < 5`) and body (`carry + 1`) each
    # iteration, forcing the carry via `mlx::core::eval` every time, until
    # cond goes false.
    test "wire format: :while instruction with nested cond/body sub-programs runs natively",
         %{worker: worker, device: device} do
      cond_subprogram = %EMLX.Native.SubProgram{
        instructions: [
          %EMLX.Native.Instruction{
            op: :less,
            operands: [{:carry, 0}, {:const, 0}],
            attrs: []
          }
        ],
        outputs: [{:result, 0}]
      }

      body_subprogram = %EMLX.Native.SubProgram{
        instructions: [
          %EMLX.Native.Instruction{
            op: :add,
            operands: [{:carry, 0}, {:const, 1}],
            attrs: []
          }
        ],
        outputs: [{:result, 0}]
      }

      wire = %EMLX.Native.Program{
        num_inputs: 1,
        captures: [],
        constants: [{5.0, :int32}, {1.0, :int32}],
        instructions: [
          %EMLX.Native.Instruction{
            op: :while,
            operands: [{:input, 0}],
            attrs: [],
            subprograms: [cond_subprogram, body_subprogram]
          }
        ],
        outputs: [{:result, 0}]
      }

      prog_ref = compile_nif!(worker, wire)

      x = Nx.tensor(0, type: :s32, backend: EMLX.Backend)
      %EMLX.Backend{ref: {_, ref_x}} = x.data

      [out_ref] = eval_nif!(worker, prog_ref, [ref_x])
      out = EMLX.Backend.to_nx({device, out_ref}, x)

      assert Nx.to_number(out) == 5
    end

    # Same native `:while` interpreter, but with two independent carry
    # slots — proves multi-slot carry indexing (`{:carry, 0}`/`{:carry, 1}`)
    # and multi-output EMLXWhile arity (one output array per input carry
    # slot) both work, matching how a real `Nx.Defn.while` over a tuple
    # accumulator would flatten to several carry leaves.
    test "wire format: :while with two independent carry slots", %{worker: worker, device: device} do
      cond_subprogram = %EMLX.Native.SubProgram{
        instructions: [
          %EMLX.Native.Instruction{
            op: :less,
            operands: [{:carry, 0}, {:const, 0}],
            attrs: []
          }
        ],
        outputs: [{:result, 0}]
      }

      body_subprogram = %EMLX.Native.SubProgram{
        instructions: [
          %EMLX.Native.Instruction{
            op: :add,
            operands: [{:carry, 0}, {:const, 1}],
            attrs: []
          },
          %EMLX.Native.Instruction{
            op: :multiply,
            operands: [{:carry, 1}, {:const, 2}],
            attrs: []
          }
        ],
        outputs: [{:result, 0}, {:result, 1}]
      }

      wire = %EMLX.Native.Program{
        num_inputs: 2,
        captures: [],
        constants: [{5.0, :int32}, {1.0, :int32}, {2.0, :int32}],
        instructions: [
          %EMLX.Native.Instruction{
            op: :while,
            operands: [{:input, 0}, {:input, 1}],
            attrs: [],
            subprograms: [cond_subprogram, body_subprogram]
          }
        ],
        outputs: [{:result, 0}, {:result, 1}]
      }

      prog_ref = compile_nif!(worker, wire)

      counter = Nx.tensor(0, type: :s32, backend: EMLX.Backend)
      acc = Nx.tensor(1, type: :s32, backend: EMLX.Backend)
      %EMLX.Backend{ref: {_, ref_counter}} = counter.data
      %EMLX.Backend{ref: {_, ref_acc}} = acc.data

      [out_counter_ref, out_acc_ref] = eval_nif!(worker, prog_ref, [ref_counter, ref_acc])
      out_counter = EMLX.Backend.to_nx({device, out_counter_ref}, counter)
      out_acc = EMLX.Backend.to_nx({device, out_acc_ref}, acc)

      assert Nx.to_number(out_counter) == 5
      assert Nx.to_number(out_acc) == 32
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

    test "unsupported op raises through the compiler seam (no Evaluator fallback)" do
      f = fn a -> Nx.population_count(a) end

      a = Nx.tensor([1, 2, 3], type: :s32, backend: EMLX.Backend)

      assert_raise ArgumentError, ~r/population_count is not supported by EMLX/, fn ->
        Nx.Defn.jit(f, compiler: EMLX).(a)
      end
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

  # Helper: compile + eval the defn via the native path, compare to eager backend.
  defp check_equiv(fun, inputs_eager, opts \\ []) do
    tol = Keyword.get(opts, :tol, 1.0e-4)
    templates = Enum.map(inputs_eager, &Nx.template(&1.shape, &1.type))
    compiled = Nx.Defn.compile(fun, templates, compiler: EMLX)
    result = apply(compiled, inputs_eager)
    eager = apply(Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator), inputs_eager)
    assert_close(result, eager, tol)
  end

  defp check_reduce_equiv(fun, inputs_eager, opts \\ []) do
    tol = Keyword.get(opts, :tol, 1.0e-4)
    templates = Enum.map(inputs_eager, &Nx.template(&1.shape, &1.type))
    compiled = Nx.Defn.compile(fun, templates, compiler: EMLX)
    result = apply(compiled, inputs_eager)

    bin_inputs = Enum.map(inputs_eager, &Nx.backend_copy(&1, Nx.BinaryBackend))
    prev = Nx.default_backend(Nx.BinaryBackend)

    eager =
      try do
        apply(Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator), bin_inputs)
      after
        Nx.default_backend(prev)
      end

    assert result.shape == eager.shape
    assert result.type == eager.type
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

  describe "unary elementwise" do
    # Sample unary ops over f32 and bf16 using representative positive values
    # to avoid NaN from log/sqrt/etc.
    test "abs/ceil/floor/negate/round/sign — f32" do
      x = Nx.tensor([1.7, -2.3, 0.0, -0.5], backend: EMLX.Backend)

      for fun <- [&Nx.abs/1, &Nx.ceil/1, &Nx.floor/1, &Nx.negate/1, &Nx.round/1, &Nx.sign/1] do
        check_equiv(fun, [x])
      end
    end

    test "exp/log/sqrt/tanh/sigmoid — f32" do
      x = Nx.tensor([0.5, 1.0, 2.0, 4.0], backend: EMLX.Backend)

      for fun <- [&Nx.exp/1, &Nx.log/1, &Nx.sqrt/1, &Nx.tanh/1, &Nx.sigmoid/1] do
        check_equiv(fun, [x])
      end
    end

    test "sin/cos/tan/asin/acos/atan — f32" do
      x = Nx.tensor([0.1, 0.5, 0.9, -0.5], backend: EMLX.Backend)

      for fun <- [&Nx.sin/1, &Nx.cos/1, &Nx.tan/1, &Nx.asin/1, &Nx.acos/1, &Nx.atan/1] do
        check_equiv(fun, [x])
      end
    end

    test "sinh/cosh/tanh/asinh/acosh/atanh — f32" do
      x = Nx.tensor([0.1, 0.5, 1.0, 1.5], backend: EMLX.Backend)

      for fun <- [&Nx.sinh/1, &Nx.cosh/1, &Nx.tanh/1, &Nx.asinh/1, &Nx.acosh/1] do
        check_equiv(fun, [x])
      end

      x2 = Nx.tensor([0.1, 0.5, 0.9, -0.5], backend: EMLX.Backend)
      check_equiv(&Nx.atanh/1, [x2])
    end

    test "erf/erf_inv/erfc/rsqrt/expm1/log1p — f32" do
      x = Nx.tensor([0.1, 0.5, 1.0, 2.0], backend: EMLX.Backend)

      for fun <- [&Nx.erf/1, &Nx.erf_inv/1, &Nx.erfc/1, &Nx.rsqrt/1, &Nx.expm1/1, &Nx.log1p/1] do
        check_equiv(fun, [x])
      end
    end

    test "cbrt — f32 positive values" do
      x = Nx.tensor([1.0, 8.0, 27.0, 0.125], backend: EMLX.Backend)
      check_equiv(&Nx.cbrt/1, [x], tol: 1.0e-3)
    end

    test "is_nan/is_infinity — f32" do
      x = Nx.tensor([1.0, :nan, :infinity, :neg_infinity], type: :f32, backend: EMLX.Backend)
      check_equiv(&Nx.is_nan/1, [x])
      check_equiv(&Nx.is_infinity/1, [x])
    end

    test "bitwise_not — s32" do
      x = Nx.tensor([0, 1, -1, 255], type: :s32, backend: EMLX.Backend)
      check_equiv(&Nx.bitwise_not/1, [x])
    end

    test "logical_not — u8" do
      x = Nx.tensor([0, 1, 1, 0], type: :u8, backend: EMLX.Backend)
      check_equiv(&Nx.logical_not/1, [x])
    end

    test "real/imag/conjugate — c64" do
      # Complex tensor: values are reals with zero imaginary parts.
      c = Nx.tensor([1.5, 2.5, -1.0], type: {:c, 64}, backend: EMLX.Backend)
      check_equiv(&Nx.real/1, [c])
      check_equiv(&Nx.imag/1, [c])
      check_equiv(&Nx.conjugate/1, [c])
    end

    test "unary ops — bf16" do
      x = Nx.tensor([0.5, 1.0, 2.0], type: :bf16, backend: EMLX.Backend)

      for fun <- [&Nx.abs/1, &Nx.exp/1, &Nx.tanh/1, &Nx.sqrt/1] do
        check_equiv(fun, [x], tol: 1.0e-2)
      end
    end
  end

  describe "binary arithmetic + bitwise" do
    test "add/subtract/multiply — f32" do
      a = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)
      b = Nx.tensor([4.0, 5.0, 6.0], backend: EMLX.Backend)

      for fun <- [&Nx.add/2, &Nx.subtract/2, &Nx.multiply/2] do
        check_equiv(fun, [a, b])
      end
    end

    test "divide/pow/atan2 — f32" do
      a = Nx.tensor([4.0, 8.0, 1.0], backend: EMLX.Backend)
      b = Nx.tensor([2.0, 4.0, 3.0], backend: EMLX.Backend)

      for fun <- [&Nx.divide/2, &Nx.pow/2, &Nx.atan2/2] do
        check_equiv(fun, [a, b])
      end
    end

    test "min/max — f32" do
      a = Nx.tensor([1.0, 5.0, 3.0], backend: EMLX.Backend)
      b = Nx.tensor([4.0, 2.0, 3.0], backend: EMLX.Backend)
      check_equiv(&Nx.min/2, [a, b])
      check_equiv(&Nx.max/2, [a, b])
    end

    test "quotient/remainder — s32 positive" do
      a = Nx.tensor([7, 9, 15], type: :s32, backend: EMLX.Backend)
      b = Nx.tensor([3, 4, 7], type: :s32, backend: EMLX.Backend)
      check_equiv(&Nx.quotient/2, [a, b])
      check_equiv(&Nx.remainder/2, [a, b])
    end

    test "remainder — s32 negative dividend" do
      a = Nx.tensor([-7, -9, 7], type: :s32, backend: EMLX.Backend)
      b = Nx.tensor([3, 4, -3], type: :s32, backend: EMLX.Backend)
      check_equiv(&Nx.remainder/2, [a, b])
    end

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

    test "mixed dtypes: add(s32, f32) → f32 with implicit upcast" do
      a = Nx.tensor([1, 2, 3], type: :s32, backend: EMLX.Backend)
      b = Nx.tensor([0.5, 1.5, 2.5], type: :f32, backend: EMLX.Backend)
      check_equiv(&mixed_add/2, [a, b])
    end

    test "binary ops — bf16" do
      a = Nx.tensor([1.0, 2.0, 4.0], type: :bf16, backend: EMLX.Backend)
      b = Nx.tensor([2.0, 1.0, 2.0], type: :bf16, backend: EMLX.Backend)

      for fun <- [&Nx.add/2, &Nx.subtract/2, &Nx.multiply/2] do
        check_equiv(fun, [a, b], tol: 1.0e-2)
      end
    end
  end

  describe "compare and logical" do
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

    test "compare — s32" do
      a = Nx.tensor([-1, 0, 1, 2], type: :s32, backend: EMLX.Backend)
      b = Nx.tensor([0, 0, 0, 1], type: :s32, backend: EMLX.Backend)

      for fun <- [&Nx.equal/2, &Nx.less/2, &Nx.greater/2] do
        check_equiv(fun, [a, b])
      end
    end

    test "logical_and/or/xor — u8" do
      a = Nx.tensor([0, 0, 1, 1], type: :u8, backend: EMLX.Backend)
      b = Nx.tensor([0, 1, 0, 1], type: :u8, backend: EMLX.Backend)

      for fun <- [&Nx.logical_and/2, &Nx.logical_or/2, &Nx.logical_xor/2] do
        check_equiv(fun, [a, b])
      end
    end

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

    test "compare with mixed dtypes s32/f32 — output u8" do
      a = Nx.tensor([1, 3], type: :s32, backend: EMLX.Backend)
      b = Nx.tensor([2.0, 1.0], type: :f32, backend: EMLX.Backend)
      check_equiv(&cmp_mixed/2, [a, b])
    end
  end

  describe "reshape" do
    test "1D → 2D, 2D → 1D" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.reshape(t, {2, 3}) end, [x])
      check_equiv(fn t -> Nx.reshape(t, {3, 2}) end, [x])
    end

    test "2D → 1D (flatten)" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.reshape(t, {4}) end, [x])
    end

    test "rank-changing — s32" do
      x = Nx.tensor([[[1, 2], [3, 4]]], type: :s32, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.reshape(t, {2, 2}) end, [x])
      check_equiv(fn t -> Nx.reshape(t, {4}) end, [x])
    end
  end

  describe "squeeze" do
    test "remove singleton dimension" do
      x = Nx.tensor([[1.0, 2.0, 3.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.squeeze(t, axes: [0]) end, [x])
    end

    test "multiple axes, negative axis" do
      x = Nx.tensor([[[1.0], [2.0], [3.0]]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.squeeze(t, axes: [0, 2]) end, [x])
      check_equiv(fn t -> Nx.squeeze(t, axes: [-1]) end, [x])
    end
  end

  describe "transpose" do
    test "2D matrix transpose" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.transpose(t) end, [x])
      check_equiv(fn t -> Nx.transpose(t, axes: [1, 0]) end, [x])
    end

    test "3D permutation, negative perm" do
      x = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.transpose(t, axes: [2, 0, 1]) end, [x])
      check_equiv(fn t -> Nx.transpose(t, axes: [-1, -3, -2]) end, [x])
    end
  end

  describe "as_type" do
    test "f32 → s32 and back" do
      x = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.as_type(t, {:s, 32}) end, [x])
      xi = Nx.tensor([1, 2, 3], type: :s32, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.as_type(t, {:f, 32}) end, [xi])
    end

    test "f32 → bf16 and back" do
      x = Nx.tensor([0.5, 1.5, 2.5], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.as_type(t, {:bf, 16}) end, [x], tol: 1.0e-2)
      xb = Nx.tensor([0.5, 1.5, 2.5], type: {:bf, 16}, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.as_type(t, {:f, 32}) end, [xb])
    end
  end

  describe "bitcast" do
    test "u8 → s8 same bit pattern" do
      # Values chosen so the bit pattern is unambiguous for both u8 and s8.
      x = Nx.tensor([1, 2, 3, 127], type: :u8, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.bitcast(t, {:s, 8}) end, [x])
    end

    test "f32 → u32 round-trip" do
      x = Nx.tensor([1.0, 2.0, 0.0], type: :f32, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.bitcast(t, {:u, 32}) end, [x])
    end
  end

  describe "broadcast" do
    test "scalar → 1D" do
      x = Nx.tensor(1.0, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.broadcast(t, {4}) end, [x])
    end

    test "1D → 2D row broadcast" do
      x = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.broadcast(t, {2, 3}) end, [x])
    end

    test "1D column broadcast (axes: [0])" do
      x = Nx.tensor([1.0, 2.0], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.broadcast(t, {2, 3}, axes: [0]) end, [x])
    end

    test "2D → 3D, broadcast_in_dim style" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.broadcast(t, {3, 2, 2}) end, [x])
    end
  end

  describe "pad" do
    test "zero-padding on 1D" do
      x = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.pad(t, 0.0, [{1, 1, 0}]) end, [x])
    end

    test "zero-padding on 2D, asymmetric" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.pad(t, 0.0, [{0, 1, 0}, {1, 0, 0}]) end, [x])
    end

    test "scalar pad value" do
      x = Nx.tensor([1.0, 2.0], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.pad(t, -1.0, [{2, 2, 0}]) end, [x])
    end

    test "interior padding on 1D" do
      x = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.pad(t, 0.0, [{0, 0, 2}]) end, [x])
    end

    test "interior padding on 2D, both axes, non-scalar-friendly pad value" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.pad(t, -9.0, [{0, 0, 1}, {0, 0, 2}]) end, [x])
    end

    test "negative lo/hi crops on 2D" do
      x = Nx.iota({4, 5}, type: :f32, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.pad(t, 0.0, [{-1, -1, 0}, {0, -2, 0}]) end, [x])
    end

    test "mixed positive/negative/interior padding on 2D" do
      x = Nx.iota({3, 4}, type: :f32, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.pad(t, 0.0, [{2, -1, 1}, {-1, 3, 2}]) end, [x])
    end

    test "interior padding on 3D" do
      x = Nx.iota({2, 3, 4}, type: :f32, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.pad(t, 0.0, [{1, 1, 1}, {-1, 0, 0}, {0, 2, 2}]) end, [x])
    end
  end

  describe "reverse" do
    test "1D reverse" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.reverse(t) end, [x])
    end

    test "2D reverse single axis, both axes" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.reverse(t, axes: [0]) end, [x])
      check_equiv(fn t -> Nx.reverse(t, axes: [1]) end, [x])
      check_equiv(fn t -> Nx.reverse(t, axes: [0, 1]) end, [x])
    end

    test "negative axis" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.reverse(t, axes: [-1]) end, [x])
    end
  end

  describe "concatenate" do
    test "concat 1D along axis 0" do
      a = Nx.tensor([1.0, 2.0], backend: EMLX.Backend)
      b = Nx.tensor([3.0, 4.0, 5.0], backend: EMLX.Backend)
      check_equiv(fn x, y -> Nx.concatenate([x, y], axis: 0) end, [a, b])
    end

    test "concat 2D along axis 0 and axis 1" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      b = Nx.tensor([[5.0, 6.0]], backend: EMLX.Backend)
      c = Nx.tensor([[7.0, 8.0], [9.0, 10.0]], backend: EMLX.Backend)
      check_equiv(fn x, y -> Nx.concatenate([x, y], axis: 0) end, [a, b])
      check_equiv(fn x, y -> Nx.concatenate([x, y], axis: 1) end, [a, c])
    end

    test "three tensors" do
      a = Nx.tensor([1.0], backend: EMLX.Backend)
      b = Nx.tensor([2.0, 3.0], backend: EMLX.Backend)
      c = Nx.tensor([4.0], backend: EMLX.Backend)
      check_equiv(fn x, y, z -> Nx.concatenate([x, y, z]) end, [a, b, c])
    end
  end

  describe "stack" do
    test "stack 1D tensors → 2D along axis 0" do
      a = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)
      b = Nx.tensor([4.0, 5.0, 6.0], backend: EMLX.Backend)
      check_equiv(fn x, y -> Nx.stack([x, y]) end, [a, b])
    end

    test "stack along axis 1" do
      a = Nx.tensor([1.0, 2.0], backend: EMLX.Backend)
      b = Nx.tensor([3.0, 4.0], backend: EMLX.Backend)
      check_equiv(fn x, y -> Nx.stack([x, y], axis: 1) end, [a, b])
    end

    test "negative axis" do
      a = Nx.tensor([1.0, 2.0], backend: EMLX.Backend)
      b = Nx.tensor([3.0, 4.0], backend: EMLX.Backend)
      check_equiv(fn x, y -> Nx.stack([x, y], axis: -1) end, [a, b])
    end
  end

  describe "squeeze without explicit axes" do
    test "squeeze all singleton dims" do
      x = Nx.tensor([[[1.0]]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.squeeze(t) end, [x])
    end
  end

  describe "sum" do
    test "sum all axes f32" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.sum(t) end, [x])
    end

    test "sum along axis 0 with keep_axes" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.sum(t, axes: [0], keep_axes: true) end, [x])
    end

    test "sum along axis 1 f32" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.sum(t, axes: [1]) end, [x])
    end

    test "sum 3D along multiple axes" do
      x = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.sum(t, axes: [0, 2]) end, [x])
    end
  end

  describe "product" do
    test "product all axes f32" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.product(t) end, [x])
    end

    test "product along axis 0" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.product(t, axes: [0]) end, [x])
    end
  end

  describe "all / any" do
    test "all on boolean-like tensor" do
      x = Nx.tensor([[1, 1], [1, 0]], type: :u8, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.all(t) end, [x])
    end

    test "all along axis 0" do
      x = Nx.tensor([[1, 0], [1, 1]], type: :u8, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.all(t, axes: [0]) end, [x])
    end

    test "any all axes" do
      x = Nx.tensor([[0, 0], [0, 1]], type: :u8, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.any(t) end, [x])
    end

    test "any along axis 1 with keep_axes" do
      x = Nx.tensor([[0, 1], [0, 0]], type: :u8, backend: EMLX.Backend)
      check_equiv(fn t -> Nx.any(t, axes: [1], keep_axes: true) end, [x])
    end
  end

  describe "custom-fun reduce (static unroll)" do
    test "1d sum reducer" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0], backend: EMLX.Backend)
      check_reduce_equiv(fn t -> Nx.reduce(t, 0.0, fn a, b -> Nx.add(a, b) end) end, [x])
    end

    test "1d product reducer" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: EMLX.Backend)
      check_reduce_equiv(fn t -> Nx.reduce(t, 1.0, fn a, b -> Nx.multiply(a, b) end) end, [x])
    end

    test "non-commutative affine reducer (validates fold order)" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0], backend: EMLX.Backend)

      check_reduce_equiv(
        fn t -> Nx.reduce(t, 0.0, fn a, b -> Nx.add(Nx.multiply(a, 2.0), b) end) end,
        [x]
      )
    end

    test "2d reduce along single axis" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: EMLX.Backend)

      check_reduce_equiv(
        fn t -> Nx.reduce(t, 0.0, [axes: [1]], fn a, b -> Nx.add(a, b) end) end,
        [x]
      )
    end

    test "2d reduce along single axis keep_axes" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: EMLX.Backend)

      check_reduce_equiv(
        fn t -> Nx.reduce(t, 0.0, [axes: [1], keep_axes: true], fn a, b -> Nx.add(a, b) end) end,
        [x]
      )
    end

    test "3d reduce over multiple axes" do
      x = Nx.iota({2, 3, 4}, type: :f32, backend: EMLX.Backend)

      check_reduce_equiv(
        fn t -> Nx.reduce(t, 0.0, [axes: [0, 2]], fn a, b -> Nx.add(a, b) end) end,
        [x]
      )
    end

    test "integer max reducer" do
      x = Nx.tensor([3, 1, 4, 1, 5, 9, 2, 6], backend: EMLX.Backend)
      check_reduce_equiv(fn t -> Nx.reduce(t, 0, fn a, b -> Nx.max(a, b) end) end, [x])
    end

    test "runtime accumulator input" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0], backend: EMLX.Backend)
      acc = Nx.tensor(10.0, backend: EMLX.Backend)
      check_reduce_equiv(fn t, a -> Nx.reduce(t, a, fn x, y -> Nx.add(x, y) end) end, [x, acc])
    end

    test "dtype-changing reduce (s32 input, f32 acc → f32)" do
      x = Nx.tensor([1, 2, 3, 4], type: :s32, backend: EMLX.Backend)
      check_reduce_equiv(fn t -> Nx.reduce(t, 0.0, fn a, b -> Nx.add(a, b) end) end, [x])
    end
  end

  describe "custom-fun window_reduce (static unroll)" do
    test "1d window sum reducer (valid, default strides)" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0], backend: EMLX.Backend)

      check_reduce_equiv(fn t -> Nx.window_reduce(t, 0.0, {2}, fn a, b -> Nx.add(a, b) end) end, [
        x
      ])
    end

    test "1d window max reducer with :same padding" do
      x = Nx.tensor([1.0, 5.0, 2.0, 4.0, 3.0], backend: EMLX.Backend)

      check_reduce_equiv(
        fn t ->
          Nx.window_reduce(t, -100.0, {3}, [padding: :same], fn a, b -> Nx.max(a, b) end)
        end,
        [x]
      )
    end

    test "2d window sum with strides" do
      x = Nx.iota({4, 4}, type: :f32, backend: EMLX.Backend)

      check_reduce_equiv(
        fn t ->
          Nx.window_reduce(t, 0.0, {2, 2}, [strides: [2, 2]], fn a, b -> Nx.add(a, b) end)
        end,
        [x]
      )
    end

    test "1d window with dilations" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0], backend: EMLX.Backend)

      check_reduce_equiv(
        fn t ->
          Nx.window_reduce(t, 0.0, {2}, [window_dilations: [2]], fn a, b -> Nx.add(a, b) end)
        end,
        [x]
      )
    end

    test "non-commutative affine reducer (validates window fold order)" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0], backend: EMLX.Backend)

      check_reduce_equiv(
        fn t -> Nx.window_reduce(t, 0.0, {3}, fn a, b -> Nx.add(Nx.multiply(a, 2.0), b) end) end,
        [x]
      )
    end

    test "explicit padding config (asymmetric lo/hi)" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: EMLX.Backend)

      check_reduce_equiv(
        fn t ->
          Nx.window_reduce(t, 0.0, {2}, [padding: [{1, 0}]], fn a, b -> Nx.add(a, b) end)
        end,
        [x]
      )
    end

    test "integer window_reduce (s32 in/out)" do
      # window_reduce output dtype tracks the input tensor (not the acc), so the
      # dtype coverage here is the integer path; acc casts to the s32 out type.
      x = Nx.tensor([1, 2, 3, 4, 5], type: :s32, backend: EMLX.Backend)

      check_reduce_equiv(fn t -> Nx.window_reduce(t, 0, {2}, fn a, b -> Nx.add(a, b) end) end, [x])
    end

    test "runtime accumulator input" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: EMLX.Backend)
      acc = Nx.tensor(0.5, backend: EMLX.Backend)

      check_reduce_equiv(
        fn t, a -> Nx.window_reduce(t, a, {2}, fn x, y -> Nx.add(x, y) end) end,
        [x, acc]
      )
    end
  end

  describe "reduce_max / reduce_min" do
    test "reduce_max all axes f32" do
      x = Nx.tensor([[1.0, 5.0], [3.0, 2.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.reduce_max(t) end, [x])
    end

    test "reduce_max along axis 1 keep_axes" do
      x = Nx.tensor([[1.0, 5.0], [3.0, 2.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.reduce_max(t, axes: [1], keep_axes: true) end, [x])
    end

    test "reduce_min all axes" do
      x = Nx.tensor([[4.0, 2.0], [1.0, 3.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.reduce_min(t) end, [x])
    end

    test "reduce_min along axis 0" do
      x = Nx.tensor([[4.0, 2.0], [1.0, 3.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.reduce_min(t, axes: [0]) end, [x])
    end
  end

  describe "argmax / argmin" do
    test "argmax along axis 1" do
      x = Nx.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.argmax(t, axis: 1) end, [x])
    end

    test "argmax along axis 0 keep_axis" do
      x = Nx.tensor([[1.0, 3.0], [4.0, 2.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.argmax(t, axis: 0, keep_axis: true) end, [x])
    end

    test "argmin along axis 0" do
      x = Nx.tensor([[4.0, 2.0], [1.0, 5.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.argmin(t, axis: 0) end, [x])
    end

    test "argmax global (no axis)" do
      x = Nx.tensor([[1.0, 7.0], [3.0, 2.0]], backend: EMLX.Backend)
      check_equiv(fn t -> Nx.argmax(t) end, [x])
    end
  end

  describe "dot (non-batched)" do
    test "matmul {2,3} × {3,4} → {2,4}" do
      a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: EMLX.Backend)

      b =
        Nx.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
          backend: EMLX.Backend
        )

      check_equiv(fn x, y -> Nx.dot(x, y) end, [a, b])
    end

    test "inner product {4} · {4} → scalar" do
      a = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: EMLX.Backend)
      b = Nx.tensor([5.0, 6.0, 7.0, 8.0], backend: EMLX.Backend)
      check_equiv(fn x, y -> Nx.dot(x, y) end, [a, b])
    end

    test "tensordot explicit contraction axes" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      b = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], backend: EMLX.Backend)
      check_equiv(fn x, y -> Nx.dot(x, [1], [], y, [0], []) end, [a, b])
    end
  end

  describe "dot (batched)" do
    test "batched matmul {2,3,4} · {2,4,5} → {2,3,5}" do
      a = Nx.iota({2, 3, 4}, type: :f32, backend: EMLX.Backend)
      b = Nx.iota({2, 4, 5}, type: :f32, backend: EMLX.Backend)
      check_equiv(fn x, y -> Nx.dot(x, [2], [0], y, [1], [0]) end, [a, b], tol: 1.0e-3)
    end
  end

  describe "conv (1D)" do
    test "1D conv {1,1,5} input, {1,1,3} kernel" do
      # 3D: {batch=1, in_channels=1, length=5}; kernel {out=1, in=1, size=3}
      input = Nx.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]], backend: EMLX.Backend)
      kernel = Nx.tensor([[[1.0, 0.0, -1.0]]], backend: EMLX.Backend)
      check_equiv(fn x, k -> Nx.conv(x, k) end, [input, kernel], tol: 1.0e-4)
    end

    test "1D conv with stride 2 and padding" do
      input = Nx.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]], backend: EMLX.Backend)
      kernel = Nx.tensor([[[1.0, 1.0]]], backend: EMLX.Backend)

      check_equiv(fn x, k -> Nx.conv(x, k, strides: [2], padding: :same) end, [input, kernel],
        tol: 1.0e-4
      )
    end
  end

  describe "conv (2D)" do
    test "2D conv identity kernel {1,1,3,3} × 1-ch input" do
      input = Nx.iota({1, 1, 4, 4}, type: :f32, backend: EMLX.Backend)

      kernel =
        Nx.tensor([[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]], backend: EMLX.Backend)

      check_equiv(fn x, k -> Nx.conv(x, k, padding: :same) end, [input, kernel], tol: 1.0e-4)
    end

    test "2D conv with strides and dilations" do
      input = Nx.iota({1, 1, 6, 6}, type: :f32, backend: EMLX.Backend)
      kernel = Nx.tensor([[[[1.0, 1.0], [1.0, 1.0]]]], backend: EMLX.Backend)
      check_equiv(fn x, k -> Nx.conv(x, k, strides: [2, 2]) end, [input, kernel], tol: 1.0e-4)
    end

    test "2D conv multi-channel" do
      input = Nx.iota({1, 3, 4, 4}, type: :f32, backend: EMLX.Backend)
      kernel = Nx.iota({2, 3, 2, 2}, type: :f32, backend: EMLX.Backend)
      check_equiv(fn x, k -> Nx.conv(x, k) end, [input, kernel], tol: 1.0e-3)
    end
  end

  defn sum_axis1_keep(x), do: Nx.sum(x, axes: [1], keep_axes: true)
  defn argmax_axis0(x), do: Nx.argmax(x, axis: 0)
  defn matmul_defn(a, b), do: Nx.dot(a, b)
  defn reduce_max_defn(x), do: Nx.reduce_max(x)

  describe "E2E jit smoke (reductions/dot/conv)" do
    test "sum with keep_axes=true via jit" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: EMLX.Backend)
      result = Nx.Defn.jit(&sum_axis1_keep/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&sum_axis1_keep/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(result, eager, 1.0e-4)
    end

    test "argmax via jit" do
      x = Nx.tensor([[1.0, 3.0], [4.0, 2.0]], backend: EMLX.Backend)
      result = Nx.Defn.jit(&argmax_axis0/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&argmax_axis0/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(result, eager, 1.0e-4)
    end

    test "matmul via jit" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: EMLX.Backend)
      b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]], backend: EMLX.Backend)
      result = Nx.Defn.jit(&matmul_defn/2, compiler: EMLX).(a, b)
      eager = Nx.Defn.jit(&matmul_defn/2, compiler: Nx.Defn.Evaluator).(a, b)
      assert_close(result, eager, 1.0e-3)
    end

    test "reduce_max via jit" do
      x = Nx.tensor([[4.0, 1.0], [2.0, 3.0]], backend: EMLX.Backend)
      result = Nx.Defn.jit(&reduce_max_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&reduce_max_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(result, eager, 1.0e-4)
    end
  end

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

  describe "select" do
    test "equivalence vs EMLX.Backend" do
      pred = Nx.tensor([1, 0, 1], type: :u8, backend: EMLX.Backend)
      a = Nx.tensor([10.0, 20.0, 30.0], backend: EMLX.Backend)
      b = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)

      native = Nx.Defn.jit(&select_defn/3, compiler: EMLX).(pred, a, b)
      eager = Nx.Defn.jit(&select_defn/3, compiler: Nx.Defn.Evaluator).(pred, a, b)
      assert_close(native, eager)
    end

    test "select with mixed-type true/false is cast to out_type" do
      pred = Nx.tensor([1, 0], type: :u8, backend: EMLX.Backend)
      a = Nx.tensor([1, 2], type: :s32, backend: EMLX.Backend)
      b = Nx.tensor([3, 4], type: :s32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&select_defn/3, compiler: EMLX).(pred, a, b)
      eager = Nx.Defn.jit(&select_defn/3, compiler: Nx.Defn.Evaluator).(pred, a, b)
      assert_close(native, eager)
    end
  end

  describe "clip" do
    test "equivalence vs EMLX.Backend — f32" do
      x = Nx.iota({5}, type: :f32, backend: EMLX.Backend) |> Nx.subtract(2.0)
      lo = Nx.tensor(-1.0, backend: EMLX.Backend)
      hi = Nx.tensor(1.5, backend: EMLX.Backend)
      native = Nx.Defn.jit(&clip_defn/3, compiler: EMLX).(x, lo, hi)
      eager = Nx.Defn.jit(&clip_defn/3, compiler: Nx.Defn.Evaluator).(x, lo, hi)
      assert_close(native, eager)
    end

    test "equivalence vs EMLX.Backend — s32" do
      x = Nx.tensor([-5, -1, 0, 3, 10], type: :s32, backend: EMLX.Backend)
      lo = Nx.tensor(0, type: :s32, backend: EMLX.Backend)
      hi = Nx.tensor(5, type: :s32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&clip_defn/3, compiler: EMLX).(x, lo, hi)
      eager = Nx.Defn.jit(&clip_defn/3, compiler: Nx.Defn.Evaluator).(x, lo, hi)
      assert_close(native, eager)
    end
  end

  describe "slice (static indices)" do
    test "equivalence vs EMLX.Backend — static 2D slice" do
      x = Nx.iota({4, 5}, type: :f32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&slice_static_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&slice_static_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    test "equivalence vs EMLX.Backend — strided slice" do
      x = Nx.iota({4, 6}, type: :f32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&slice_strided_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&slice_strided_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end
  end

  describe "slice (dynamic index)" do
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

  describe "put_slice (static indices)" do
    test "equivalence vs EMLX.Backend — static 2D put_slice" do
      x = Nx.iota({4, 4}, type: :f32, backend: EMLX.Backend)
      patch = Nx.broadcast(Nx.tensor(99.0, backend: EMLX.Backend), {2, 2})
      native = Nx.Defn.jit(&put_slice_static_defn/2, compiler: EMLX).(x, patch)
      eager = Nx.Defn.jit(&put_slice_static_defn/2, compiler: Nx.Defn.Evaluator).(x, patch)
      assert_close(native, eager)
    end

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

  describe "gather" do
    test "equivalence vs EMLX.Backend — 2D gather on axis 0" do
      x = Nx.iota({4, 3}, type: :f32, backend: EMLX.Backend)
      idx = Nx.tensor([[0], [2], [1]], type: :s32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&gather_defn/2, compiler: EMLX).(x, idx)
      eager = Nx.Defn.jit(&gather_defn/2, compiler: Nx.Defn.Evaluator).(x, idx)
      assert_close(native, eager)
    end

    test "equivalence vs EMLX.Backend — multi-axis gather" do
      x = Nx.iota({3, 4}, type: :f32, backend: EMLX.Backend)
      idx = Nx.tensor([[0, 1], [2, 3]], type: :s32, backend: EMLX.Backend)

      gather_multi = fn x, idx -> Nx.gather(x, idx, axes: [0, 1]) end

      native = Nx.Defn.jit(gather_multi, compiler: EMLX).(x, idx)
      eager = Nx.Defn.jit(gather_multi, compiler: Nx.Defn.Evaluator).(x, idx)
      assert_close(native, eager)
    end
  end

  describe "take" do
    test "equivalence vs EMLX.Backend" do
      x = Nx.iota({3, 4}, type: :f32, backend: EMLX.Backend)
      idx = Nx.tensor([2, 0, 3, 1], type: :s32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&take_defn/2, compiler: EMLX).(x, idx)
      eager = Nx.Defn.jit(&take_defn/2, compiler: Nx.Defn.Evaluator).(x, idx)
      assert_close(native, eager)
    end
  end

  describe "take_along_axis" do
    test "equivalence vs EMLX.Backend" do
      x = Nx.iota({3, 4}, type: :f32, backend: EMLX.Backend)
      idx = Nx.tensor([[2, 0, 1, 2]], type: :s32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&take_along_axis_defn/2, compiler: EMLX).(x, idx)
      eager = Nx.Defn.jit(&take_along_axis_defn/2, compiler: Nx.Defn.Evaluator).(x, idx)
      assert_close(native, eager)
    end
  end

  describe "indexed_put" do
    test "equivalence vs EMLX.Backend" do
      x = Nx.iota({4, 3}, type: :f32, backend: EMLX.Backend)
      idx = Nx.tensor([[0], [2]], type: :s32, backend: EMLX.Backend)
      updates = Nx.tensor([[99.0, 99.0, 99.0], [88.0, 88.0, 88.0]], backend: EMLX.Backend)
      native = Nx.Defn.jit(&indexed_put_defn/3, compiler: EMLX).(x, idx, updates)
      eager = Nx.Defn.jit(&indexed_put_defn/3, compiler: Nx.Defn.Evaluator).(x, idx, updates)
      assert_close(native, eager)
    end
  end

  describe "indexed_add" do
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

  describe "E2E jit smoke (select/slice/gather/indexed)" do
    test "select via jit" do
      pred = Nx.tensor([1, 0, 1, 0], type: :u8, backend: EMLX.Backend)
      a = Nx.iota({4}, type: :f32, backend: EMLX.Backend)
      b = Nx.multiply(Nx.iota({4}, type: :f32, backend: EMLX.Backend), -1.0)
      native = Nx.Defn.jit(&select_defn/3, compiler: EMLX).(pred, a, b)
      eager = Nx.Defn.jit(&select_defn/3, compiler: Nx.Defn.Evaluator).(pred, a, b)
      assert_close(native, eager)
    end

    test "static slice + add via jit" do
      x = Nx.iota({4, 5}, type: :f32, backend: EMLX.Backend)
      add_slice = fn x -> Nx.add(Nx.slice(x, [0, 0], [2, 3]), 1.0) end
      native = Nx.Defn.jit(add_slice, compiler: EMLX).(x)
      eager = Nx.Defn.jit(add_slice, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end
  end

  # ── private helpers ───────────────────────────────────────────────────────

  defp compile_nif!(worker, %EMLX.Native.Program{} = wire) do
    EMLX.NIF.compile_program(worker, wire)
    |> unwrap!()
    |> await_worker!()
  end

  defp eval_nif!(worker, prog_ref, input_refs) do
    EMLX.NIF.eval_program(worker, prog_ref, input_refs)
    |> unwrap!()
    |> await_worker!()
  end

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

  describe "sort" do
    test "sort ascending, axis 0 vs EMLX.Backend — f32" do
      x =
        Nx.tensor([[3.0, 1.0, 2.0], [9.0, 4.0, 7.0]], backend: EMLX.Backend)

      native = Nx.Defn.jit(&sort_asc_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&sort_asc_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    test "sort descending, axis 1 vs EMLX.Backend — f32" do
      x =
        Nx.tensor([[3.0, 1.0, 2.0], [9.0, 4.0, 7.0]], backend: EMLX.Backend)

      native = Nx.Defn.jit(&sort_desc_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&sort_desc_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

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

    test "sort s32 ascending" do
      x = Nx.tensor([5, 3, 1, 4, 2], type: :s32, backend: EMLX.Backend)

      native =
        Nx.Defn.jit(fn t -> Nx.sort(t, axis: 0) end, compiler: EMLX).(x)

      eager =
        Nx.Defn.jit(fn t -> Nx.sort(t, axis: 0) end, compiler: Nx.Defn.Evaluator).(x)

      assert_close(native, eager)
    end
  end

  describe "argsort" do
    test "argsort ascending, axis 0 vs EMLX.Backend" do
      x =
        Nx.tensor([[3.0, 1.0, 2.0], [9.0, 4.0, 7.0]], backend: EMLX.Backend)

      native = Nx.Defn.jit(&argsort_asc_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&argsort_asc_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    test "argsort descending, axis 1 vs EMLX.Backend" do
      x =
        Nx.tensor([[3.0, 1.0, 2.0], [9.0, 4.0, 7.0]], backend: EMLX.Backend)

      native = Nx.Defn.jit(&argsort_desc_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&argsort_desc_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    test "argsort output type matches out tensor type" do
      x = Nx.tensor([3.0, 1.0, 2.0], backend: EMLX.Backend)
      prog = Expr.lower(Nx.Defn.debug_expr_apply(fn t -> Nx.argsort(t, axis: 0) end, [x]))
      # argsort output should be :u64 (Nx default for argsort)
      assert Enum.any?(prog.instructions, fn {_, op, _, _} -> op == :argsort end)
    end
  end

  describe "window reductions" do
    test "window_sum 2x2, no padding vs EMLX.Backend — f32" do
      x = Nx.iota({4, 4}, type: :f32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&window_sum_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&window_sum_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    test "window_max 2x2, no padding vs EMLX.Backend — f32" do
      x = Nx.iota({4, 4}, type: :f32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&window_max_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&window_max_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    test "window_min 2x2, no padding vs EMLX.Backend — f32" do
      x = Nx.iota({4, 4}, type: :f32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&window_min_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&window_min_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    test "window_product 2x2, no padding vs EMLX.Backend — f32" do
      x =
        Nx.iota({4, 4}, type: :f32, backend: EMLX.Backend)
        |> Nx.add(1.0)

      native = Nx.Defn.jit(&window_product_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&window_product_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    test "window_sum with padding vs EMLX.Backend" do
      x = Nx.iota({3, 3}, type: :f32, backend: EMLX.Backend)

      f = fn t -> Nx.window_sum(t, {2, 2}, padding: :same) end
      native = Nx.Defn.jit(f, compiler: EMLX).(x)
      eager = Nx.Defn.jit(f, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    test "window_max with strides vs EMLX.Backend" do
      x = Nx.iota({6, 6}, type: :f32, backend: EMLX.Backend)

      f = fn t -> Nx.window_max(t, {2, 2}, strides: [2, 2]) end
      native = Nx.Defn.jit(f, compiler: EMLX).(x)
      eager = Nx.Defn.jit(f, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    test "window_sum 1D vs EMLX.Backend" do
      x = Nx.iota({6}, type: :f32, backend: EMLX.Backend)

      f = fn t -> Nx.window_sum(t, {3}) end
      native = Nx.Defn.jit(f, compiler: EMLX).(x)
      eager = Nx.Defn.jit(f, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end
  end

  describe "cumulative reductions" do
    test "cumulative_sum axis 1 vs EMLX.Backend — f32" do
      x = Nx.iota({2, 4}, type: :f32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&cumulative_sum_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&cumulative_sum_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    test "cumulative_product axis 0 vs EMLX.Backend — f32" do
      x = Nx.iota({3, 3}, type: :f32, backend: EMLX.Backend) |> Nx.add(1.0)

      native = Nx.Defn.jit(&cumulative_product_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&cumulative_product_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    test "cumulative_min axis 0 vs EMLX.Backend — s32" do
      x = Nx.tensor([[3, 1, 4], [1, 5, 9], [2, 6, 5]], type: :s32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&cumulative_min_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&cumulative_min_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    test "cumulative_max axis 0 reverse vs EMLX.Backend — f32" do
      x = Nx.iota({4, 3}, type: :f32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&cumulative_max_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&cumulative_max_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    test "cumulative_sum s32 axis 0 vs EMLX.Backend" do
      x = Nx.tensor([1, 2, 3, 4], type: :s32, backend: EMLX.Backend)

      f = fn t -> Nx.cumulative_sum(t, axis: 0) end
      native = Nx.Defn.jit(f, compiler: EMLX).(x)
      eager = Nx.Defn.jit(f, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end
  end

  describe "fft / ifft" do
    test "fft 1D vs EMLX.Backend" do
      x = Nx.tensor([1.0, 1.0, 0.0, 0.0], type: :f32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&fft_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&fft_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_complex_close(native, eager)
    end

    test "ifft 1D vs EMLX.Backend" do
      x = Nx.tensor([1.0, 1.0, 0.0, 0.0], type: :f32, backend: EMLX.Backend)
      x_fft = Nx.Defn.jit(&fft_defn/1, compiler: EMLX).(x)

      native = Nx.Defn.jit(&ifft_defn/1, compiler: EMLX).(x_fft)
      eager = Nx.Defn.jit(&ifft_defn/1, compiler: Nx.Defn.Evaluator).(x_fft)
      assert_complex_close(native, eager)
    end

    test "fft with explicit length vs EMLX.Backend" do
      x = Nx.tensor([1.0, 1.0], type: :f32, backend: EMLX.Backend)

      f = fn t -> Nx.fft(t, length: 4) end
      native = Nx.Defn.jit(f, compiler: EMLX).(x)
      eager = Nx.Defn.jit(f, compiler: Nx.Defn.Evaluator).(x)
      assert_complex_close(native, eager)
    end

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

    test "rfft via default_expr descent vs EMLX.Backend" do
      x = Nx.tensor([1.0, 1.0, 0.0, 0.0], type: :f32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&rfft_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&rfft_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_complex_close(native, eager)
    end
  end

  describe "iota" do
    test "iota flat: IR has :iota instruction, no operands" do
      prog = Expr.lower(Nx.Defn.debug_expr_apply(&iota_flat_defn/0, []))

      assert Enum.any?(prog.instructions, fn {_, op, operands, _} ->
               op == :iota and operands == []
             end)
    end

    test "iota flat lowering: iattrs encode shape and axis=-1" do
      prog = Expr.lower(Nx.Defn.debug_expr_apply(&iota_flat_defn/0, []))

      {_, :iota, [], [_dtype, n_dims, axis_int | shape]} =
        Enum.find(prog.instructions, fn {_, op, _, _} -> op == :iota end)

      assert n_dims == 2
      assert axis_int == -1
      assert shape == [3, 4]
    end

    test "iota flat: C++ replay (to_native) matches Nx.iota" do
      prog = Expr.lower(Nx.Defn.debug_expr_apply(&iota_flat_defn/0, []))
      [nif_out] = run_nif(prog, [])
      expected = Nx.iota({3, 4}, backend: EMLX.Backend)
      assert_close(nif_out, expected)
    end

    test "iota flat: E2E jit vs Nx.Defn.Evaluator" do
      native = Nx.Defn.jit(&iota_flat_defn/0, compiler: EMLX).()
      eager = Nx.Defn.jit(&iota_flat_defn/0, compiler: Nx.Defn.Evaluator).()
      assert_close(native, eager)
    end

    test "iota with axis=1: E2E jit vs Nx.Defn.Evaluator" do
      native = Nx.Defn.jit(&iota_axis1_defn/0, compiler: EMLX).()
      eager = Nx.Defn.jit(&iota_axis1_defn/0, compiler: Nx.Defn.Evaluator).()
      assert_close(native, eager)
    end

    test "iota f32 flat: E2E jit vs Nx.Defn.Evaluator" do
      native = Nx.Defn.jit(&iota_f32_defn/0, compiler: EMLX).()
      eager = Nx.Defn.jit(&iota_f32_defn/0, compiler: Nx.Defn.Evaluator).()
      assert_close(native, eager)
    end
  end

  describe "eye" do
    test "eye: IR has :eye instruction, no operands" do
      prog = Expr.lower(Nx.Defn.debug_expr_apply(&eye_3x3_defn/0, []))

      assert Enum.any?(prog.instructions, fn {_, op, operands, _} ->
               op == :eye and operands == []
             end)
    end

    test "eye 3x3: iattrs encode [dtype, 3, 3]" do
      prog = Expr.lower(Nx.Defn.debug_expr_apply(&eye_3x3_defn/0, []))

      {_, :eye, [], [_dtype, m, n]} =
        Enum.find(prog.instructions, fn {_, op, _, _} -> op == :eye end)

      assert m == 3
      assert n == 3
    end

    test "eye 3x3: C++ replay (to_native) matches Nx.eye" do
      prog = Expr.lower(Nx.Defn.debug_expr_apply(&eye_3x3_defn/0, []))
      [nif_out] = run_nif(prog, [])
      expected = Nx.eye({3, 3}, backend: EMLX.Backend)
      assert_close(nif_out, expected)
    end

    test "eye 3x3: E2E jit vs Nx.Defn.Evaluator" do
      native = Nx.Defn.jit(&eye_3x3_defn/0, compiler: EMLX).()
      eager = Nx.Defn.jit(&eye_3x3_defn/0, compiler: Nx.Defn.Evaluator).()
      assert_close(native, eager)
    end

    test "eye 2x4 rectangular: E2E jit vs Nx.Defn.Evaluator" do
      native = Nx.Defn.jit(&eye_2x4_defn/0, compiler: EMLX).()
      eager = Nx.Defn.jit(&eye_2x4_defn/0, compiler: Nx.Defn.Evaluator).()
      assert_close(native, eager)
    end
  end

  describe "Nx.Random" do
    test "Nx.Random.uniform: native matches evaluator for fixed key" do
      key = Nx.Random.key(42) |> Nx.backend_transfer(EMLX.Backend)

      native = Nx.Defn.jit(&rng_uniform_defn/1, compiler: EMLX).(key)
      eager = Nx.Defn.jit(&rng_uniform_defn/1, compiler: Nx.Defn.Evaluator).(key)

      assert_close(native, eager, 1.0e-5)
      # Samples should be in [0, 1)
      assert Nx.all(Nx.greater_equal(native, 0.0)) |> Nx.to_number() == 1
      assert Nx.all(Nx.less(native, 1.0)) |> Nx.to_number() == 1
    end

    test "Nx.Random.uniform: same key → same samples (deterministic)" do
      key = Nx.Random.key(99) |> Nx.backend_transfer(EMLX.Backend)

      out1 = Nx.Defn.jit(&rng_uniform_defn/1, compiler: EMLX).(key)
      out2 = Nx.Defn.jit(&rng_uniform_defn/1, compiler: EMLX).(key)

      assert_close(out1, out2)
    end

    test "Nx.Random.normal: native matches evaluator for fixed key" do
      key = Nx.Random.key(7) |> Nx.backend_transfer(EMLX.Backend)

      native = Nx.Defn.jit(&rng_normal_defn/1, compiler: EMLX).(key)
      eager = Nx.Defn.jit(&rng_normal_defn/1, compiler: Nx.Defn.Evaluator).(key)

      assert_close(native, eager, 1.0e-4)
    end
  end

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

  defn while_counter_only_cond(x, i, n) do
    while {x, i, n}, Nx.less(i, n) do
      {Nx.add(x, 1), Nx.add(i, 1), n}
    end
  end

  # A) A `while` followed by a deeply-shared post-DAG. `add(acc, acc)` references
  # `acc` twice, so after N levels the graph is N nodes but 2^N tree paths. The
  # split's rewrite pass must be id-memoized or this never terminates.
  deftransformp deep_shared(v) do
    Enum.reduce(1..28, v, fn _, acc -> Nx.add(acc, acc) end)
  end

  defn while_then_shared(x) do
    {acc, _} =
      while {a = x, k = x}, Nx.less(Nx.sum(a), 5.0) do
        {Nx.add(a, k), k}
      end

    deep_shared(acc)
  end

  # B) A `while` whose initial carry is computed through a runtime_call
  # (EMLX.Fast.rms_norm packs its operands in a `{x, weight}` tuple). The before
  # stage must collect the runtime_call's operand parameters, or it under-counts
  # its args and the remapped param indices overflow the stage input list.
  defn while_after_runtime_call(x, w, k) do
    s = Nx.add(EMLX.Fast.rms_norm(x, w, 1.0e-6), k)

    {acc, _} =
      while {a = s, kk = k}, Nx.less(Nx.sum(a), 10.0) do
        {Nx.add(a, kk), kk}
      end

    acc
  end

  # C) A while-chain whose output is a MAP container; Graph.run/3 must return the
  # non-tuple container from the final stage instead of trying to Tuple.to_list it.
  defn while_then_map(x) do
    {acc, _} =
      while {a = x, k = x}, Nx.less(Nx.sum(a), 10.0) do
        {Nx.add(a, k), k}
      end

    %{value: Nx.add(acc, 1.0), doubled: Nx.multiply(acc, 2.0)}
  end

  describe "cond" do
    test "cond two-branch: native matches evaluator (negative input)" do
      x = Nx.tensor(-3.0, backend: EMLX.Backend)
      native = Nx.Defn.jit(&cond_two_branch/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&cond_two_branch/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    test "cond two-branch: native matches evaluator (positive input)" do
      x = Nx.tensor(3.0, backend: EMLX.Backend)
      native = Nx.Defn.jit(&cond_two_branch/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&cond_two_branch/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    test "cond three-branch: all three branches produce correct results" do
      for {input, _branch} <- [{-2.0, :first}, {0.0, :second}, {5.0, :third}] do
        x = Nx.tensor(input, backend: EMLX.Backend)
        native = Nx.Defn.jit(&cond_three_branch/1, compiler: EMLX).(x)
        eager = Nx.Defn.jit(&cond_three_branch/1, compiler: Nx.Defn.Evaluator).(x)
        assert_close(native, eager)
      end
    end
  end

  describe "while" do
    test "while: count from 0 to 10" do
      x = Nx.tensor(0, type: :s32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&count_to_10/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&count_to_10/1, compiler: Nx.Defn.Evaluator).(x)
      assert Nx.to_number(native) == Nx.to_number(eager)
    end

    test "while: trip count depends on runtime value (count from 5)" do
      x = Nx.tensor(5, type: :s32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&count_to_10/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&count_to_10/1, compiler: Nx.Defn.Evaluator).(x)
      assert Nx.to_number(native) == Nx.to_number(eager)
    end

    test "while: already past condition — zero iterations" do
      x = Nx.tensor(15, type: :s32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&count_to_10/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&count_to_10/1, compiler: Nx.Defn.Evaluator).(x)
      assert Nx.to_number(native) == Nx.to_number(eager)
    end

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

    test "while: counter-only cond ignores a scalar carry slot" do
      x = Nx.tensor(0, type: :s32, backend: EMLX.Backend)
      i0 = Nx.tensor(0, type: :s32, backend: EMLX.Backend)
      n = Nx.tensor(5, type: :s32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&while_counter_only_cond/3, compiler: EMLX).(x, i0, n)
      eager = Nx.Defn.jit(&while_counter_only_cond/3, compiler: Nx.Defn.Evaluator).(x, i0, n)
      {nx, ni, _} = native
      {ex, ei, _} = eager
      assert Nx.to_number(nx) == Nx.to_number(ex)
      assert Nx.to_number(ni) == Nx.to_number(ei)
      assert Nx.to_number(ni) == 5
    end

    test "while: counter-only cond ignores a non-scalar carry slot" do
      x = Nx.tensor([1.0, 2.0, 3.0], type: :f32, backend: EMLX.Backend)
      i0 = Nx.tensor(0, type: :s32, backend: EMLX.Backend)
      n = Nx.tensor(4, type: :s32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&while_counter_only_cond/3, compiler: EMLX).(x, i0, n)
      eager = Nx.Defn.jit(&while_counter_only_cond/3, compiler: Nx.Defn.Evaluator).(x, i0, n)
      {nx, ni, _} = native
      {ex, ei, _} = eager
      assert_close(nx, ex)
      assert Nx.to_number(ni) == Nx.to_number(ei)
      assert Nx.to_number(ni) == 4
    end

    # Pins that a plain dynamic `while` actually takes the native lowering
    # path (a single `:while` wire instruction, no raise) rather than merely
    # "happening to still produce a correct result via the old host-driven
    # fallback" -- the two are indistinguishable from a native-vs-eager
    # equivalence assertion alone, so this checks the lowered wire program
    # directly. See EMLX.Native.Expr's `:while` moduledoc section /
    # `native_while_eligible?/2`.
    test "while: dynamic trip count lowers to a single native :while instruction" do
      expr = Nx.Defn.debug_expr_apply(&count_to_10/1, [Nx.template({}, :s32)])
      wire = expr |> Expr.lower() |> Expr.to_native()

      assert [%EMLX.Native.Instruction{op: :while, subprograms: [cond_sub, body_sub]}] =
               wire.instructions

      assert %EMLX.Native.SubProgram{outputs: [{:result, _}]} = cond_sub
      assert %EMLX.Native.SubProgram{outputs: [{:result, _}]} = body_sub
    end

    test "native_while_eligible?/2: plain condition/body (no runtime_call, hook, or nesting)" do
      expr = Nx.Defn.debug_expr_apply(&count_to_10/1, [Nx.template({}, :s32)])

      while_node =
        expr
        |> EMLX.Defn.Tree.post_order(&Expr.scope_dependencies/1)
        |> Enum.find(&(&1.data.op == :while))

      [_initial, _arg, condition, body] = while_node.data.args
      assert Expr.native_while_eligible?(condition, body)
    end

    test "native_while_eligible?/2: false when the body contains a runtime_call" do
      weight = Nx.iota({2, 64}, type: :f32) |> Nx.divide(10) |> Nx.backend_transfer(EMLX.Backend)
      qw = EMLX.quantize(weight, [])
      x0 = Nx.broadcast(Nx.tensor(0.0, type: :f32, backend: EMLX.Backend), {2, 64})
      n = Nx.tensor(3, type: :s32, backend: EMLX.Backend)

      expr = Nx.Defn.debug_expr_apply(&runtime_call_inside_while/3, [x0, n, qw])

      while_node =
        expr
        |> EMLX.Defn.Tree.post_order(&Expr.scope_dependencies/1)
        |> Enum.find(&(&1.data.op == :while))

      [_initial, _arg, condition, body] = while_node.data.args
      refute Expr.native_while_eligible?(condition, body)
    end

    test "native_while_eligible?/2: false when the body contains a hook" do
      expr = Nx.Defn.debug_expr_apply(&hook_in_while_body/1, [Nx.template({}, :s32)])

      while_node =
        expr
        |> EMLX.Defn.Tree.post_order(&Expr.scope_dependencies/1)
        |> Enum.find(&(&1.data.op == :while))

      [_initial, _arg, condition, body] = while_node.data.args
      refute Expr.native_while_eligible?(condition, body)
    end

    test "native_while_eligible?/2: true when the body contains a nested while" do
      expr = Nx.Defn.debug_expr_apply(&nested_while_top_level/1, [Nx.template({3}, :f32)])

      while_node =
        expr
        |> EMLX.Defn.Tree.post_order(&Expr.scope_dependencies/1)
        |> Enum.find(&(&1.data.op == :while))

      [_initial, _arg, condition, body] = while_node.data.args
      assert Expr.native_while_eligible?(condition, body)
    end

    # Pins that a nested `while` lowers to a genuinely nested wire `:while`
    # instruction (a `:while` `Instruction` inside another `:while`'s body
    # `SubProgram`), not merely "correct end-to-end result via some fallback"
    # — see `EMLX.Native.Expr`'s `:while` moduledoc section for why this is
    # safe (validated 2- and 3-level nesting against the checkpoint (a) spike).
    test "while: nested dynamic while lowers to a nested native :while instruction" do
      expr = Nx.Defn.debug_expr_apply(&nested_while_top_level/1, [Nx.template({3}, :f32)])
      wire = expr |> Expr.lower() |> Expr.to_native()

      assert [%EMLX.Native.Instruction{op: :while, subprograms: [_outer_cond, outer_body]}] =
               wire.instructions

      assert %EMLX.Native.SubProgram{instructions: outer_body_instructions} = outer_body

      assert Enum.any?(
               outer_body_instructions,
               &match?(%EMLX.Native.Instruction{op: :while, subprograms: [_, _]}, &1)
             )
    end

    test "while: nested dynamic while matches eager evaluation" do
      x = Nx.tensor([1.0, 2.0, 3.0], type: :f32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&nested_while_top_level/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&nested_while_top_level/1, compiler: Nx.Defn.Evaluator).(x)

      assert_close(native, eager)
    end

    # Two independent top-level native `:while`s (not nested inside each
    # other) each get their own `EMLXWhile` primitive instance sharing the
    # same read-only captures/constants tables (see `EMLXWhile`'s moduledoc
    # in emlx_compiler.cpp) -- pins that there's no cross-instance mutable
    # state that could leak between sibling loops.
    defn sibling_whiles(x) do
      {a, _} = while {a = x, i = 0}, Nx.less(i, 3), do: {a + 1.0, i + 1}
      {b, _} = while {b = x, j = 0}, Nx.less(j, 5), do: {b * 2.0, j + 1}
      {a, b}
    end

    test "while: two sibling top-level whiles don't share mutable state" do
      x = Nx.tensor([1.0, 2.0], type: :f32, backend: EMLX.Backend)

      {native_a, native_b} = Nx.Defn.jit(&sibling_whiles/1, compiler: EMLX).(x)
      {eager_a, eager_b} = Nx.Defn.jit(&sibling_whiles/1, compiler: Nx.Defn.Evaluator).(x)

      assert_close(native_a, eager_a)
      assert_close(native_b, eager_b)
    end

    # Defensive depth cap (`@max_while_nesting_depth` in `EMLX.Native.Expr`):
    # protects against unbounded native C++ call-stack recursion from
    # pathologically deep (e.g. generated) `while`-inside-`while` nesting --
    # not a realistic hand-written shape, so it's exercised via the
    # `deeply_nested_while_20`/`deeply_nested_while_over_cap` defns generated
    # near the top of this module (see `NestedWhileAst.build/2`).
    test "while: deeply nested (20-level) while still lowers and evaluates correctly" do
      x = Nx.tensor([0.0, 1.0], type: :f32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&deeply_nested_while_20/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&deeply_nested_while_20/1, compiler: Nx.Defn.Evaluator).(x)

      assert_close(native, eager)
    end

    test "while: nesting beyond the depth cap raises a clear error" do
      x = Nx.tensor([0.0, 1.0], type: :f32, backend: EMLX.Backend)

      assert_raise ArgumentError, ~r/nested.*64 levels deep/, fn ->
        Nx.Defn.jit(&deeply_nested_while_over_cap/1, compiler: EMLX).(x)
      end
    end
  end

  describe "splitter regressions" do
    # A) Without id-memoization in rewrite_subtree this hangs (2^28 tree walk),
    # so a generous-but-bounded timeout turns the hang into a test failure.
    @tag timeout: 30_000
    test "while + deeply-shared post-DAG compiles without exponential blowup" do
      x = Nx.tensor([1.0, 1.0], type: :f32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&while_then_shared/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&while_then_shared/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    # B) runtime_call (rms_norm) feeding a while carry: the before stage must
    # collect the runtime_call's tuple operands or param indices overflow.
    test "runtime_call operands feeding a while carry are fully collected" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0]], type: :f32, backend: EMLX.Backend)
      w = Nx.tensor([1.0, 1.0, 1.0, 1.0], type: :f32, backend: EMLX.Backend)
      k = Nx.tensor([[0.1, 0.1, 0.1, 0.1]], type: :f32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&while_after_runtime_call/3, compiler: EMLX).(x, w, k)
      eager = Nx.Defn.jit(&while_after_runtime_call/3, compiler: Nx.Defn.Evaluator).(x, w, k)
      assert_close(native, eager)
    end

    # C) Map container as the final stage output of a while-chain.
    test "while-chain returning a map container runs end-to-end" do
      x = Nx.tensor([1.0, 1.0], type: :f32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&while_then_map/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&while_then_map/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native.value, eager.value)
      assert_close(native.doubled, eager.doubled)
    end
  end

  defn cholesky_defn(x), do: Nx.LinAlg.cholesky(x)
  defn det_defn(x), do: Nx.LinAlg.determinant(x)

  defn all_close_defn(a, b), do: Nx.all_close(a, b)
  defn phase_defn(x), do: Nx.phase(x)
  defn top_k_defn(x), do: Nx.top_k(x, k: 3)

  # A small helper: a well-conditioned SPD matrix t·tᵀ + n·I.
  defp spd_matrix(n) do
    t = Nx.iota({n, n}, type: :f32, backend: EMLX.Backend) |> Nx.add(1.0)
    Nx.add(Nx.dot(t, Nx.transpose(t)), Nx.multiply(Nx.eye(n, backend: EMLX.Backend), n * 1.0))
  end

  describe "LinAlg.cholesky (native)" do
    test "cholesky of an SPD matrix vs EMLX.Backend — f32" do
      x = Nx.tensor([[4.0, 2.0], [2.0, 3.0]], type: :f32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&cholesky_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&cholesky_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    test "cholesky composed with surrounding ops (in-graph) vs EMLX.Backend" do
      f = fn t ->
        spd = Nx.add(Nx.dot(t, Nx.transpose(t)), Nx.multiply(Nx.eye(3), 1.0))
        Nx.LinAlg.cholesky(spd) |> Nx.add(1.0)
      end

      t = Nx.iota({3, 3}, type: :f32, backend: EMLX.Backend) |> Nx.add(1.0)
      native = Nx.Defn.jit(f, compiler: EMLX).(t)
      eager = Nx.Defn.jit(f, compiler: Nx.Defn.Evaluator).(t)
      assert_close(native, eager)
    end
  end

  describe "LinAlg.solve / triangular_solve (native)" do
    test "solve A x = b vs EMLX.Backend" do
      a = Nx.tensor([[3.0, 1.0], [1.0, 2.0]], type: :f32, backend: EMLX.Backend)
      b = Nx.tensor([9.0, 8.0], type: :f32, backend: EMLX.Backend)

      f = fn a, b -> Nx.LinAlg.solve(a, b) end
      native = Nx.Defn.jit(f, compiler: EMLX).(a, b)
      eager = Nx.Defn.jit(f, compiler: Nx.Defn.Evaluator).(a, b)
      assert_close(native, eager)
    end

    test "triangular_solve (lower, left side) vs EMLX.Backend" do
      a = Nx.tensor([[2.0, 0.0], [3.0, 1.0]], type: :f32, backend: EMLX.Backend)
      b = Nx.tensor([4.0, 5.0], type: :f32, backend: EMLX.Backend)

      f = fn a, b -> Nx.LinAlg.triangular_solve(a, b, lower: true) end
      native = Nx.Defn.jit(f, compiler: EMLX).(a, b)
      eager = Nx.Defn.jit(f, compiler: Nx.Defn.Evaluator).(a, b)
      assert_close(native, eager)
    end

    test "chained cholesky -> solve composes natively (contiguous across linalg ops)" do
      spd = Nx.tensor([[4.0, 2.0], [2.0, 3.0]], type: :f32, backend: EMLX.Backend)
      b = Nx.tensor([1.0, 2.0], type: :f32, backend: EMLX.Backend)

      f = fn s, b -> Nx.LinAlg.solve(Nx.LinAlg.cholesky(s), b) end
      native = Nx.Defn.jit(f, compiler: EMLX).(spd, b)
      eager = Nx.Defn.jit(f, compiler: Nx.Defn.Evaluator).(spd, b)
      assert_close(native, eager)
    end
  end

  describe "LinAlg batched" do
    test "batched cholesky (rank-3) vs EMLX.Backend" do
      a =
        Nx.tensor(
          [[[4.0, 2.0], [2.0, 3.0]], [[9.0, 3.0], [3.0, 5.0]]],
          type: :f32,
          backend: EMLX.Backend
        )

      native = Nx.Defn.jit(&Nx.LinAlg.cholesky/1, compiler: EMLX).(a)
      eager = Nx.Defn.jit(&Nx.LinAlg.cholesky/1, compiler: Nx.Defn.Evaluator).(a)
      assert_close(native, eager)
    end
  end

  describe "LinAlg.qr / eigh / svd (native, reconstruction)" do
    test "qr: Q·R reconstructs A and Q is orthonormal" do
      a =
        Nx.iota({3, 3}, type: :f32, backend: EMLX.Backend)
        |> Nx.add(Nx.eye(3, backend: EMLX.Backend))

      {q, r} = Nx.Defn.jit(&Nx.LinAlg.qr/1, compiler: EMLX).(a)
      assert_close(Nx.dot(q, r), a)
      assert_close(Nx.dot(Nx.transpose(q), q), Nx.eye(3, type: :f32, backend: EMLX.Backend))
    end

    test "eigh: V·diag(W)·Vᵀ reconstructs a symmetric A" do
      a = spd_matrix(3)
      n = 3

      {w, v} = Nx.Defn.jit(&Nx.LinAlg.eigh/1, compiler: EMLX).(a)
      recon = Nx.dot(Nx.multiply(v, Nx.reshape(w, {1, n})), Nx.transpose(v))
      assert_close(recon, a)
    end

    test "svd: U·diag(S)·Vᵀ reconstructs A" do
      a =
        Nx.iota({3, 3}, type: :f32, backend: EMLX.Backend)
        |> Nx.add(Nx.eye(3, backend: EMLX.Backend))

      n = 3

      {u, s, vt} = Nx.Defn.jit(&Nx.LinAlg.svd/1, compiler: EMLX).(a)
      recon = Nx.dot(Nx.multiply(u, Nx.reshape(s, {1, n})), vt)
      assert_close(recon, a)
    end
  end

  describe "LinAlg.lu (native)" do
    test "lu factors vs EMLX.Backend" do
      a = Nx.tensor([[4.0, 3.0], [6.0, 3.0]], type: :f32, backend: EMLX.Backend)

      {pn, ln, un} = Nx.Defn.jit(&Nx.LinAlg.lu/1, compiler: EMLX).(a)
      {pe, le, ue} = Nx.Defn.jit(&Nx.LinAlg.lu/1, compiler: Nx.Defn.Evaluator).(a)
      assert_close(pn, pe)
      assert_close(ln, le)
      assert_close(un, ue)
    end

    test "lu: P·L·U reconstructs A" do
      a = Nx.tensor([[4.0, 3.0], [6.0, 3.0]], type: :f32, backend: EMLX.Backend)

      {p, l, u} = Nx.Defn.jit(&Nx.LinAlg.lu/1, compiler: EMLX).(a)
      assert_close(Nx.dot(p, Nx.dot(l, u)), a)
    end
  end

  describe "LinAlg.determinant (default_expr descent)" do
    test "determinant 2x2 (pure primitives) vs EMLX.Backend" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f32, backend: EMLX.Backend)
      native = Nx.Defn.jit(&det_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&det_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    test "determinant 3x3 (pure primitives) vs EMLX.Backend" do
      x =
        Nx.tensor([[1.0, 2.0, 3.0], [1.0, -2.0, 3.0], [7.0, 8.0, 9.0]],
          type: :f32,
          backend: EMLX.Backend
        )

      native = Nx.Defn.jit(&det_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&det_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    test "determinant 3x3 sign (row-swap permutation) = -1" do
      # A single row swap permutation has det = -1; exercises the cofactor sign.
      x =
        Nx.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
          type: :f32,
          backend: EMLX.Backend
        )

      native = Nx.Defn.jit(&det_defn/1, compiler: EMLX).(x)
      assert_in_delta(Nx.to_number(native), -1.0, 1.0e-5)
    end

    test "determinant 4x4 (descends through native LU) vs BinaryBackend reference" do
      x = spd_matrix(4)
      native = Nx.Defn.jit(&det_defn/1, compiler: EMLX).(x)

      # EMLX.Backend's eager N>3 determinant has a pre-existing type bug, so
      # compute the reference entirely on the BinaryBackend (Nx.LinAlg.determinant
      # is a defn whose intermediate ops otherwise use the global default backend).
      prev = Nx.default_backend(Nx.BinaryBackend)

      ref =
        try do
          x |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.LinAlg.determinant()
        after
          Nx.default_backend(prev)
        end

      # f32 LU accumulates rounding error proportional to the magnitude of
      # the determinant (~2e5 here), so an absolute delta of 1.0 is too
      # tight and flakes under normal f32 rounding — use a relative delta.
      ref_num = Nx.to_number(ref)
      assert_in_delta(Nx.to_number(native), ref_num, abs(ref_num) * 1.0e-4)
    end
  end

  describe "block-descent completeness (AllClose/Phase/TopK)" do
    test "all_close: close tensors vs EMLX.Backend" do
      a = Nx.tensor([1.0, 2.0, 3.0], type: :f32, backend: EMLX.Backend)
      b = Nx.tensor([1.0, 2.0, 3.0 + 1.0e-7], type: :f32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&all_close_defn/2, compiler: EMLX).(a, b)
      eager = Nx.Defn.jit(&all_close_defn/2, compiler: Nx.Defn.Evaluator).(a, b)
      assert_close(native, eager)
      assert Nx.to_number(native) == 1
    end

    test "all_close: not-close tensors vs EMLX.Backend" do
      a = Nx.tensor([1.0, 2.0, 3.0], type: :f32, backend: EMLX.Backend)
      b = Nx.tensor([1.0, 2.0, 4.0], type: :f32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&all_close_defn/2, compiler: EMLX).(a, b)
      eager = Nx.Defn.jit(&all_close_defn/2, compiler: Nx.Defn.Evaluator).(a, b)
      assert_close(native, eager)
      assert Nx.to_number(native) == 0
    end

    test "phase: complex tensor vs EMLX.Backend" do
      x =
        Nx.complex(
          Nx.tensor([1.0, -2.0, 0.0], type: :f32, backend: EMLX.Backend),
          Nx.tensor([2.0, 1.0, -3.0], type: :f32, backend: EMLX.Backend)
        )

      native = Nx.Defn.jit(&phase_defn/1, compiler: EMLX).(x)
      eager = Nx.Defn.jit(&phase_defn/1, compiler: Nx.Defn.Evaluator).(x)
      assert_close(native, eager)
    end

    test "top_k (tuple-output block): values + indices vs EMLX.Backend" do
      x = Nx.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0], type: :f32, backend: EMLX.Backend)

      {native_values, native_indices} = Nx.Defn.jit(&top_k_defn/1, compiler: EMLX).(x)
      {eager_values, eager_indices} = Nx.Defn.jit(&top_k_defn/1, compiler: Nx.Defn.Evaluator).(x)

      assert_close(native_values, eager_values)
      assert Nx.to_flat_list(native_indices) == Nx.to_flat_list(eager_indices)
    end

    test "top_k on a batched input (rank 2) vs EMLX.Backend" do
      x =
        Nx.tensor([[3.0, 1.0, 4.0, 1.0, 5.0], [9.0, 2.0, 6.0, 5.0, 3.0]],
          type: :f32,
          backend: EMLX.Backend
        )

      {native_values, native_indices} = Nx.Defn.jit(&top_k_defn/1, compiler: EMLX).(x)
      {eager_values, eager_indices} = Nx.Defn.jit(&top_k_defn/1, compiler: Nx.Defn.Evaluator).(x)

      assert_close(native_values, eager_values)
      assert Nx.to_flat_list(native_indices) == Nx.to_flat_list(eager_indices)
    end
  end

  # Routing/IR-shape assertions are pure Elixir (no NIF) — run without GPU.
  describe "fused-kernel recognition (lowering)" do
    test "rms_norm runtime_call lowers to a single :fast_rms_norm instruction" do
      fun = fn x, w -> EMLX.Fast.rms_norm(x, w, 1.0e-5) end
      expr = Nx.Defn.debug_expr_apply(fun, [Nx.template({2, 16}, :f32), Nx.template({16}, :f32)])
      prog = Expr.lower(expr)

      assert [{_, :fast_rms_norm, [_x, _w], [eps_bits]}] = prog.instructions
      assert_in_delta(Expr.bits_to_f64(eps_bits), 1.0e-5, 1.0e-12)
    end

    test "f64_bits/1 ↔ bits_to_f64/1 round-trips through the int64 attr channel" do
      for v <- [1.0e-6, 1.0e-5, 0.08838834764831845, 10_000.0, 1_000_000.0] do
        assert Expr.bits_to_f64(Expr.f64_bits(v)) == v
      end
    end

    test "swiglu / layer_norm / sdpa each lower to a single fused opcode" do
      cases = [
        {fn g, u -> EMLX.Fast.swiglu(g, u) end,
         [Nx.template({2, 8}, :f32), Nx.template({2, 8}, :f32)], :fast_swiglu},
        {fn x, w, b -> EMLX.Fast.layer_norm(x, w, b, 1.0e-5) end,
         [Nx.template({2, 16}, :f32), Nx.template({16}, :f32), Nx.template({16}, :f32)],
         :fast_layer_norm},
        {fn x, w -> EMLX.Fast.layer_norm(x, w, 1.0e-5) end,
         [Nx.template({2, 16}, :f32), Nx.template({16}, :f32)], :fast_layer_norm_no_bias},
        {fn q, k, v -> EMLX.Fast.scaled_dot_product_attention_causal(q, k, v, 0.125) end,
         [
           Nx.template({1, 2, 4, 8}, :f32),
           Nx.template({1, 2, 4, 8}, :f32),
           Nx.template({1, 2, 4, 8}, :f32)
         ], :fast_sdpa_causal},
        {fn q, k, v, s -> EMLX.Fast.scaled_dot_product_attention(q, k, v, 0.125, sinks: s) end,
         [
           Nx.template({1, 2, 4, 8}, :f32),
           Nx.template({1, 2, 4, 8}, :f32),
           Nx.template({1, 2, 4, 8}, :f32),
           Nx.template({2}, :f32)
         ], :fast_sdpa_sinks},
        {fn q, k, v, s ->
           EMLX.Fast.scaled_dot_product_attention_causal(q, k, v, 0.125, sinks: s)
         end,
         [
           Nx.template({1, 2, 4, 8}, :f32),
           Nx.template({1, 2, 4, 8}, :f32),
           Nx.template({1, 2, 4, 8}, :f32),
           Nx.template({2}, :f32)
         ], :fast_sdpa_causal_sinks}
      ]

      for {fun, templates, opcode} <- cases do
        prog = Expr.lower(Nx.Defn.debug_expr_apply(fun, templates))
        assert [{_, ^opcode, _operands, _attrs}] = prog.instructions
      end
    end

    test "prefill RoPE with positions (T>1) lowers to a single :fast_rope_positions instruction" do
      fun = fn a, pos -> EMLX.Fast.rope_with_positions(a, pos, 64, false, 10_000.0, 1.0) end

      expr =
        Nx.Defn.debug_expr_apply(fun, [
          Nx.template({2, 4, 2, 64}, :f32),
          Nx.template({2, 4}, :s32)
        ])

      prog = Expr.lower(expr)
      assert [{_, :fast_rope_positions, [_a, _pos], _attrs}] = prog.instructions
    end

    test "prefill RoPE with freqs (T>1) lowers to a single :fast_rope_with_freqs_positions instruction" do
      fun = fn a, pos, freqs -> EMLX.Fast.rope_with_freqs(a, pos, 64, false, 1.0, freqs) end

      expr =
        Nx.Defn.debug_expr_apply(fun, [
          Nx.template({2, 4, 2, 64}, :f32),
          Nx.template({2, 4}, :s32),
          Nx.template({32}, :f32)
        ])

      prog = Expr.lower(expr)

      assert [{_, :fast_rope_with_freqs_positions, [_a, _pos, _freqs], _attrs}] =
               prog.instructions
    end
  end

  describe "fused kernels vs eager + primitive (Metal)" do
    @describetag :metal

    test "rms_norm: fused replay matches eager and hand-written primitive" do
      fun = fn x, w -> EMLX.Fast.rms_norm(x, w, 1.0e-5) end
      x = Nx.iota({2, 4, 16}, type: :f32) |> Nx.divide(50) |> gpu_t()
      w = Nx.broadcast(Nx.tensor(1.0, type: :f32), {16}) |> gpu_t()

      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(x, w)
      eager = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(x, w)
      assert_all_close(native, eager, tol: 1.0e-3)

      prim_fun = fn x, w ->
        rms = Nx.sqrt(Nx.add(Nx.mean(Nx.pow(x, 2), axes: [-1], keep_axes: true), 1.0e-5))
        Nx.divide(x, rms) |> Nx.multiply(w)
      end

      prim = Nx.Defn.jit(prim_fun, compiler: EMLX, device: :gpu).(x, w)
      assert_all_close(native, prim, tol: 1.0e-3)
    end

    test "layer_norm (with bias) and no-bias: fused vs eager and primitive" do
      x = Nx.iota({2, 16}, type: :f32) |> Nx.divide(30) |> gpu_t()
      w = Nx.broadcast(Nx.tensor(1.2, type: :f32), {16}) |> gpu_t()
      b = Nx.broadcast(Nx.tensor(0.3, type: :f32), {16}) |> gpu_t()

      with_bias = fn x, w, b -> EMLX.Fast.layer_norm(x, w, b, 1.0e-5) end
      native = Nx.Defn.jit(with_bias, compiler: EMLX, device: :gpu).(x, w, b)
      eager = Nx.Defn.jit(with_bias, compiler: Nx.Defn.Evaluator).(x, w, b)
      assert_all_close(native, eager, tol: 1.0e-3)

      ln_prim = fn x, w, b ->
        mean = Nx.mean(x, axes: [-1], keep_axes: true)
        var = Nx.mean(Nx.pow(Nx.subtract(x, mean), 2), axes: [-1], keep_axes: true)
        normed = Nx.divide(Nx.subtract(x, mean), Nx.sqrt(Nx.add(var, 1.0e-5)))
        Nx.add(Nx.multiply(normed, w), b)
      end

      prim = Nx.Defn.jit(ln_prim, compiler: EMLX, device: :gpu).(x, w, b)
      assert_all_close(native, prim, tol: 1.0e-3)

      no_bias = fn x, w -> EMLX.Fast.layer_norm(x, w, 1.0e-5) end
      nb_native = Nx.Defn.jit(no_bias, compiler: EMLX, device: :gpu).(x, w)
      nb_eager = Nx.Defn.jit(no_bias, compiler: Nx.Defn.Evaluator).(x, w)
      assert_all_close(nb_native, nb_eager, tol: 1.0e-3)
    end

    test "swiglu: fused replay matches hand-written silu(gate)*up" do
      fun = fn g, u -> EMLX.Fast.swiglu(g, u) end
      gate = Nx.iota({2, 8}, type: :f32) |> Nx.divide(10) |> gpu_t()
      up = Nx.iota({2, 8}, type: :f32) |> Nx.divide(7) |> gpu_t()

      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(gate, up)
      prim = Nx.multiply(Nx.multiply(gate, Nx.sigmoid(gate)), up)
      assert_all_close(native, prim, tol: 1.0e-4)
    end

    test "sdpa (no mask): fused replay matches eager and softmax(QKᵀ)·V primitive" do
      scale = 0.125
      fun = fn q, k, v -> EMLX.Fast.scaled_dot_product_attention(q, k, v, scale) end
      q = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(100) |> gpu_t()
      k = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(90) |> gpu_t()
      v = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(80) |> gpu_t()

      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(q, k, v)
      eager = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(q, k, v)
      assert_all_close(native, eager, tol: 1.0e-3)

      prim_fun = fn q, k, v ->
        scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.multiply(scale)
        Nx.dot(Nx.exp(scores) |> normalize_rows(), [3], [0, 1], v, [2], [0, 1])
      end

      prim = Nx.Defn.jit(prim_fun, compiler: EMLX, device: :gpu).(q, k, v)
      assert_all_close(native, prim, tol: 1.0e-3)
    end

    test "sdpa (causal): fused replay matches eager and masked-softmax primitive" do
      scale = 0.125
      fun = fn q, k, v -> EMLX.Fast.scaled_dot_product_attention_causal(q, k, v, scale) end
      q = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(100) |> gpu_t()
      k = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(90) |> gpu_t()
      v = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(80) |> gpu_t()

      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(q, k, v)
      eager = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(q, k, v)
      assert_all_close(native, eager, tol: 1.0e-3)

      # Primitive causal attention (T_q == T_kv): row i attends keys 0..i.
      prim_fun = fn q, k, v ->
        scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.multiply(scale)
        rows = Nx.iota({4, 4}, axis: 0)
        cols = Nx.iota({4, 4}, axis: 1)
        bias = Nx.select(Nx.greater_equal(rows, cols), 0.0, -1.0e9) |> Nx.reshape({1, 1, 4, 4})
        attn = normalize_rows(Nx.exp(Nx.add(scores, bias)))
        Nx.dot(attn, [3], [0, 1], v, [2], [0, 1])
      end

      prim = Nx.Defn.jit(prim_fun, compiler: EMLX, device: :gpu).(q, k, v)
      assert_all_close(native, prim, tol: 1.0e-3)
    end

    test "sdpa (additive mask): fused replay matches eager and masked-softmax primitive" do
      scale = 0.125
      fun = fn q, k, v, m -> EMLX.Fast.scaled_dot_product_attention(q, k, v, scale, m) end
      q = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(100) |> gpu_t()
      k = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(90) |> gpu_t()
      v = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(80) |> gpu_t()
      # Mask out the last key position for every query (non-trivial additive mask).
      mask =
        Nx.tensor([[[[0.0, 0.0, 0.0, -1.0e9]]]], type: :f32)
        |> Nx.broadcast({1, 1, 4, 4})
        |> gpu_t()

      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(q, k, v, mask)
      eager = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(q, k, v, mask)
      assert_all_close(native, eager, tol: 1.0e-3)

      prim_fun = fn q, k, v, m ->
        scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.multiply(scale)
        attn = normalize_rows(Nx.exp(Nx.add(scores, m)))
        Nx.dot(attn, [3], [0, 1], v, [2], [0, 1])
      end

      prim = Nx.Defn.jit(prim_fun, compiler: EMLX, device: :gpu).(q, k, v, mask)
      assert_all_close(native, prim, tol: 1.0e-3)
    end

    test "sdpa (causal + key_mask, decode): fused replay matches eager (padded and all-present)" do
      fun = fn q, k, v, km ->
        EMLX.Fast.scaled_dot_product_attention_causal_key_masked(q, k, v, 0.125, km)
      end

      q = Nx.iota({2, 2, 1, 8}, type: :f32) |> Nx.divide(100) |> gpu_t()
      k = Nx.iota({2, 2, 4, 8}, type: :f32) |> Nx.divide(90) |> gpu_t()
      v = Nx.iota({2, 2, 4, 8}, type: :f32) |> Nx.divide(80) |> gpu_t()

      # Padded batch: the eager NIF builds an additive mask; so does the opcode.
      padded = Nx.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], type: :s32) |> gpu_t()
      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(q, k, v, padded)
      eager = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(q, k, v, padded)
      assert_all_close(native, eager, tol: 1.0e-3)

      # All keys present: the eager NIF host-fast-paths to pure "causal" (no mask
      # alloc) while the compiled opcode always builds the additive mask. They
      # must still agree — this exercises that branch divergence.
      all_present = Nx.broadcast(Nx.tensor(1, type: :s32), {2, 4}) |> gpu_t()
      native_ap = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(q, k, v, all_present)
      eager_ap = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(q, k, v, all_present)
      assert_all_close(native_ap, eager_ap, tol: 1.0e-3)
    end

    test "sdpa (no mask, + sinks): fused replay matches eager" do
      scale = 0.125
      fun = fn q, k, v, s -> EMLX.Fast.scaled_dot_product_attention(q, k, v, scale, sinks: s) end
      q = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(100) |> gpu_t()
      k = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(90) |> gpu_t()
      v = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(80) |> gpu_t()
      sinks = Nx.tensor([0.1, -0.2], type: :f32) |> gpu_t()

      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(q, k, v, sinks)
      eager = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(q, k, v, sinks)
      assert_all_close(native, eager, tol: 1.0e-3)
    end

    test "sdpa (additive mask, + sinks): fused replay matches eager" do
      scale = 0.125

      fun = fn q, k, v, m, s ->
        EMLX.Fast.scaled_dot_product_attention(q, k, v, scale, m, sinks: s)
      end

      q = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(100) |> gpu_t()
      k = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(90) |> gpu_t()
      v = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(80) |> gpu_t()

      mask =
        Nx.tensor([[[[0.0, 0.0, 0.0, -1.0e9]]]], type: :f32)
        |> Nx.broadcast({1, 1, 4, 4})
        |> gpu_t()

      sinks = Nx.tensor([0.1, -0.2], type: :f32) |> gpu_t()

      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(q, k, v, mask, sinks)
      eager = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(q, k, v, mask, sinks)
      assert_all_close(native, eager, tol: 1.0e-3)
    end

    test "sdpa (causal, + sinks): fused replay matches eager" do
      scale = 0.125

      fun = fn q, k, v, s ->
        EMLX.Fast.scaled_dot_product_attention_causal(q, k, v, scale, sinks: s)
      end

      q = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(100) |> gpu_t()
      k = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(90) |> gpu_t()
      v = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(80) |> gpu_t()
      sinks = Nx.tensor([0.1, -0.2], type: :f32) |> gpu_t()

      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(q, k, v, sinks)
      eager = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(q, k, v, sinks)
      assert_all_close(native, eager, tol: 1.0e-3)
    end

    test "sdpa (causal + key_mask, decode, + sinks): fused replay matches eager (padded and all-present)" do
      fun = fn q, k, v, km, s ->
        EMLX.Fast.scaled_dot_product_attention_causal_key_masked(q, k, v, 0.125, km, sinks: s)
      end

      q = Nx.iota({2, 2, 1, 8}, type: :f32) |> Nx.divide(100) |> gpu_t()
      k = Nx.iota({2, 2, 4, 8}, type: :f32) |> Nx.divide(90) |> gpu_t()
      v = Nx.iota({2, 2, 4, 8}, type: :f32) |> Nx.divide(80) |> gpu_t()
      sinks = Nx.tensor([0.1, -0.2], type: :f32) |> gpu_t()

      padded = Nx.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], type: :s32) |> gpu_t()
      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(q, k, v, padded, sinks)
      eager = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(q, k, v, padded, sinks)
      assert_all_close(native, eager, tol: 1.0e-3)

      all_present = Nx.broadcast(Nx.tensor(1, type: :s32), {2, 4}) |> gpu_t()
      native_ap = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(q, k, v, all_present, sinks)
      eager_ap = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(q, k, v, all_present, sinks)
      assert_all_close(native_ap, eager_ap, tol: 1.0e-3)
    end

    test "rope (scalar offset): fused replay matches eager" do
      fun = fn a -> EMLX.Fast.rope(a, 64, false, 10_000.0, 1.0, 3) end
      a = Nx.iota({1, 4, 1, 64}, type: :f32) |> Nx.divide(100) |> gpu_t()

      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(a)
      eager = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(a)
      assert_all_close(native, eager, tol: 1.0e-3)
    end

    test "rope_with_positions (decode T=1): fused replay matches eager" do
      fun = fn a, pos -> EMLX.Fast.rope_with_positions(a, pos, 64, false, 10_000.0, 1.0) end
      a = Nx.iota({2, 1, 2, 64}, type: :f32) |> Nx.divide(100) |> gpu_t()
      pos = Nx.tensor([[3], [5]], type: :s32) |> gpu_t()

      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(a, pos)
      eager = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(a, pos)
      assert_all_close(native, eager, tol: 1.0e-3)
    end

    test "rope_with_freqs (decode T=1): fused replay matches eager" do
      fun = fn a, pos, freqs -> EMLX.Fast.rope_with_freqs(a, pos, 64, false, 1.0, freqs) end
      a = Nx.iota({2, 1, 2, 64}, type: :f32) |> Nx.divide(100) |> gpu_t()
      pos = Nx.tensor([[3], [5]], type: :s32) |> gpu_t()
      freqs = Nx.iota({32}, type: :f32) |> Nx.add(1) |> Nx.divide(1000) |> gpu_t()

      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(a, pos, freqs)
      eager = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(a, pos, freqs)
      assert_all_close(native, eager, tol: 1.0e-3)
    end

    # https://github.com/elixir-nx/emlx/issues/121 — mlx::core::fast::rope's
    # head_seq_transpose stride-detection guard doesn't trigger for a
    # freshly-allocated, contiguous {B,T,H,D} tensor, so its row_contiguous
    # fallback used to rotate head h at angle `position + h` instead of
    # `position`, for every h > 0. "fused replay matches eager" above only
    # proves native and eager *agree*, which they trivially did even while
    # both called the same buggy primitive underneath — these tests instead
    # check each head against an independent single-head computation (every
    # head, computed alone, must use only its own position).
    test "rope_with_positions (decode T=1, H>1): every head matches its single-head computation" do
      dims = 64
      base = 10_000.0
      fun = fn a, pos -> EMLX.Fast.rope_with_positions(a, pos, dims, false, base, 1.0) end

      a = Nx.iota({1, 1, 3, dims}, type: :f32) |> Nx.divide(100) |> gpu_t()
      pos = Nx.tensor([[6]], type: :s32) |> gpu_t()

      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(a, pos)
      eager = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(a, pos)

      for head <- 0..2 do
        a_head = a[[.., .., head..head, ..]]
        alone = EMLX.Fast.rope_with_positions(a_head, pos, dims, false, base, 1.0)

        assert_all_close(native[[.., .., head..head, ..]], alone, tol: 1.0e-3)
        assert_all_close(eager[[.., .., head..head, ..]], alone, tol: 1.0e-3)
      end
    end

    test "rope_with_freqs (decode T=1, H>1): every head matches its single-head computation" do
      dims = 64
      half = div(dims, 2)
      fun = fn a, pos, freqs -> EMLX.Fast.rope_with_freqs(a, pos, dims, false, 1.0, freqs) end

      a = Nx.iota({1, 1, 3, dims}, type: :f32) |> Nx.divide(100) |> gpu_t()
      pos = Nx.tensor([[6]], type: :s32) |> gpu_t()
      freqs = Nx.iota({half}, type: :f32) |> Nx.add(1) |> Nx.divide(1000) |> gpu_t()

      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(a, pos, freqs)
      eager = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(a, pos, freqs)

      for head <- 0..2 do
        a_head = a[[.., .., head..head, ..]]
        alone = EMLX.Fast.rope_with_freqs(a_head, pos, dims, false, 1.0, freqs)

        assert_all_close(native[[.., .., head..head, ..]], alone, tol: 1.0e-3)
        assert_all_close(eager[[.., .., head..head, ..]], alone, tol: 1.0e-3)
      end
    end

    test "decode-shaped block: fused path improves over primitive replay" do
      # A small attention+norm decode step: RMSNorm → causal SDPA → RMSNorm.
      scale = 0.125

      fused = fn q, k, v, w ->
        a = EMLX.Fast.scaled_dot_product_attention_causal(q, k, v, scale)
        flat = Nx.reshape(a, {1, 16})
        EMLX.Fast.rms_norm(flat, w, 1.0e-5)
      end

      primitive = fn q, k, v, w ->
        scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.multiply(scale)
        a = Nx.dot(normalize_rows(Nx.exp(scores)), [3], [0, 1], v, [2], [0, 1])
        flat = Nx.reshape(a, {1, 16})
        rms = Nx.sqrt(Nx.add(Nx.mean(Nx.pow(flat, 2), axes: [-1], keep_axes: true), 1.0e-5))
        Nx.divide(flat, rms) |> Nx.multiply(w)
      end

      q = Nx.iota({1, 2, 1, 8}, type: :f32) |> Nx.divide(100) |> gpu_t()
      k = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(90) |> gpu_t()
      v = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(80) |> gpu_t()
      w = Nx.broadcast(Nx.tensor(1.0, type: :f32), {16}) |> gpu_t()

      fused_c = Nx.Defn.jit(fused, compiler: EMLX, device: :gpu)
      prim_c = Nx.Defn.jit(primitive, compiler: EMLX, device: :gpu)

      # Correctness: same result within fused-kernel tolerance.
      assert_all_close(fused_c.(q, k, v, w), prim_c.(q, k, v, w), tol: 1.0e-2)

      # Warm both compiled graphs, then time the replay-only hot path.
      for _ <- 1..5, do: fused_c.(q, k, v, w) |> Nx.backend_transfer()
      for _ <- 1..5, do: prim_c.(q, k, v, w) |> Nx.backend_transfer()

      fused_us = bench_us(200, fn -> fused_c.(q, k, v, w) |> Nx.backend_transfer() end)
      prim_us = bench_us(200, fn -> prim_c.(q, k, v, w) |> Nx.backend_transfer() end)

      assert fused_us <= prim_us * 1.1
    end
  end

  describe "prefill RoPE (Metal)" do
    @describetag :metal

    test "rope_with_positions (T>1, sequential positions): native matches eager" do
      fun = fn a, pos -> EMLX.Fast.rope_with_positions(a, pos, 64, false, 10_000.0, 1.0) end
      a = Nx.iota({2, 4, 2, 64}, type: :f32) |> Nx.divide(100) |> gpu_t()
      pos = Nx.tensor([[3, 4, 5, 6], [10, 11, 12, 13]], type: :s32) |> gpu_t()

      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(a, pos)
      eager = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(a, pos)
      assert_all_close(native, eager, tol: 1.0e-3)
    end

    test "rope_with_positions (T>1, left-padded/non-sequential positions): native matches eager" do
      fun = fn a, pos -> EMLX.Fast.rope_with_positions(a, pos, 64, false, 10_000.0, 1.0) end
      a = Nx.iota({2, 5, 2, 64}, type: :f32) |> Nx.divide(100) |> gpu_t()
      # Left-padded batch: row 0 has 2 pad tokens (position 0 repeated), row 1 has none.
      pos = Nx.tensor([[0, 0, 1, 2, 3], [8, 9, 10, 11, 12]], type: :s32) |> gpu_t()

      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(a, pos)
      eager = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(a, pos)
      assert_all_close(native, eager, tol: 1.0e-3)
    end

    test "rope_with_positions (T>1, dims < D): native matches eager on the pass-through tail" do
      fun = fn a, pos -> EMLX.Fast.rope_with_positions(a, pos, 32, false, 10_000.0, 1.0) end
      a = Nx.iota({2, 3, 2, 64}, type: :f32) |> Nx.divide(100) |> gpu_t()
      pos = Nx.tensor([[0, 1, 2], [4, 5, 6]], type: :s32) |> gpu_t()

      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(a, pos)
      eager = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(a, pos)
      assert_all_close(native, eager, tol: 1.0e-3)
    end

    # `mlx::fast::rope`'s freqs overload reciprocates internally (both the CPU
    # fallback and the Metal kernel compute `inv_freq = 1/freqs[i]` — see
    # mlx/fast.cpp's default_inv_freqs-vs-freqs branch and
    # backend/metal/kernels/rope.metal's `1.0 / freqs[...]`), so `freqs` here is
    # a *raw frequency* tensor, not an inv_freq tensor: realistic magnitudes
    # are ~1..base (reciprocal lands back in the usual ~1e-4..1 inv_freq
    # range). Using inv_freq-scale values directly as `freqs` (as one might
    # naively expect from the name) reciprocates into huge angles and makes
    # native-vs-eager float32 cos/sin agreement numerically meaningless — a
    # test-data pitfall, not a lowering bug.
    test "rope_with_freqs (T>1, H=1, sequential positions): native matches eager" do
      fun = fn a, pos, freqs -> EMLX.Fast.rope_with_freqs(a, pos, 64, false, 1.0, freqs) end
      a = Nx.iota({2, 4, 1, 64}, type: :f32) |> Nx.divide(100) |> gpu_t()
      pos = Nx.tensor([[3, 4, 5, 6], [10, 11, 12, 13]], type: :s32) |> gpu_t()
      freqs = Nx.pow(10_000.0, Nx.divide(Nx.iota({32}, type: :f32), 32)) |> gpu_t()

      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(a, pos, freqs)
      eager = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(a, pos, freqs)
      assert_all_close(native, eager, tol: 1.0e-3)
    end

    test "rope_with_freqs (T>1, H=2, sequential positions): native matches pure-Nx primitive" do
      fun = fn a, pos, freqs -> EMLX.Fast.rope_with_freqs(a, pos, 64, false, 1.0, freqs) end
      a = Nx.iota({2, 4, 2, 64}, type: :f32) |> Nx.divide(100) |> gpu_t()
      pos = Nx.tensor([[3, 4, 5, 6], [10, 11, 12, 13]], type: :s32) |> gpu_t()
      freqs = Nx.pow(10_000.0, Nx.divide(Nx.iota({32}, type: :f32), 32)) |> gpu_t()

      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(a, pos, freqs)
      prim_fun = fn a, pos, freqs -> rope_freqs_prim(a, pos, freqs, 64, 1.0) end
      prim = Nx.Defn.jit(prim_fun, compiler: EMLX, device: :gpu).(a, pos, freqs)
      assert_all_close(native, prim, tol: 1.0e-3)
    end

    test "rope_with_freqs (T>1, H=2, left-padded/non-sequential positions): native matches pure-Nx primitive" do
      fun = fn a, pos, freqs -> EMLX.Fast.rope_with_freqs(a, pos, 64, false, 1.0, freqs) end
      a = Nx.iota({2, 5, 2, 64}, type: :f32) |> Nx.divide(100) |> gpu_t()
      pos = Nx.tensor([[0, 0, 1, 2, 3], [8, 9, 10, 11, 12]], type: :s32) |> gpu_t()
      freqs = Nx.pow(10_000.0, Nx.divide(Nx.iota({32}, type: :f32), 32)) |> gpu_t()

      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(a, pos, freqs)
      prim_fun = fn a, pos, freqs -> rope_freqs_prim(a, pos, freqs, 64, 1.0) end
      prim = Nx.Defn.jit(prim_fun, compiler: EMLX, device: :gpu).(a, pos, freqs)
      assert_all_close(native, prim, tol: 1.0e-3)
    end

    test "rope_with_freqs (T>1, H=2, dims < D): native matches pure-Nx primitive on the pass-through tail" do
      fun = fn a, pos, freqs -> EMLX.Fast.rope_with_freqs(a, pos, 32, false, 1.0, freqs) end
      a = Nx.iota({2, 6, 2, 64}, type: :f32) |> Nx.divide(100) |> gpu_t()
      pos = Nx.tensor([[0, 0, 0, 1, 2, 3], [20, 21, 22, 23, 24, 25]], type: :s32) |> gpu_t()
      # dims=32 → freqs shape {dims/2} = {16}.
      freqs = Nx.pow(10_000.0, Nx.divide(Nx.iota({16}, type: :f32), 16)) |> gpu_t()

      native = Nx.Defn.jit(fun, compiler: EMLX, device: :gpu).(a, pos, freqs)
      prim_fun = fn a, pos, freqs -> rope_freqs_prim(a, pos, freqs, 32, 1.0) end
      prim = Nx.Defn.jit(prim_fun, compiler: EMLX, device: :gpu).(a, pos, freqs)
      assert_all_close(native, prim, tol: 1.0e-3)
    end
  end

  describe ":fun node unreachability (doc audit)" do
    test "custom-fun reduce: :fun never becomes an instruction" do
      templates = [Nx.template({5}, :f32)]

      expr =
        Nx.Defn.debug_expr_apply(
          fn t -> Nx.reduce(t, 0.0, fn a, b -> Nx.add(a, b) end) end,
          templates
        )

      prog = Expr.lower(expr)

      assert Enum.all?(prog.instructions, fn {_id, op, _operands, _attrs} -> op != :fun end)
    end

    test "custom-fun window_reduce: :fun never becomes an instruction" do
      templates = [Nx.template({6}, :f32)]

      expr =
        Nx.Defn.debug_expr_apply(
          fn t -> Nx.window_reduce(t, 0.0, {2}, fn a, b -> Nx.add(a, b) end) end,
          templates
        )

      prog = Expr.lower(expr)

      assert Enum.all?(prog.instructions, fn {_id, op, _operands, _attrs} -> op != :fun end)
    end
  end

  describe "while-in-default_expr descent (static unroll)" do
    # QR `mode: :complete` and SVD `full_matrices?: false` both fall through
    # to `expand_block_via_default`; QR's Householder decomposition carries a
    # statically-counted `while` (trip count fixed by the input shape at
    # trace time) that previously raised "does not yet lower op :while" and
    # fired the Evaluator fallback. `expand_node`'s new `:while` clause
    # detects that shape and unrolls it in place.
    test "qr :complete lowers natively — Q orthonormal, Q·R reconstructs A (square)" do
      a =
        Nx.iota({3, 3}, type: :f32, backend: EMLX.Backend)
        |> Nx.add(Nx.eye(3, backend: EMLX.Backend))

      {q, r} = Nx.Defn.jit(fn t -> Nx.LinAlg.qr(t, mode: :complete) end, compiler: EMLX).(a)

      assert q.shape == {3, 3}
      assert r.shape == {3, 3}
      assert_close(Nx.dot(q, r), a)
      assert_close(Nx.dot(Nx.transpose(q), q), Nx.eye(3, type: :f32, backend: EMLX.Backend))
    end

    test "qr :complete lowers natively — tall input, Q is m×m (not m×n)" do
      a = Nx.iota({4, 3}, type: :f32, backend: EMLX.Backend) |> Nx.add(1)

      {q, r} = Nx.Defn.jit(fn t -> Nx.LinAlg.qr(t, mode: :complete) end, compiler: EMLX).(a)

      assert q.shape == {4, 4}
      assert r.shape == {4, 3}
      assert_close(Nx.dot(q, r), a)
      assert_close(Nx.dot(Nx.transpose(q), q), Nx.eye(4, type: :f32, backend: EMLX.Backend))
    end

    test "qr :complete matches eager EMLX.Backend (Evaluator) on Q·R and orthonormality" do
      a = Nx.iota({5, 2}, type: :f32, backend: EMLX.Backend) |> Nx.add(1)

      {q_native, r_native} =
        Nx.Defn.jit(fn t -> Nx.LinAlg.qr(t, mode: :complete) end, compiler: EMLX).(a)

      {q_eager, r_eager} =
        Nx.Defn.jit(fn t -> Nx.LinAlg.qr(t, mode: :complete) end, compiler: Nx.Defn.Evaluator).(a)

      assert_close(Nx.dot(q_native, r_native), Nx.dot(q_eager, r_eager))

      assert_close(
        Nx.dot(Nx.transpose(q_native), q_native),
        Nx.eye(5, type: :f32, backend: EMLX.Backend)
      )
    end

    test "svd full_matrices?: false lowers natively — U·diag(S)·Vᵗ reconstructs A (tall)" do
      a = Nx.iota({4, 3}, type: :f32, backend: EMLX.Backend) |> Nx.add(1)

      {u, s, vt} =
        Nx.Defn.jit(fn t -> Nx.LinAlg.svd(t, full_matrices?: false) end, compiler: EMLX).(a)

      assert u.shape == {4, 3}
      assert s.shape == {3}
      assert vt.shape == {3, 3}
      recon = Nx.dot(Nx.multiply(u, Nx.reshape(s, {1, 3})), vt)
      assert_close(recon, a)
    end

    test "svd full_matrices?: false lowers natively — wide and square inputs" do
      wide = Nx.iota({3, 4}, type: :f32, backend: EMLX.Backend) |> Nx.add(1)

      {u, s, vt} =
        Nx.Defn.jit(fn t -> Nx.LinAlg.svd(t, full_matrices?: false) end, compiler: EMLX).(wide)

      assert u.shape == {3, 3}
      assert s.shape == {3}
      assert vt.shape == {3, 4}
      assert_close(Nx.dot(Nx.multiply(u, Nx.reshape(s, {1, 3})), vt), wide)

      square =
        Nx.iota({3, 3}, type: :f32, backend: EMLX.Backend)
        |> Nx.add(Nx.eye(3, backend: EMLX.Backend))

      {u2, s2, vt2} =
        Nx.Defn.jit(fn t -> Nx.LinAlg.svd(t, full_matrices?: false) end, compiler: EMLX).(square)

      assert_close(Nx.dot(Nx.multiply(u2, Nx.reshape(s2, {1, 3})), vt2), square)
    end

    test "svd full_matrices?: false matches eager EMLX.Backend singular values" do
      a = Nx.iota({4, 3}, type: :f32, backend: EMLX.Backend) |> Nx.add(1)

      {_u, s_native, _vt} =
        Nx.Defn.jit(fn t -> Nx.LinAlg.svd(t, full_matrices?: false) end, compiler: EMLX).(a)

      {_u, s_eager, _vt} =
        Nx.Defn.jit(fn t -> Nx.LinAlg.svd(t, full_matrices?: false) end,
          compiler: Nx.Defn.Evaluator
        ).(a)

      assert_close(s_native, s_eager)
    end

    test "qr :complete lowers with no :while instruction in the compiled program" do
      templates = [Nx.template({4, 3}, :f32)]

      expr =
        Nx.Defn.debug_expr_apply(
          fn t -> Nx.LinAlg.qr(t, mode: :complete) end,
          templates
        )

      prog = Expr.lower(expr, 1)

      assert Enum.all?(prog.instructions, fn {_id, op, _operands, _attrs} -> op != :while end)
    end

    test "triangular_solve left_side: false (2D b) vs EMLX.Backend" do
      a = Nx.tensor([[3.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 1.0, 1.0]], backend: EMLX.Backend)
      b = Nx.tensor([[1.0, 2.0, 3.0]], backend: EMLX.Backend)

      f = fn a, b -> Nx.LinAlg.triangular_solve(a, b, left_side: false) end
      native = Nx.Defn.jit(f, compiler: EMLX).(a, b)
      eager = Nx.Defn.jit(f, compiler: Nx.Defn.Evaluator).(a, b)
      assert_close(native, eager)
    end

    test "triangular_solve left_side: false (1D b) vs EMLX.Backend" do
      a = Nx.tensor([[3.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 1.0, 1.0]], backend: EMLX.Backend)
      b = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)

      f = fn a, b -> Nx.LinAlg.triangular_solve(a, b, left_side: false) end
      native = Nx.Defn.jit(f, compiler: EMLX).(a, b)
      eager = Nx.Defn.jit(f, compiler: Nx.Defn.Evaluator).(a, b)
      assert_close(native, eager)
    end

    test "triangular_solve transform_a: :transpose vs EMLX.Backend" do
      a = Nx.tensor([[3.0, 1.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], backend: EMLX.Backend)
      b = Nx.tensor([4.0, 5.0, 6.0], backend: EMLX.Backend)

      f = fn a, b -> Nx.LinAlg.triangular_solve(a, b, lower: false, transform_a: :transpose) end
      native = Nx.Defn.jit(f, compiler: EMLX).(a, b)
      eager = Nx.Defn.jit(f, compiler: Nx.Defn.Evaluator).(a, b)
      assert_close(native, eager)
    end

    test "triangular_solve left_side: false + transform_a: :transpose vs EMLX.Backend" do
      a = Nx.tensor([[3.0, 1.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], backend: EMLX.Backend)
      b = Nx.tensor([[4.0, 5.0, 6.0]], backend: EMLX.Backend)

      f = fn a, b ->
        Nx.LinAlg.triangular_solve(a, b, left_side: false, lower: false, transform_a: :transpose)
      end

      native = Nx.Defn.jit(f, compiler: EMLX).(a, b)
      eager = Nx.Defn.jit(f, compiler: Nx.Defn.Evaluator).(a, b)
      assert_close(native, eager)
    end
  end

  describe "hooks (token/attach_token, extra-output design)" do
    # `Nx.Defn.Kernel.hook/2,3` is fire-and-forget, not control flow (see
    # `EMLX.Native.Expr`'s moduledoc "Hooks" section): `attach_token`'s value
    # is its wrapped expr unchanged, so the hook is lowered in the *same*
    # single NIF-call program as everything else -- the hooked value(s) ride
    # along as extra outputs, and the callback fires host-side right after
    # `eval_program` returns. No `Nx.Defn.Graph.split` host round-trip needed.
    test "top-level hook fires once with the correct value; result unaffected" do
      a = Nx.tensor(1, backend: EMLX.Backend)
      b = Nx.tensor(2, backend: EMLX.Backend)

      result = Nx.Defn.jit(&hook_top_level/2, compiler: EMLX).(a, b)
      assert Nx.to_number(result) == 5

      assert_receive {:mid, hooked_value}
      assert Nx.to_number(hooked_value) == 6
      refute_receive {:mid, _}
    end

    test "a hook whose value is unreferenced by the output never fires (matches Evaluator's dead-code elimination)" do
      a = Nx.tensor(4, backend: EMLX.Backend)
      b = Nx.tensor(3, backend: EMLX.Backend)

      native = Nx.Defn.jit(&hook_unused_value/2, compiler: EMLX).(a, b)
      assert Nx.to_number(native) == 7
      refute_receive {:dbg, _}

      eager = Nx.Defn.jit(&hook_unused_value/2, compiler: Nx.Defn.Evaluator).(a, b)
      assert Nx.to_number(eager) == 7
      refute_receive {:dbg, _}
    end

    test "a name-only hook (no trace-time callback, no runtime override) is a silent no-op" do
      a = Nx.tensor(4, backend: EMLX.Backend)
      b = Nx.tensor(3, backend: EMLX.Backend)

      result = Nx.Defn.jit(&hook_name_only/2, compiler: EMLX).(a, b)
      assert Nx.to_number(result) == 7
    end

    test "a hook on a tuple payload fires with the matching container shape" do
      a = Nx.tensor(4, backend: EMLX.Backend)
      b = Nx.tensor(3, backend: EMLX.Backend)

      {s, d} = Nx.Defn.jit(&hook_tuple_payload/2, compiler: EMLX).(a, b)
      assert Nx.to_number(s) == 7
      assert Nx.to_number(d) == 1

      assert_receive {:pair, sum, diff}
      assert Nx.to_number(sum) == 7
      assert Nx.to_number(diff) == 1
    end

    test "a hook nested inside a cond branch raises instead of silently double-firing" do
      a = Nx.tensor(1, backend: EMLX.Backend)
      b = Nx.tensor(2, backend: EMLX.Backend)
      templates = [Nx.template(a.shape, a.type), Nx.template(b.shape, b.type)]

      expr = Nx.Defn.debug_expr_apply(&hook_in_cond_branch/2, templates)

      assert_raise ArgumentError, ~r/cannot lower a hook nested inside a cond branch/, fn ->
        Expr.lower(expr, 2)
      end
    end

    test "a hook inside a while body fires once per iteration, matching Evaluator" do
      a = Nx.tensor(3, backend: EMLX.Backend)

      native = Nx.Defn.jit(&hook_in_while_body/1, compiler: EMLX).(a)
      assert Nx.to_number(native) == 6
      assert_receive {:iter, v1}
      assert_receive {:iter, v2}
      assert_receive {:iter, v3}
      native_values = Enum.map([v1, v2, v3], &Nx.to_number/1)
      refute_receive {:iter, _}

      eager = Nx.Defn.jit(&hook_in_while_body/1, compiler: Nx.Defn.Evaluator).(a)
      assert Nx.to_number(eager) == 6
      assert_receive {:iter, e1}
      assert_receive {:iter, e2}
      assert_receive {:iter, e3}
      eager_values = Enum.map([e1, e2, e3], &Nx.to_number/1)
      refute_receive {:iter, _}

      assert native_values == eager_values
    end

    # Regression: hooks straddling a non-bare `while` (surrounding work on
    # both sides) route through `Nx.Defn.Graph.split`'s multi-stage chain
    # (`EMLX.build_while_chain_eval_fn`). This previously crashed the NIF
    # with a wire-arity mismatch ("vector in NIF.eval_program/2") because
    # `Nx.Defn.Graph`'s `do_rewrite_subtree/3` had no `:token` clause, so a
    # hook payload depending on a stage-boundary-hoisted value kept its
    # stale, pre-remap parameter position. Fixed upstream in the `nx` fork
    # (see Results); this test pins the fix from the EMLX side.
    test "a hook before AND after a while (Graph.split chain) matches Evaluator" do
      a = Nx.tensor(2, backend: EMLX.Backend)

      native = Nx.Defn.jit(&hook_around_while/1, compiler: EMLX).(a)
      assert Nx.to_number(native) == 11
      assert_receive {:seed, seed_v}
      assert_receive {:final, final_v}
      assert Nx.to_number(seed_v) == 4
      assert Nx.to_number(final_v) == 11

      eager = Nx.Defn.jit(&hook_around_while/1, compiler: Nx.Defn.Evaluator).(a)
      assert Nx.to_number(eager) == 11
      assert_receive {:seed, seed_v2}
      assert_receive {:final, final_v2}
      assert Nx.to_number(seed_v2) == 4
      assert Nx.to_number(final_v2) == 11
    end

    # Regression for a bug the reviewer subagent caught: a hook inside a
    # custom-fun `reduce` body was wrongly rejected by the cond-branch guard
    # (false positive -- `scope_ids/1` never walks into a `:fun` body, so the
    # hook's id looked "not top-scope" even though it isn't cond-branch-local
    # either), AND separately, `lower_fun_body/3` was dropping any hooks
    # registered while lowering the body (only `instructions`/`captures`/
    # `constants`/`inputs` were carried out of its local `state`, not `hooks`).
    test "a hook inside a custom-fun reduce body fires once per fold step, matching Evaluator" do
      t = Nx.tensor([1, 2, 3], backend: EMLX.Backend)

      native = Nx.Defn.jit(&hook_in_reduce_body/1, compiler: EMLX).(t)
      assert Nx.to_number(native) == 6
      assert_receive {:step, v1}
      assert_receive {:step, v2}
      assert_receive {:step, v3}
      native_values = Enum.map([v1, v2, v3], &Nx.to_number/1)
      refute_receive {:step, _}
      assert native_values == [1, 3, 6]

      bin_t = Nx.backend_copy(Nx.tensor([1, 2, 3]), Nx.BinaryBackend)
      prev = Nx.default_backend(Nx.BinaryBackend)

      eager =
        try do
          Nx.Defn.jit(&hook_in_reduce_body/1, compiler: Nx.Defn.Evaluator).(bin_t)
        after
          Nx.default_backend(prev)
        end

      assert Nx.to_number(eager) == 6
      assert_receive {:step, e1}
      assert_receive {:step, e2}
      assert_receive {:step, e3}
      eager_values = Enum.map([e1, e2, e3], &Nx.to_number/1)
      refute_receive {:step, _}

      assert native_values == eager_values
    end

    test "a hook inside a cond that's nested inside a reduce body still raises" do
      t = Nx.tensor(1, backend: EMLX.Backend)
      template = Nx.template({3}, t.type)

      expr = Nx.Defn.debug_expr_apply(&hook_in_cond_in_reduce_body/1, [template])

      assert_raise ArgumentError, ~r/cannot lower a hook nested inside a cond branch/, fn ->
        Expr.lower(expr, 1)
      end
    end
  end

  describe "quantized Nx.dot input (root-caused)" do
    test "the same defn runs correctly under Nx.Defn.Evaluator (not a model/graph bug)" do
      weight =
        Nx.iota({128, 64}, type: :f32) |> Nx.divide(100) |> Nx.backend_transfer(EMLX.Backend)

      qw = EMLX.quantize(weight, [])
      x = Nx.iota({4, 128}, type: :f32) |> Nx.backend_transfer(EMLX.Backend)

      result = Nx.Defn.jit(&Nx.dot/2, compiler: Nx.Defn.Evaluator).(x, qw)
      assert Nx.shape(result) == {4, 64}
    end
  end

  describe "full fix: quantized Nx.dot via call-time program specialization" do
    # See workdir/native-compiler/25-quantized-dot-full-fix.md. A quantized
    # right-operand `Nx.dot` now specializes to a `:quantized_matmul` opcode
    # once real (call-time) tensors reveal the quantization signature —
    # equivalence-tested against eager EMLX.Backend.dot/7 and
    # Nx.Defn.Evaluator, both used as independent references.
    test "a quantized weight bound to a native-compiled defn now runs end-to-end" do
      weight =
        Nx.iota({128, 64}, type: :f32) |> Nx.divide(100) |> Nx.backend_transfer(EMLX.Backend)

      qw = EMLX.quantize(weight, [])
      x = Nx.iota({4, 128}, type: :f32) |> Nx.divide(37) |> Nx.backend_transfer(EMLX.Backend)

      native = Nx.Defn.jit(&Nx.dot/2, compiler: EMLX).(x, qw)
      eager = EMLX.Backend.dot(Nx.template({4, 64}, native.type), x, [1], [], qw, [0], [])

      assert Nx.shape(native) == {4, 64}
      assert_all_close(native, eager)
    end

    test "repeated calls with the same quantized weight reuse the cached specialized program" do
      weight =
        Nx.iota({128, 64}, type: :f32) |> Nx.divide(100) |> Nx.backend_transfer(EMLX.Backend)

      qw = EMLX.quantize(weight, [])
      x1 = Nx.iota({4, 128}, type: :f32) |> Nx.divide(37) |> Nx.backend_transfer(EMLX.Backend)
      x2 = Nx.iota({4, 128}, type: :f32) |> Nx.divide(11) |> Nx.backend_transfer(EMLX.Backend)

      jitted = Nx.Defn.jit(&Nx.dot/2, compiler: EMLX)
      r1 = jitted.(x1, qw)
      r2 = jitted.(x2, qw)

      eager1 = EMLX.Backend.dot(Nx.template({4, 64}, r1.type), x1, [1], [], qw, [0], [])
      eager2 = EMLX.Backend.dot(Nx.template({4, 64}, r2.type), x2, [1], [], qw, [0], [])

      assert_all_close(r1, eager1)
      assert_all_close(r2, eager2)
    end

    test "a defn with two independently-quantized weights specializes both dots" do
      w1 =
        Nx.iota({128, 64}, type: :f32) |> Nx.divide(100) |> Nx.backend_transfer(EMLX.Backend)

      w2 =
        Nx.iota({128, 64}, type: :f32) |> Nx.divide(50) |> Nx.backend_transfer(EMLX.Backend)

      qw1 = EMLX.quantize(w1, [])
      qw2 = EMLX.quantize(w2, [])
      x = Nx.iota({4, 128}, type: :f32) |> Nx.divide(37) |> Nx.backend_transfer(EMLX.Backend)

      native = Nx.Defn.jit(&two_quantized_dots/3, compiler: EMLX).(x, qw1, qw2)

      eager1 = EMLX.Backend.dot(Nx.template({4, 64}, native.type), x, [1], [], qw1, [0], [])
      eager2 = EMLX.Backend.dot(Nx.template({4, 64}, native.type), x, [1], [], qw2, [0], [])
      expected = Nx.concatenate([eager1, eager2], axis: 1)

      assert_all_close(native, expected)
    end

    test "a microscaled-quantized weight (no biases) specializes correctly" do
      weight =
        Nx.iota({128, 64}, type: :f32) |> Nx.divide(100) |> Nx.backend_transfer(EMLX.Backend)

      qw = EMLX.quantize(weight, mode: "mxfp4", group_size: 32)
      refute qw.data.quantization_config.biases

      x = Nx.iota({4, 128}, type: :f32) |> Nx.divide(37) |> Nx.backend_transfer(EMLX.Backend)

      native = Nx.Defn.jit(&Nx.dot/2, compiler: EMLX).(x, qw)
      eager = EMLX.Backend.dot(Nx.template({4, 64}, native.type), x, [1], [], qw, [0], [])

      assert_all_close(native, eager)
    end

    test "a quantized weight threaded through unchanged (pass-through output) round-trips" do
      # Mirrors the shape a `while`-split stage boundary produces: a
      # loop-invariant quantized weight carried through without being
      # consumed by any op in *this* stage — its output leaf is a bare
      # :parameter node. See build_native_eval_fn/5's output_param_positions.
      weight =
        Nx.iota({128, 64}, type: :f32) |> Nx.divide(100) |> Nx.backend_transfer(EMLX.Backend)

      qw = EMLX.quantize(weight, [])
      x = Nx.iota({4, 128}, type: :f32) |> Nx.divide(37) |> Nx.backend_transfer(EMLX.Backend)

      passthrough_and_dot = fn x, qw -> {qw, Nx.dot(x, qw)} end

      {qw_out, native} = Nx.Defn.jit(passthrough_and_dot, compiler: EMLX).(x, qw)
      eager = EMLX.Backend.dot(Nx.template({4, 64}, native.type), x, [1], [], qw, [0], [])

      assert Nx.shape(qw_out) == Nx.shape(qw)
      assert qw_out.data.quantization_config
      assert_all_close(EMLX.dequantize(qw_out), EMLX.dequantize(qw))
      assert_all_close(native, eager)
    end

    test "a quantized left operand still raises a clear ArgumentError (matches EMLX.Backend.dot/7)" do
      weight =
        Nx.iota({128, 64}, type: :f32) |> Nx.divide(100) |> Nx.backend_transfer(EMLX.Backend)

      qw = EMLX.quantize(weight, [])
      y = Nx.iota({64, 4}, type: :f32) |> Nx.backend_transfer(EMLX.Backend)

      assert_raise ArgumentError,
                   ~r/does not yet lower op :dot with a quantized left operand/,
                   fn ->
                     Nx.Defn.jit(&Nx.dot/2, compiler: EMLX).(qw, y)
                   end
    end
  end

  describe "unrecognized :runtime_call as a graph-split point (like while)" do
    # See workdir/native-compiler/31-runtime-call-graph-split.md. An
    # unrecognized :runtime_call (e.g. EMLX.Quantization.dequantize's
    # callback, not an EMLX.Fast.* fused kernel) previously hard-raised
    # "does not yet lower op :runtime_call". It now becomes a
    # Nx.Defn.Graph.split point exactly like `while`: the callback runs once,
    # directly, with real materialised tensors, so ordinary surrounding
    # `compiler: EMLX` computation still compiles to flat native programs.
    test "a bare tail runtime_call (no surrounding work) runs directly" do
      weight =
        Nx.iota({4, 64}, type: :f32) |> Nx.divide(10) |> Nx.backend_transfer(EMLX.Backend)

      qw = EMLX.quantize(weight, [])

      native = Nx.Defn.jit(&dequant_only/1, compiler: EMLX).(qw)
      eager = EMLX.dequantize(qw)

      assert_all_close(native, eager)
    end

    test "a runtime_call surrounded by ordinary ops splits into a native/host/native chain" do
      weight =
        Nx.iota({4, 64}, type: :f32) |> Nx.divide(10) |> Nx.backend_transfer(EMLX.Backend)

      qw = EMLX.quantize(weight, [])
      x = Nx.iota({4, 64}, type: :f32) |> Nx.divide(3) |> Nx.backend_transfer(EMLX.Backend)

      native = Nx.Defn.jit(&dequant_surrounded/2, compiler: EMLX).(x, qw)
      eager = Nx.Defn.jit(&dequant_surrounded/2, compiler: Nx.Defn.Evaluator).(x, qw)

      assert_all_close(native, eager)
    end

    test "two independent runtime_calls in one defn both split correctly" do
      w1 = Nx.iota({4, 64}, type: :f32) |> Nx.divide(10) |> Nx.backend_transfer(EMLX.Backend)
      w2 = Nx.iota({4, 64}, type: :f32) |> Nx.divide(5) |> Nx.backend_transfer(EMLX.Backend)
      qw1 = EMLX.quantize(w1, [])
      qw2 = EMLX.quantize(w2, [])
      x = Nx.iota({4, 64}, type: :f32) |> Nx.divide(3) |> Nx.backend_transfer(EMLX.Backend)

      native = Nx.Defn.jit(&two_runtime_calls/3, compiler: EMLX).(x, qw1, qw2)
      eager = Nx.Defn.jit(&two_runtime_calls/3, compiler: Nx.Defn.Evaluator).(x, qw1, qw2)

      assert_all_close(native, eager)
    end

    test "a runtime_call inside a while body re-enters the compiler correctly" do
      weight =
        Nx.iota({2, 64}, type: :f32) |> Nx.divide(10) |> Nx.backend_transfer(EMLX.Backend)

      qw = EMLX.quantize(weight, [])
      x0 = Nx.broadcast(Nx.tensor(0.0, type: :f32, backend: EMLX.Backend), {2, 64})
      n = Nx.tensor(3, type: :s32, backend: EMLX.Backend)

      native = Nx.Defn.jit(&runtime_call_inside_while/3, compiler: EMLX).(x0, n, qw)
      eager = Nx.Defn.jit(&runtime_call_inside_while/3, compiler: Nx.Defn.Evaluator).(x0, n, qw)

      assert_all_close(native, eager)
    end

    test "a runtime_call with a tuple (multi-tensor) operand container splits correctly" do
      # Unlike dequantize's single bare-tensor operand, quantized_matmul's
      # runtime_call operand is a tuple {activation, qw} -- the same
      # multi-tensor-container shape as the real target use case
      # (EMLXAxon.native_kv_attn_callback/2), which no other test here
      # exercises.
      w = Nx.iota({4, 64}, type: :f32) |> Nx.divide(10) |> Nx.backend_transfer(EMLX.Backend)
      qw = EMLX.quantize(w, [])
      x = Nx.iota({2, 64}, type: :f32) |> Nx.divide(3) |> Nx.backend_transfer(EMLX.Backend)

      native = Nx.Defn.jit(&quantized_matmul_surrounded/2, compiler: EMLX).(x, qw)
      eager = Nx.Defn.jit(&quantized_matmul_surrounded/2, compiler: Nx.Defn.Evaluator).(x, qw)

      assert_all_close(native, eager)
    end

    test "a recognized EMLX.Fast.* runtime_call is still lowered in-graph, not split" do
      fun = fn x, w -> EMLX.Fast.rms_norm(x, w, 1.0e-6) end
      expr = Nx.Defn.debug_expr_apply(fun, [Nx.template({2, 4}, :f32), Nx.template({4}, :f32)])
      prog = Expr.lower(expr)

      assert [{_, :fast_rms_norm, [_x, _w], [_eps_bits]}] = prog.instructions

      x = Nx.iota({2, 4}, type: :f32, backend: EMLX.Backend)
      w = Nx.broadcast(Nx.tensor(1.0, type: :f32, backend: EMLX.Backend), {4})
      native = Nx.Defn.jit(fun, compiler: EMLX).(x, w)
      eager = fun.(x, w)

      assert_all_close(native, eager)
    end
  end

  defp dispatch_cache_entries_mentioning(shape) do
    :emlx_native_dispatch_cache
    |> :ets.tab2list()
    |> Enum.filter(fn {key, _resource, _hooks, _runtime_calls} -> term_mentions?(key, shape) end)
    |> Enum.map(&elem(&1, 0))
    |> Enum.uniq()
  end

  defp term_mentions?(term, needle) when term == needle, do: true

  defp term_mentions?(term, needle) when is_tuple(term) do
    term |> Tuple.to_list() |> term_mentions?(needle)
  end

  defp term_mentions?(term, needle) when is_list(term) do
    Enum.any?(term, &term_mentions?(&1, needle))
  end

  defp term_mentions?(term, needle) when is_map(term) do
    term |> Map.to_list() |> term_mentions?(needle)
  end

  defp term_mentions?(_term, _needle), do: false

  describe "runtime_call split-point dispatch cache (compile once, reuse)" do
    test "calling the same runtime_call-split defn twice compiles its flat stages once" do
      shape = {19, 128}
      w1 = Nx.iota(shape, type: :f32) |> Nx.divide(10) |> Nx.backend_transfer(EMLX.Backend)
      w2 = Nx.iota(shape, type: :f32) |> Nx.divide(7) |> Nx.backend_transfer(EMLX.Backend)
      qw1 = EMLX.quantize(w1, [])
      qw2 = EMLX.quantize(w2, [])
      x = Nx.iota(shape, type: :f32) |> Nx.divide(3) |> Nx.backend_transfer(EMLX.Backend)

      native1 = Nx.Defn.jit(&dequant_surrounded/2, compiler: EMLX).(x, qw1)
      entries_after_first = dispatch_cache_entries_mentioning(shape)
      assert entries_after_first != []

      native2 = Nx.Defn.jit(&dequant_surrounded/2, compiler: EMLX).(x, qw2)
      entries_after_second = dispatch_cache_entries_mentioning(shape)

      assert entries_after_second == entries_after_first

      eager1 = Nx.Defn.jit(&dequant_surrounded/2, compiler: Nx.Defn.Evaluator).(x, qw1)
      eager2 = Nx.Defn.jit(&dequant_surrounded/2, compiler: Nx.Defn.Evaluator).(x, qw2)
      assert_all_close(native1, eager1)
      assert_all_close(native2, eager2)
    end

    test "two structurally-identical but distinct call sites share one cache entry" do
      shape = {23, 192}
      w1 = Nx.iota(shape, type: :f32) |> Nx.divide(10) |> Nx.backend_transfer(EMLX.Backend)
      w2 = Nx.iota(shape, type: :f32) |> Nx.divide(7) |> Nx.backend_transfer(EMLX.Backend)
      qw1 = EMLX.quantize(w1, [])
      qw2 = EMLX.quantize(w2, [])
      x = Nx.iota(shape, type: :f32) |> Nx.divide(3) |> Nx.backend_transfer(EMLX.Backend)

      # Two separately-defined defns with the identical op sequence stand in
      # for "two of Qwen3's 28 structurally-identical attention layers":
      # each traces to a fresh `Expr` (different ids), but the same shape of
      # computation.
      native1 = Nx.Defn.jit(&dequant_surrounded/2, compiler: EMLX).(x, qw1)
      native2 = Nx.Defn.jit(&dequant_surrounded_other_site/2, compiler: EMLX).(x, qw2)

      entries = dispatch_cache_entries_mentioning(shape)
      assert length(entries) == 1

      eager1 = Nx.Defn.jit(&dequant_surrounded/2, compiler: Nx.Defn.Evaluator).(x, qw1)
      eager2 = Nx.Defn.jit(&dequant_surrounded_other_site/2, compiler: Nx.Defn.Evaluator).(x, qw2)
      assert_all_close(native1, eager1)
      assert_all_close(native2, eager2)
    end
  end

  describe "native block dispatch-key caching (skips default_expr for recognized blocks)" do
    test "svd full_matrices?: true: distinct retraces of the same shape share one dispatch-key entry" do
      shape = {17, 17}
      a1 = Nx.iota(shape, type: :f32) |> Nx.divide(3) |> Nx.backend_transfer(EMLX.Backend)

      a2 =
        Nx.iota(shape, type: :f32)
        |> Nx.divide(7)
        |> Nx.add(1)
        |> Nx.backend_transfer(EMLX.Backend)

      fun = fn t -> Nx.LinAlg.svd(t, full_matrices?: true) end

      # `Nx.Defn.jit_apply/3` retraces `fun` from scratch on every call (fresh
      # `Expr` ids each time), so `dispatch_key/3`'s id-based fast path always
      # misses and the structural signature walk always runs -- this is the
      # retrace-per-call pattern the PRD targets. Both calls must still land
      # in the *same* dispatch-cache entry: `default_expr`'s content (data-
      # independent, since tracing is symbolic) was never what discriminated
      # these before this change either, so this also guards against a
      # regression that would split them.
      {u1, s1, vt1} = Nx.Defn.jit_apply(fun, [a1], compiler: EMLX)
      {u2, s2, vt2} = Nx.Defn.jit_apply(fun, [a2], compiler: EMLX)

      entries = dispatch_cache_entries_mentioning(shape)
      assert length(entries) == 1

      assert_close(Nx.dot(Nx.multiply(u1, Nx.reshape(s1, {1, 17})), vt1), a1)
      assert_close(Nx.dot(Nx.multiply(u2, Nx.reshape(s2, {1, 17})), vt2), a2)
    end

    test "svd full_matrices?: false: default_expr still discriminates the dispatch key" do
      # full_matrices?: false lowers via `expand_block_via_default`'s
      # while-unroll (see "while-in-default_expr descent" above) -- it
      # genuinely consults `default_expr`, so two traces whose `default_expr`
      # differs (here, via `max_iter`, which changes the unrolled Jacobi
      # iteration count) must land in *different* dispatch-cache entries.
      # G3 regression guard: `native_lowerable_block?/2` must not (and does
      # not) match `full_matrices?: false`, so this path is untouched by P1.
      shape = {13, 13}
      a = Nx.iota(shape, type: :f32) |> Nx.add(1) |> Nx.backend_transfer(EMLX.Backend)

      fun_50 = fn t -> Nx.LinAlg.svd(t, full_matrices?: false, max_iter: 50) end
      fun_100 = fn t -> Nx.LinAlg.svd(t, full_matrices?: false, max_iter: 100) end

      {u1, s1, vt1} = Nx.Defn.jit_apply(fun_50, [a], compiler: EMLX)
      entries_after_50 = dispatch_cache_entries_mentioning(shape)

      {u2, s2, vt2} = Nx.Defn.jit_apply(fun_100, [a], compiler: EMLX)
      entries_after_both = dispatch_cache_entries_mentioning(shape)

      assert length(entries_after_both) > length(entries_after_50)

      # Loose tolerance: the Jacobi rotation algorithm's reconstruction
      # accuracy at these iteration counts isn't what this test is about --
      # only that both traces still compute a valid SVD.
      assert_close(Nx.dot(Nx.multiply(u1, Nx.reshape(s1, {1, 13})), vt1), a, 1.0e-2)
      assert_close(Nx.dot(Nx.multiply(u2, Nx.reshape(s2, {1, 13})), vt2), a, 1.0e-2)
    end
  end

  # Softmax normalisation over the last axis (primitive SDPA reference helper).
  defp normalize_rows(t) do
    Nx.divide(t, Nx.sum(t, axes: [-1], keep_axes: true))
  end

  # Pure-Nx primitive reference for prefill RoPE against a precomputed `freqs`
  # tensor (mirrors emlx_compiler.cpp's fast_rope_with_freqs_positions lambda:
  # inv_freq = reciprocal(freqs), half-rotate cos/sin blend). Kept as an
  # independent oracle for H>1 prefill: the T>1 lowering path never calls
  # mlx::core::fast::rope directly (it always uses the hand cos/sin/rotate
  # composition, broadcast across heads), but a hand-written primitive here
  # still guards against regressions in that composition without relying on
  # the code under test to check itself. (T=1 decode used to have a related,
  # now-fixed, multi-head bug — see elixir-nx/emlx#121 and the "multi-head
  # correctness" tests below.) `a` is {B, T, H, D}; `pos` is {B, T}; `freqs`
  # is {dims/2}.
  defp rope_freqs_prim(a, pos, freqs, dims, scale) do
    {b, t, _h, d} = Nx.shape(a)
    half = div(dims, 2)

    inv_freq = Nx.divide(1.0, freqs) |> Nx.reshape({1, 1, half})
    pos_bt1 = Nx.as_type(pos, :f32) |> Nx.reshape({b, t, 1})
    angles = Nx.multiply(Nx.multiply(pos_bt1, inv_freq), scale)

    cos_full =
      Nx.cos(angles) |> Nx.reshape({b, t, 1, half}) |> then(&Nx.concatenate([&1, &1], axis: 3))

    sin_full =
      Nx.sin(angles) |> Nx.reshape({b, t, 1, half}) |> then(&Nx.concatenate([&1, &1], axis: 3))

    x1 = a[[.., .., .., 0..(half - 1)]]
    x2 = a[[.., .., .., half..(dims - 1)]]
    rotated = Nx.concatenate([Nx.negate(x2), x1], axis: 3)

    a_head = a[[.., .., .., 0..(dims - 1)]]
    rope_head = Nx.add(Nx.multiply(a_head, cos_full), Nx.multiply(rotated, sin_full))

    if dims == d do
      rope_head
    else
      tail = a[[.., .., .., dims..(d - 1)]]
      Nx.concatenate([rope_head, tail], axis: 3)
    end
  end

  defp gpu_t(t), do: Nx.backend_transfer(t, {EMLX.Backend, device: :gpu})

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
    wire = Expr.to_native(prog)

    input_refs =
      Enum.map(inputs, fn %Nx.Tensor{data: %EMLX.Backend{ref: {_, r}}} -> r end)

    prog_ref = compile_nif!(worker, wire)
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
