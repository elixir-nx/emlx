defmodule EMLX.GradEquivalenceTest do
  @moduledoc """
  Stage 28 (`grad-equivalence-suite`): permanent grad-equivalence regression
  suite, widening Stage 23's 8-scenario triage (`grad_triage_test.exs`) into
  materially more op-class combinations, shapes, and dtypes (Emily M9
  testing-half parity).

  Two adjustments to the stage doc's original plan (user directive + advisor
  sign-off — see `workdir/native-compiler/28-grad-equivalence-suite.md`):

    * **No `StreamData`.** Breadth comes from table-driven fixed
      scenario × shape × dtype combinations (looped at test-run time), not
      generated inputs.
    * **Non-differentiable ops are included, not excluded.**
      `Nx.Defn.Grad`'s `@constants` list (`deps/nx/nx/lib/nx/defn/grad.ex`)
      already treats `argmax`/`argmin`/`floor`/`sign`/comparisons as a
      stop-gradient boundary when they appear as an operand of a
      differentiable op — grad is always well-defined for them (typically
      zero contribution through that operand). The real EMLX-specific
      question this suite checks is whether the *native* compiled backward
      pass applies the exact same stop-gradient rule as the Evaluator, not
      whether grad exists at all.

  Reference convention (same as `grad_triage_test.exs`): `reference/2` runs
  `Nx.Defn.jit_apply/3` with `compiler: Nx.Defn.Evaluator` on
  `Nx.BinaryBackend` tensors; `native/2` runs the same with `compiler: EMLX`.
  """

  use EMLX.Case, async: false
  import Nx.Defn

  # ── shared helpers ─────────────────────────────────────────────────────────

  # `EMLX.Case`'s setup sets the *process-global* default backend to
  # `EMLX.Backend`. `Nx.Defn.Evaluator` uses that default backend for any
  # tensor it synthesizes internally (e.g. `window_max`'s padding fill value)
  # rather than matching the explicit-backend args passed in — so without
  # this scoping, the reference silently mixes in `EMLX.Backend` and stops being
  # a pure BinaryBackend/Evaluator reference. Found via this stage's
  # `window_max`-with-explicit-padding scenario (see Results below).
  defp reference(fun, args) do
    previous = Nx.default_backend()
    Nx.default_backend(Nx.BinaryBackend)

    try do
      Nx.Defn.jit_apply(fun, args, compiler: Nx.Defn.Evaluator)
    after
      Nx.default_backend(previous)
    end
  end

  defp native(fun, args), do: Nx.Defn.jit_apply(fun, args, compiler: EMLX)

  # Unique-valued, zero-free, tie-free tensor generator — safe for argmax
  # (no ties), sign/floor (no zero/no exact-integer boundary), and division
  # (no zero denominator) all at once, across any shape/dtype combination.
  defp bin(shape, dtype) do
    size = Tuple.product(shape)

    values =
      for i <- 0..(size - 1) do
        (i - size / 2) * 0.37 + 0.6
      end

    values
    |> Nx.tensor(type: dtype, backend: Nx.BinaryBackend)
    |> Nx.reshape(shape)
  end

  defp assert_grad_equivalent(fun, args, tol \\ [atol: 1.0e-3, rtol: 1.0e-3]) do
    assert_all_close(native(fun, args), reference(fun, args), tol)
  end

  @shapes [{}, {4}, {2, 3}, {2, 2, 3}]
  @rank1plus_shapes [{4}, {2, 3}, {2, 2, 3}]
  @dtypes [{:f, 32}, {:f, 64}]

  # ── 1. smooth elementwise chain (broader op mix than Stage 23's sin*cos) ───

  defn smooth_elementwise_loss(x) do
    x
    |> Nx.sin()
    |> Nx.multiply(Nx.cos(x))
    |> Nx.add(Nx.log1p(Nx.abs(x)))
    |> Nx.subtract(Nx.tanh(x))
    |> Nx.multiply(Nx.sigmoid(x))
    |> Nx.add(Nx.sqrt(Nx.abs(x)))
    |> Nx.sum()
  end

  defn smooth_elementwise_grad(x), do: grad(x, &smooth_elementwise_loss/1)

  describe "smooth elementwise chain grad (sin/cos/log1p/tanh/sigmoid/sqrt/abs composed)" do
    test "matches the Evaluator reference across shapes and dtypes" do
      for shape <- @shapes, dtype <- @dtypes do
        x = bin(shape, dtype)
        assert_grad_equivalent(&smooth_elementwise_grad/1, [x])
      end
    end
  end

  # ── 2. reduction chain (sum/mean/reduce_max/reduce_min composed) ───────────

  defn reduction_zoo_loss(x) do
    x
    |> Nx.abs()
    |> Nx.add(1)
    |> Nx.log()
    |> Nx.sum(axes: [-1])
    |> Nx.mean()
    |> Nx.add(Nx.multiply(Nx.reduce_max(x), 0.1))
    |> Nx.add(Nx.multiply(Nx.reduce_min(x), 0.1))
  end

  defn reduction_zoo_grad(x), do: grad(x, &reduction_zoo_loss/1)

  describe "reduction chain grad (sum/mean/reduce_max/reduce_min composed)" do
    test "matches the Evaluator reference across rank>=1 shapes and dtypes" do
      for shape <- @rank1plus_shapes, dtype <- @dtypes do
        x = bin(shape, dtype)
        assert_grad_equivalent(&reduction_zoo_grad/1, [x])
      end
    end
  end

  # ── 3. dot chain (matmul backward through tanh + sum) ──────────────────────

  defn dot_chain_loss(a, b) do
    a |> Nx.dot(b) |> Nx.tanh() |> Nx.sum()
  end

  defn dot_chain_grad(a, b), do: grad(a, fn a -> dot_chain_loss(a, b) end)

  describe "dot chain grad (dot -> tanh -> sum)" do
    test "matches the Evaluator reference across shape pairs and dtypes" do
      for {shape_a, shape_b} <- [{{2, 3}, {3, 2}}, {{4, 4}, {4, 4}}, {{3, 4}, {4, 2}}],
          dtype <- @dtypes do
        a = bin(shape_a, dtype)
        b = bin(shape_b, dtype)
        assert_grad_equivalent(&dot_chain_grad/2, [a, b])
      end
    end
  end

  # ── 4. nested cond (cond inside cond, all four leaf branches exercised) ────

  defn nested_cond_loss(x) do
    cond do
      Nx.all(Nx.greater(x, 0)) ->
        cond do
          Nx.all(Nx.greater(Nx.sum(x), 10)) -> Nx.sum(Nx.multiply(x, x))
          true -> Nx.sum(Nx.exp(x))
        end

      true ->
        cond do
          Nx.all(Nx.less(x, -10)) -> Nx.sum(Nx.abs(x))
          true -> Nx.sum(Nx.multiply(x, 3))
        end
    end
  end

  defn nested_cond_grad(x), do: grad(x, &nested_cond_loss/1)

  describe "nested cond grad (cond inside cond, all four leaf branches)" do
    test "matches the Evaluator reference for each of the four branch combinations" do
      # {all positive?, sum > 10 or all < -10?}
      cases = [
        Nx.tensor([1.0, 2.0, 8.0]),
        Nx.tensor([0.1, 0.2, 0.3]),
        Nx.tensor([-11.0, -12.0, -13.0]),
        Nx.tensor([-1.0, 2.0, -3.0])
      ]

      for x <- cases do
        assert_grad_equivalent(&nested_cond_grad/1, [Nx.backend_transfer(x, Nx.BinaryBackend)])
      end
    end
  end

  # ── 5. while whose body itself contains a cond ─────────────────────────────

  defn while_with_cond_loss(x) do
    {out, _i} =
      while {out = x, i = 0}, Nx.less(i, 4) do
        out =
          cond do
            Nx.all(Nx.greater(Nx.sum(out), 0)) -> Nx.multiply(out, 1.1)
            true -> Nx.add(out, 0.05)
          end

        {out, i + 1}
      end

    Nx.sum(out)
  end

  defn while_with_cond_grad(x), do: grad(x, &while_with_cond_loss/1)

  describe "while-body-contains-cond grad" do
    # Not checked against the `Nx.Defn.Evaluator` reference here — a genuine
    # `Nx.Defn.Grad` bug (not an EMLX bug) makes that reference itself wrong for
    # this exact scenario (backward `:while` + nested data-dependent `cond`).
    # See `workdir/native-compiler/nx-grad-while-cond-bugreport.md`: EMLX's
    # native result is finite-difference-correct; `Nx.Defn.Evaluator`'s
    # (pure-BinaryBackend, no EMLX involved) is off by up to 20 orders of
    # magnitude on some elements. So this scenario is checked against a
    # finite-difference reference instead.
    test "matches a finite-difference reference (not the known-broken Evaluator path)" do
      eps = 1.0e-4

      for dtype <- @dtypes do
        x = bin({3}, dtype)
        native_grad = native(&while_with_cond_grad/1, [x])

        fd_grad =
          for i <- 0..2 do
            plus = Nx.indexed_add(x, Nx.tensor([[i]]), Nx.tensor([eps], type: dtype))
            minus = Nx.indexed_add(x, Nx.tensor([[i]]), Nx.tensor([-eps], type: dtype))

            (Nx.to_number(native(&while_with_cond_loss/1, [plus])) -
               Nx.to_number(native(&while_with_cond_loss/1, [minus]))) / (2 * eps)
          end
          |> Nx.tensor(type: dtype)

        assert_all_close(native_grad, fd_grad, atol: 5.0e-3, rtol: 5.0e-3)
      end
    end
  end

  # ── 6. multi-output while carries (3 carried tensors) ──────────────────────

  defn while_multi_carry_loss(x) do
    {acc1, acc2, _i} =
      while {acc1 = x, acc2 = Nx.multiply(x, 2), i = 0}, Nx.less(i, 3) do
        {Nx.add(acc1, acc2), Nx.multiply(acc2, 1.05), i + 1}
      end

    Nx.sum(Nx.add(acc1, acc2))
  end

  defn while_multi_carry_grad(x), do: grad(x, &while_multi_carry_loss/1)

  describe "multi-output while carries grad (3 carried tensors)" do
    test "matches the Evaluator reference" do
      for dtype <- @dtypes do
        x = bin({4}, dtype)
        assert_grad_equivalent(&while_multi_carry_grad/1, [x])
      end
    end
  end

  # ── 7. nested while (while inside while) ───────────────────────────────────

  defn nested_while_loss(x) do
    {out, _i} =
      while {out = x, i = 0}, Nx.less(i, 2) do
        {inner, _j} =
          while {inner = out, j = 0}, Nx.less(j, 2) do
            {Nx.add(inner, 0.1), j + 1}
          end

        {inner, i + 1}
      end

    Nx.sum(out)
  end

  defn nested_while_grad(x), do: grad(x, &nested_while_loss/1)

  describe "nested while grad (while inside while)" do
    test "matches the Evaluator reference" do
      for dtype <- @dtypes do
        x = bin({3}, dtype)
        assert_grad_equivalent(&nested_while_grad/1, [x])
      end
    end
  end

  # ── 8. windowed ops with non-default strides/padding ───────────────────────

  defn window_sum_strided_loss(x), do: Nx.sum(Nx.window_sum(x, {2, 2}, strides: [2, 1]))
  defn window_sum_strided_grad(x), do: grad(x, &window_sum_strided_loss/1)

  defn window_max_padded_loss(x),
    do: Nx.sum(Nx.window_max(x, {2, 2}, padding: [{1, 1}, {1, 1}]))

  defn window_max_padded_grad(x), do: grad(x, &window_max_padded_loss/1)

  defn window_sum_strided_3d_loss(x),
    do: Nx.sum(Nx.window_sum(x, {2, 2, 2}, strides: [2, 1, 2]))

  defn window_sum_strided_3d_grad(x), do: grad(x, &window_sum_strided_3d_loss/1)

  describe "windowed ops grad with non-default strides/padding" do
    # Closed by Stage 33: `window_sum`'s backward (grad.ex's
    # `grad(:window_sum, …)`) un-strides the cotangent via `Nx.pad` with
    # *interior* padding whenever `strides != 1`. `:pad` with interior padding
    # (and negative lo/hi) now lowers natively (see `EMLX.Native.Expr.expand_pad_general/5`)
    # instead of raising — this scenario used to assert the known raise
    # (`EXPR_NODES.md`'s "pad (simple: non-negative lo/hi, interior=0; …)"),
    # now it asserts equivalence like every other scenario in this suite.
    # Stage 23's `window_sum` grad scenario used default (unit) strides, so
    # it never exercised this path.
    test "window_sum with non-unit strides matches the Evaluator reference" do
      for shape <- [{4, 4}, {3, 5}], dtype <- @dtypes do
        x = bin(shape, dtype)
        assert_grad_equivalent(&window_sum_strided_grad/1, [x])
      end
    end

    test "window_sum with non-unit strides matches the Evaluator reference (3D)" do
      for dtype <- @dtypes do
        x = bin({4, 3, 5}, dtype)
        assert_grad_equivalent(&window_sum_strided_3d_grad/1, [x])
      end
    end

    test "window_max with padding matches the Evaluator reference" do
      for shape <- [{4, 4}, {3, 5}], dtype <- @dtypes do
        x = bin(shape, dtype)
        assert_grad_equivalent(&window_max_padded_grad/1, [x])
      end
    end
  end

  # ── 8b. direct :pad / :slice grad (Stage 33) ────────────────────────────────
  #
  # Broader than window ops (found during Stage 33's advisor review): grad.ex's
  # own `grad(:pad, …)` un-pads the cotangent via *negative* lo/hi whenever the
  # forward `Nx.pad` had positive lo/hi, unconditionally (no strides needed);
  # `grad(:slice, …)` re-inserts *interior* padding into the cotangent whenever
  # the forward `Nx.slice` used non-unit strides. Both are more common than the
  # window_sum path above and exercise the same `:pad` decomposition from a
  # different direction.

  defn pad_positive_loss(x), do: Nx.sum(Nx.pad(x, 0.0, [{2, 1, 0}, {0, 3, 0}]))
  defn pad_positive_grad(x), do: grad(x, &pad_positive_loss/1)

  defn slice_strided_loss(x), do: Nx.sum(Nx.slice(x, [0, 0], [2, 3], strides: [2, 3]))
  defn slice_strided_grad(x), do: grad(x, &slice_strided_loss/1)

  describe "direct :pad / :slice grad (Stage 33)" do
    test "pad with positive lo/hi (negative-lo/hi backward) matches the Evaluator reference" do
      for shape <- [{4, 4}, {3, 5}], dtype <- @dtypes do
        x = bin(shape, dtype)
        assert_grad_equivalent(&pad_positive_grad/1, [x])
      end
    end

    test "slice with non-unit strides (interior-pad backward) matches the Evaluator reference" do
      for shape <- [{5, 8}], dtype <- @dtypes do
        x = bin(shape, dtype)
        assert_grad_equivalent(&slice_strided_grad/1, [x])
      end
    end
  end

  # ── 9. non-differentiable ops used as an operand (stop-gradient boundary) ──
  #
  # Per this stage's plan adjustment: these aren't excluded — `Nx.Defn.Grad`'s
  # `@constants` list makes them act as a stop-gradient boundary when used as
  # an operand of a differentiable op, so grad is well-defined. What's tested
  # here is that EMLX's native backward lowering applies that exact same
  # boundary as the Evaluator, at points chosen away from each op's
  # discontinuity (no exact zero for `sign`, no argmax ties, no integer
  # boundary for `floor`).

  defn sign_operand_loss(x), do: Nx.sum(Nx.multiply(x, Nx.sign(x)))
  defn sign_operand_grad(x), do: grad(x, &sign_operand_loss/1)

  defn floor_operand_loss(x), do: Nx.sum(Nx.multiply(x, Nx.floor(Nx.abs(x) |> Nx.add(1))))
  defn floor_operand_grad(x), do: grad(x, &floor_operand_loss/1)

  defn comparison_select_loss(x) do
    Nx.sum(Nx.select(Nx.greater(x, 0), Nx.multiply(x, x), Nx.multiply(x, -1)))
  end

  defn comparison_select_grad(x), do: grad(x, &comparison_select_loss/1)

  defn argmax_gather_loss(x) do
    idx = Nx.argmax(x, axis: -1, keep_axis: true)
    Nx.sum(Nx.take_along_axis(x, idx, axis: -1))
  end

  defn argmax_gather_grad(x), do: grad(x, &argmax_gather_loss/1)

  describe "non-differentiable-op-as-operand grad (stop-gradient boundary parity)" do
    test "sign(x) used as a multiplicative operand matches the Evaluator reference" do
      for shape <- @shapes, dtype <- @dtypes do
        x = bin(shape, dtype)
        assert_grad_equivalent(&sign_operand_grad/1, [x])
      end
    end

    test "floor(x) used as a multiplicative operand matches the Evaluator reference" do
      for shape <- @shapes, dtype <- @dtypes do
        x = bin(shape, dtype)
        assert_grad_equivalent(&floor_operand_grad/1, [x])
      end
    end

    test "a comparison feeding select matches the Evaluator reference" do
      for shape <- @shapes, dtype <- @dtypes do
        x = bin(shape, dtype)
        assert_grad_equivalent(&comparison_select_grad/1, [x])
      end
    end

    test "argmax used to gather (max-pooling-style pattern) matches the Evaluator reference" do
      for shape <- @rank1plus_shapes, dtype <- @dtypes do
        x = bin(shape, dtype)
        assert_grad_equivalent(&argmax_gather_grad/1, [x])
      end
    end
  end

  # ── 10. finite-difference reference (smooth ops only) ─────────────────────────
  #
  # FD is meaningless exactly at a non-differentiable op's discontinuity, so
  # this is restricted to the smooth subset (per advisor guidance) and run at
  # points chosen away from any domain edge (all positive, away from zero).
  # A vector shape is used (not just scalar): `Nx.Defn.grad/2` treats extra
  # dims as batch dims, so this also validates the gradient is correct
  # *per element*, not just in aggregate.

  @smooth_unary_ops [:sin, :cos, :exp, :tanh, :sigmoid, :sqrt, :log, :cbrt, :expm1, :log1p]
  @fd_eps 1.0e-3
  # f32 central differences bottom out ~1e-3 relative (re-verified here, not
  # copied from Emily's own number) — use a matching tolerance.
  @fd_tol [atol: 5.0e-3, rtol: 5.0e-3]

  describe "finite-difference reference (smooth unary ops, points away from discontinuities)" do
    test "native grad matches central-difference FD for each smooth unary op" do
      x = Nx.tensor([0.3, 0.8, 1.4, 2.1], type: {:f, 64}, backend: Nx.BinaryBackend)

      for op <- @smooth_unary_ops do
        grad_fn = fn x -> Nx.Defn.grad(x, &apply(Nx, op, [&1])) end
        native_grad = Nx.Defn.jit_apply(grad_fn, [x], compiler: EMLX)

        fd_grad =
          apply(Nx, op, [Nx.add(x, @fd_eps)])
          |> Nx.subtract(apply(Nx, op, [Nx.subtract(x, @fd_eps)]))
          |> Nx.divide(2 * @fd_eps)

        assert_all_close(native_grad, fd_grad, @fd_tol)
      end
    end
  end
end
