defmodule EMLX.ConvPoolTrainingCanaryTest do
  @moduledoc """
  Stage 30 (`conv-pool-training-curve-canary`): a training-curve-matching
  canary for `window`-reduction-backed conv/pool training (window ops off
  `via_binary` — already closed per Stage 23's finding, re-verified here by
  grepping `lib/emlx/backend.ex` for `window_op/5`/`window_scatter_function/7`
  as the sole implementations, no `via_binary`).

  A small handwritten conv → relu → max-pool → dense classifier trains for
  `@steps` SGD steps against a fixed, deterministic 3-batch dataset, and the
  resulting per-step loss curve is compared against a `Nx.BinaryBackend` /
  `Nx.Defn.Evaluator` reference — not just "does it run," but "does it
  converge the same way." Per advisor sign-off (`/tackle-step` pre-work
  check): only `Nx.window_max` (max-pool) is used, not strided `window_sum`
  (avg-pool), to stay clear of Stage 33's known strided-`window_sum`-grad
  gap; both the eager `EMLX.Backend` lane and the native `compiler: EMLX`
  lane are exercised against the same reference, since Stage 23/28 covered
  grad correctness on both.

  Reference convention matches `grad_equivalence_test.exs`: scope the
  process-global default backend to `Nx.BinaryBackend` around
  `Nx.Defn.Evaluator` so any tensor the Evaluator synthesizes internally
  (e.g. a window op's padding fill) doesn't silently pull in `EMLX.Backend`.
  """

  use EMLX.Case, async: false
  import Nx.Defn

  @batch 4
  @in_hw 8
  @kernel_hw 3
  @out_channels 4
  @pooled_hw 3
  @flat_size @out_channels * @pooled_hw * @pooled_hw
  @num_classes 3
  @lr 0.05
  @steps 20
  @num_batches 3

  # ── deterministic, backend-agnostic data/param generation ──────────────────
  # Avoids Nx.Random entirely so the two backends under comparison start from
  # bit-identical inputs without relying on cross-backend RNG parity (not
  # this stage's concern — see grad_equivalence_test.exs for that).
  defp seeded(shape, seed) do
    size = Tuple.product(shape)

    for i <- 0..(size - 1) do
      :math.sin((i + seed) * 0.7) * 0.5
    end
    |> Nx.tensor(type: {:f, 32}, backend: Nx.BinaryBackend)
    |> Nx.reshape(shape)
  end

  defp one_hot_targets(offset) do
    for i <- 0..(@batch - 1) do
      label = rem(i + offset, @num_classes)
      for c <- 0..(@num_classes - 1), do: if(c == label, do: 1.0, else: 0.0)
    end
    |> Nx.tensor(type: {:f, 32}, backend: Nx.BinaryBackend)
  end

  defp init_params do
    %{
      w1: seeded({@out_channels, 1, @kernel_hw, @kernel_hw}, 1),
      w2: seeded({@flat_size, @num_classes}, 2),
      b2: seeded({@num_classes}, 3)
    }
  end

  defp dataset do
    for b <- 0..(@num_batches - 1) do
      {seeded({@batch, 1, @in_hw, @in_hw}, 10 + b), one_hot_targets(b)}
    end
  end

  # ── model ────────────────────────────────────────────────────────────────

  defn forward(params, x) do
    x
    |> Nx.conv(params.w1, padding: :valid)
    |> Nx.max(0.0)
    |> Nx.window_max({1, 1, 2, 2}, strides: [1, 1, 2, 2], padding: :valid)
    |> Nx.reshape({@batch, @flat_size})
    |> Nx.dot(params.w2)
    |> Nx.add(params.b2)
  end

  defn loss_fn(params, x, y) do
    preds = forward(params, x)
    Nx.mean(Nx.pow(Nx.subtract(preds, y), 2))
  end

  defn train_step(params, x, y) do
    {loss, grads} = value_and_grad(params, &loss_fn(&1, x, y))

    new_params = %{
      w1: Nx.subtract(params.w1, Nx.multiply(@lr, grads.w1)),
      w2: Nx.subtract(params.w2, Nx.multiply(@lr, grads.w2)),
      b2: Nx.subtract(params.b2, Nx.multiply(@lr, grads.b2))
    }

    {loss, new_params}
  end

  # ── runners ──────────────────────────────────────────────────────────────

  defp transfer_params(params, backend) do
    %{
      w1: Nx.backend_transfer(params.w1, backend),
      w2: Nx.backend_transfer(params.w2, backend),
      b2: Nx.backend_transfer(params.b2, backend)
    }
  end

  defp run_steps(params0, dataset, jit_fun) do
    {_final_params, losses} =
      Enum.reduce(1..@steps, {params0, []}, fn step, {params, losses} ->
        {x, y} = Enum.at(dataset, rem(step - 1, length(dataset)))
        {loss, new_params} = jit_fun.(params, x, y)
        {new_params, [Nx.to_number(loss) | losses]}
      end)

    Enum.reverse(losses)
  end

  defp reference_curve(params0, dataset) do
    previous = Nx.default_backend()
    Nx.default_backend(Nx.BinaryBackend)

    try do
      run_steps(params0, dataset, fn params, x, y ->
        Nx.Defn.jit_apply(&train_step/3, [params, x, y], compiler: Nx.Defn.Evaluator)
      end)
    after
      Nx.default_backend(previous)
    end
  end

  defp eager_emlx_curve(params0, dataset) do
    params = transfer_params(params0, EMLX.Backend)

    dataset =
      Enum.map(dataset, fn {x, y} ->
        {Nx.backend_transfer(x, EMLX.Backend), Nx.backend_transfer(y, EMLX.Backend)}
      end)

    run_steps(params, dataset, fn params, x, y ->
      Nx.Defn.jit_apply(&train_step/3, [params, x, y], compiler: Nx.Defn.Evaluator)
    end)
  end

  defp native_curve(params0, dataset) do
    run_steps(params0, dataset, fn params, x, y ->
      Nx.Defn.jit_apply(&train_step/3, [params, x, y], compiler: EMLX)
    end)
  end

  defp assert_curves_match(curve, reference, tol) do
    curve
    |> Enum.zip(reference)
    |> Enum.with_index()
    |> Enum.each(fn {{a, b}, i} ->
      assert_in_delta(a, b, tol, "loss curves diverge at step #{i}: #{a} vs #{b}")
    end)
  end

  describe "conv+max_pool small-CNN training curve" do
    test "eager EMLX.Backend matches the Nx.BinaryBackend/Evaluator reference" do
      params0 = init_params()
      data = dataset()

      reference = reference_curve(params0, data)
      eager = eager_emlx_curve(params0, data)

      assert_curves_match(eager, reference, 1.0e-3)
      assert List.last(reference) < List.first(reference) * 0.5
    end

    test "native compiler: EMLX matches the Nx.BinaryBackend/Evaluator reference" do
      params0 = init_params()
      data = dataset()

      reference = reference_curve(params0, data)
      native = native_curve(params0, data)

      assert_curves_match(native, reference, 1.0e-3)
      assert List.last(reference) < List.first(reference) * 0.5
    end
  end
end
