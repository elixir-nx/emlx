defmodule EMLX.PoolingTest do
  use EMLX.Case, async: true

  # Helper: run on both backends and compare (accepts tensors from any backend)
  defp assert_close(emlx_t, binary_t) do
    emlx_vals = Nx.to_flat_list(Nx.as_type(emlx_t, {:f, 32}))
    binary_vals = Nx.to_flat_list(Nx.as_type(binary_t, {:f, 32}))

    Enum.zip(emlx_vals, binary_vals)
    |> Enum.each(fn {e, b} ->
      assert_in_delta(e, b, 1.0e-5)
    end)
  end

  describe "window_scatter_max: NIF matches Elixir reference" do
    test "1-D basic, no padding" do
      t = Nx.tensor([1.0, 3.0, 2.0, 5.0], backend: EMLX.Backend)
      ref = Nx.tensor([1.0, 3.0, 2.0, 5.0], backend: Nx.BinaryBackend)

      emlx_result = Nx.window_max(t, {2}, strides: [1], padding: :valid)
      ref_result = Nx.window_max(ref, {2}, strides: [1], padding: :valid)

      assert_close(emlx_result, ref_result)
    end

    test "2-D window_scatter_max: grad matches BinaryBackend" do
      t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: EMLX.Backend)
      ref = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: Nx.BinaryBackend)

      grad_fn = fn x -> Nx.window_max(x, {2, 2}, strides: [1, 1], padding: :valid) end

      emlx_grad = Nx.Defn.grad(t, grad_fn)
      ref_grad = Nx.Defn.grad(ref, grad_fn)

      assert_close(emlx_grad, ref_grad)
    end

    test "window_scatter_max with non-unit strides" do
      t = Nx.tensor([1.0, 3.0, 2.0, 5.0, 4.0, 6.0], backend: EMLX.Backend)
      ref = Nx.tensor([1.0, 3.0, 2.0, 5.0, 4.0, 6.0], backend: Nx.BinaryBackend)

      grad_fn = fn x -> Nx.window_max(x, {3}, strides: [2], padding: :valid) end

      emlx_grad = Nx.Defn.grad(t, grad_fn)
      ref_grad = Nx.Defn.grad(ref, grad_fn)

      assert_close(emlx_grad, ref_grad)
    end

    test "window_scatter_max with same padding" do
      t = Nx.tensor([1.0, 3.0, 2.0, 5.0], backend: EMLX.Backend)
      ref = Nx.tensor([1.0, 3.0, 2.0, 5.0], backend: Nx.BinaryBackend)

      grad_fn = fn x -> Nx.window_max(x, {3}, strides: [1], padding: :same) end

      emlx_grad = Nx.Defn.grad(t, grad_fn)
      ref_grad = Nx.Defn.grad(ref, grad_fn)

      assert_close(emlx_grad, ref_grad)
    end

    test "window_scatter_max non-zero init_value: unwritten positions carry init_value" do
      # Call window_scatter_max directly with a source that has one element
      t = Nx.tensor([1.0, 3.0, 2.0], backend: EMLX.Backend)
      source = Nx.tensor([10.0], backend: EMLX.Backend)
      init_value = Nx.tensor(0.0, backend: EMLX.Backend)
      out_shape = {3}

      # window [3] on a [3] tensor: one window, max is at index 1 (value 3.0)
      result =
        EMLX.Backend.window_scatter_max(
          %Nx.Tensor{shape: out_shape, type: {:f, 32}, names: [nil]},
          t,
          source,
          init_value,
          {3},
          padding: [{0, 0}],
          strides: [1]
        )

      result_list = Nx.to_flat_list(Nx.backend_transfer(result, Nx.BinaryBackend))
      assert Enum.at(result_list, 0) == 0.0
      assert Enum.at(result_list, 1) == 10.0
      assert Enum.at(result_list, 2) == 0.0
    end
  end

  describe "window_scatter_min: NIF matches Elixir reference" do
    test "2-D window_scatter_min: grad matches BinaryBackend" do
      t = Nx.tensor([[4.0, 2.0, 3.0], [1.0, 5.0, 6.0]], backend: EMLX.Backend)
      ref = Nx.tensor([[4.0, 2.0, 3.0], [1.0, 5.0, 6.0]], backend: Nx.BinaryBackend)

      grad_fn = fn x -> Nx.window_min(x, {2, 2}, strides: [1, 1], padding: :valid) end

      emlx_grad = Nx.Defn.grad(t, grad_fn)
      ref_grad = Nx.Defn.grad(ref, grad_fn)

      assert_close(emlx_grad, ref_grad)
    end

    test "window_scatter_min tie-breaking: last-occurrence wins" do
      # Two equal minimum values in the window — the LAST one should receive the gradient.
      # Window of size 3 over [2.0, 1.0, 1.0]: min=1.0 appears at indices 1 and 2;
      # tie_break: :high means last occurrence (index 2) wins.
      t = Nx.tensor([2.0, 1.0, 1.0], backend: EMLX.Backend)
      ref = Nx.tensor([2.0, 1.0, 1.0], backend: Nx.BinaryBackend)

      grad_fn = fn x -> Nx.window_min(x, {3}, strides: [1], padding: :valid) end

      emlx_grad = Nx.Defn.grad(t, grad_fn)
      ref_grad = Nx.Defn.grad(ref, grad_fn)

      # Transfer once, then use the copied result for both assertions.
      grad_binary = Nx.backend_transfer(emlx_grad, Nx.BinaryBackend)
      assert_close(grad_binary, ref_grad)

      # Confirm that index 2 (last occurrence) received the gradient, not index 1.
      grad_list = Nx.to_flat_list(grad_binary)
      assert Enum.at(grad_list, 2) > 0.0
      assert Enum.at(grad_list, 1) == 0.0
    end

    test "window_scatter_min tie-breaking: all-equal window" do
      t = Nx.tensor([1.0, 1.0, 1.0, 1.0], backend: EMLX.Backend)
      ref = Nx.tensor([1.0, 1.0, 1.0, 1.0], backend: Nx.BinaryBackend)

      grad_fn = fn x -> Nx.window_min(x, {2}, strides: [1], padding: :valid) end

      emlx_grad = Nx.Defn.grad(t, grad_fn)
      ref_grad = Nx.Defn.grad(ref, grad_fn)

      assert_close(emlx_grad, ref_grad)
    end

    test "window_scatter_min 3-D input" do
      shape = {2, 3, 4}
      t = Nx.iota(shape, type: :f32, backend: EMLX.Backend)
      ref = Nx.iota(shape, type: :f32, backend: Nx.BinaryBackend)

      grad_fn = fn x -> Nx.window_min(x, {1, 2, 2}, strides: [1, 1, 1], padding: :valid) end

      emlx_grad = Nx.Defn.grad(t, grad_fn)
      ref_grad = Nx.Defn.grad(ref, grad_fn)

      assert_close(emlx_grad, ref_grad)
    end

    test "window_scatter_min non-zero init_value: unwritten positions carry init_value" do
      # Window [3] over [1.0, 2.0, 3.0] (valid, no padding): one window, min at index 0.
      # scatter_add ADDS source to init_value, so selected = init_value + source.
      t = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)
      source = Nx.tensor([5.0], backend: EMLX.Backend)
      init_value = Nx.tensor(-99.0, backend: EMLX.Backend)

      result =
        EMLX.Backend.window_scatter_min(
          %Nx.Tensor{shape: {3}, type: {:f, 32}, names: [nil]},
          t,
          source,
          init_value,
          {3},
          padding: [{0, 0}],
          strides: [1]
        )

      result_list = Nx.to_flat_list(Nx.as_type(result, {:f, 32}))
      # min is at index 0 → init_value + source = -99.0 + 5.0 = -94.0
      # other positions carry init_value = -99.0
      assert_in_delta(Enum.at(result_list, 0), -94.0, 1.0e-5)
      assert_in_delta(Enum.at(result_list, 1), -99.0, 1.0e-5)
      assert_in_delta(Enum.at(result_list, 2), -99.0, 1.0e-5)
    end
  end

  describe "window scatter: additional coverage" do
    test "window_scatter_max first-occurrence tie-break" do
      # Window of size 3 over [1.0, 3.0, 3.0]: max=3.0 at indices 1 and 2.
      # MLX argmax is first-occurrence, so index 1 wins.
      t = Nx.tensor([1.0, 3.0, 3.0], backend: EMLX.Backend)
      ref = Nx.tensor([1.0, 3.0, 3.0], backend: Nx.BinaryBackend)

      grad_fn = fn x -> Nx.window_max(x, {3}, strides: [1], padding: :valid) end

      emlx_grad = Nx.Defn.grad(t, grad_fn)
      ref_grad = Nx.Defn.grad(ref, grad_fn)

      assert_close(emlx_grad, ref_grad)

      grad_list = Nx.to_flat_list(Nx.as_type(emlx_grad, {:f, 32}))
      assert Enum.at(grad_list, 1) > 0.0
      assert Enum.at(grad_list, 2) == 0.0
    end

    test "non-overlapping windows (strides == window size)" do
      t = Nx.tensor([1.0, 3.0, 2.0, 5.0, 4.0, 6.0], backend: EMLX.Backend)
      ref = Nx.tensor([1.0, 3.0, 2.0, 5.0, 4.0, 6.0], backend: Nx.BinaryBackend)

      grad_fn = fn x -> Nx.window_max(x, {2}, strides: [2], padding: :valid) end

      emlx_grad = Nx.Defn.grad(t, grad_fn)
      ref_grad = Nx.Defn.grad(ref, grad_fn)

      assert_close(emlx_grad, ref_grad)
    end

    test "integer dtype (s32)" do
      t = Nx.tensor([1, 3, 2, 5], type: {:s, 32}, backend: EMLX.Backend)
      ref = Nx.tensor([1, 3, 2, 5], type: {:s, 32}, backend: Nx.BinaryBackend)

      emlx_result = Nx.window_max(t, {2}, strides: [1], padding: :valid)
      ref_result = Nx.window_max(ref, {2}, strides: [1], padding: :valid)

      assert_close(emlx_result, ref_result)
    end
  end
end
