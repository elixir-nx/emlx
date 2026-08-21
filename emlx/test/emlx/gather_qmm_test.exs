defmodule EMLX.GatherQmmTest do
  @moduledoc """
  Tests for `EMLX.gather_qmm/11` — quantized matmul with per-row weight
  selection (`mx::gather_qmm`), the primitive behind mixture-of-experts layers.
  """
  use EMLX.Case, async: true

  @moduletag :metal

  # Device refs are consumed by the NIF, so every call gets a fresh one.
  # backend_copy, not backend_transfer: the latter moves the tensor and leaves
  # the source deallocated, which breaks the second use of the same input.
  defp ref(tensor) do
    tensor
    |> Nx.backend_copy({EMLX.Backend, device: :gpu})
    |> EMLX.Backend.from_nx()
  end

  # Stacked per-expert weights: {experts, out, in}. Returned as a plain tensor
  # and quantized at each point of use.
  defp expert_weights(count, out, inp) do
    Nx.iota({count, out, inp}, type: :f32)
    |> Nx.divide(count * out * inp)
    |> Nx.subtract(0.5)
  end

  defp quantized(weights), do: EMLX.quantize(ref(weights), 64, 4)

  defp gather(x, weights, indices, opts \\ []) do
    {w, scales, biases} = quantized(weights)

    EMLX.gather_qmm(
      ref(x),
      w,
      scales,
      biases,
      nil,
      ref(indices),
      true,
      64,
      4,
      "affine",
      Keyword.get(opts, :sorted, false)
    )
    |> EMLX.Backend.to_nx()
  end

  describe "EMLX.gather_qmm/11" do
    test "each row uses the expert named by rhs_indices" do
      weights = expert_weights(4, 32, 128)
      x = Nx.iota({4, 1, 1, 128}, type: :f32) |> Nx.divide(128)
      indices = Nx.tensor([[0], [1], [2], [3]], type: :u32)

      gathered = gather(x, weights, indices)
      assert Nx.shape(gathered) == {4, 1, 1, 32}

      # Each row must match a plain quantized_matmul against that expert alone.
      for expert <- 0..3 do
        row = x |> Nx.slice([expert, 0, 0, 0], [1, 1, 1, 128]) |> Nx.reshape({1, 128})
        one = weights |> Nx.slice([expert, 0, 0], [1, 32, 128]) |> Nx.reshape({32, 128})
        {w, scales, biases} = quantized(one)

        direct =
          EMLX.quantized_matmul(ref(row), w, scales, biases, true, 64, 4)
          |> EMLX.Backend.to_nx()

        taken = gathered |> Nx.slice([expert, 0, 0, 0], [1, 1, 1, 32]) |> Nx.reshape({1, 32})
        assert Nx.all_close(taken, direct, atol: 1.0e-5) |> Nx.to_number() == 1
      end
    end

    test "repeated indices produce identical rows" do
      weights = expert_weights(4, 32, 128)
      x = Nx.iota({1, 1, 1, 128}, type: :f32) |> Nx.divide(128)

      out = gather(x, weights, Nx.tensor([[2, 2, 2]], type: :u32))
      assert Nx.shape(out) == {1, 3, 1, 32}

      first = Nx.slice(out, [0, 0, 0, 0], [1, 1, 1, 32])

      for k <- 1..2 do
        other = Nx.slice(out, [0, k, 0, 0], [1, 1, 1, 32])
        assert Nx.all_close(first, other) |> Nx.to_number() == 1
      end
    end

    test "sorted_indices does not change the result" do
      weights = expert_weights(8, 32, 128)
      x = Nx.iota({4, 1, 1, 128}, type: :f32) |> Nx.divide(128)
      indices = Nx.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], type: :u32)

      unsorted = gather(x, weights, indices, sorted: false)
      sorted = gather(x, weights, indices, sorted: true)

      assert Nx.all_close(unsorted, sorted, atol: 1.0e-5) |> Nx.to_number() == 1
    end

    test "output shape follows the index shape" do
      weights = expert_weights(8, 32, 128)
      x = Nx.iota({2, 1, 1, 128}, type: :f32) |> Nx.divide(128)

      for per_row <- [1, 2, 4] do
        indices =
          Nx.iota({2, per_row}, type: :u32)
          |> Nx.remainder(Nx.tensor(8, type: :u32))

        assert Nx.shape(gather(x, weights, indices)) == {2, per_row, 1, 32}
      end
    end
  end
end
