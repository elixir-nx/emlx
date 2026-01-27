defmodule EMLX.Quantization.ModuleTest do
  @moduledoc """
  Tests for the EMLX.Quantization module - the primary user-facing API
  for quantized tensor operations.

  These tests verify the high-level API that users should prefer over
  the lower-level EMLX.quantize/EMLX.dequantize/EMLX.quantized_matmul functions.
  """
  use EMLX.Case

  alias EMLX.Quantization

  describe "Quantization.quantize/2" do
    test "quantizes an Nx.Tensor" do
      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      {q_weight, scales, biases} = Quantization.quantize(weight)

      # Returns EMLX device refs
      assert is_tuple(q_weight)
      assert is_tuple(scales)
      assert is_tuple(biases)

      # Quantized weights are uint32 (packed int4)
      {_dev, ref} = q_weight
      assert EMLX.scalar_type({:gpu, ref}) == :uint32
    end

    test "quantizes with custom group_size" do
      weight = Nx.iota({128, 128}, type: :f32) |> Nx.divide(100)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      {_q_weight, scales, _biases} = Quantization.quantize(weight, group_size: 128)

      # With group_size=128, scales shape is [128, 128/128] = [128, 1]
      {_dev, s_ref} = scales
      assert EMLX.shape({:gpu, s_ref}) == {128, 1}
    end

    test "accepts EMLX device ref directly" do
      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      emlx_ref = EMLX.Backend.from_nx(weight)

      {q_weight, scales, biases} = Quantization.quantize(emlx_ref)

      assert is_tuple(q_weight)
      assert is_tuple(scales)
      assert is_tuple(biases)
    end
  end

  describe "Quantization.tensor/5" do
    test "creates quantized Nx.Tensor with {:s, 4} type" do
      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      emlx_weight = EMLX.Backend.from_nx(weight)
      {q_weight, scales, biases} = EMLX.quantize(emlx_weight, 64, 4)

      qt = Quantization.tensor(q_weight, scales, biases, {64, 64})

      assert %Nx.Tensor{} = qt
      assert Nx.type(qt) == {:s, 4}
      assert Nx.shape(qt) == {64, 64}
    end

    test "creates tensor with 8-bit quantization" do
      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      emlx_weight = EMLX.Backend.from_nx(weight)
      {q_weight, scales, biases} = EMLX.quantize(emlx_weight, 64, 8)

      qt = Quantization.tensor(q_weight, scales, biases, {64, 64}, bits: 8)

      assert Nx.type(qt) == {:s, 8}
    end

    test "stores group_size in backend struct" do
      weight = Nx.iota({128, 128}, type: :f32) |> Nx.divide(100)
      emlx_weight = EMLX.Backend.from_nx(weight)
      {q_weight, scales, biases} = EMLX.quantize(emlx_weight, 128, 4)

      qt = Quantization.tensor(q_weight, scales, biases, {128, 128}, group_size: 128)

      opts = Quantization.options(qt)
      assert opts.group_size == 128
    end
  end

  describe "Quantization.dequantize/4" do
    test "converts quantized weights back to float" do
      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      emlx_weight = EMLX.Backend.from_nx(weight)

      {q_weight, scales, biases} = Quantization.quantize(emlx_weight)
      dequantized = Quantization.dequantize(q_weight, scales, biases)

      # Returns EMLX device ref
      {_dev, d_ref} = dequantized
      assert EMLX.shape({:gpu, d_ref}) == {64, 64}
    end

    test "roundtrip preserves approximate values" do
      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(10)
      emlx_weight = EMLX.Backend.from_nx(weight)

      {q_weight, scales, biases} = Quantization.quantize(emlx_weight)
      dequantized = Quantization.dequantize(q_weight, scales, biases)

      original = EMLX.Backend.to_nx(emlx_weight)
      recovered = EMLX.Backend.to_nx(dequantized)

      # 4-bit is lossy, but mean should be in ballpark
      original_mean = Nx.mean(original) |> Nx.to_number()
      recovered_mean = Nx.mean(recovered) |> Nx.to_number()

      assert abs(original_mean - recovered_mean) / abs(original_mean) < 0.5
    end
  end

  describe "Quantization.quantized?/1" do
    test "returns true for quantized tensors" do
      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      emlx_weight = EMLX.Backend.from_nx(weight)
      {q_weight, scales, biases} = EMLX.quantize(emlx_weight, 64, 4)

      qt = Quantization.tensor(q_weight, scales, biases, {64, 64})

      assert Quantization.quantized?(qt)
    end

    test "returns false for regular tensors" do
      tensor = Nx.iota({4, 4}, type: :f32)
      tensor = Nx.backend_transfer(tensor, {EMLX.Backend, device: :gpu})

      refute Quantization.quantized?(tensor)
    end

    test "returns false for non-tensors" do
      refute Quantization.quantized?(nil)
      refute Quantization.quantized?(%{})
      refute Quantization.quantized?("not a tensor")
    end
  end

  describe "Quantization.options/1" do
    test "returns options map for quantized tensor" do
      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      emlx_weight = EMLX.Backend.from_nx(weight)
      {q_weight, scales, biases} = EMLX.quantize(emlx_weight, 64, 4)

      qt = Quantization.tensor(q_weight, scales, biases, {64, 64})

      opts = Quantization.options(qt)

      assert is_map(opts)
      assert Map.has_key?(opts, :scales)
      assert Map.has_key?(opts, :biases)
      assert Map.has_key?(opts, :group_size)
      assert opts.group_size == 64
    end

    test "returns nil for regular tensors" do
      tensor = Nx.iota({4, 4}, type: :f32)
      tensor = Nx.backend_transfer(tensor, {EMLX.Backend, device: :gpu})

      assert Quantization.options(tensor) == nil
    end

    test "returns nil for non-tensors" do
      assert Quantization.options(nil) == nil
      assert Quantization.options(%{}) == nil
    end
  end

  describe "Nx.dot integration" do
    test "Nx.dot automatically dispatches to quantized_matmul" do
      # Create input tensor
      input = Nx.iota({1, 4, 64}, type: :f32) |> Nx.divide(100)
      input = Nx.backend_transfer(input, {EMLX.Backend, device: :gpu})

      # Create and quantize weight
      weight = Nx.iota({128, 64}, type: :f32) |> Nx.divide(1000)
      {q_weight, scales, biases} = Quantization.quantize(weight)
      qt = Quantization.tensor(q_weight, scales, biases, {128, 64})

      # Nx.dot should work transparently
      result = Nx.dot(input, [2], qt, [1])

      assert Nx.shape(result) == {1, 4, 128}
    end

    test "quantized dot produces reasonable results" do
      input = Nx.iota({1, 4, 64}, type: :f32) |> Nx.divide(100)
      input = Nx.backend_transfer(input, {EMLX.Backend, device: :gpu})

      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      weight_gpu = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      # Full precision reference
      expected = Nx.dot(input, [2], Nx.transpose(weight_gpu), [1])

      # Quantized path
      {q_weight, scales, biases} = Quantization.quantize(weight_gpu)
      qt = Quantization.tensor(q_weight, scales, biases, {64, 64})
      result = Nx.dot(input, [2], qt, [1])

      # Both should produce positive values of similar magnitude
      expected_mean = Nx.mean(expected) |> Nx.to_number()
      result_mean = Nx.mean(result) |> Nx.to_number()

      assert expected_mean > 0
      assert result_mean > 0
      assert result_mean / expected_mean > 0.1
      assert result_mean / expected_mean < 10
    end
  end

  describe "end-to-end workflow" do
    test "complete quantization workflow" do
      # 1. Create a weight matrix (using iota for determinism)
      weight = Nx.iota({256, 128}, type: :f32) |> Nx.divide(1000)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      # 2. Quantize it
      {q_weight, scales, biases} = Quantization.quantize(weight, group_size: 64, bits: 4)

      # 3. Create quantized tensor for Nx operations
      qt = Quantization.tensor(q_weight, scales, biases, {256, 128}, group_size: 64, bits: 4)

      # 4. Verify it's marked as quantized
      assert Quantization.quantized?(qt)
      assert Nx.type(qt) == {:s, 4}

      # 5. Use with Nx.dot
      input = Nx.iota({1, 8, 128}, type: :f32) |> Nx.divide(100)
      input = Nx.backend_transfer(input, {EMLX.Backend, device: :gpu})

      result = Nx.dot(input, [2], qt, [1])
      assert Nx.shape(result) == {1, 8, 256}

      # 6. Optionally dequantize for debugging
      dequantized = Quantization.dequantize(q_weight, scales, biases, group_size: 64, bits: 4)
      dequant_nx = EMLX.Backend.to_nx(dequantized)
      assert Nx.shape(dequant_nx) == {256, 128}
    end

    test "LLM-style inference pattern" do
      # Simulate a transformer linear layer:
      # hidden_states @ weight.T where weight is quantized

      batch_size = 1
      seq_len = 4
      hidden_dim = 128
      output_dim = 256

      # Hidden states from previous layer (using iota for determinism)
      hidden = Nx.iota({batch_size, seq_len, hidden_dim}, type: :f32) |> Nx.divide(100)
      hidden = Nx.backend_transfer(hidden, {EMLX.Backend, device: :gpu})

      # Quantized projection weight
      weight = Nx.iota({output_dim, hidden_dim}, type: :f32) |> Nx.divide(1000)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      {q_weight, scales, biases} = Quantization.quantize(weight)
      qt = Quantization.tensor(q_weight, scales, biases, {output_dim, hidden_dim})

      # Forward pass: hidden @ weight.T
      output = Nx.dot(hidden, [2], qt, [1])

      assert Nx.shape(output) == {batch_size, seq_len, output_dim}
      assert Nx.type(output) == {:f, 32}  # Output is float, not quantized
    end
  end
end
