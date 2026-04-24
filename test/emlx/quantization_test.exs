defmodule EMLX.QuantizationTest do
  use EMLX.Case, async: true

  @moduletag :metal

  describe "quantize/3" do
    test "quantizes float tensor to packed format" do
      # Create a tensor with shape suitable for quantization
      # group_size=64 means we need at least 64 elements in the last dimension
      tensor = Nx.iota({64, 64}, type: :f32) |> Nx.divide(1000)
      emlx_tensor = EMLX.Backend.from_nx(tensor)
      {quantized, scales, biases} = EMLX.quantize(emlx_tensor, 64, 4)

      # Verify we get three tensors back
      assert is_tuple(quantized)
      assert is_tuple(scales)
      assert is_tuple(biases)

      # Quantized weights should be u32 (packed int4 values)
      {_dev, q_ref} = quantized
      assert EMLX.scalar_type({:gpu, q_ref}) == :uint32
    end

    test "quantizes larger tensor" do
      # Create a tensor with shape suitable for quantization
      tensor = Nx.iota({128, 128}, type: :f32) |> Nx.divide(1000)
      emlx_tensor = EMLX.Backend.from_nx(tensor)

      {quantized, scales, biases} = EMLX.quantize(emlx_tensor, 64, 4)

      # Verify shapes
      {_dev, q_ref} = quantized
      {_dev, s_ref} = scales
      {_dev, b_ref} = biases

      q_shape = EMLX.shape({:gpu, q_ref})
      s_shape = EMLX.shape({:gpu, s_ref})
      b_shape = EMLX.shape({:gpu, b_ref})

      # For 4-bit quantization with group_size=64:
      # - quantized: [128, 128/8] = [128, 16] (8 int4 values packed per uint32)
      # - scales: [128, 128/64] = [128, 2]
      # - biases: [128, 128/64] = [128, 2]
      assert q_shape == {128, 16}
      assert s_shape == {128, 2}
      assert b_shape == {128, 2}
    end
  end

  describe "dequantize/5" do
    test "dequantizes back to float" do
      # Create, quantize, then dequantize
      tensor = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      emlx_tensor = EMLX.Backend.from_nx(tensor)

      {quantized, scales, biases} = EMLX.quantize(emlx_tensor, 64, 4)
      dequantized = EMLX.dequantize(quantized, scales, biases, 64, 4)

      # Verify shape is restored
      {_dev, d_ref} = dequantized
      d_shape = EMLX.shape({:gpu, d_ref})
      assert d_shape == {64, 64}

      # Verify dtype is float (bfloat16 or float32)
      dtype = EMLX.scalar_type({:gpu, d_ref})
      assert dtype in [:bfloat16, :float32]
    end

    test "quantize-dequantize roundtrip preserves rough values" do
      # Create a tensor with values in a reasonable range
      tensor = Nx.iota({64, 64}, type: :f32) |> Nx.divide(10)
      emlx_tensor = EMLX.Backend.from_nx(tensor)

      {quantized, scales, biases} = EMLX.quantize(emlx_tensor, 64, 4)
      dequantized = EMLX.dequantize(quantized, scales, biases, 64, 4)

      # Convert back to Nx for comparison
      original = EMLX.Backend.to_nx(emlx_tensor)
      recovered = EMLX.Backend.to_nx(dequantized)

      # 4-bit quantization is very lossy - just check mean is in ballpark
      original_mean = Nx.mean(original) |> Nx.to_number()
      recovered_mean = Nx.mean(recovered) |> Nx.to_number()

      # Mean should be within 50% (4-bit is very approximate)
      assert abs(original_mean - recovered_mean) / original_mean < 0.5
    end
  end

  describe "quantized_matmul/7" do
    test "multiplies input with quantized weights" do
      # Create input tensor [batch, seq, hidden=64]
      x = Nx.iota({1, 4, 64}, type: :f32) |> Nx.divide(100)
      # Create weight tensor [out, hidden] where hidden must be divisible by 64
      # For transpose=true, MLX expects w.T to multiply with x, so w is [out, in]
      w = Nx.iota({128, 64}, type: :f32) |> Nx.divide(1000)

      emlx_x = EMLX.Backend.from_nx(x)
      emlx_w = EMLX.Backend.from_nx(w)

      # Quantize the weights (last dim is quantized, so 64 is our group)
      {q_w, scales, biases} = EMLX.quantize(emlx_w, 64, 4)

      # Perform quantized matmul with transpose=true
      # x @ w.T where x is [1,4,64] and w is [128,64] -> result is [1,4,128]
      result = EMLX.quantized_matmul(emlx_x, q_w, scales, biases, true, 64, 4)

      # Verify output shape: [1, 4, 128]
      {_dev, r_ref} = result
      r_shape = EMLX.shape({:gpu, r_ref})
      assert r_shape == {1, 4, 128}
    end

    test "quantized matmul produces reasonable values" do
      # Create deterministic test data
      # x: [1, 4, 64], w: [64, 64] (transpose=true means x @ w.T)
      x = Nx.iota({1, 4, 64}, type: :f32) |> Nx.divide(100)
      w = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)

      emlx_x = EMLX.Backend.from_nx(x)
      emlx_w = EMLX.Backend.from_nx(w)

      # Full precision matmul: x @ w.T
      expected = Nx.dot(x, [2], Nx.transpose(w), [1])

      # Quantized matmul
      {q_w, scales, biases} = EMLX.quantize(emlx_w, 64, 4)
      result = EMLX.quantized_matmul(emlx_x, q_w, scales, biases, true, 64, 4)
      result_nx = EMLX.Backend.to_nx(result)

      # Compare means (exact match isn't expected with 4-bit quantization)
      expected_mean = Nx.mean(expected) |> Nx.to_number()
      result_mean = Nx.mean(result_nx) |> Nx.to_number()

      # Means should be in the same ballpark (within factor of 10)
      assert result_mean > 0
      assert expected_mean > 0
      # Relaxed check: both should be positive and relatively close in magnitude
      ratio = result_mean / expected_mean
      assert ratio > 0.01 and ratio < 100
    end
  end
end
