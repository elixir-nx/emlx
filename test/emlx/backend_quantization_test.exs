defmodule EMLX.BackendQuantizationTest do
  @moduledoc """
  Tests for backend-level quantization support.

  This tests Paulo's suggested approach: storing quantization options
  in the EMLX.Backend struct so that Nx.dot automatically dispatches
  to quantized_matmul.
  """
  use EMLX.Case

  describe "EMLX.Backend.quantized_tensor/5" do
    test "creates Nx.Tensor with quantization options" do
      # Create and quantize a weight matrix
      weight = Nx.iota({128, 64}, type: :f32) |> Nx.divide(1000)
      emlx_weight = EMLX.Backend.from_nx(weight)

      {q_weight, scales, biases} = EMLX.quantize(emlx_weight, 64, 4)

      # Create quantized tensor via backend
      qt = EMLX.Backend.quantized_tensor(q_weight, scales, biases, {128, 64},
        bits: 4,
        group_size: 64
      )

      # Should be an Nx.Tensor
      assert %Nx.Tensor{} = qt

      # Should have quantization options in backend
      assert EMLX.Backend.quantized?(qt)
      opts = EMLX.Backend.quantization_options(qt)
      assert opts.group_size == 64

      # Bits is now derived from tensor type, not stored in quant_opts
      # Per Paulo's feedback: "you can remove :bits from quant opts given
      # that the tensor type carries that info"
      assert Nx.type(qt) == {:s, 4}
    end

    test "quantized?/1 returns false for regular tensors" do
      tensor = Nx.iota({4, 4}, type: :f32)
      emlx_tensor = Nx.backend_transfer(tensor, {EMLX.Backend, device: :gpu})

      refute EMLX.Backend.quantized?(emlx_tensor)
      assert EMLX.Backend.quantization_options(emlx_tensor) == nil
    end
  end

  describe "EMLX.quantized_tensor/5 convenience function" do
    test "delegates to Backend.quantized_tensor" do
      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(1000)
      emlx_weight = EMLX.Backend.from_nx(weight)
      {q_weight, scales, biases} = EMLX.quantize(emlx_weight, 64, 4)

      qt = EMLX.quantized_tensor(q_weight, scales, biases, {64, 64})

      assert %Nx.Tensor{} = qt
      assert EMLX.Backend.quantized?(qt)
    end
  end

  describe "Nx.dot with backend-quantized tensors" do
    test "automatically dispatches to quantized_matmul" do
      # Create input: [1, 4, 64]
      input = Nx.iota({1, 4, 64}, type: :f32) |> Nx.divide(100)
      input = Nx.backend_transfer(input, {EMLX.Backend, device: :gpu})

      # Create and quantize weight: [128, 64]
      weight = Nx.iota({128, 64}, type: :f32) |> Nx.divide(1000)
      emlx_weight = EMLX.Backend.from_nx(weight)
      {q_weight, scales, biases} = EMLX.quantize(emlx_weight, 64, 4)

      # Create quantized tensor with backend options
      qt = EMLX.quantized_tensor(q_weight, scales, biases, {128, 64})

      # Nx.dot should automatically use quantized_matmul
      result = Nx.dot(input, [2], qt, [1])

      # Verify output shape: [1, 4, 128]
      assert Nx.shape(result) == {1, 4, 128}
    end

    test "produces same results as direct quantized_matmul" do
      # Create input tensor and keep on GPU
      input = Nx.iota({1, 4, 64}, type: :f32) |> Nx.divide(100)
      input_gpu = Nx.backend_transfer(input, {EMLX.Backend, device: :gpu})
      emlx_input = EMLX.Backend.from_nx(input_gpu)

      # Create weight tensor on GPU
      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      weight_gpu = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})
      emlx_weight = EMLX.Backend.from_nx(weight_gpu)
      {q_weight, scales, biases} = EMLX.quantize(emlx_weight, 64, 4)

      # Method 1: Direct quantized_matmul
      result_direct = EMLX.quantized_matmul(emlx_input, q_weight, scales, biases, true, 64, 4)
      result_direct_nx = EMLX.Backend.to_nx(result_direct)

      # Method 2: Nx.dot with quantized tensor (need fresh quantization since refs are consumed)
      {q_weight2, scales2, biases2} = EMLX.quantize(emlx_weight, 64, 4)
      qt = EMLX.quantized_tensor(q_weight2, scales2, biases2, {64, 64})
      result_nx_dot = Nx.dot(input_gpu, [2], qt, [1])

      # Should produce identical results
      assert Nx.all_close(result_direct_nx, result_nx_dot, atol: 1.0e-6) |> Nx.to_number() == 1
    end

    test "produces reasonable values compared to full precision" do
      input = Nx.iota({1, 4, 64}, type: :f32) |> Nx.divide(100)
      input_gpu = Nx.backend_transfer(input, {EMLX.Backend, device: :gpu})

      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      weight_gpu = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      # Full precision: input @ weight.T
      expected = Nx.dot(input_gpu, [2], Nx.transpose(weight_gpu), [1])

      # Quantized path - get EMLX ref from GPU tensor
      emlx_weight = EMLX.Backend.from_nx(weight_gpu)
      {q_weight, scales, biases} = EMLX.quantize(emlx_weight, 64, 4)
      qt = EMLX.quantized_tensor(q_weight, scales, biases, {64, 64})
      result = Nx.dot(input_gpu, [2], qt, [1])

      # Compare means (4-bit quantization is approximate)
      expected_mean = Nx.mean(expected) |> Nx.to_number()
      result_mean = Nx.mean(result) |> Nx.to_number()

      # Should be in same order of magnitude
      assert result_mean > 0
      assert expected_mean > 0
      ratio = result_mean / expected_mean
      assert ratio > 0.1 and ratio < 10
    end
  end

  describe "LoRA with backend-quantized tensors" do
    test "LoRA addition works with quantized base" do
      # Input
      input = Nx.iota({1, 4, 64}, type: :f32) |> Nx.divide(100)
      input_gpu = Nx.backend_transfer(input, {EMLX.Backend, device: :gpu})

      # Quantized base weight
      base_weight = Nx.iota({128, 64}, type: :f32) |> Nx.divide(1000)
      emlx_base = EMLX.Backend.from_nx(base_weight)
      {q_base, scales, biases} = EMLX.quantize(emlx_base, 64, 4)

      base_qt = EMLX.quantized_tensor(q_base, scales, biases, {128, 64})

      # LoRA matrices (full precision on GPU)
      lora_a = Nx.iota({64, 8}, type: :f32) |> Nx.divide(100)
      lora_a = Nx.backend_transfer(lora_a, {EMLX.Backend, device: :gpu})
      lora_b = Nx.iota({8, 128}, type: :f32) |> Nx.divide(100)
      lora_b = Nx.backend_transfer(lora_b, {EMLX.Backend, device: :gpu})
      scaling = 20.0

      # Base output via Nx.dot (uses quantized_matmul)
      base_out = Nx.dot(input_gpu, [2], base_qt, [1])

      # LoRA output (full precision)
      lora_intermediate = Nx.dot(input_gpu, [2], lora_a, [0])
      lora_out = Nx.dot(lora_intermediate, [2], lora_b, [0])
      lora_scaled = Nx.multiply(lora_out, scaling)

      # Combined
      combined = Nx.add(base_out, lora_scaled)

      # Verify shape
      assert Nx.shape(combined) == {1, 4, 128}

      # Verify LoRA has an effect
      base_mean = Nx.mean(base_out) |> Nx.to_number()
      combined_mean = Nx.mean(combined) |> Nx.to_number()
      refute_in_delta base_mean, combined_mean, 1.0
    end
  end

  describe "edge cases" do
    test "non-quantized Nx.dot still works" do
      a = Nx.iota({4, 3}, type: :f32)
      b = Nx.iota({3, 5}, type: :f32)

      a_gpu = Nx.backend_transfer(a, {EMLX.Backend, device: :gpu})
      b_gpu = Nx.backend_transfer(b, {EMLX.Backend, device: :gpu})

      result = Nx.dot(a_gpu, b_gpu)

      assert Nx.shape(result) == {4, 5}
    end

    test "mixed backends still work" do
      a = Nx.iota({4, 3}, type: :f32)  # Binary backend
      b = Nx.iota({3, 5}, type: :f32)
      b_gpu = Nx.backend_transfer(b, {EMLX.Backend, device: :gpu})

      # This should transfer a to EMLX and compute
      result = Nx.dot(a, b_gpu)

      assert Nx.shape(result) == {4, 5}
    end
  end
end
