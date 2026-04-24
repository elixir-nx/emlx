defmodule EMLX.BackendQuantizationTest do
  @moduledoc """
  Tests for the EMLX.Backend quantization path — struct annotation, transparent
  Nx.dot dispatch, and edge cases.
  """
  use EMLX.Case, async: true

  alias EMLX.Quantization

  describe "EMLX.Backend struct" do
    test "has :quantization_config field, nil by default" do
      backend = %EMLX.Backend{}
      assert Map.has_key?(backend, :quantization_config)
      assert is_nil(backend.quantization_config)
    end

    test "Quantization.quantized?/1 returns false for regular Nx.Tensor" do
      tensor = Nx.iota({4, 4}, type: :f32)
      emlx_tensor = Nx.backend_transfer(tensor, {EMLX.Backend, device: :gpu})
      refute Quantization.quantized?(emlx_tensor)
    end

    test "quantize/2 sets quantization_config on Backend" do
      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      qw = EMLX.quantize(weight, [])

      assert %EMLX.Backend{quantization_config: %EMLX.Quantization.Config{} = cfg} = qw.data
      assert cfg.bits == 4
      assert cfg.group_size == 64
    end
  end

  describe "transparent Nx.dot dispatch" do
    test "produces output with correct shape (right quantized)" do
      input = Nx.iota({1, 4, 64}, type: :f32) |> Nx.divide(100)
      input = Nx.backend_transfer(input, {EMLX.Backend, device: :gpu})

      weight = Nx.iota({128, 64}, type: :f32) |> Nx.divide(1000)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})
      qw = EMLX.quantize(weight, [])

      result = Nx.dot(input, [2], qw, [1])

      assert Nx.shape(result) == {1, 4, 128}
    end

    test "matches direct NIF call" do
      input = Nx.iota({1, 4, 64}, type: :f32) |> Nx.divide(100)
      input_gpu = Nx.backend_transfer(input, {EMLX.Backend, device: :gpu})
      emlx_input = EMLX.Backend.from_nx(input_gpu)

      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      weight_gpu = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      qw = EMLX.quantize(weight_gpu, [])
      result_dot = Nx.dot(input_gpu, [2], qw, [1])

      # Re-quantize for the NIF call (refs are per-call)
      emlx_weight = EMLX.Backend.from_nx(weight_gpu)
      {q_w, scales, biases} = EMLX.quantize(emlx_weight, 64, 4)
      result_nif = EMLX.quantized_matmul(emlx_input, q_w, scales, biases, true, 64, 4)
      result_nif_nx = EMLX.Backend.to_nx(result_nif)

      assert Nx.all_close(result_dot, result_nif_nx, atol: 1.0e-4) |> Nx.to_number() == 1
    end

    test "produces reasonable values compared to full-precision dot" do
      input = Nx.iota({1, 4, 64}, type: :f32) |> Nx.divide(100)
      input_gpu = Nx.backend_transfer(input, {EMLX.Backend, device: :gpu})

      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      weight_gpu = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      expected = Nx.dot(input_gpu, [2], Nx.transpose(weight_gpu), [1])

      qw = EMLX.quantize(weight_gpu, [])
      result = Nx.dot(input_gpu, [2], qw, [1])

      expected_mean = Nx.mean(expected) |> Nx.to_number()
      result_mean = Nx.mean(result) |> Nx.to_number()

      assert result_mean > 0
      assert expected_mean > 0
      ratio = result_mean / expected_mean
      assert ratio > 0.1 and ratio < 10
    end

    test "raises when both dot operands are quantized" do
      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})
      qw = EMLX.quantize(weight, [])

      assert_raise ArgumentError, fn ->
        Nx.dot(qw, [1], qw, [0])
      end
    end
  end

  describe "LoRA with quantized base weight" do
    test "LoRA addition works with quantized base via Nx.dot" do
      input = Nx.iota({1, 4, 64}, type: :f32) |> Nx.divide(100)
      input_gpu = Nx.backend_transfer(input, {EMLX.Backend, device: :gpu})

      base_weight = Nx.iota({128, 64}, type: :f32) |> Nx.divide(1000)
      base_weight_gpu = Nx.backend_transfer(base_weight, {EMLX.Backend, device: :gpu})
      base_qw = EMLX.quantize(base_weight_gpu, [])

      lora_a = Nx.iota({64, 8}, type: :f32) |> Nx.divide(100)
      lora_a = Nx.backend_transfer(lora_a, {EMLX.Backend, device: :gpu})
      lora_b = Nx.iota({8, 128}, type: :f32) |> Nx.divide(100)
      lora_b = Nx.backend_transfer(lora_b, {EMLX.Backend, device: :gpu})
      scaling = 20.0

      # Transparent Nx.dot dispatch for the quantized base
      base_out = Nx.dot(input_gpu, [2], base_qw, [1])

      # LoRA: standard dense path
      lora_intermediate = Nx.dot(input_gpu, [2], lora_a, [0])
      lora_out = Nx.dot(lora_intermediate, [2], lora_b, [0])
      lora_scaled = Nx.multiply(lora_out, scaling)

      combined = Nx.add(base_out, lora_scaled)

      assert Nx.shape(combined) == {1, 4, 128}

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
      a = Nx.iota({4, 3}, type: :f32)
      b = Nx.iota({3, 5}, type: :f32)
      b_gpu = Nx.backend_transfer(b, {EMLX.Backend, device: :gpu})

      result = Nx.dot(a, b_gpu)

      assert Nx.shape(result) == {4, 5}
    end
  end
end
