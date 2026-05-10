defmodule EMLX.Quantization.ModuleTest do
  @moduledoc """
  Tests for the EMLX.Quantization module — quantize/2, quantize/2,
  dequantize/1, dequantize/1, quantized_matmul/2, quantized?/1.
  """
  use EMLX.Case, async: true

  @moduletag :metal

  alias EMLX.Quantization

  describe "EMLX.quantize/2" do
    test "quantizes a rank-2 float tensor with defaults" do
      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      qw = EMLX.quantize(weight, [])

      assert %Nx.Tensor{} = qw
      assert Nx.type(qw) == {:s, 4}
      assert Nx.shape(qw) == {64, 64}
      assert Quantization.quantized?(qw)
    end

    test "quantizes with custom type and group_size" do
      weight = Nx.iota({128, 128}, type: :f32) |> Nx.divide(100)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      qw = EMLX.quantize(weight, type: {:s, 8}, group_size: 128)

      %EMLX.Backend{quantization_config: cfg} = qw.data
      assert cfg.bits == 8
      assert cfg.group_size == 128
    end

    test "backend carries packed shape (4-bit, group_size=64)" do
      weight = Nx.iota({128, 128}, type: :f32) |> Nx.divide(100)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      qw = EMLX.quantize(weight, type: {:s, 4}, group_size: 64)

      %EMLX.Backend{shape: packed_shape, quantization_config: cfg} = qw.data
      # 4-bit: 8 values per uint32 → {128, 128/8} = {128, 16}
      assert packed_shape == {128, 16}
      # scales: {128, 128/64} = {128, 2}
      assert Nx.shape(cfg.scales) == {128, 2}
      assert Nx.shape(cfg.biases) == {128, 2}
    end

    test "raises on non-rank-2 tensor" do
      weight = Nx.iota({2, 64, 64}, type: :f32)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      assert_raise ArgumentError, ~r/rank-2/, fn ->
        EMLX.quantize(weight, [])
      end
    end

    test "raises when in_features not divisible by group_size" do
      weight = Nx.iota({64, 70}, type: :f32)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      assert_raise ArgumentError, ~r/divisible/, fn ->
        EMLX.quantize(weight, group_size: 64)
      end
    end
  end

  describe "Quantization.quantized?/1" do
    test "returns true for a quantized tensor" do
      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})
      qw = EMLX.quantize(weight, group_size: 64, bits: 4)

      assert Quantization.quantized?(qw)
    end

    test "returns false for a regular EMLX tensor" do
      tensor = Nx.iota({4, 4}, type: :f32)
      tensor = Nx.backend_transfer(tensor, {EMLX.Backend, device: :gpu})

      refute Quantization.quantized?(tensor)
    end

    test "returns false for non-tensor values" do
      refute Quantization.quantized?(nil)
      refute Quantization.quantized?(%{})
      refute Quantization.quantized?("not a tensor")
    end
  end

  describe "Quantization.quantized_matmul/2" do
    test "produces output with correct shape" do
      input = Nx.iota({1, 4, 64}, type: :f32) |> Nx.divide(100)
      input = Nx.backend_transfer(input, {EMLX.Backend, device: :gpu})

      weight = Nx.iota({128, 64}, type: :f32) |> Nx.divide(1000)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})
      qw = EMLX.quantize(weight, group_size: 64, bits: 4)

      result = Quantization.quantized_matmul(input, qw)

      assert Nx.shape(result) == {1, 4, 128}
    end

    test "raises ArgumentError when both operands are quantized" do
      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})
      qw = EMLX.quantize(weight, group_size: 64, bits: 4)

      assert_raise ArgumentError, ~r/two quantized tensors/, fn ->
        Quantization.quantized_matmul(qw, qw)
      end
    end

    test "produces values comparable to dense dot (within int4 tolerance)" do
      input = Nx.iota({1, 4, 64}, type: :f32) |> Nx.divide(100)
      input_gpu = Nx.backend_transfer(input, {EMLX.Backend, device: :gpu})

      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      weight_gpu = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      # Dense reference: input @ weight.T
      expected = Nx.dot(input_gpu, [2], Nx.transpose(weight_gpu), [1])

      qw = EMLX.quantize(weight_gpu, group_size: 64, bits: 4)
      result = Quantization.quantized_matmul(input_gpu, qw)

      expected_mean = Nx.mean(expected) |> Nx.to_number()
      result_mean = Nx.mean(result) |> Nx.to_number()

      assert expected_mean > 0
      assert result_mean > 0
      ratio = result_mean / expected_mean
      assert ratio > 0.1 and ratio < 10
    end
  end

  describe "Quantization.dequantize/1" do
    test "returns a float tensor of the correct shape" do
      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      qw = EMLX.quantize(weight, group_size: 64, bits: 4)
      result = Quantization.dequantize(qw)

      assert %Nx.Tensor{} = result
      assert Nx.shape(result) == {64, 64}
    end
  end

  describe "transparent Nx.dot dispatch" do
    test "Nx.dot dispatches to quantized_matmul when right operand is quantized" do
      input = Nx.iota({1, 4, 64}, type: :f32) |> Nx.divide(100)
      input = Nx.backend_transfer(input, {EMLX.Backend, device: :gpu})

      weight = Nx.iota({128, 64}, type: :f32) |> Nx.divide(1000)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})
      qw = EMLX.quantize(weight, group_size: 64, bits: 4)

      # Transparent dispatch: Nx.dot detects quantized right operand
      result = Nx.dot(input, [2], qw, [1])

      assert Nx.shape(result) == {1, 4, 128}
    end

    test "Nx.dot result matches explicit quantized_matmul" do
      input = Nx.iota({1, 4, 64}, type: :f32) |> Nx.divide(100)
      input = Nx.backend_transfer(input, {EMLX.Backend, device: :gpu})

      weight = Nx.iota({128, 64}, type: :f32) |> Nx.divide(1000)
      weight_gpu = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      qw = EMLX.quantize(weight_gpu, group_size: 64, bits: 4)
      result_explicit = Quantization.quantized_matmul(input, qw)

      qw2 = EMLX.quantize(weight_gpu, group_size: 64, bits: 4)
      result_dot = Nx.dot(input, [2], qw2, [1])

      assert Nx.all_close(result_explicit, result_dot, atol: 1.0e-4) |> Nx.to_number() == 1
    end

    test "raises when left operand is quantized" do
      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})
      qw = EMLX.quantize(weight, group_size: 64, bits: 4)

      assert_raise ArgumentError, ~r/quantized left operand/, fn ->
        Nx.dot(qw, [1], qw, [0])
      end
    end
  end

  describe "runtime_call deftransform variants" do
    test "dequantize/1 matches dequantize/1" do
      weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      qw = EMLX.quantize(weight, group_size: 64, bits: 4)
      qw2 = EMLX.quantize(weight, [])

      via_mx = EMLX.dequantize(qw)
      via_runtime = Quantization.dequantize(qw2)

      assert Nx.shape(via_mx) == Nx.shape(via_runtime)
      assert Nx.all_close(via_mx, via_runtime, atol: 1.0e-4) |> Nx.to_number() == 1
    end

    test "quantize/2 returns a quantized tensor with the same logical shape" do
      weight = Nx.iota({128, 64}, type: :f32) |> Nx.divide(100)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      qw = Quantization.quantize(weight, group_size: 64, bits: 4)

      assert Nx.type(qw) == {:s, 4}
      assert Nx.shape(qw) == {128, 64}
      assert Quantization.quantized?(qw)
    end

    test "quantized_matmul/2 matches eager quantized_matmul/2" do
      input = Nx.iota({1, 4, 64}, type: :f32) |> Nx.divide(100)
      input = Nx.backend_transfer(input, {EMLX.Backend, device: :gpu})

      weight = Nx.iota({128, 64}, type: :f32) |> Nx.divide(1000)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      qw_eager = EMLX.quantize(weight, group_size: 64, bits: 4)
      qw_defn = EMLX.quantize(weight, [])

      result_eager = EMLX.quantized_matmul(input, qw_eager)
      result_defn = Quantization.quantized_matmul(input, qw_defn)

      assert Nx.shape(result_defn) == {1, 4, 128}
      assert Nx.all_close(result_eager, result_defn, atol: 1.0e-4) |> Nx.to_number() == 1
    end
  end

  describe "end-to-end workflow" do
    test "quantize → Nx.dot → reasonable output" do
      batch_size = 1
      seq_len = 4
      hidden_dim = 128
      output_dim = 256

      hidden = Nx.iota({batch_size, seq_len, hidden_dim}, type: :f32) |> Nx.divide(100)
      hidden = Nx.backend_transfer(hidden, {EMLX.Backend, device: :gpu})

      weight = Nx.iota({output_dim, hidden_dim}, type: :f32) |> Nx.divide(1000)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      qw = EMLX.quantize(weight, [])
      output = Nx.dot(hidden, [2], qw, [1])

      assert Nx.shape(output) == {batch_size, seq_len, output_dim}
    end
  end
end
