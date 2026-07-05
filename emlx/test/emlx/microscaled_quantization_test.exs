defmodule EMLX.MicroscaledQuantizationTest do
  use EMLX.Case, async: true

  @moduletag :metal

  alias EMLX.Quantization

  # {mode, group_size, bits}. Each microscaled mode pins an exact
  # {group_size, bits} pair (mlx::core::quantize's mode-specific constraint).
  @modes [
    {"affine", 64, 4},
    {"mxfp4", 32, 4},
    {"mxfp8", 32, 8},
    {"nvfp4", 16, 4}
  ]

  describe "EMLX.quantize/2 per-mode" do
    for {mode, group_size, bits} <- @modes do
      test "#{mode}: round-trips through EMLX.quantize/EMLX.dequantize" do
        mode = unquote(mode)
        group_size = unquote(group_size)
        bits = unquote(bits)

        weight = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
        weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

        qw = EMLX.quantize(weight, type: {:s, bits}, group_size: group_size, mode: mode)

        %EMLX.Backend{quantization_config: cfg} = qw.data
        assert cfg.mode == mode
        assert cfg.group_size == group_size
        assert cfg.bits == bits

        if mode == "affine" do
          assert %Nx.Tensor{} = cfg.biases
        else
          assert cfg.biases == nil
        end

        dense = EMLX.dequantize(qw)
        assert Nx.shape(dense) == {64, 64}

        original = Nx.backend_transfer(weight, Nx.BinaryBackend)
        recovered = Nx.backend_transfer(dense, Nx.BinaryBackend)

        original_mean = Nx.mean(original) |> Nx.to_number()
        recovered_mean = Nx.mean(recovered) |> Nx.to_number()

        # Lossy quantization (esp. 4-bit) — assert same ballpark, not exact match.
        assert abs(original_mean - recovered_mean) / original_mean < 0.5
      end
    end

    test "raises on an unknown :mode" do
      weight = Nx.iota({64, 64}, type: :f32)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      assert_raise ArgumentError, ~r/:mode must be one of/, fn ->
        EMLX.quantize(weight, mode: "not_a_real_mode")
      end
    end

    for {mode, expected_gs, expected_bits} <- Enum.reject(@modes, &(elem(&1, 0) == "affine")) do
      test "#{mode}: raises when group_size doesn't match the mode's fixed constraint" do
        mode = unquote(mode)
        expected_gs = unquote(expected_gs)
        expected_bits = unquote(expected_bits)
        wrong_gs = if expected_gs == 64, do: 32, else: 64

        weight = Nx.iota({64, 128}, type: :f32)
        weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

        assert_raise ArgumentError, ~r/requires group_size=#{expected_gs}/, fn ->
          EMLX.quantize(weight, mode: mode, group_size: wrong_gs, type: {:s, expected_bits})
        end
      end
    end
  end

  describe "EMLX.quantized_matmul/2 per-mode vs dense dot" do
    for {mode, group_size, bits} <- @modes do
      test "#{mode}: matches Nx.dot(x, transpose(dense)) within quantization tolerance" do
        mode = unquote(mode)
        group_size = unquote(group_size)
        bits = unquote(bits)

        x = Nx.iota({1, 4, 128}, type: :f32) |> Nx.divide(100)
        x = Nx.backend_transfer(x, {EMLX.Backend, device: :gpu})

        w = Nx.iota({64, 128}, type: :f32) |> Nx.divide(1000)
        w = Nx.backend_transfer(w, {EMLX.Backend, device: :gpu})

        qw = EMLX.quantize(w, type: {:s, bits}, group_size: group_size, mode: mode)
        dense = EMLX.dequantize(qw)

        # dense (and qw) is {out_features, in_features}; quantized_matmul's
        # transpose: true means x @ dense^T, i.e. contract x's last axis
        # directly against dense's in_features axis (no explicit transpose).
        expected = Nx.dot(x, [2], dense, [1])
        result = EMLX.quantized_matmul(x, qw)

        assert Nx.shape(result) == {1, 4, 64}

        # quantized_matmul must agree with dequantize-then-dense-dot using the
        # *same* quantized weight (not the pre-quantization original) — this
        # isolates matmul correctness from quantization lossiness.
        assert Nx.all_close(result, expected, atol: 1.0e-2, rtol: 1.0e-2) |> Nx.to_number() == 1
      end
    end
  end

  describe "transparent Nx.dot dispatch per-mode" do
    for {mode, group_size, bits} <- @modes do
      test "#{mode}: Nx.dot dispatches to quantized_matmul for a quantized right operand" do
        mode = unquote(mode)
        group_size = unquote(group_size)
        bits = unquote(bits)

        x = Nx.iota({1, 4, 128}, type: :f32) |> Nx.divide(100)
        x = Nx.backend_transfer(x, {EMLX.Backend, device: :gpu})

        w = Nx.iota({64, 128}, type: :f32) |> Nx.divide(1000)
        w = Nx.backend_transfer(w, {EMLX.Backend, device: :gpu})

        qw = EMLX.quantize(w, type: {:s, bits}, group_size: group_size, mode: mode)

        result_dot = Nx.dot(x, [2], qw, [1])
        result_explicit = EMLX.quantized_matmul(x, qw)

        assert Nx.shape(result_dot) == {1, 4, 64}
        assert Nx.all_close(result_dot, result_explicit, atol: 1.0e-4) |> Nx.to_number() == 1
      end
    end
  end

  describe "EMLX.Quantization (runtime_call deftransform) per-mode" do
    for {mode, group_size, bits} <- @modes do
      test "#{mode}: Quantization.quantize/2 + dequantize/1 + quantized_matmul/2 agree with the eager path" do
        mode = unquote(mode)
        group_size = unquote(group_size)
        bits = unquote(bits)

        x = Nx.iota({1, 4, 128}, type: :f32) |> Nx.divide(100)
        x = Nx.backend_transfer(x, {EMLX.Backend, device: :gpu})

        w = Nx.iota({64, 128}, type: :f32) |> Nx.divide(1000)
        w = Nx.backend_transfer(w, {EMLX.Backend, device: :gpu})

        qw_eager = EMLX.quantize(w, type: {:s, bits}, group_size: group_size, mode: mode)
        qw_defn = Quantization.quantize(w, type: {:s, bits}, group_size: group_size, mode: mode)

        assert Quantization.quantized?(qw_defn)

        dequant_eager = EMLX.dequantize(qw_eager)
        dequant_defn = Quantization.dequantize(qw_defn)
        assert Nx.all_close(dequant_eager, dequant_defn, atol: 1.0e-4) |> Nx.to_number() == 1

        matmul_eager = EMLX.quantized_matmul(x, qw_eager)
        matmul_defn = Quantization.quantized_matmul(x, qw_defn)
        assert Nx.all_close(matmul_eager, matmul_defn, atol: 1.0e-4) |> Nx.to_number() == 1
      end
    end
  end
end
