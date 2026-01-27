defmodule EMLX.Quantization do
  @moduledoc """
  Utilities for creating and working with quantized tensors on Apple Silicon.

  This module provides the primary user-facing API for 4-bit and 8-bit
  quantization, enabling efficient LLM inference with MLX.

  ## Why Quantize Data?

  Large language models like Qwen3-8B or LLaMA-7B require 16GB+ of memory
  at float16 precision. 4-bit quantization reduces this to ~4-5GB while
  maintaining reasonable quality, enabling inference on consumer hardware.

  Performance on Apple M-series:
  - **Memory**: 4-5GB vs 16GB for fp16
  - **Speed**: ~135 tok/s with quantized_matmul
  - **Quality**: ~95% of fp16 perplexity for most tasks

  ## MLX 4-bit Format

  MLX uses group-wise affine quantization:

      dequantized[i] = scales[i/group_size] * (packed_int4[i] - biases[i/group_size])

  Weights are packed as uint32 (8 int4 values per uint32). With `group_size=64`:
  - Weight `[out, in]` becomes `[out, in/8]` as uint32
  - Scales: `[out, in/group_size]` as bfloat16
  - Biases: `[out, in/group_size]` as bfloat16

  ## Basic Usage

      # 1. Quantize a weight matrix
      weight = Nx.iota({512, 4096}, type: :f32)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})

      {q_weight, scales, biases} = EMLX.Quantization.quantize(weight)

      # 2. Create a quantized tensor for Nx operations
      qt = EMLX.Quantization.tensor(q_weight, scales, biases, {512, 4096})

      # 3. Use with standard Nx.dot - automatically dispatches to quantized_matmul
      input = Nx.iota({1, 8, 4096}, type: :f32)
      result = Nx.dot(input, [2], qt, [1])  # [1, 8, 512]

  ## Transparent Nx Integration

  Quantized tensors work with standard Nx operations. The EMLX backend
  detects quantization metadata and dispatches to optimized kernels:

      # This calls EMLX.quantized_matmul under the hood
      result = Nx.dot(input, quantized_weight)

  The tensor type `{:s, 4}` indicates 4-bit signed quantization.
  Bits are derived from the type, not stored separately.

  ## Loading Pre-quantized Models

  For models already in MLX 4-bit format (e.g., from Hugging Face):

      # Load from safetensors
      weight = load_tensor("model.layers.0.self_attn.q_proj.weight")
      scales = load_tensor("model.layers.0.self_attn.q_proj.scales")
      biases = load_tensor("model.layers.0.self_attn.q_proj.biases")

      # Convert to EMLX refs
      w_ref = EMLX.Backend.from_nx(weight)
      s_ref = EMLX.Backend.from_nx(scales)
      b_ref = EMLX.Backend.from_nx(biases)

      # Create quantized tensor
      qt = EMLX.Quantization.tensor(w_ref, s_ref, b_ref, {out_dim, in_dim})

  ## Debugging with Dequantization

      # Dequantize to verify values
      {q_weight, scales, biases} = EMLX.Quantization.quantize(weight)
      recovered = EMLX.Quantization.dequantize(q_weight, scales, biases)

      # Compare (4-bit is lossy, ~5-10% error typical)
      original_mean = Nx.mean(weight) |> Nx.to_number()
      recovered_mean = Nx.mean(EMLX.Backend.to_nx(recovered)) |> Nx.to_number()

  ## Options

  - `:bits` - 4 or 8 (default: 4). 4-bit is more memory efficient, 8-bit is more accurate.
  - `:group_size` - Number of weights sharing a scale factor (default: 64).
    Smaller groups = better accuracy but more overhead.

  ## See Also

  - `EMLX.quantize/3` - Low-level quantization NIF
  - `EMLX.dequantize/5` - Low-level dequantization NIF
  - `EMLX.quantized_matmul/7` - Low-level quantized matrix multiply NIF
  - `EMLX.Backend` - Backend struct with quantization fields
  """

  alias Nx.Tensor, as: T

  @doc """
  Creates a quantized Nx.Tensor from packed weights and scales/biases.

  The returned tensor has type `{:s, bits}` and can be used directly with
  `Nx.dot`, which will automatically dispatch to `quantized_matmul`.

  ## Parameters

  - `weight_ref` - EMLX device ref for packed uint32 weights
  - `scales_ref` - EMLX device ref for per-group scale factors
  - `biases_ref` - EMLX device ref for per-group zero points
  - `original_shape` - Shape before quantization `{out_features, in_features}`
  - `opts` - Options:
    - `:bits` - Bits per weight (default 4)
    - `:group_size` - Quantization group size (default 64)

  ## Example

      {q_weight, scales, biases} = EMLX.Quantization.quantize(weight)
      qt = EMLX.Quantization.tensor(q_weight, scales, biases, {512, 4096})

      # Use with standard Nx operations
      result = Nx.dot(input, qt)
  """
  def tensor(weight_ref, scales_ref, biases_ref, original_shape, opts \\ []) do
    EMLX.Backend.quantized_tensor(weight_ref, scales_ref, biases_ref, original_shape, opts)
  end

  @doc """
  Quantizes a float tensor to packed 4-bit or 8-bit format.

  Returns `{quantized_weights, scales, biases}` where:
  - `quantized_weights` - Packed uint32 tensor (8 int4 values per uint32 for 4-bit)
  - `scales` - Per-group scale factors (bfloat16)
  - `biases` - Per-group zero points (bfloat16)

  ## Parameters

  - `tensor` - Float tensor to quantize (Nx.Tensor or EMLX device ref)
  - `opts` - Options:
    - `:group_size` - Number of weights sharing a scale factor (default 64)
    - `:bits` - Bits per weight, 4 or 8 (default 4)

  ## Example

      weight = Nx.iota({512, 4096}, type: :f32)
      {q_weight, scales, biases} = EMLX.Quantization.quantize(weight)

      # With custom options
      {q_weight, scales, biases} = EMLX.Quantization.quantize(weight, group_size: 128, bits: 4)
  """
  def quantize(tensor, opts \\ [])

  def quantize(%T{} = tensor, opts) do
    group_size = Keyword.get(opts, :group_size, 64)
    bits = Keyword.get(opts, :bits, 4)

    tensor
    |> EMLX.Backend.from_nx()
    |> EMLX.quantize(group_size, bits)
  end

  def quantize({device, ref} = device_ref, opts) when is_atom(device) and is_reference(ref) do
    group_size = Keyword.get(opts, :group_size, 64)
    bits = Keyword.get(opts, :bits, 4)

    EMLX.quantize(device_ref, group_size, bits)
  end

  @doc """
  Dequantizes packed weights back to floating point.

  Useful for debugging and verification. The dequantized values will be
  approximate due to quantization loss.

  ## Parameters

  - `weights` - Packed uint32 weights (EMLX device ref)
  - `scales` - Per-group scale factors (EMLX device ref)
  - `biases` - Per-group zero points (EMLX device ref)
  - `opts` - Options:
    - `:group_size` - Quantization group size (default 64)
    - `:bits` - Bits per weight (default 4)

  ## Example

      {q_weight, scales, biases} = EMLX.Quantization.quantize(weight)
      recovered = EMLX.Quantization.dequantize(q_weight, scales, biases)

      # Check roundtrip accuracy
      Nx.all_close(weight, recovered, atol: 0.1)
  """
  def dequantize(weights, scales, biases, opts \\ []) do
    group_size = Keyword.get(opts, :group_size, 64)
    bits = Keyword.get(opts, :bits, 4)

    EMLX.dequantize(weights, scales, biases, group_size, bits)
  end

  @doc """
  Returns true if the tensor is quantized.

  ## Example

      qt = EMLX.Quantization.tensor(q_weight, scales, biases, {512, 4096})
      EMLX.Quantization.quantized?(qt)  #=> true

      regular = Nx.iota({4, 4})
      EMLX.Quantization.quantized?(regular)  #=> false
  """
  def quantized?(%T{} = tensor) do
    EMLX.Backend.quantized?(tensor)
  end

  def quantized?(_), do: false

  @doc """
  Gets quantization options from a tensor, or nil if not quantized.

  Returns a map with `:scales`, `:biases`, and `:group_size`.
  The bit width is derived from the tensor type (`{:s, 4}` or `{:s, 8}`).

  ## Example

      qt = EMLX.Quantization.tensor(q_weight, scales, biases, {512, 4096}, bits: 4)

      opts = EMLX.Quantization.options(qt)
      opts.group_size  #=> 64

      Nx.type(qt)  #=> {:s, 4}
  """
  def options(%T{} = tensor) do
    EMLX.Backend.quantization_options(tensor)
  end

  def options(_), do: nil
end
