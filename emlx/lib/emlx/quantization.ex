defmodule EMLX.Quantization do
  @moduledoc """
  Affine group-wise int2/int4/int8 quantization for Apple Silicon inference.

  Quantized weights are represented as annotated `Nx.Tensor` values — the
  tensor carries the original logical shape and type (e.g. `{:s, 4}` for
  4-bit), while the `EMLX.Backend` struct stores the packed uint32 data and
  a `EMLX.Quantization.Config` with scales, biases, group_size, and bits.

  `Nx.dot` automatically dispatches to `mx::quantized_matmul` when it detects
  a quantized operand on `EMLX.Backend` — no explicit call site changes needed.

  ## Basic usage

      # Quantize a dense weight
      weight = Nx.iota({512, 4096}, type: :f32)
      weight = Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu})
      qw = EMLX.Quantization.from_dense(weight)

      # Standard Nx.dot dispatches to mx::quantized_matmul automatically
      input = Nx.iota({1, 8, 4096}, type: :f32)
      result = Nx.dot(input, [2], qw, [1])

  ## Inside defn

  `quantize/2`, `dequantize/1`, and `quantized_matmul/2` are all
  `deftransform` functions backed by `Nx.runtime_call`, so they are safe
  to call inside `Nx.Defn.jit`-traced forward passes:

      defn my_layer(x, qw) do
        dense = EMLX.Quantization.dequantize(qw)
        Nx.dot(x, [2], dense, [1])
      end

  `Nx.dot` with a quantized tensor also dispatches transparently via
  `EMLX.Backend.dot/7` at execution time, so explicit `dequantize` is
  only needed when you want the dense weight for a non-dot operation.

  ## See also

  * `EMLX.Quantization.Config` — internal metadata struct
  """

  import Nx.Defn
  alias EMLX.Quantization.Config

  @doc """
  Quantize a dense 2-D tensor via `Nx.runtime_call`.

  At execution time the callback receives the real tensor and runs
  `mx::quantize`. Returns an annotated quantized `Nx.Tensor` with the same
  logical shape as the input and type `{:s, N}`. Safe to call inside
  `Nx.Defn.jit`-traced forward passes.

  ## Options

  * `:type` — Nx storage type: `{:s, 2}`, `{:s, 4}` (default), or `{:s, 8}`.
  * `:group_size` — 32, 64, or 128 (default 64). Must evenly divide the last
    dimension of `tensor`.
  """
  deftransform quantize(tensor, opts \\ []) do
    type = Keyword.get(opts, :type, {:s, 4})
    out = Nx.template(Nx.shape(tensor), type)
    Nx.runtime_call(out, tensor, opts, &__MODULE__.quantize_callback/2)
  end

  @doc false
  def quantize_callback(%Nx.Tensor{} = tensor, opts) do
    EMLX.quantize(tensor, opts)
  end

  @doc """
  Construct an annotated quantized `Nx.Tensor` from pre-computed device refs.

  Use this when you already have packed weights from a checkpoint. For
  quantizing a dense tensor from scratch, prefer `from_dense/2`.

  ## Options

  * `:type` — Nx storage type: `{:s, 2}`, `{:s, 4}` (default), or `{:s, 8}`.
  * `:group_size` — quantization group size (default 64).
  """
  @spec quantized_tensor(term(), term(), term(), tuple(), keyword()) :: Nx.Tensor.t()
  def quantized_tensor(weight_ref, scales_ref, biases_ref, original_shape, opts \\ []) do
    type = Keyword.get(opts, :type, {:s, 4})
    {_, bits} = type
    group_size = Keyword.get(opts, :group_size, 64)

    scales = EMLX.Backend.to_nx(scales_ref)
    biases = EMLX.Backend.to_nx(biases_ref)
    config = %Config{scales: scales, biases: biases, group_size: group_size, bits: bits}

    weight_shape = EMLX.shape(weight_ref)
    template = Nx.template(original_shape, type)

    %Nx.Tensor{
      template
      | data: %EMLX.Backend{
          ref: weight_ref,
          shape: weight_shape,
          type: {:u, 32},
          quantization_config: config
        }
    }
  end

  @doc """
  Dequantize a quantized `Nx.Tensor` via `Nx.runtime_call`.

  At execution time the callback receives the real tensor, unpacks
  `quantization_config`, and calls `mx::dequantize`. Safe to call inside
  `Nx.Defn.jit`-traced forward passes.

  The output has the same shape as the input (the quantized tensor's logical
  shape equals the dense shape).
  """
  deftransform dequantize(qw) do
    # Infer float type from scales when we have the real backend (eager / evaluator).
    # Falls back to :f32 during JIT tracing where qw.data is Nx.Defn.Expr.
    out_type =
      case qw do
        %Nx.Tensor{data: %EMLX.Backend{quantization_config: %Config{scales: s}}} -> Nx.type(s)
        _ -> :f32
      end

    out = Nx.template(Nx.shape(qw), out_type)
    Nx.runtime_call(out, qw, [], &__MODULE__.dequantize_callback/2)
  end

  @doc false
  def dequantize_callback(%Nx.Tensor{} = qw, _opts) do
    EMLX.dequantize(qw)
  end

  @doc """
  Run a quantized matmul via `Nx.runtime_call`.

  At execution time the callback unpacks `quantization_config` from `qw` and
  calls `mx::quantized_matmul`. Safe to call inside `Nx.Defn.jit`-traced
  forward passes.

  Output shape is `{batch_dims..., out_features}` where `out_features` is the
  first dimension of `qw`. Output type is `:bf16`.
  """
  deftransform quantized_matmul(activation, qw) do
    {out_features, _} = Nx.shape(qw)
    act_batch = Nx.shape(activation) |> Tuple.to_list() |> Enum.drop(-1)
    out_shape = List.to_tuple(act_batch ++ [out_features])

    out_type =
      case qw do
        %Nx.Tensor{data: %EMLX.Backend{quantization_config: %Config{scales: s}}} -> Nx.type(s)
        _ -> Nx.Type.merge(Nx.type(activation), Nx.type(qw))
      end

    out = Nx.template(out_shape, out_type)
    Nx.runtime_call(out, {activation, qw}, [], &__MODULE__.quantized_matmul_callback/2)
  end

  @doc false
  def quantized_matmul_callback({%Nx.Tensor{} = activation, %Nx.Tensor{} = qw}, _opts) do
    EMLX.quantized_matmul(activation, qw)
  end

  @doc """
  Returns `true` if the tensor has quantization metadata on its backend.
  """
  @spec quantized?(term()) :: boolean()
  def quantized?(%Nx.Tensor{data: %EMLX.Backend{quantization_config: cfg}}) when not is_nil(cfg),
    do: true

  def quantized?(_), do: false
end
