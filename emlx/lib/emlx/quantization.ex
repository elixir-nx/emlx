defmodule EMLX.Quantization do
  @moduledoc """
  Affine group-wise int2/int4/int8 quantization for Apple Silicon inference,
  plus microscaled floating-point modes (`"mxfp4"`/`"mxfp8"`/`"nvfp4"`, via
  `:mode` on `quantize/2` — see `EMLX.quantize/2`).

  Quantized weights are represented as annotated `Nx.Tensor` values — the
  tensor carries the original logical shape and type (e.g. `{:s, 4}` for
  4-bit), while the `EMLX.Backend` struct stores the packed uint32 data and
  a `EMLX.Quantization.Config` with scales, biases, group_size, bits, and
  mode. Microscaled modes have no biases (`Config.biases` is `nil` — MLX's
  `fp_quantize` returns only `(wq, scales)` for them).

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

  ## Limitation: freshly-quantized values can't feed a traced `Nx.dot`

  `Nx.dot`'s `mx::quantized_matmul` dispatch (see
  `EMLX.Native.Expr.quantizable_param_positions/1`) only recognizes a
  quantized tensor that started life as a bound top-level parameter of the
  `Nx.Defn.jit`-compiled function — it's resolved via a call-time
  `quant_signature` keyed by parameter position, baked into a specialized
  compiled program. A value produced by `quantize/2` *inside* the traced
  program (e.g. `dequantize/1` → some op → `quantize/2` again) has no such
  position and isn't recognized as a quantized `:dot` operand; it may only
  be used as a direct output of the traced function (see `quantize/2`'s doc).

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
  `Nx.Defn.jit`-traced forward passes — including on a value that is itself
  freshly computed in-graph (e.g. `dequantize/1` followed by some op followed
  by `quantize/2` again), not just a bare top-level parameter.

  Note: the *result* of a traced `quantize/2` call may only be used as a
  direct output of the `Nx.Defn.jit`-compiled function (returned as-is, or
  passed through unchanged). Feeding it into another op in the same traced
  program (e.g. `Nx.dot`) is not yet supported and raises during lowering —
  only a quantized tensor that started life as a bound top-level parameter
  dispatches to `mx::quantized_matmul` today.

  ## Options

  * `:type` — Nx storage type: `{:s, 2}`, `{:s, 4}` (default), or `{:s, 8}`.
  * `:group_size` — 32, 64, or 128 (default 64). Must evenly divide the last
    dimension of `tensor`.
  * `:mode` — `"affine"` (default), or a microscaled mode (`"mxfp4"`,
    `"mxfp8"`, `"nvfp4"`) — see `EMLX.quantize/2`.
  """
  deftransform quantize(tensor, opts \\ []) do
    if match?(%Nx.Tensor{data: %Nx.Defn.Expr{}}, tensor) do
      quantize_traced(tensor, opts)
    else
      EMLX.quantize(tensor, opts)
    end
  end

  # `EMLX.quantize/2`'s result is a single annotated tensor whose Nx-visible
  # shape/type (e.g. `{:s, 4}`) is a *logical* fiction that never matches the
  # physical packed array MLX actually produces — so a bare `Nx.runtime_call`
  # with that logical template can't round-trip it: the generic
  # `Nx.to_binary`/`from_binary` wire marshalling it relies on has no
  # `{:u, 32} -> {:s, 4}` conversion (nor could it — unpacking sub-byte-width
  # lanes isn't a byte-for-byte reinterpretation). Instead we run a *second*,
  # real-shaped `Nx.runtime_call` that decomposes the result into its
  # genuinely dense physical pieces (packed weight, scales, biases), each
  # with its own real (non-fictional) shape/type, and stash their traced
  # leaves in an `Nx.Defn.Expr.metadata/2` node — mirroring `EMLX.Fast`'s
  # `:__EMLX__` pattern — so the rest of the traced program still sees one
  # ordinary, logical-shaped tensor. `EMLX.Native.Expr.lower/2`'s output-ref
  # builder and `EMLX.native_compile/3`'s output reconstruction both know how
  # to read this `__EMLX_QUANT__` metadata back into a proper annotated
  # quantized tensor when such a leaf reaches a JIT function's output.
  defp quantize_traced(tensor, opts) do
    shape = Nx.shape(tensor)
    type = Keyword.get(opts, :type, {:s, 4})
    {_, bits} = type
    group_size = Keyword.get(opts, :group_size, 64)
    mode = Keyword.get(opts, :mode, "affine")

    {weight_template, scales_template, biases_template} = quantize_output_templates(shape, opts)

    # Fallback path for compilers other than EMLX (e.g. Nx.Defn.Evaluator,
    # Nx.Defn.Grad): a plain runtime_call recomputing the real, correct,
    # single annotated tensor. EMLX's own lowering never evaluates this —
    # see `EMLX.Native.Expr.scope_dependencies/1`'s `__EMLX_QUANT__` clause,
    # which redirects traversal to the decomposition leaves below instead.
    inner = Nx.runtime_call(Nx.template(shape, type), tensor, opts, &quantize_callback/2)

    {decomposition_out, has_biases} =
      if biases_template do
        {{weight_template, scales_template, biases_template}, true}
      else
        {{weight_template, scales_template}, false}
      end

    decomposition =
      Nx.runtime_call(decomposition_out, tensor, opts, &quantize_decompose_callback/2)

    {weight_leaf, scales_leaf, biases_leaf} =
      if has_biases do
        {elem(decomposition, 0), elem(decomposition, 1), elem(decomposition, 2)}
      else
        {elem(decomposition, 0), elem(decomposition, 1), nil}
      end

    Nx.Defn.Expr.metadata(inner, %{
      __EMLX_QUANT__: %{
        weight: weight_leaf,
        scales: scales_leaf,
        biases: biases_leaf,
        group_size: group_size,
        bits: bits,
        mode: mode
      }
    })
  end

  # Runs a tiny real quantize call (values are discarded) purely to learn the
  # physical weight/scales/biases shapes and dtypes MLX will actually
  # produce for this logical shape + opts — avoids duplicating
  # `mx::quantize`'s internal packing/group-size math (which varies by
  # mode/bits) in Elixir.
  defp quantize_output_templates(shape, opts) do
    dummy = Nx.iota(shape, type: {:f, 32}, backend: EMLX.Backend)

    %Nx.Tensor{data: %EMLX.Backend{shape: w_shape, type: w_type, quantization_config: cfg}} =
      EMLX.quantize(dummy, opts)

    weight_template = Nx.template(w_shape, w_type)
    scales_template = Nx.to_template(cfg.scales)
    biases_template = cfg.biases && Nx.to_template(cfg.biases)

    {weight_template, scales_template, biases_template}
  end

  @doc false
  def quantize_callback(%Nx.Tensor{} = tensor, opts) do
    EMLX.quantize(tensor, opts)
  end

  @doc false
  def quantize_decompose_callback(%Nx.Tensor{} = tensor, opts) do
    type = Keyword.get(opts, :type, {:s, 4})
    {_, bits} = type
    group_size = Keyword.get(opts, :group_size, 64)
    mode = Keyword.get(opts, :mode, "affine")

    device_ref = EMLX.Backend.from_nx(tensor)
    {weight_ref, scales_ref, biases_ref} = EMLX.quantize(device_ref, group_size, bits, mode)

    weight = EMLX.Backend.to_nx(weight_ref)
    scales = EMLX.Backend.to_nx(scales_ref)

    if biases_ref do
      {weight, scales, EMLX.Backend.to_nx(biases_ref)}
    else
      {weight, scales}
    end
  end

  @doc """
  Construct an annotated quantized `Nx.Tensor` from pre-computed device refs.

  Use this when you already have packed weights from a checkpoint. For
  quantizing a dense tensor from scratch, prefer `from_dense/2`.

  ## Options

  * `:type` — Nx storage type: `{:s, 2}`, `{:s, 4}` (default), or `{:s, 8}`.
  * `:group_size` — quantization group size (default 64).
  """
  def quantized_tensor(weight_ref, scales_ref, biases_ref, original_shape, opts \\ []) do
    type = Keyword.get(opts, :type, {:s, 4})
    {_, bits} = type
    group_size = Keyword.get(opts, :group_size, 64)

    scales = EMLX.Backend.to_nx(scales_ref)
    biases = EMLX.Backend.to_nx(biases_ref)
    config = %Config{scales: scales, biases: biases, group_size: group_size, bits: bits}

    weight_shape = EMLX.shape(weight_ref)
    %Nx.Tensor{} = template = Nx.template(original_shape, type)

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
  shape equals the dense shape). Supports every mode `EMLX.quantize/2`
  accepts.
  """
  deftransform dequantize(qw) do
    # Infer float type from scales when we have the real backend (eager / evaluator).
    # Falls back to :f32 during JIT tracing where qw.data is Nx.Defn.Expr.
    # Microscaled modes store scales as :u8 (a packed exponent byte, not a
    # float dtype), so `Nx.type(s)` doesn't apply there — MLX's `dequantize`
    # reconstructs a float array regardless of mode; :bf16 matches the
    # convention used elsewhere for microscaled outputs.
    out_type =
      case qw do
        %Nx.Tensor{data: %EMLX.Backend{quantization_config: %Config{mode: "affine", scales: s}}} ->
          Nx.type(s)

        %Nx.Tensor{data: %EMLX.Backend{quantization_config: %Config{}}} ->
          {:bf, 16}

        _ ->
          :f32
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
  first dimension of `qw`. Output type matches the scales dtype for
  `"affine"` (historically the activation's own float type); for microscaled
  modes scales are `:u8` (a packed exponent byte), so the output type
  follows the activation's dtype instead.
  """
  deftransform quantized_matmul(activation, qw) do
    {out_features, _} = Nx.shape(qw)
    act_batch = Nx.shape(activation) |> Tuple.to_list() |> Enum.drop(-1)
    out_shape = List.to_tuple(act_batch ++ [out_features])

    out_type =
      case qw do
        %Nx.Tensor{data: %EMLX.Backend{quantization_config: %Config{mode: "affine", scales: s}}} ->
          Nx.type(s)

        %Nx.Tensor{data: %EMLX.Backend{quantization_config: %Config{}}} ->
          Nx.type(activation)

        _ ->
          Nx.Type.merge(Nx.type(activation), Nx.type(qw))
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
  def quantized?(%Nx.Tensor{data: %EMLX.Backend{quantization_config: cfg}}) when not is_nil(cfg),
    do: true

  def quantized?(_), do: false
end
