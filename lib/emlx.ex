defmodule EMLX.NIFError do
  defexception [:message]
end

defmodule EMLX.Macro do
  @moduledoc false

  defmacro __using__(_opts) do
    quote do
      import unquote(__MODULE__)
      Module.register_attribute(EMLX, :mlx_function, accumulate: true)

      @before_compile EMLX.Macro
    end
  end

  @doc false
  defmacro __before_compile__(env) do
    mlx_functions = Module.get_attribute(env.module, :mlx_function)

    quote do
      def __mlx_functions__ do
        unquote(mlx_functions)
      end
    end
  end

  @doc """
  Function that receives a device and allocates a tensor.
  """
  defmacro defdevice(call) do
    {name, args} = Macro.decompose_call(call)

    unless has_device?(args) do
      raise("At least one argument of defdevice function should be named 'device'.")
    end

    tensors =
      case tensors(args) do
        [] -> :ok
        tensors -> quote do: {unquote(tensors), _} = prepare_tensors!(unquote(tensors))
      end

    quote do
      @mlx_function {unquote(name), unquote(length(args))}
      def unquote(name)(unquote_splicing(args)) do
        unquote(tensors)
        {user_device, index} = normalize_device!(var!(device))
        var!(device) = mlx_device!(user_device, index)

        EMLX.NIF.unquote(name)(unquote_splicing(args))
        |> unwrap_tensor!(user_device)
      end
    end
  end

  @doc """
  Generates a call that returns a tensor (or a tuple/list of tensors).

  All tensor variables must start with the name tensor.
  """
  defmacro deftensor(call) do
    defcall(call, :unwrap_tensor!, [Macro.var(:device, __MODULE__)])
  end

  @doc """
  Generates a call that returns a value (not a tensor).

  All tensor variables must start with the name tensor.
  """
  defmacro defvalue(call) do
    defcall(call, :unwrap!, [])
  end

  defp defcall(call, unwrapper, extra) do
    {name, args} = Macro.decompose_call(call)
    tensors = tensors(args)

    if tensors == [] do
      raise ArgumentError, "at least one tensor required in #{name}/#{length(args)}"
    end

    quote do
      @mlx_function {unquote(name), unquote(length(args) + length(extra))}
      def unquote(name)(unquote_splicing(args)) do
        {unquote(tensors), device} = prepare_tensors!(unquote(tensors))

        EMLX.NIF.unquote(name)(unquote_splicing(args ++ extra))
        |> unquote(unwrapper)(unquote_splicing(extra))
      end
    end
  end

  defp has_device?(args) do
    Enum.any?(args, &match?({:device, _, nil}, &1))
  end

  defp tensors(args) do
    Enum.filter(args, fn {name, _, _} -> match?("tensor" <> _, Atom.to_string(name)) end)
  end
end

defmodule EMLX do
  use EMLX.Macro

  defguard is_tensor(device, ref) when is_reference(ref) and is_atom(device)

  ## Macro callbacks

  defp normalize_device!({device, index}) when is_atom(device) and is_integer(index),
    do: {device, index}

  defp normalize_device!(device) when is_atom(device),
    do: {device, -1}

  defp normalize_device!(device),
    do: raise(ArgumentError, "expected device to be {atom, index} or atom, got: #{device}")

  defp mlx_device!(device, _index) do
    case device do
      :cpu -> :cpu
      :gpu -> :gpu
      _ -> raise ArgumentError, "unknown device #{inspect(device)}"
    end
  end

  ## Creation / conversion
  defdevice eye(m, n, type, device)
  defdevice from_blob(blob, shape, type, device)
  defdevice scalar_tensor(scalar, type, device)
  defdevice ones(shape, type, device)
  defdevice full(value, shape, type, device)
  defdevice arange(start, stop, step, integer?, device)

  ## Manipulation
  deftensor reshape(tensor, shape)
  deftensor broadcast_to(tensor, shape)
  deftensor astype(tensor, type)
  deftensor as_strided(tensor, shape, strides, offset)
  deftensor view(tensor, type)

  ## Binary ops
  deftensor add(tensorA, tensorB)
  deftensor subtract(tensorA, tensorB)
  deftensor multiply(tensorA, tensorB)
  deftensor pow(tensorA, tensorB)
  deftensor remainder(tensorA, tensorB)
  deftensor divide(tensorA, tensorB)
  deftensor atan2(tensorA, tensorB)
  deftensor bitwise_and(tensorA, tensorB)
  deftensor bitwise_or(tensorA, tensorB)
  deftensor bitwise_xor(tensorA, tensorB)
  deftensor bitwise_not(tensor)
  deftensor left_shift(tensorA, tensorB)
  deftensor right_shift(tensorA, tensorB)
  deftensor minimum(tensorA, tensorB)
  deftensor maximum(tensorA, tensorB)
  deftensor quotient(tensorA, tensorB)
  deftensor equal(tensorA, tensorB)
  deftensor not_equal(tensorA, tensorB)
  deftensor greater(tensorA, tensorB)
  deftensor less(tensorA, tensorB)
  deftensor greater_equal(tensorA, tensorB)
  deftensor less_equal(tensorA, tensorB)
  deftensor logical_and(tensorA, tensorB)
  deftensor logical_or(tensorA, tensorB)
  deftensor logical_xor(tensorA, tensorB)

  deftensor fft(tensor, n, axis)
  deftensor ifft(tensor, n, axis)
  deftensor fft2(tensor, s, axes)
  deftensor ifft2(tensor, s, axes)

  deftensor allclose(tensorA, tensorB, rtol, atol, equal_nan)
  deftensor isclose(tensorA, tensorB, rtol, atol, equal_nan)

  deftensor tensordot(tensorA, tensorB, axesA, axesB)
  deftensor einsum(tensorA, tensorB, spec_string)
  deftensor transpose(tensor, axes)
  deftensor pad(tensor, axes, low_pad_size, high_pad_size, tensor_pad_value)
  deftensor sort(tensor, axis)
  deftensor argsort(tensor, axis)
  deftensor tri_inv(tensor, upper)

  deftensor conv_general(
              tensor_input,
              tensor_kernel,
              strides,
              padding_low,
              padding_high,
              kernel_dilation,
              input_dilation,
              feature_group_count
            )

  ## Unary ops
  deftensor abs(tensor)
  deftensor ceil(tensor)
  deftensor conjugate(tensor)
  deftensor floor(tensor)
  deftensor negate(tensor)
  deftensor round(tensor)
  deftensor sign(tensor)
  deftensor real(tensor)
  deftensor imag(tensor)
  deftensor is_nan(tensor)
  deftensor is_infinity(tensor)
  deftensor logical_not(tensor)
  deftensor sigmoid(tensor)

  deftensor asin(tensor)
  deftensor asinh(tensor)
  deftensor acos(tensor)
  deftensor acosh(tensor)
  deftensor atan(tensor)
  deftensor atanh(tensor)
  deftensor cos(tensor)
  deftensor cosh(tensor)
  deftensor erf(tensor)
  deftensor erf_inv(tensor)
  deftensor exp(tensor)
  deftensor expm1(tensor)
  deftensor log(tensor)
  deftensor log1p(tensor)
  deftensor rsqrt(tensor)
  deftensor sin(tensor)
  deftensor sinh(tensor)
  deftensor sqrt(tensor)
  deftensor tan(tensor)
  deftensor tanh(tensor)

  ## Aggregation
  deftensor all(tensor, axes, keep_axes)
  deftensor any(tensor, axes, keep_axes)
  deftensor sum(tensor, axes, keep_axes)
  deftensor product(tensor, axes, keep_axes)
  deftensor argmax(tensor, keep_axes)
  deftensor argmax(tensor, axes, keep_axes)
  deftensor argmin(tensor, keep_axes)
  deftensor argmin(tensor, axes, keep_axes)
  deftensor cumulative_sum(tensor, axis, reverse, inclusive)
  deftensor cumulative_product(tensor, axis, reverse, inclusive)
  deftensor cumulative_max(tensor, axis, reverse, inclusive)
  deftensor cumulative_min(tensor, axis, reverse, inclusive)
  deftensor stack(tensors, axis)
  deftensor where(tensorPred, tensorTrue, tensorFalse)
  deftensor concatenate(tensors, axis)
  deftensor take_along_axis(tensor, tensorIndices, axis)
  deftensor take(tensor, tensorIndices, axis)
  deftensor gather(tensor, indices, axes, slice_sizes)
  deftensor scatter_add(tensor, indices, tensor_updates, axes)
  deftensor scatter(tensor, indices, tensor_updates, axes)
  deftensor max(tensor, axes, keep_axes)
  deftensor min(tensor, axes, keep_axes)
  deftensor clip(tensor, tensor_min, tensor_max)

  ## Dirty non-tensor return values
  defvalue scalar_type(tensor)
  defvalue shape(tensor)

  ## Quantization operations (for 4-bit model support)

  @doc """
  Performs quantized matrix multiplication.

  This is the key operation for efficient 4-bit inference. It multiplies `x` with
  quantized weights `w` (packed as uint32), using scales and biases for
  dequantization during the computation.

  ## Parameters
    - `x` - Input tensor (e.g., {batch, seq, hidden})
    - `w` - Quantized weights as uint32 (8 int4 values packed per uint32)
    - `scales` - Per-group scale factors (bfloat16)
    - `biases` - Per-group zero points (bfloat16)
    - `transpose` - Whether to transpose weights (default: true)
    - `group_size` - Number of weights per scale/bias group (default: 64)
    - `bits` - Quantization bits (default: 4)
  """
  @mlx_function {:quantized_matmul, 8}
  def quantized_matmul(
        {dev_x, ref_x} = _tensor_x,
        {dev_w, ref_w} = _tensor_w,
        {dev_s, ref_s} = _tensor_scales,
        {dev_b, ref_b} = _tensor_biases,
        transpose \\ true,
        group_size \\ 64,
        bits \\ 4
      )
      when is_tensor(dev_x, ref_x) and is_tensor(dev_w, ref_w) and
           is_tensor(dev_s, ref_s) and is_tensor(dev_b, ref_b) do
    device = merge_device(merge_device(dev_x, dev_w), merge_device(dev_s, dev_b))
    mlx_device = mlx_device!(device, -1)

    EMLX.NIF.quantized_matmul(ref_x, ref_w, ref_s, ref_b, transpose, group_size, bits, mlx_device)
    |> unwrap_tensor!(device)
  end

  @doc """
  Dequantizes packed weights to floating point.

  Converts quantized weights back to their original floating point representation.
  Useful for debugging and verification.

  ## Parameters
    - `w` - Quantized weights as uint32 (packed int4 values)
    - `scales` - Per-group scale factors
    - `biases` - Per-group zero points
    - `group_size` - Number of weights per group (default: 64)
    - `bits` - Quantization bits (default: 4)
  """
  @mlx_function {:dequantize, 6}
  def dequantize(
        {dev_w, ref_w} = _tensor_w,
        {dev_s, ref_s} = _tensor_scales,
        {dev_b, ref_b} = _tensor_biases,
        group_size \\ 64,
        bits \\ 4
      )
      when is_tensor(dev_w, ref_w) and is_tensor(dev_s, ref_s) and is_tensor(dev_b, ref_b) do
    device = merge_device(dev_w, merge_device(dev_s, dev_b))
    mlx_device = mlx_device!(device, -1)

    EMLX.NIF.dequantize(ref_w, ref_s, ref_b, group_size, bits, mlx_device)
    |> unwrap_tensor!(device)
  end

  @doc """
  Quantizes a floating point tensor to packed format.

  Returns a tuple of `{quantized_weights, scales, biases}` where:
    - `quantized_weights` - Packed uint32 tensor (8 int4 values per uint32)
    - `scales` - Per-group scale factors
    - `biases` - Per-group zero points

  ## Parameters
    - `w` - Float tensor to quantize
    - `group_size` - Number of weights per group (default: 64)
    - `bits` - Quantization bits (default: 4)
  """
  @mlx_function {:quantize, 4}
  def quantize({dev_w, ref_w} = _tensor_w, group_size \\ 64, bits \\ 4)
      when is_tensor(dev_w, ref_w) do
    device = dev_w
    mlx_device = mlx_device!(device, -1)

    {weights_ref, scales_ref, biases_ref} =
      EMLX.NIF.quantize(ref_w, group_size, bits, mlx_device) |> unwrap!()

    {{device, weights_ref}, {device, scales_ref}, {device, biases_ref}}
  end

  def to_blob({device, ref} = tensor) when is_tensor(device, ref) do
    # Two-step to_blob: eval on main scheduler, then copy on dirty scheduler
    eval(tensor)
    EMLX.NIF.to_blob(ref) |> unwrap!()
  end

  def to_blob({device, ref} = tensor, limit) when is_tensor(device, ref) do
    # Two-step to_blob: eval on main scheduler, then copy on dirty scheduler
    eval(tensor)
    EMLX.NIF.to_blob(ref, limit) |> unwrap!()
  end

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, result}), do: result
  defp unwrap!({:error, error}), do: raise(EMLX.NIFError, List.to_string(error))

  defp unwrap_tensor!(tagged_result, device) do
    case unwrap!(tagged_result) do
      ref when is_reference(ref) ->
        {device, ref}

      list when is_list(list) ->
        Enum.map(list, &{device, &1})

      tuple when is_tuple(tuple) ->
        tuple |> Tuple.to_list() |> Enum.map(&{device, &1}) |> List.to_tuple()
    end
  end

  defp prepare_tensors_list!(tensors_list, device) do
    Enum.map_reduce(tensors_list, device, fn
      {dev, ref}, device when is_tensor(dev, ref) ->
        {ref, merge_device(device, dev)}

      bad_tensor, _device ->
        raise ArgumentError, "expected a EMLX tensor, got: #{inspect(bad_tensor)}"
    end)
  end

  defp prepare_tensors!(tensors) do
    Enum.map_reduce(tensors, :cpu, fn
      {dev, ref}, device when is_tensor(dev, ref) ->
        {ref, merge_device(device, dev)}

      [{dev, ref} | _] = tensors, device when is_tensor(dev, ref) ->
        prepare_tensors_list!(tensors, device)

      bad_tensor, _device ->
        raise ArgumentError, "expected a EMLX tensor, got: #{inspect(bad_tensor)}"
    end)
  end

  defp merge_device(:gpu, _), do: :gpu
  defp merge_device(_, :gpu), do: :gpu
  defp merge_device(_, _), do: :cpu

  defvalue deallocate(tensor_ref)
  defvalue eval(tensor)

  deftensor slice(tensor, starts, stops, strides)
  deftensor slice_update(tensor, tensor_updates, starts, stops)
  deftensor squeeze(tensor, axes)
  defvalue item(tensor)
  defvalue strides(tensor)

  # ============================================================================
  # Quantized Tensor Operations (Backend-Integrated)
  # ============================================================================

  @doc """
  Creates a quantized Nx.Tensor with backend-level quantization options.

  This creates an Nx.Tensor where the EMLX.Backend struct contains
  quantization metadata. When this tensor is used in `Nx.dot`, the
  backend automatically dispatches to `quantized_matmul`.

  ## Parameters

  - `weight_ref` - EMLX device ref for packed uint32 weights
  - `scales_ref` - EMLX device ref for per-group scale factors
  - `biases_ref` - EMLX device ref for per-group zero points
  - `original_shape` - Shape before quantization {out_features, in_features}

  ## Options

  - `:bits` - Quantization bits (default: 4)
  - `:group_size` - Weights per scale/bias group (default: 64)

  ## Example

      # Quantize weights
      {q_weight, scales, biases} = EMLX.quantize(weight_tensor, 64, 4)

      # Create quantized Nx.Tensor
      quantized = EMLX.quantized_tensor(q_weight, scales, biases, {512, 4096})

      # Standard Nx.dot automatically uses quantized_matmul!
      result = Nx.dot(input, quantized)
  """
  def quantized_tensor(weight_ref, scales_ref, biases_ref, original_shape, opts \\ []) do
    EMLX.Backend.quantized_tensor(weight_ref, scales_ref, biases_ref, original_shape, opts)
  end

  @doc """
  Converts an EMLX device ref back to an Nx.Tensor.

  ## Example

      result_ref = EMLX.some_operation(input)
      result_tensor = EMLX.to_nx(result_ref)
  """
  def to_nx({device, ref} = device_ref) when is_atom(device) and is_reference(ref) do
    EMLX.Backend.to_nx(device_ref)
  end

  @behaviour Nx.Defn.Compiler

  @impl Nx.Defn.Compiler
  defdelegate __jit__(key, vars, fun, args_list, opts), to: Nx.Defn.Evaluator

  @impl Nx.Defn.Compiler
  defdelegate __compile__(key, vars, fun, opts), to: Nx.Defn.Evaluator

  @impl Nx.Defn.Compiler
  defdelegate __partitions_options__(opts), to: Nx.Defn.Evaluator

  @impl Nx.Defn.Compiler
  def __to_backend__(opts) do
    device = Keyword.get(opts, :device, :gpu)
    {EMLX.Backend, device: device}
  end
end
