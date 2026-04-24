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

  Routes through an `EMLX.CommandQueue` worker because MLX 0.31.2 pins
  every GPU stream to the OS thread that created it. The macro injects
  `worker = EMLX.resolve_worker(device)` and prepends `worker` to the
  NIF call, then awaits the tagged-ref reply via `await_worker/1`.
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
      # NIF arity is original + 1 because the wrapper prepends `worker` as
      # argv[0]. The original `device` arg is preserved (kept for MLX's
      # `to_stream(device) -> default_stream(device)` lookup, which on the
      # worker thread returns the worker's stream — see emlx_async.hpp).
      @mlx_function {unquote(name), unquote(length(args) + 1)}
      def unquote(name)(unquote_splicing(args)) do
        unquote(tensors)
        {user_device, index} = normalize_device!(var!(device))
        {worker, effective_device} = resolve_worker(user_device)
        var!(device) = mlx_device!(effective_device, index)

        job_ref =
          EMLX.NIF.unquote(name)(worker, unquote_splicing(args)) |> unwrap!()

        await_worker(job_ref) |> wrap_tensor(effective_device)
      end
    end
  end

  @doc """
  Generates a call that returns a tensor (or a tuple/list of tensors).

  All tensor variables must start with the name tensor. Routes through an
  `EMLX.CommandQueue` worker (see `defdevice/1`).
  """
  defmacro deftensor(call) do
    {name, args} = Macro.decompose_call(call)
    tensors = tensors(args)

    if tensors == [] do
      raise ArgumentError, "at least one tensor required in #{name}/#{length(args)}"
    end

    quote do
      # NIF arity = original + 2: leading `worker` + trailing `device`. The
      # device atom is still passed through so the underlying sync NIF body
      # (e.g. `mlx::core::add(*a, *b, device)`) gets a `StreamOrDevice` that
      # MLX resolves to the worker's stream via the worker thread's default
      # stream slot.
      @mlx_function {unquote(name), unquote(length(args) + 2)}
      def unquote(name)(unquote_splicing(args)) do
        {unquote(tensors), device} = prepare_tensors!(unquote(tensors))
        {worker, effective_device} = resolve_worker(device)

        job_ref =
          EMLX.NIF.unquote(name)(worker, unquote_splicing(args), effective_device)
          |> unwrap!()

        await_worker(job_ref) |> wrap_tensor(effective_device)
      end
    end
  end

  @doc """
  Generates a call that returns a value (not a tensor). NOT worker-routed —
  use only for pure metadata / refcount NIFs (`scalar_type`, `shape`,
  `strides`, `deallocate`). Graph-touching value NIFs (e.g. `item`) must be
  hand-written so they thread a worker through `EMLX.NIF.<op>(worker, ...)`
  and await a tagged-ref reply.
  """
  defmacro defvalue(call) do
    {name, args} = Macro.decompose_call(call)
    tensors = tensors(args)

    if tensors == [] do
      raise ArgumentError, "at least one tensor required in #{name}/#{length(args)}"
    end

    quote do
      @mlx_function {unquote(name), unquote(length(args))}
      def unquote(name)(unquote_splicing(args)) do
        {unquote(tensors), _device} = prepare_tensors!(unquote(tensors))
        EMLX.NIF.unquote(name)(unquote_splicing(args)) |> unwrap!()
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

  ## Native linalg ops
  deftensor linalg_lu(tensor)
  deftensor linalg_qr(tensor)
  deftensor linalg_svd(tensor, compute_uv)
  deftensor linalg_cholesky(tensor, upper)
  deftensor linalg_eigh(tensor, uplo)
  deftensor linalg_inv(tensor)
  deftensor linalg_pinv(tensor)
  deftensor linalg_solve(tensorA, tensorB)
  deftensor linalg_solve_triangular(tensorA, tensorB, upper)

  ## Native pooling (window scatter) ops
  deftensor window_scatter_max(
              tensor_t,
              tensor_source,
              tensor_init_value,
              window,
              low_pad,
              high_pad,
              strides
            )

  deftensor window_scatter_min(
              tensor_t,
              tensor_source,
              tensor_init_value,
              window,
              low_pad,
              high_pad,
              strides
            )

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
  @mlx_function {:quantized_matmul, 9}
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
    {worker, effective_device} = resolve_worker(device)

    job_ref =
      EMLX.NIF.quantized_matmul(
        worker,
        ref_x,
        ref_w,
        ref_s,
        ref_b,
        transpose,
        group_size,
        bits,
        effective_device
      )
      |> unwrap!()

    await_worker(job_ref) |> wrap_tensor(effective_device)
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
  @mlx_function {:dequantize, 7}
  def dequantize(
        {dev_w, ref_w} = _tensor_w,
        {dev_s, ref_s} = _tensor_scales,
        {dev_b, ref_b} = _tensor_biases,
        group_size,
        bits
      )
      when is_tensor(dev_w, ref_w) and is_tensor(dev_s, ref_s) and is_tensor(dev_b, ref_b) do
    device = merge_device(dev_w, merge_device(dev_s, dev_b))
    {worker, effective_device} = resolve_worker(device)

    job_ref =
      EMLX.NIF.dequantize(worker, ref_w, ref_s, ref_b, group_size, bits, effective_device)
      |> unwrap!()

    await_worker(job_ref) |> wrap_tensor(effective_device)
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
  @mlx_function {:quantize, 5}
  def quantize({dev_w, ref_w}, group_size, bits)
      when is_tensor(dev_w, ref_w) do
    device = dev_w
    {worker, effective_device} = resolve_worker(device)

    {weights_ref, scales_ref, biases_ref} =
      EMLX.NIF.quantize(worker, ref_w, group_size, bits, effective_device)
      |> unwrap!()
      |> await_worker()

    {{effective_device, weights_ref}, {effective_device, scales_ref},
     {effective_device, biases_ref}}
  end

  @doc """
  Quantize a dense 2-D `Nx.Tensor` and return an annotated quantized tensor.

  The returned tensor carries the original logical shape and type (e.g.
  `{:s, 4}`). Its backend stores the packed uint32 data and a
  `EMLX.Quantization.Config` with scales, biases, `group_size`, and `bits`.

  ## Options

  * `:type` — storage type: `{:s, 2}`, `{:s, 4}` (default), or `{:s, 8}`.
  * `:group_size` — 32, 64, or 128 (default 64). Must evenly divide the last
    dimension of `tensor`.
  """
  @spec quantize(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def quantize(%Nx.Tensor{} = tensor, opts) when is_list(opts) do
    type = Keyword.get(opts, :type, {:s, 4})
    {_, bits} = type
    group_size = Keyword.get(opts, :group_size, 64)

    unless Nx.rank(tensor) == 2 do
      raise ArgumentError,
            "EMLX.quantize/2 requires a rank-2 tensor, got rank #{Nx.rank(tensor)}"
    end

    {_out_features, in_features} = Nx.shape(tensor)

    unless rem(in_features, group_size) == 0 do
      raise ArgumentError,
            "EMLX.quantize/2 requires the last dimension (#{in_features}) " <>
              "to be divisible by group_size (#{group_size})"
    end

    device_ref = EMLX.Backend.from_nx(tensor)
    {weight_ref, scales_ref, biases_ref} = EMLX.quantize(device_ref, group_size, bits)

    scales = EMLX.Backend.to_nx(scales_ref)
    biases = EMLX.Backend.to_nx(biases_ref)

    config = %EMLX.Quantization.Config{
      scales: scales,
      biases: biases,
      group_size: group_size,
      bits: bits
    }

    weight_shape = EMLX.shape(weight_ref)
    template = Nx.template(Nx.shape(tensor), type)

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
  Dequantize a quantized `Nx.Tensor` (created by `EMLX.quantize/2`) to a
  dense float tensor by calling `mx::dequantize`.
  """
  @spec dequantize(Nx.Tensor.t()) :: Nx.Tensor.t()
  def dequantize(
        %Nx.Tensor{
          data: %EMLX.Backend{ref: weight_ref, quantization_config: cfg}
        } = _qw
      )
      when not is_nil(cfg) do
    EMLX.dequantize(
      weight_ref,
      EMLX.Backend.from_nx(cfg.scales),
      EMLX.Backend.from_nx(cfg.biases),
      cfg.group_size,
      cfg.bits
    )
    |> EMLX.Backend.to_nx()
  end

  @doc """
  Run `activation @ dequantize(qw)` using `mx::quantized_matmul`.

  `qw` must be a quantized tensor produced by `EMLX.quantize/2`. Raises
  `ArgumentError` if both arguments are quantized.
  """
  @spec quantized_matmul(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def quantized_matmul(%Nx.Tensor{} = activation, %Nx.Tensor{} = qw) do
    cfg = qw.data.quantization_config

    if is_nil(cfg) do
      raise ArgumentError,
            "EMLX.quantized_matmul/2: second argument must be a quantized tensor"
    end

    if not is_nil(activation.data.quantization_config) do
      raise ArgumentError,
            "EMLX.quantized_matmul/2 requires a dense activation as the first " <>
              "argument; got two quantized tensors. Dequantize one of them first."
    end

    result =
      EMLX.quantized_matmul(
        EMLX.Backend.from_nx(activation),
        qw.data.ref,
        EMLX.Backend.from_nx(cfg.scales),
        EMLX.Backend.from_nx(cfg.biases),
        true,
        cfg.group_size,
        cfg.bits
      )

    EMLX.Backend.to_nx(result)
  end

  def to_blob({device, ref} = tensor) when is_tensor(device, ref) do
    # Eval first so the underlying MLX array is materialised; then ask the
    # worker for the contiguous-copy + zero-copy resource binary. Both
    # operations are routed through the same worker resolution path so
    # that the contiguous fallback in `to_blob_term` runs on the same OS
    # thread that owns the tensor's stream encoder.
    eval(tensor)
    {worker, _effective_device} = resolve_worker(device)
    job_ref = EMLX.NIF.to_blob(worker, ref) |> unwrap!()
    await_worker(job_ref)
  end

  def to_blob({device, ref} = tensor, limit) when is_tensor(device, ref) do
    eval(tensor)
    {worker, _effective_device} = resolve_worker(device)
    job_ref = EMLX.NIF.to_blob(worker, ref, limit) |> unwrap!()
    await_worker(job_ref)
  end

  @doc """
  Returns `{address, byte_size}` for the tensor's raw GPU buffer.

  Evals the tensor first (same pattern as `to_blob/1`). The pointer is valid
  as long as no further MLX evaluation is triggered on the array and the
  Elixir tensor term is kept alive. On Apple Silicon the address is accessible
  from both CPU and GPU due to unified memory.
  """
  def tensor_data_ptr({device, ref} = tensor) when is_tensor(device, ref) do
    eval(tensor)
    EMLX.NIF.tensor_data_ptr(ref) |> unwrap!()
  end

  @doc """
  Copies tensor data into a new POSIX shared-memory segment and returns
  `{shm_name, byte_size}`.

  Note: this involves a **memcpy** — MLX arrays are immutable so zero-copy
  cross-process sharing is not possible.  `permissions` is a Unix mode integer
  (e.g. `0o400` for owner-read-only).

  The shm name persists until the receiver opens and unlinks it (which
  `EMLX.NIF.array_from_shm/4` does automatically).
  """
  def tensor_to_shm({device, ref} = tensor, permissions) when is_tensor(device, ref) do
    eval(tensor)
    {worker, _effective_device} = resolve_worker(device)
    # Worker-routed: `tensor_to_shm`'s NIF body may call `mx::contiguous`
    # + `mx::eval` on a non-contiguous tensor, which both touch the
    # thread-local Metal encoder. Must run on the worker.
    job_ref = EMLX.NIF.tensor_to_shm(worker, ref, permissions) |> unwrap!()
    await_worker(job_ref)
  end

  @doc """
  Unlinks a POSIX shared-memory segment by its handle name.

  Call this if the receiver never opens the `%Nx.Pointer{kind: :ipc}` returned
  by `Nx.to_pointer/2` — otherwise the shm name persists until the next reboot.
  Safe to call even if the segment has already been unlinked (ENOENT is ignored).
  """
  def shm_unlink(name) when is_binary(name) do
    EMLX.NIF.shm_unlink_handle(name) |> unwrap!()
  end

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, result}), do: result
  defp unwrap!({:error, error}), do: raise(EMLX.NIFError, List.to_string(error))

  # Wraps a worker-thread payload in {device, ref} envelopes.
  # Already-unwrapped (no leading {:ok, _}) — `await_worker/1` peels that
  # off when the worker delivers the reply.
  defp wrap_tensor(ref, device) when is_reference(ref), do: {device, ref}

  defp wrap_tensor(list, device) when is_list(list),
    do: Enum.map(list, &{device, &1})

  defp wrap_tensor(tuple, device) when is_tuple(tuple),
    do: tuple |> Tuple.to_list() |> Enum.map(&{device, &1}) |> List.to_tuple()

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

  @doc """
  Evaluates a (possibly lazy) MLX tensor by routing the work through an
  `EMLX.CommandQueue`. Blocks the caller until the worker thread has
  finished `mlx::core::eval/1` for this tensor.

  Resolves the queue via `resolve_worker/1`:

    1. If the calling process has bound a queue with
       `EMLX.CommandQueue.with_queue/2`, that queue is used.
    2. Otherwise the application-default worker for the tensor's device
       (CPU or GPU) is used — see `EMLX.Application`.
  """
  def eval({device, ref}) when is_tensor(device, ref) do
    {worker, _effective_device} = resolve_worker(device)
    job_ref = EMLX.NIF.eval(worker, ref) |> unwrap!()
    await_worker(job_ref)
  end

  # ── Worker resolution ──────────────────────────────────────────────────────
  #
  # `Process.get(:emlx_command_queue)` is set by EMLX.CommandQueue.with_queue/2.
  # The value is `{worker_ref, device}` so we know the queue's device without a
  # second lookup. Returns `{worker, effective_device}` — callers must use
  # `effective_device` for both the NIF device argument and `wrap_tensor/2`.
  #
  # When no queue is bound, we fall back to the application-default worker for
  # the requested device.

  @doc false
  def resolve_worker(device) do
    case Process.get(:emlx_command_queue) do
      nil -> {EMLX.Application.default_worker(device), device}
      {worker, ^device} -> {worker, device}
      {worker, bound_device} -> resolve_cross_device(device, worker, bound_device)
    end
  end

  # CPU and GPU operations do not share thread-local Metal encoder state, so
  # routing a CPU tensor through a GPU queue (or vice-versa) is safe — MLX
  # inserts the necessary cross-stream synchronization internally. We therefore
  # do NOT force an intermediate eval; we let MLX manage the graph dependency.
  defp resolve_cross_device(requested, worker, bound) do
    if Application.get_env(:emlx, :cross_device_promotion, false) do
      if Application.get_env(:emlx, :warn_cross_device, false) do
        require Logger

        Logger.warning(
          "[EMLX] cross-device promotion: #{requested} tensor routed to #{bound} queue"
        )
      end

      {worker, bound}
    else
      {EMLX.Application.default_worker(requested), requested}
    end
  end

  defp await_worker(job_ref) do
    receive do
      # Worker NIFs (sync bodies) return one of:
      #   nx::nif::ok(env)         => :ok
      #   nx::nif::ok(env, term)   => {:ok, term}
      #   nx::nif::error(env, msg) => {:error, msg}
      # The async wrapper forwards the payload as-is in {ref, payload}.
      {^job_ref, :ok} -> :ok
      {^job_ref, {:ok, result}} -> result
      {^job_ref, {:error, reason}} -> raise(EMLX.NIFError, List.to_string(reason))
    end
  end

  deftensor slice(tensor, starts, stops, strides)
  deftensor slice_update(tensor, tensor_updates, starts, stops)
  deftensor squeeze(tensor, axes)
  defvalue strides(tensor)

  @doc """
  Converts an EMLX device ref back to an Nx.Tensor.

  ## Example

      result_ref = EMLX.some_operation(input)
      result_tensor = EMLX.to_nx(result_ref)
  """
  def to_nx({device, ref} = device_ref) when is_atom(device) and is_reference(ref) do
    EMLX.Backend.to_nx(device_ref)
  end

  @doc """
  Returns the scalar value of a 0-d tensor as a number.

  Worker-routed: the NIF body calls `mlx::core::eval(*t)` and `t->item<T>()`,
  both of which require running on the OS thread that owns the tensor's
  stream encoder.
  """
  def item({device, ref}) when is_tensor(device, ref) do
    {worker, _effective_device} = resolve_worker(device)
    job_ref = EMLX.NIF.item(worker, ref) |> unwrap!()
    await_worker(job_ref)
  end

  @behaviour Nx.Defn.Compiler

  # Known EMLX-specific compiler opts. `:command_queue` is injected by
  # `__partitions_options__/1` but may also be passed directly by callers
  # that manage their own queues (equivalent to a manual `with_queue`).
  @valid_compiler_keys [:device, :max_concurrency, :command_queue]

  @impl Nx.Defn.Compiler
  def __jit__(key, vars, fun, args_list, opts) do
    __compile__(key, vars, fun, opts).(args_list)
  end

  @impl Nx.Defn.Compiler
  def __compile__(key, vars, fun, opts) do
    Keyword.validate!(opts, @valid_compiler_keys)
    {compiler_opts, rest_opts} = split_compiler_opts(opts)
    queue = Keyword.get(compiler_opts, :command_queue)

    inner =
      Nx.Defn.Evaluator.__compile__(
        key,
        vars,
        fun,
        Keyword.put(rest_opts, :compiler, Nx.Defn.Evaluator)
      )

    if queue do
      # Capture the queue ref in a closure so each invocation of the compiled
      # function routes through the correct CommandQueue. The queue lives as
      # long as the Nx.Serving module_state that holds this compiled function.
      fn inputs -> EMLX.CommandQueue.with_queue(queue, fn -> inner.(inputs) end) end
    else
      inner
    end
  end

  @impl Nx.Defn.Compiler
  def __partitions_options__(opts) do
    n = Keyword.get(opts, :max_concurrency, 1)
    device = Keyword.get(opts, :device, default_device())

    # Allocate one CommandQueue (and its OS thread) per partition. This runs
    # inside Nx.Serving's GenServer init/1 — queues are owned by module_state.
    # For N ≤ 8 the synchronous pthread_create calls take a few milliseconds.
    for _i <- 1..n do
      [device: device, command_queue: EMLX.CommandQueue.new!(device)]
    end
  end

  @impl Nx.Defn.Compiler
  def __to_backend__(opts) do
    device = Keyword.get(opts, :device, default_device())
    {EMLX.Backend, device: device}
  end

  @doc """
  Returns the default MLX device for this process.

  Reads `:default_device` from the `:emlx` application environment, falling
  back to `:gpu`. Override in tests or config via:

      Application.put_env(:emlx, :default_device, :cpu)
  """
  def default_device do
    Application.get_env(:emlx, :default_device, :gpu)
  end

  # Splits opts into {emlx_compiler_opts, rest_opts}. The rest_opts are
  # forwarded to Nx.Defn.Evaluator; EMLX-specific keys are consumed here.
  defp split_compiler_opts(opts) do
    Enum.split_with(opts, fn {k, _v} -> k in @valid_compiler_keys end)
  end

  @doc """
  Returns a map with current memory usage information.

  Keys:
    * `:active_memory` - bytes currently allocated and in use
    * `:peak_memory` - highest active memory since last reset
    * `:cache_memory` - bytes in the allocator cache (freed but not returned to OS)

  ## Examples

      iex> info = EMLX.memory_info()
      iex> is_integer(info.active_memory) and is_integer(info.peak_memory) and is_integer(info.cache_memory)
      true
  """
  def memory_info, do: EMLX.NIF.memory_info() |> unwrap!()

  @doc """
  Clears the MLX memory cache, releasing unused GPU memory back to the system.

  Useful after inference batches to prevent memory growth. Does not affect
  tensors that are still referenced.

  ## Examples

      EMLX.clear_cache()
      #=> :ok
  """
  @spec clear_cache() :: :ok
  def clear_cache, do: EMLX.NIF.clear_cache() |> unwrap!()

  @doc """
  Resets the peak memory counter to zero.

  ## Examples

      EMLX.reset_peak_memory()
      #=> :ok
  """
  @spec reset_peak_memory() :: :ok
  def reset_peak_memory, do: EMLX.NIF.reset_peak_memory() |> unwrap!()

  @doc """
  Sets the memory limit in bytes. Returns the previous limit.

  The memory limit is a guideline for maximum memory usage during graph
  evaluation. Defaults to 1.5× the device's recommended working set size.

  ## Examples

      prev = EMLX.set_memory_limit(8_000_000_000)
      EMLX.set_memory_limit(prev)
  """
  def set_memory_limit(limit) when is_integer(limit) and limit >= 0 do
    EMLX.NIF.set_memory_limit(limit) |> unwrap!()
  end

  @doc """
  Sets the cache limit in bytes. Returns the previous limit.

  When cached memory exceeds this limit, it will be reclaimed on the next
  allocation. Set to 0 to disable caching entirely.

  ## Examples

      prev = EMLX.set_cache_limit(500_000_000)
      EMLX.set_cache_limit(prev)
  """
  def set_cache_limit(limit) when is_integer(limit) and limit >= 0 do
    EMLX.NIF.set_cache_limit(limit) |> unwrap!()
  end
end
