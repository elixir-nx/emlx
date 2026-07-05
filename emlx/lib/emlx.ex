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
  @moduledoc """
  Low-level MLX NIF wrappers and the native `Nx.Defn.Compiler` for
  `EMLX.Backend` tensors.

  Most users don't call this module directly — set `EMLX.Backend` as your
  `Nx` backend (see the [README](readme.html)) and, optionally, `EMLX` as
  your `Nx.Defn` compiler for the JIT-compiled native path:

      Nx.Defn.default_options(compiler: EMLX)

  ## Compile once, replay many times

  `Nx.Defn.jit/2` and `Nx.Defn.jit_apply/3` **retrace** the given function
  from scratch on every call. For most ops that's cheap relative to the
  actual computation, but for a hot loop (e.g. a decode step, or any call
  site invoked with the same input shapes/types repeatedly) it means paying
  full retrace + dispatch-key computation cost on every single call — for
  some ops (e.g. `Nx.LinAlg.svd/1`, which traces a large, normally-unused
  fallback algorithm as part of every trace) this cost can dominate over the
  actual native computation.

  If you know you'll call the same function repeatedly with the same input
  shapes/types, prefer `Nx.Defn.compile/3`, which traces and lowers **once**
  and returns a plain closure that replays the already-compiled program on
  every subsequent call — no retrace, no re-lowering, no dispatch-key
  recomputation:

      svd = Nx.Defn.compile(&Nx.LinAlg.svd/1, [Nx.template({128, 128}, {:f, 32})], compiler: EMLX)
      # Hold onto `svd` (e.g. in a GenServer, or a module attribute at startup)
      # and call it repeatedly — each call below is pure replay:
      {u, s, vt} = svd.(a)

  This is exactly the strategy `Nx.Serving` and Bumblebee already rely on for
  other compilers, and it works identically here — no `EMLX`-specific code
  needed. EMLX additionally keeps a persistent, structural (shape/op-based,
  not object-identity) dispatch-key cache across calls (see `dispatch_key/3`
  in the source), which is what makes `Nx.Defn.Graph.run/3`'s per-call
  re-tracing and structurally-identical-but-distinct call sites (e.g. many
  copies of the same layer in a model) cheap too — but a caller-held
  `Nx.Defn.compile/3` closure is always cheaper still, since it skips
  retracing entirely.

  ## Compile-time debug flags

  Several development-only checks are gated by `Application.compile_env/3`
  at compile time — when a flag is `false`, the check is erased entirely
  (no BEAM opcodes, zero runtime cost). After changing a flag, run
  `mix compile --force`; values are baked in at compile time, not read at
  runtime. See `config/dev.exs` for commented examples.

      config :emlx, enable_bounds_check: true
      config :emlx, detect_non_finites: true
      config :emlx, compiler_debug: true

  * `:enable_bounds_check` — raises on out-of-bounds indices in gather,
    take, take_along_axis, indexed_add, and indexed_put.
  * `:detect_non_finites` — raises when dot, conv, or `EMLX.Fast` fused
    kernels produce NaN or Inf. Forces extra `EMLX.eval` syncs and breaks
    MLX lazy-graph fusion; never enable in production.
  * `:compiler_debug` — raises on internal `EMLX.Native.Expr` lowering /
    `to_native` invariant violations that would otherwise silently miscompile.
    Cheap (no extra eval syncs); off by default.

  WARNING: `:enable_bounds_check` and `:detect_non_finites` break MLX
  lazy-graph fusion. On non-unified-memory targets (Linux GPU),
  `:enable_bounds_check` also incurs an extra GPU→CPU copy per indexed op.

  ## CPU JIT compilation and SIGCHLD

  On the CPU backend, MLX JIT-compiles fused kernels the first time it sees
  a new graph shape by shelling out to `popen("g++ ...")` and reading the
  result back via `pclose()`. The BEAM sets `SIGCHLD` to `SIG_IGN` by
  default (so it can run as PID 1 in a container without leaking zombies);
  under Linux/POSIX semantics that makes the kernel auto-reap any child the
  instant it exits, so by the time MLX's `pclose()` calls `waitpid()` there
  is nothing left to collect — it fails with `ECHILD`
  (`** (EMLX.NIFError) ... pclose() failed.`), independent of whether the
  compile itself would have succeeded.

  EMLX does not change this VM-wide setting itself — doing so from a
  dependency would silently change zombie-reaping behavior for the whole
  host application. If you hit this error (typically on Linux CPU backend,
  the first time a given fused-op shape runs), restore the default
  disposition yourself, as early as possible in your own application
  (e.g. in your own `Application.start/2`, before `:emlx` or `:nx` start
  doing real work):

      :os.set_signal(:sigchld, :default)

  This is the same fix TensorFlow's Erlang bindings needed for the
  identical reason — see
  https://erlang.org/pipermail/erlang-questions/2020-November/100109.html.
  Only skip this if your VM runs as PID 1 in a container and can't
  tolerate the (small, `g++`-subprocess-shaped) risk of zombie processes.
  See `EMLX.Application` for more detail.
  """

  use EMLX.Macro

  @profile_eval Application.compile_env(:emlx, :profile_eval, false)
  # Emits the profiling call only when `config :emlx, :profile_eval, true` is
  # set at compile time; otherwise expands to `:ok` with zero runtime cost.
  defmacrop maybe_profile(call) do
    if @profile_eval do
      call
    else
      :ok
    end
  end

  defguard is_tensor(device, ref) when is_reference(ref) and is_atom(device)
  @type tensor_ref :: {atom(), reference()}

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
  deftensor copy(tensor)
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
  deftensor einsum(tensors, spec_string)
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
  quantized weights `w` (packed as uint32), using scales and (for `"affine"`)
  biases for dequantization during the computation.

  ## Parameters
    - `x` - Input tensor (e.g., {batch, seq, hidden})
    - `w` - Quantized weights as uint32 (8 int4 values packed per uint32)
    - `scales` - Per-group scale factors (bfloat16, or u8 for microscaled modes)
    - `biases` - Per-group zero points (bfloat16), or `nil` for microscaled modes
    - `transpose` - Whether to transpose weights (default: true)
    - `group_size` - Number of weights per scale/bias group (default: 64)
    - `bits` - Quantization bits (default: 4)
    - `mode` - `"affine"` (default), `"mxfp4"`, `"mxfp8"`, or `"nvfp4"`
  """
  @mlx_function {:quantized_matmul, 10}
  def quantized_matmul(
        {dev_x, ref_x} = _tensor_x,
        {dev_w, ref_w} = _tensor_w,
        {dev_s, ref_s} = _tensor_scales,
        biases,
        transpose \\ true,
        group_size \\ 64,
        bits \\ 4,
        mode \\ "affine"
      )
      when is_tensor(dev_x, ref_x) and is_tensor(dev_w, ref_w) and is_tensor(dev_s, ref_s) do
    {ref_b, biases_device} = unwrap_optional_tensor(biases)
    device = merge_device(merge_device(dev_x, dev_w), merge_device(dev_s, biases_device))
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
        mode,
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
    - `scales` - Per-group scale factors (or u8 for microscaled modes)
    - `biases` - Per-group zero points, or `nil` for microscaled modes
    - `group_size` - Number of weights per group (default: 64)
    - `bits` - Quantization bits (default: 4)
    - `mode` - `"affine"` (default), `"mxfp4"`, `"mxfp8"`, or `"nvfp4"`
  """
  @mlx_function {:dequantize, 8}
  def dequantize(
        {dev_w, ref_w} = _tensor_w,
        {dev_s, ref_s} = _tensor_scales,
        biases,
        group_size,
        bits,
        mode \\ "affine"
      )
      when is_tensor(dev_w, ref_w) and is_tensor(dev_s, ref_s) do
    {ref_b, biases_device} = unwrap_optional_tensor(biases)
    device = merge_device(dev_w, merge_device(dev_s, biases_device))
    {worker, effective_device} = resolve_worker(device)

    job_ref =
      EMLX.NIF.dequantize(worker, ref_w, ref_s, ref_b, group_size, bits, mode, effective_device)
      |> unwrap!()

    await_worker(job_ref) |> wrap_tensor(effective_device)
  end

  @doc """
  Quantizes a floating point tensor to packed format.

  Returns a tuple of `{quantized_weights, scales, biases}` where:
    - `quantized_weights` - Packed uint32 tensor (8 int4 values per uint32)
    - `scales` - Per-group scale factors (or u8 for microscaled modes)
    - `biases` - Per-group zero points, or `nil` for microscaled modes
      (`"mxfp4"`/`"mxfp8"`/`"nvfp4"` — `mx::fp_quantize` doesn't emit biases)

  ## Parameters
    - `w` - Float tensor to quantize
    - `group_size` - Number of weights per group (default: 64)
    - `bits` - Quantization bits (default: 4)
    - `mode` - `"affine"` (default), `"mxfp4"`, `"mxfp8"`, or `"nvfp4"`
  """
  @mlx_function {:quantize, 6}
  def quantize({dev_w, ref_w}, group_size, bits, mode \\ "affine")
      when is_tensor(dev_w, ref_w) do
    device = dev_w
    {worker, effective_device} = resolve_worker(device)

    {weights_ref, scales_ref, biases_ref} =
      EMLX.NIF.quantize(worker, ref_w, group_size, bits, mode, effective_device)
      |> unwrap!()
      |> await_worker()

    {{effective_device, weights_ref}, {effective_device, scales_ref},
     wrap_optional_tensor(biases_ref, effective_device)}
  end

  # `nil` (microscaled modes have no biases) passes through as the atom `nil`
  # to the NIF layer; a real tensor unwraps to its raw ref for device merging.
  defp unwrap_optional_tensor(nil), do: {nil, nil}
  defp unwrap_optional_tensor({device, ref}), do: {ref, device}

  defp wrap_optional_tensor(nil, _device), do: nil
  defp wrap_optional_tensor(ref, device), do: {device, ref}

  # ── mlx::fast ops ───────────────────────────────────────────────────────────

  @doc """
  Fused RMS normalisation (`mlx::fast::rms_norm`).

  Single Metal shader. Normalises over the last axis of `x` and scales by
  `weight`. Output shape and type match `x`.

  Prefer `EMLX.Fast.rms_norm/3` inside `defn`; call this directly only from
  eager (non-defn) code.
  """
  @mlx_function {:fast_rms_norm, 5}
  def fast_rms_norm({dev_x, ref_x}, {dev_w, ref_w}, eps)
      when is_tensor(dev_x, ref_x) and is_tensor(dev_w, ref_w) do
    device = merge_device(dev_x, dev_w)
    {worker, effective_device} = resolve_worker(device)

    job_ref =
      EMLX.NIF.fast_rms_norm(worker, ref_x, ref_w, eps * 1.0, effective_device)
      |> unwrap!()

    await_worker(job_ref) |> wrap_tensor(effective_device)
  end

  @doc """
  Fused rotary position embedding (`mlx::fast::rope`).

  Single Metal shader. Applies RoPE with a scalar position `offset`.

  - `a`           — input `{B, ..., T, D}`; **`T` must be the second-to-last
    axis** — e.g. `{B, T, D}` or, with a heads axis, `{B, H, T, D}`
    (heads-transposed). If `H` (or any other middle axis) is placed *before*
    `T` instead — e.g. `{B, T, H, D}` — `mlx::core::fast::rope` silently
    rotates row `h` at angle `position + h` instead of `position` for every
    `h > 0` (only row 0 is correct); see
    [elixir-nx/emlx#121](https://github.com/elixir-nx/emlx/issues/121).
    For Bumblebee's heads-not-yet-transposed `{B, T, H, D}` convention, use
    `EMLX.fast_rope_ids/6` or `EMLX.fast_rope_positions/6` instead, which
    guard against this.
  - `dims`        — number of feature dims to rotate (≤ last-axis size, must be even)
  - `traditional` — `false` for split-half (Qwen3); `true` for interleaved
  - `base`        — angular frequency base (e.g. 10_000 or 1_000_000)
  - `scale`       — position scale (1.0 unless using NTK-aware scaling)
  - `offset`      — integer token position (length of KV cache already filled)

  Prefer `EMLX.Fast.rope/6` inside `defn`.
  """
  @mlx_function {:fast_rope, 8}
  def fast_rope({dev_a, ref_a}, dims, traditional, base, scale, offset)
      when is_tensor(dev_a, ref_a) do
    {worker, effective_device} = resolve_worker(dev_a)

    job_ref =
      EMLX.NIF.fast_rope(
        worker,
        ref_a,
        dims,
        traditional,
        base * 1.0,
        scale * 1.0,
        offset,
        effective_device
      )
      |> unwrap!()

    await_worker(job_ref) |> wrap_tensor(effective_device)
  end

  @doc """
  Flash-attention style SDPA, no mask (`mlx::fast::scaled_dot_product_attention`).

  GQA-native: `k`/`v` may have fewer heads than `q` — no pre-tiling needed.

  - `q`     — `{B, N_q,  T_q,  D}`
  - `k`     — `{B, N_kv, T_kv, D}`
  - `v`     — `{B, N_kv, T_kv, D}`
  - `scale` — scalar (typically `1 / sqrt(D)`)
  - `sinks` — optional learned per-head attention-sink logits tensor, or `nil`

  Prefer `EMLX.Fast.scaled_dot_product_attention/4` inside `defn`.
  """
  @mlx_function {:fast_sdpa, 7}
  def fast_sdpa({dev_q, ref_q}, {dev_k, ref_k}, {dev_v, ref_v}, scale, sinks \\ nil)
      when is_tensor(dev_q, ref_q) and is_tensor(dev_k, ref_k) and is_tensor(dev_v, ref_v) do
    {ref_s, sinks_device} = unwrap_optional_tensor(sinks)
    device = merge_device(dev_q, merge_device(dev_k, merge_device(dev_v, sinks_device)))
    {worker, effective_device} = resolve_worker(device)

    job_ref =
      EMLX.NIF.fast_sdpa(worker, ref_q, ref_k, ref_v, scale * 1.0, ref_s, effective_device)
      |> unwrap!()

    await_worker(job_ref) |> wrap_tensor(effective_device)
  end

  @doc """
  Flash-attention SDPA with an additive or boolean `mask`.

  `mask` must be broadcast-compatible with `{B, N_q, T_q, T_kv}`.
  Boolean `false` entries are treated as `-∞`.

  `sinks` — optional learned per-head attention-sink logits tensor, or `nil`.

  Prefer `EMLX.Fast.scaled_dot_product_attention/5` inside `defn`.
  """
  @mlx_function {:fast_sdpa_masked, 8}
  def fast_sdpa_masked(
        {dev_q, ref_q},
        {dev_k, ref_k},
        {dev_v, ref_v},
        {dev_m, ref_m},
        scale,
        sinks \\ nil
      )
      when is_tensor(dev_q, ref_q) and is_tensor(dev_k, ref_k) and
             is_tensor(dev_v, ref_v) and is_tensor(dev_m, ref_m) do
    {ref_s, sinks_device} = unwrap_optional_tensor(sinks)

    device =
      merge_device(
        dev_q,
        merge_device(dev_k, merge_device(dev_v, merge_device(dev_m, sinks_device)))
      )

    {worker, effective_device} = resolve_worker(device)

    job_ref =
      EMLX.NIF.fast_sdpa_masked(
        worker,
        ref_q,
        ref_k,
        ref_v,
        scale * 1.0,
        ref_m,
        ref_s,
        effective_device
      )
      |> unwrap!()

    await_worker(job_ref) |> wrap_tensor(effective_device)
  end

  @doc """
  Fused layer normalisation (`mlx::fast::layer_norm`).

  - `x`      — input tensor; normalised over the last axis.
  - `weight` — `{hidden}` scale vector (gamma).
  - `bias`   — `{hidden}` bias vector (beta).
  - `eps`    — numerical stability constant (e.g. `1.0e-5`).

  Prefer `EMLX.Fast.layer_norm/4` inside `defn`.
  """
  @mlx_function {:fast_layer_norm, 6}
  def fast_layer_norm({dev_x, ref_x}, {dev_w, ref_w}, {dev_b, ref_b}, eps)
      when is_tensor(dev_x, ref_x) and is_tensor(dev_w, ref_w) and is_tensor(dev_b, ref_b) do
    device = merge_device(dev_x, merge_device(dev_w, dev_b))
    {worker, effective_device} = resolve_worker(device)

    job_ref =
      EMLX.NIF.fast_layer_norm(worker, ref_x, ref_w, ref_b, eps * 1.0, effective_device)
      |> unwrap!()

    await_worker(job_ref) |> wrap_tensor(effective_device)
  end

  @doc """
  Fused layer normalisation without bias (`mlx::fast::layer_norm`, weight-only variant).

  Prefer `EMLX.Fast.layer_norm/3` inside `defn`.
  """
  @mlx_function {:fast_layer_norm_no_bias, 5}
  def fast_layer_norm_no_bias({dev_x, ref_x}, {dev_w, ref_w}, eps)
      when is_tensor(dev_x, ref_x) and is_tensor(dev_w, ref_w) do
    device = merge_device(dev_x, dev_w)
    {worker, effective_device} = resolve_worker(device)

    job_ref =
      EMLX.NIF.fast_layer_norm_no_bias(worker, ref_x, ref_w, eps * 1.0, effective_device)
      |> unwrap!()

    await_worker(job_ref) |> wrap_tensor(effective_device)
  end

  @doc """
  Fused RoPE with a per-batch offset array (`mlx::fast::rope`, array-offset overload).

  Calls the `const array& offset` overload of `mlx::fast::rope`, where `offset`
  has shape `{B}` — one starting position per batch example. Positions within each
  example are assumed to be sequential: `[offset[b], offset[b]+1, ..., offset[b]+T-1]`.

  Typically you build `offset` by slicing `position_ids[:, 0]` (first token's
  position for each batch example) before calling this function.

  Prefer `EMLX.Fast.rope_with_positions/6` inside `defn`.
  """
  @mlx_function {:fast_rope_ids, 8}
  def fast_rope_ids({dev_a, ref_a}, dims, traditional, base, scale, {dev_off, ref_off})
      when is_tensor(dev_a, ref_a) and is_tensor(dev_off, ref_off) do
    device = merge_device(dev_a, dev_off)
    {worker, effective_device} = resolve_worker(device)

    job_ref =
      EMLX.NIF.fast_rope_ids(
        worker,
        ref_a,
        dims,
        traditional,
        base * 1.0,
        scale * 1.0,
        ref_off,
        effective_device
      )
      |> unwrap!()

    await_worker(job_ref) |> wrap_tensor(effective_device)
  end

  @doc """
  Fused RoPE for arbitrary per-token `position_ids`.

  Uses full `{B, T}` position IDs (not just an offset) and mirrors Bumblebee's
  per-token cos/sin lookup, avoiding the sequential-offset assumption of
  `fast_rope_ids`.

  - `a`            — input tensor `{B, T, H, D}`.
  - `dims`         — number of feature dims to rotate.
  - `traditional`  — currently only `false` is supported.
  - `base`         — angular frequency base (e.g. `1_000_000`).
  - `scale`        — position scale.
  - `position_ids` — `{B, T}` integer tensor.
  """
  @mlx_function {:fast_rope_positions, 8}
  def fast_rope_positions(
        {dev_a, ref_a},
        dims,
        traditional,
        base,
        scale,
        {dev_pos, ref_pos}
      )
      when is_tensor(dev_a, ref_a) and is_tensor(dev_pos, ref_pos) do
    device = merge_device(dev_a, dev_pos)
    {worker, effective_device} = resolve_worker(device)

    job_ref =
      EMLX.NIF.fast_rope_positions(
        worker,
        ref_a,
        dims,
        traditional,
        base * 1.0,
        scale * 1.0,
        ref_pos,
        effective_device
      )
      |> unwrap!()

    await_worker(job_ref) |> wrap_tensor(effective_device)
  end

  @doc """
  Fused RoPE with precomputed inv-frequency vector (`mlx::fast::rope`, freqs overload).

  - `a`          — input tensor; shape `{B, T, ..., D}`.
  - `dims`       — number of feature dims to rotate.
  - `traditional`— `false` for split-half (Qwen3/Bumblebee); `true` for interleaved.
  - `scale`      — position scale (typically `1.0` when using precomputed freqs).
  - `offset`     — `{B}` per-batch starting position tensor.
  - `freqs`      — `{dims/2}` precomputed inverse-frequency tensor.

  Prefer `EMLX.Fast.rope_with_freqs/6` inside `defn`.
  """
  @mlx_function {:fast_rope_with_freqs, 8}
  def fast_rope_with_freqs(
        {dev_a, ref_a},
        dims,
        traditional,
        scale,
        {dev_off, ref_off},
        {dev_f, ref_f}
      )
      when is_tensor(dev_a, ref_a) and is_tensor(dev_off, ref_off) and is_tensor(dev_f, ref_f) do
    device = merge_device(dev_a, merge_device(dev_off, dev_f))
    {worker, effective_device} = resolve_worker(device)

    job_ref =
      EMLX.NIF.fast_rope_with_freqs(
        worker,
        ref_a,
        dims,
        traditional,
        scale * 1.0,
        ref_off,
        ref_f,
        effective_device
      )
      |> unwrap!()

    await_worker(job_ref) |> wrap_tensor(effective_device)
  end

  @doc """
  Flash-attention SDPA with built-in causal mask (`mlx::fast::scaled_dot_product_attention`,
  `mask_mode="causal"`).

  MLX constructs the upper-triangular causal mask internally — no explicit mask
  tensor required. GQA-native: `k`/`v` may have fewer heads than `q`.

  - `q`     — `{B, N_q,  T_q,  D}`
  - `k`     — `{B, N_kv, T_kv, D}`
  - `v`     — `{B, N_kv, T_kv, D}`
  - `scale` — scalar (typically `1 / sqrt(D)`)

  `sinks` — optional learned per-head attention-sink logits tensor, or `nil`.

  Prefer `EMLX.Fast.scaled_dot_product_attention_causal/4` inside `defn`.
  """
  @mlx_function {:fast_sdpa_causal, 7}
  def fast_sdpa_causal({dev_q, ref_q}, {dev_k, ref_k}, {dev_v, ref_v}, scale, sinks \\ nil)
      when is_tensor(dev_q, ref_q) and is_tensor(dev_k, ref_k) and is_tensor(dev_v, ref_v) do
    {ref_s, sinks_device} = unwrap_optional_tensor(sinks)
    device = merge_device(dev_q, merge_device(dev_k, merge_device(dev_v, sinks_device)))
    {worker, effective_device} = resolve_worker(device)

    job_ref =
      EMLX.NIF.fast_sdpa_causal(worker, ref_q, ref_k, ref_v, scale * 1.0, ref_s, effective_device)
      |> unwrap!()

    await_worker(job_ref) |> wrap_tensor(effective_device)
  end

  @doc """
  Causal SDPA with a runtime `key_mask` check performed at the C++ level.

  At the NIF level: evaluates `all(key_mask == 1)` (cheap for small/constant
  tensors). If true → uses the pure causal Metal kernel (no mask allocation).
  If false → builds a combined causal + key_mask additive float mask and calls
  the masked kernel.

  - `q`        — `{B, N_q,  T_q,  D}`
  - `k`        — `{B, N_kv, T_kv, D}`
  - `v`        — `{B, N_kv, T_kv, D}`
  - `scale`    — scalar (typically `1 / sqrt(D)`)
  - `key_mask` — `{B, T_kv}` boolean/int tensor (1 = attend, 0 = padding)
  - `sinks`    — optional learned per-head attention-sink logits tensor, or `nil`

  Prefer `EMLX.Fast.scaled_dot_product_attention_causal_key_masked/5` inside `defn`.
  """
  @mlx_function {:fast_sdpa_causal_key_masked, 9}
  def fast_sdpa_causal_key_masked(
        {dev_q, ref_q},
        {dev_k, ref_k},
        {dev_v, ref_v},
        scale,
        {dev_m, ref_m},
        kv_offset,
        sinks \\ nil
      )
      when is_tensor(dev_q, ref_q) and is_tensor(dev_k, ref_k) and
             is_tensor(dev_v, ref_v) and is_tensor(dev_m, ref_m) and is_integer(kv_offset) do
    {ref_s, sinks_device} = unwrap_optional_tensor(sinks)

    device =
      merge_device(
        dev_q,
        merge_device(dev_k, merge_device(dev_v, merge_device(dev_m, sinks_device)))
      )

    {worker, effective_device} = resolve_worker(device)

    job_ref =
      EMLX.NIF.fast_sdpa_causal_key_masked(
        worker,
        ref_q,
        ref_k,
        ref_v,
        scale * 1.0,
        ref_m,
        kv_offset,
        ref_s,
        effective_device
      )
      |> unwrap!()

    await_worker(job_ref) |> wrap_tensor(effective_device)
  end

  @doc """
  Fused SwiGLU activation: `silu(gate) * up` where `silu(x) = x * sigmoid(x)`.

  - `gate` — gate tensor; silu is applied element-wise.
  - `up`   — up-projection tensor; same shape as `gate`.

  Output has the same shape and dtype as `gate`.

  Prefer `EMLX.Fast.swiglu/2` inside `defn`.
  """
  @mlx_function {:fast_swiglu, 4}
  def fast_swiglu({dev_gate, ref_gate}, {dev_up, ref_up})
      when is_tensor(dev_gate, ref_gate) and is_tensor(dev_up, ref_up) do
    device = merge_device(dev_gate, dev_up)
    {worker, effective_device} = resolve_worker(device)

    job_ref =
      EMLX.NIF.fast_swiglu(worker, ref_gate, ref_up, effective_device)
      |> unwrap!()

    await_worker(job_ref) |> wrap_tensor(effective_device)
  end

  @doc """
  Fused KV cache update + variable-length SDPA in a single Metal command buffer.

  Receives tensors in Bumblebee `{B, T, N, D}` convention. Internally transposes
  to MLX `{B, N, T, D}` for `mlx::fast::scaled_dot_product_attention`, then
  transposes the result back. Returns a 3-tuple of EMLX `{device, ref}` pairs.

  - `q`       — `{B, T_q,   N_q,  D}` post-RoPE query
  - `new_k`   — `{B, T_new, N_kv, D}` current key projection (post-RoPE)
  - `new_v`   — `{B, T_new, N_kv, D}` current value projection
  - `k_cache` — `{B, T_max, N_kv, D}` preallocated key buffer
  - `v_cache` — `{B, T_max, N_kv, D}` preallocated value buffer
  - `offset`  — integer, number of positions already in cache
  - `scale`   — float, `1 / sqrt(head_dim)`

  Returns `{{dev, attn_ref}, {dev, k_upd_ref}, {dev, v_upd_ref}}`.
  """
  @mlx_function {:kv_cache_attention, 9}
  def kv_cache_attention(
        {dev_q, ref_q},
        {_dev_k, ref_k},
        {_dev_v, ref_v},
        {_dev_kc, ref_kc},
        {_dev_vc, ref_vc},
        offset,
        scale
      )
      when is_tensor(dev_q, ref_q) and is_integer(offset) and is_float(scale) do
    device = dev_q
    {worker, effective_device} = resolve_worker(device)

    {attn_ref, k_upd_ref, v_upd_ref} =
      EMLX.NIF.kv_cache_attention(
        worker,
        ref_q,
        ref_k,
        ref_v,
        ref_kc,
        ref_vc,
        offset,
        scale,
        effective_device
      )
      |> unwrap!()
      |> await_worker()

    {{effective_device, attn_ref}, {effective_device, k_upd_ref}, {effective_device, v_upd_ref}}
  end

  @doc """
  Like `kv_cache_attention/7` but also applies `key_mask` to exclude padding
  positions from attention in both prefill and decode steps.

  - `key_mask` — `{B, T_kv}` integer or boolean tensor with `1` = attend,
    `0` = skip (padding). Must cover exactly `valid_len = offset + T_new` positions.

  The combined additive mask applies causal AND key_mask constraints without
  calling `mlx::core::all()`, avoiding Metal sort-kernel compilation issues for
  small tensor shapes.

  Returns `{{dev, attn_ref}, {dev, k_upd_ref}, {dev, v_upd_ref}}`.
  """
  @mlx_function {:kv_cache_attention_masked, 10}
  def kv_cache_attention_masked(
        {dev_q, ref_q},
        {_dev_k, ref_k},
        {_dev_v, ref_v},
        {_dev_kc, ref_kc},
        {_dev_vc, ref_vc},
        offset,
        scale,
        {_dev_m, ref_m}
      )
      when is_tensor(dev_q, ref_q) and is_integer(offset) and is_float(scale) do
    device = dev_q
    {worker, effective_device} = resolve_worker(device)

    {attn_ref, k_upd_ref, v_upd_ref} =
      EMLX.NIF.kv_cache_attention_masked(
        worker,
        ref_q,
        ref_k,
        ref_v,
        ref_kc,
        ref_vc,
        offset,
        scale,
        ref_m,
        effective_device
      )
      |> unwrap!()
      |> await_worker()

    {{effective_device, attn_ref}, {effective_device, k_upd_ref}, {effective_device, v_upd_ref}}
  end

  @doc """
  Fused KV cache update + SDPA for the native NIF loop (BNHD layout).

  Accepts `q`, `new_k`, `new_v` already transposed to `{B, N, T, D}` and a
  pre-allocated cache in `{B, N_kv, T_max, D}` layout.

  Internally, the cache arrays are **move-extracted** from their ENIF resources
  before `slice_update` so that MLX's donation optimisation fires at eval time:
  the existing Metal buffer is reused in-place — no new allocation.

  Returns `{{dev, attn_ref}, {dev, k_upd_ref}, {dev, v_upd_ref}}`.
  """
  @mlx_function {:kv_cache_sdpa_update, 9}
  def kv_cache_sdpa_update(
        {dev_q, ref_q},
        {_dev_k, ref_k},
        {_dev_v, ref_v},
        {_dev_kc, ref_kc},
        {_dev_vc, ref_vc},
        offset,
        scale
      )
      when is_tensor(dev_q, ref_q) and is_integer(offset) and is_float(scale) do
    device = dev_q
    {worker, effective_device} = resolve_worker(device)

    {attn_ref, k_upd_ref, v_upd_ref} =
      EMLX.NIF.kv_cache_sdpa_update(
        worker,
        ref_q,
        ref_k,
        ref_v,
        ref_kc,
        ref_vc,
        offset,
        scale,
        effective_device
      )
      |> unwrap!()
      |> await_worker()

    {{effective_device, attn_ref}, {effective_device, k_upd_ref}, {effective_device, v_upd_ref}}
  end

  # Microscaled modes pin an exact {group_size, bits} pair (MLX's
  # `fp_quantize` — see `mlx::core::quantize` in `mlx/ops.h`), checked
  # directly against the same MLX version this repo vendors.
  @microscaled_constraints %{"mxfp4" => {32, 4}, "mxfp8" => {32, 8}, "nvfp4" => {16, 4}}
  @valid_quantization_modes ~w(affine mxfp4 mxfp8 nvfp4)

  defp validate_quantization_mode!(mode) when mode in @valid_quantization_modes, do: :ok

  defp validate_quantization_mode!(mode) do
    raise ArgumentError,
          "EMLX.quantize/2: :mode must be one of #{inspect(@valid_quantization_modes)}, " <>
            "got: #{inspect(mode)}"
  end

  defp validate_microscaled_constraints!("affine", _group_size, _bits), do: :ok

  defp validate_microscaled_constraints!(mode, group_size, bits) do
    {expected_gs, expected_bits} = Map.fetch!(@microscaled_constraints, mode)

    cond do
      group_size != expected_gs ->
        raise ArgumentError,
              "EMLX.quantize/2: mode #{inspect(mode)} requires group_size=#{expected_gs}, " <>
                "got: #{inspect(group_size)}"

      bits != expected_bits ->
        raise ArgumentError,
              "EMLX.quantize/2: mode #{inspect(mode)} requires bits=#{expected_bits}, " <>
                "got: #{inspect(bits)}"

      true ->
        :ok
    end
  end

  @doc """
  Stable wrapper for causal self attention with an owned KV cache.

  This is the public semantic API for decode paths like Qwen and Llama that need
  compact GQA KV heads, fused cache update, valid prefix slicing, and causal
  SDPA through EMLX's native MLX kernels.

  Inputs use Bumblebee convention:

  - `query` — `{B, T_q, N_q, D}`
  - `new_key` / `new_value` — `{B, T_new, N_kv, D}`
  - `key_cache` / `value_cache` — `{B, T_max, N_kv, D}`
  - `offset` — integer count of positions already present in the cache

  Options:

  - `:scale` — required float, usually `1 / sqrt(head_dim)`
  - `:key_mask` — optional `{B, offset + T_new}` mask with `1` = attend and
    `0` = skip padding

  Returns `{attention, updated_key_cache, updated_value_cache}` as raw EMLX
  `{device, ref}` pairs. Keeping the caches in this representation lets callers
  store and reuse device resident cache buffers without converting them back to
  `Nx.Tensor` between decode steps.
  """
  def causal_kv_attention(query, new_key, new_value, key_cache, value_cache, offset, opts)
      when is_integer(offset) and offset >= 0 and is_list(opts) do
    scale = Keyword.fetch!(opts, :scale)

    case Keyword.get(opts, :key_mask) do
      nil ->
        kv_cache_attention(query, new_key, new_value, key_cache, value_cache, offset, scale)

      key_mask ->
        kv_cache_attention_masked(
          query,
          new_key,
          new_value,
          key_cache,
          value_cache,
          offset,
          scale,
          key_mask
        )
    end
  end

  @doc """
  Quantize a dense 2-D `Nx.Tensor` and return an annotated quantized tensor.

  The returned tensor carries the original logical shape and type (e.g.
  `{:s, 4}`). Its backend stores the packed uint32 data and a
  `EMLX.Quantization.Config` with scales, biases, `group_size`, and `bits`.

  ## Options

  * `:type` — storage type: `{:s, 2}`, `{:s, 4}` (default), or `{:s, 8}`.
  * `:group_size` — 32, 64, or 128 (default 64). Must evenly divide the last
    dimension of `tensor`. Microscaled modes pin this to a specific value
    (see `:mode` below).
  * `:mode` — `"affine"` (default, real biases), or a microscaled mode —
    `"mxfp4"` (group_size 32, bits 4), `"mxfp8"` (group_size 32, bits 8), or
    `"nvfp4"` (group_size 16, bits 4). Microscaled modes have no biases
    (`mx::fp_quantize` returns only `(wq, scales)`); the returned tensor's
    `EMLX.Quantization.Config.biases` is `nil`.
  """
  def quantize(%Nx.Tensor{} = tensor, opts) when is_list(opts) do
    type = Keyword.get(opts, :type, {:s, 4})
    {_, bits} = type
    group_size = Keyword.get(opts, :group_size, 64)
    mode = Keyword.get(opts, :mode, "affine")

    validate_quantization_mode!(mode)
    validate_microscaled_constraints!(mode, group_size, bits)

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
    {weight_ref, scales_ref, biases_ref} = EMLX.quantize(device_ref, group_size, bits, mode)

    scales = EMLX.Backend.to_nx(scales_ref)
    biases = biases_ref && EMLX.Backend.to_nx(biases_ref)

    config = %EMLX.Quantization.Config{
      scales: scales,
      biases: biases,
      group_size: group_size,
      bits: bits,
      mode: mode
    }

    weight_shape = EMLX.shape(weight_ref)
    %Nx.Tensor{} = template = Nx.template(Nx.shape(tensor), type)

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
  dense float tensor by calling `mx::dequantize`. Supports every mode
  `EMLX.quantize/2` accepts (`"affine"` and the microscaled variants).
  """
  def dequantize(
        %Nx.Tensor{
          data: %EMLX.Backend{ref: weight_ref, quantization_config: cfg}
        } = _qw
      )
      when not is_nil(cfg) do
    biases_ref = cfg.biases && EMLX.Backend.from_nx(cfg.biases)

    EMLX.dequantize(
      weight_ref,
      EMLX.Backend.from_nx(cfg.scales),
      biases_ref,
      cfg.group_size,
      cfg.bits,
      cfg.mode
    )
    |> EMLX.Backend.to_nx()
  end

  @doc """
  Run `activation @ dequantize(qw)` using `mx::quantized_matmul`.

  `qw` must be a quantized tensor produced by `EMLX.quantize/2`. Raises
  `ArgumentError` if both arguments are quantized.
  """
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

    biases_ref = cfg.biases && EMLX.Backend.from_nx(cfg.biases)

    result =
      EMLX.quantized_matmul(
        EMLX.Backend.from_nx(activation),
        qw.data.ref,
        EMLX.Backend.from_nx(cfg.scales),
        biases_ref,
        true,
        cfg.group_size,
        cfg.bits,
        cfg.mode
      )

    EMLX.Backend.to_nx(result)
  end

  def to_blob({device, ref} = tensor) when is_tensor(device, ref) do
    maybe_profile(EMLX.Profiling.inc_to_blob())
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
    maybe_profile(EMLX.Profiling.inc_to_blob())
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

  @doc false
  def unwrap!(:ok), do: :ok
  def unwrap!({:ok, result}), do: result
  def unwrap!({:error, error}), do: raise(EMLX.NIFError, List.to_string(error))

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
    maybe_profile(EMLX.Profiling.inc_eval())

    EMLX.Telemetry.span_eval(fn ->
      {worker, _effective_device} = resolve_worker(device)
      job_ref = EMLX.NIF.eval(worker, ref) |> unwrap!()
      await_worker(job_ref)
    end)
  end

  @doc """
  Moves a tensor to `target_device` (`:cpu` or `:gpu`) without deallocating
  the source. This is a no-op if the tensor is already on the target device.

  Unlike `Nx.backend_transfer/2`, this does not round-trip through a binary
  and does not call `backend_deallocate` on the original tensor. Internally
  it schedules `mlx::core::contiguous(arr, target_device)` on the target
  device's worker thread, which on Apple Silicon (unified memory) avoids any
  physical data copy.
  """
  def to_device(tensor, target_device)

  def to_device({dev, _} = tensor, dev), do: tensor

  def to_device({old_device, ref} = tensor, target_device)
      when is_tensor(old_device, ref) and target_device in [:cpu, :gpu] do
    # Materialize the source on its own device's thread before constructing the
    # contiguous op. Without this, evaluating contiguous(lazy_gpu_array, cpu)
    # on the CPU worker would traverse the graph and hit a missing GPU stream.
    eval(tensor)
    {worker, effective_device} = resolve_worker(target_device)
    job_ref = EMLX.NIF.to_device(worker, ref, effective_device) |> unwrap!()
    await_worker(job_ref) |> wrap_tensor(effective_device)
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

  # `runtime_calls` (see `EMLX.Native.Expr`'s moduledoc "Runtime calls"
  # section) is only ever non-empty for an `eval_program` job whose compiled
  # program contains an inlined `:runtime_call` node — every other call site
  # keeps the 1-arg form and can never receive an `:emlx_runtime_call`
  # request in the first place. A request is served here, mid-flight, before
  # the final `{^job_ref, _}` reply for *this* `eval_program` call arrives:
  # the worker thread is blocked inside `EMLXRuntimeCall::eval_cpu`/`eval_gpu`
  # (emlx_compiler.cpp) waiting on exactly the reply this loop sends back via
  # `EMLX.NIF.resolve_runtime_call/3`.
  @doc false
  def await_worker(job_ref, runtime_calls \\ [], tensors \\ [], dev \\ nil) do
    receive do
      # Worker NIFs (sync bodies) return one of:
      #   nx::nif::ok(env)         => :ok
      #   nx::nif::ok(env, term)   => {:ok, term}
      #   nx::nif::error(env, msg) => {:error, msg}
      # The async wrapper forwards the payload as-is in {ref, payload}.
      {^job_ref, :ok} ->
        :ok

      {^job_ref, {:ok, result}} ->
        result

      {^job_ref, {:error, reason}} ->
        raise(EMLX.NIFError, List.to_string(reason))

      {:emlx_runtime_call, pending, callback_index, args_binaries} ->
        handle_runtime_call(pending, callback_index, args_binaries, runtime_calls, tensors, dev)
        await_worker(job_ref, runtime_calls, tensors, dev)
    end
  end

  # Runs the real Elixir callback for one `:emlx_runtime_call` request and
  # replies via `EMLX.NIF.resolve_runtime_call/3`, waking the worker thread
  # blocked inside `EMLXRuntimeCall::eval_cpu`/`eval_gpu`. A raising callback
  # is caught and reported as an `:error` reply instead of crashing this
  # process (which would otherwise leave the worker thread blocked forever).
  #
  # `tensors` is *this* `eval_program` call's own materialised input list
  # (`build_native_eval_fn/7`'s own `tensors`, matching this program's own
  # `:parameter` numbering exactly — including for a `:runtime_call` nested
  # inside a `while` body, whose own re-entrant compile has its own
  # independent parameter numbering over its own materialised inputs). Used
  # only to substitute back a quantized argument's original bound tensor —
  # see `EMLX.Native.Expr`'s moduledoc "Runtime calls" section and
  # `arg_param_positions`'s doc.
  #
  # The real callback runs inside `EMLX.CommandQueue.with_queue/2` bound to
  # `EMLX.Application.runtime_call_worker(dev)` — see that function's doc
  # for why: any eager EMLX call the callback itself makes (e.g.
  # `EMLX.Quantization.dequantize_callback/2` calling `EMLX.dequantize/1`)
  # must never be routed back to the worker that is, right now, blocked
  # inside `EMLXRuntimeCall::eval_cpu`/`eval_gpu` waiting for exactly this
  # callback to return.
  defp handle_runtime_call(pending, callback_index, args_binaries, runtime_calls, tensors, dev) do
    %{
      callback: callback,
      args_template: args_template,
      arg_param_positions: positions,
      opts: opts
    } = Enum.at(runtime_calls, callback_index)

    {args_container, {[], []}} =
      Nx.Defn.Composite.traverse(
        args_template,
        {args_binaries, positions},
        fn leaf, {[bin | bins_rest], [pos | pos_rest]} ->
          value =
            case pos && Enum.at(tensors, pos) do
              %Nx.Tensor{data: %EMLX.Backend{quantization_config: %EMLX.Quantization.Config{}}} =
                  t ->
                t

              _ ->
                bin |> Nx.from_binary(leaf.type) |> Nx.reshape(leaf.shape)
            end

          {value, {bins_rest, pos_rest}}
        end
      )

    callback_queue = %EMLX.CommandQueue{
      ref: EMLX.Application.runtime_call_worker(dev),
      device: dev
    }

    reply =
      try do
        result =
          EMLX.CommandQueue.with_queue(callback_queue, fn -> callback.(args_container, opts) end)

        binaries =
          [result]
          |> Nx.Defn.Composite.flatten_list()
          |> Enum.map(&Nx.to_binary/1)

        {:ok, binaries}
      rescue
        e -> {:error, Exception.format(:error, e, __STACKTRACE__)}
      end

    case reply do
      {:ok, binaries} -> EMLX.NIF.resolve_runtime_call(pending, :ok, binaries)
      {:error, message} -> EMLX.NIF.resolve_runtime_call(pending, :error, message)
    end
    |> unwrap!()
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
    maybe_profile(EMLX.Profiling.inc_item())
    {worker, _effective_device} = resolve_worker(device)
    job_ref = EMLX.NIF.item(worker, ref) |> unwrap!()
    await_worker(job_ref)
  end

  @behaviour Nx.Defn.Compiler

  # Known EMLX-specific compiler opts. `:command_queue` is injected by
  # `__partitions_options__/1` but may also be passed directly by callers
  # that manage their own queues (equivalent to a manual `with_queue`).
  @valid_compiler_keys [:device, :max_concurrency, :command_queue]

  # Process-lifetime dispatch cache backing `dispatch_key/3` +
  # `get_or_compile_program/6` (see their docs) — a compiled program is keyed
  # by a *structural* signature of its `Expr` (not object identity), so
  # it survives across `Nx.Defn.Graph.run/3`'s per-call re-tracing and is
  # shared across structurally-identical call sites (e.g. every one of
  # Qwen3's 28 attention layers), not just within one closure's lifetime.
  @native_dispatch_cache_table :emlx_native_dispatch_cache

  # A second, process-lifetime cache in front of `dispatch_key/3`'s own
  # (expensive — O(nodes), plus per-opaque-scope SHA256 hashing) structural
  # walk, keyed by `output_expr`'s own node identity rather than its
  # structural signature. This matters for `run_while_loop/3`'s host-driven
  # `cond_fn`/`body_fn`: `Nx.Defn.jit/2` retraces `fn _ -> body_expr end`
  # once and caches *that* trace by argument template, so every subsequent
  # call re-enters `__jit__`/`build_eval_fn` with the *exact same* `Expr`
  # (identical ids, not just structurally identical) — walking it again on
  # every decode step is pure waste. (This is unlike `Nx.Defn.Graph.run/3`'s
  # per-stage re-tracing, which *does* mint fresh ids each call — hence
  # `dispatch_key/3` still has to fall back to the structural walk on a miss.)
  @dispatch_key_by_id_table :emlx_dispatch_key_by_id

  @impl Nx.Defn.Compiler
  def __jit__(key, vars, fun, args_list, opts) do
    __compile__(key, vars, fun, opts).(args_list)
  end

  @impl Nx.Defn.Compiler
  def __compile__(_key, vars, fun, opts) do
    Keyword.validate!(opts, @valid_compiler_keys)
    queue = Keyword.get(opts, :command_queue)
    device = Keyword.get(opts, :device, default_device())

    wrap_with_queue(queue, native_compile(vars, fun, device))
  end

  # Attempts to lower `fun.(vars)` to an `EMLX.Native.Expr` program and build a
  # compiled program resource, returning the resulting eval closure. Single-mode:
  # any op not yet implemented raises `ArgumentError` ("does not yet lower op
  # ...") straight through to the caller — there is no whole-`defn` fallback lane.
  defp native_compile(vars, fun, device) do
    output_expr = fun.(vars)
    {worker, effective_device} = resolve_worker(device)
    # `vars`' flattened leaf count is the true call arity — passed through so
    # `lower/2` can densify its input list even when `output_expr` doesn't
    # reference every parameter position (see EMLX.Native.Expr.lower/2 doc).
    num_inputs = vars |> Nx.Defn.Composite.flatten_list() |> length()
    build_eval_fn(output_expr, worker, effective_device, num_inputs, fun, vars)
  end

  # Wraps a compiled eval closure so each invocation routes through `queue`.
  #
  # A CommandQueue is a *transient* execution scope: it is bound only for the
  # duration of the call and torn down immediately after. The compiled program
  # therefore runs on the queue's worker thread, producing a lazy result whose
  # MLX graph is tied to that thread's stream. We must materialise the outputs
  # *before* the binding is restored — otherwise a later read (e.g. `to_blob`)
  # resolves to the device-default worker, whose thread does not own the
  # queue's stream ("There is no Stream(gpu, N) in current thread").
  defp wrap_with_queue(nil, eval_fn), do: eval_fn

  defp wrap_with_queue(queue, eval_fn) do
    fn inputs ->
      EMLX.CommandQueue.with_queue(queue, fn ->
        result = eval_fn.(inputs)
        force_eval_outputs(result)
        result
      end)
    end
  end

  # Evals every EMLX-backed output tensor on the currently-bound worker so the
  # result is materialised (not a dangling lazy graph) once the queue unbinds.
  defp force_eval_outputs(result) do
    result
    |> List.wrap()
    |> Enum.each(fn container ->
      Nx.Defn.Composite.traverse(container, fn
        %Nx.Tensor{data: %EMLX.Backend{ref: ref}} = tensor ->
          eval(ref)
          tensor

        leaf ->
          leaf
      end)
    end)
  end

  # Routes a traced expression to the right eval-closure builder. A `while`
  # is a structural split point (`Nx.Defn.Graph`) only when it can't be
  # natively lowered to a single `EMLXWhile` primitive (see
  # `EMLX.Native.Expr.native_while_eligible?/2` and `split_point?/1`) — e.g.
  # it has a `:runtime_call`/hook in its condition or body (nesting alone
  # doesn't disqualify it: a nested `while` lowers to a nested `EMLXWhile`
  # primitive). An eligible `while` lowers in-graph like any other op, so the
  # whole loop runs inside one `eval_program` NIF call instead of driving it from
  # Elixir. `:runtime_call` outside a `while` never splits the graph either:
  # it lowers in-graph to a real `:runtime_call` opcode backed by a genuine
  # `mx::core::Primitive` (see `EMLX.Native.Expr`'s moduledoc "Runtime
  # calls" section and `emlx_compiler.cpp`'s `EMLXRuntimeCall`), whose
  # callback fires from `await_worker/2`'s `:emlx_runtime_call` receive
  # clause while the single `eval_program` NIF call for that stage is still
  # in flight. An `EMLX.Fast.*` fused kernel is not a `:runtime_call` node at
  # all (it's a plain `:metadata`-tagged expr — see `EMLX.Fast`'s
  # moduledoc), so it never splits either; it lowers in-graph to a single
  # fused opcode (`EMLX.Native.Expr`'s `:metadata` clause).
  #
  #   * no split point in the parent scope -> one flat native program
  #     (possibly containing one or more inlined `:runtime_call` nodes
  #     and/or native `:while` loops).
  #   * a bare tail `while` that still needs a split (base case) ->
  #     host-driven; the condition/body run by re-entering this compiler, so
  #     a nested split point recurses through the same path.
  #   * a split point with surrounding work -> `Nx.Defn.Graph.split/2` on
  #     every non-native-lowerable `while`, replayed by `Nx.Defn.Graph.run/3`
  #     with `compiler: EMLX`; each stage re-enters this compiler (flat
  #     stages compile flat, isolated split-point stages hit the base case
  #     above).
  defp build_eval_fn(output_expr, worker, effective_device, num_inputs, fun, vars) do
    cond do
      # Only the legacy host-driven path when the bare tail `while` still
      # needs a split (i.e. isn't natively lowerable) -- an eligible bare
      # `while` falls through to the flat native path below instead, same as
      # any other native-lowerable expression.
      bare_while?(output_expr) and contains_split_point?(output_expr) ->
        build_while_base_eval_fn(output_expr, effective_device)

      not contains_split_point?(output_expr) ->
        base_key = dispatch_key(output_expr, num_inputs, effective_device)

        {resource, hooks, runtime_calls} =
          get_or_compile_program(
            base_key,
            %{},
            fn -> output_expr end,
            num_inputs,
            worker,
            effective_device
          )

        build_native_eval_fn(
          base_key,
          resource,
          hooks,
          runtime_calls,
          output_expr,
          num_inputs,
          effective_device,
          fun,
          vars
        )

      true ->
        build_split_chain_eval_fn(output_expr, effective_device)
    end
  end

  # True when the parent scope contains a `while` split point. `post_order/2`
  # treats it as an opaque leaf, so this only sees parent-scope split
  # points — a nested one inside a sub-scope surfaces when that sub-scope is
  # compiled.
  defp contains_split_point?(output_expr) do
    output_expr
    |> EMLX.Defn.Tree.post_order(&EMLX.Native.Expr.scope_dependencies/1)
    |> Enum.any?(&split_point?/1)
  end

  # A `while` is only a split point when it can't be natively lowered to a
  # single `EMLXWhile` primitive (see `EMLX.Native.Expr.native_while_eligible?/2`
  # and its `:while` moduledoc section) -- e.g. it has a `:runtime_call` or a
  # hook in its condition/body (a nested `while` alone is fine -- it lowers to
  # a nested `EMLXWhile`). An eligible `while` is left in place:
  # `EMLX.Native.Expr.lower/1` (via `expand_node`) lowers it straight into a
  # flat native program below, no split needed.
  defp split_point?(%Nx.Tensor{
         data: %Nx.Defn.Expr{op: :while, args: [_initial, _arg, condition, body]}
       }) do
    not EMLX.Native.Expr.native_while_eligible?(condition, body)
  end

  defp split_point?(%Nx.Tensor{}), do: false

  # Compiles an `EMLX.Native.Expr` program to a NIF resource on `worker`.
  # Captured host tensors are copied onto `device` first: a `defn`-embedded
  # constant tensor (e.g. an RNG algorithm constant) is traced with the default
  # backend, so it must be moved to EMLX before `to_native` can extract its ref.
  defp compile_native_program(worker, device, %EMLX.Native.Expr{} = program) do
    program = ensure_emlx_captures(program, device)
    wire_program = EMLX.Native.Expr.to_native(program)

    EMLX.NIF.compile_program(worker, wire_program)
    |> unwrap!()
    |> await_worker()
  end

  # Ensures every captured tensor is EMLX-backed on `device` (copies any that
  # were traced with another backend), so `to_native` can read a NIF ref per
  # capture.
  defp ensure_emlx_captures(%EMLX.Native.Expr{captures: captures} = program, device) do
    captures =
      Enum.map(captures, fn
        {ref, %Nx.Tensor{data: %EMLX.Backend{}} = tensor} ->
          {ref, tensor}

        {ref, %Nx.Tensor{} = tensor} ->
          {ref, Nx.backend_copy(tensor, {EMLX.Backend, device: device})}
      end)

    %{program | captures: captures}
  end

  # Materialises defn input lazy refs to real bound %Nx.Tensor{} values on
  # `dev` (copying any non-EMLX-backed tensor).
  defp materialise_input_tensors(params, dev) do
    Enum.map(params, fn lazy ->
      case lazy.() do
        %Nx.Tensor{data: %EMLX.Backend{}} = t -> t
        %Nx.Tensor{} = t -> Nx.backend_copy(t, {EMLX.Backend, device: dev})
      end
    end)
  end

  # Derives the "quantization signature" a call's bound inputs need. A
  # quantized weight's Nx-visible `.shape`/`.type` deliberately mirror its
  # logical dense shape (so eager `Nx.dot` can transparently reroute to
  # `EMLX.quantized_matmul` via `EMLX.Backend.dot/7`'s runtime dispatch) —
  # invisible to `EMLX.Native.Expr.lower/2` at trace time. Once real tensors
  # are bound (here, at call time) their `quantization_config` is visible,
  # so a specialized program can be built that lowers
  # the specific `:dot` nodes consuming these positions to
  # `:quantized_matmul` instead of plain `:dot`. Returns `%{}` when nothing
  # is quantized — the common, zero-overhead case.
  #
  # `allowed_positions` (`EMLX.Native.Expr.quantizable_param_positions/1`,
  # computed once from `output_expr`) restricts this to positions actually
  # consumed as a `:dot` right operand somewhere in the program: a
  # quantized tensor merely *passed through* to something else (e.g. an
  # `EMLX.Fast.*` fused kernel's `:__EMLX__` metadata operands, or
  # `EMLX.Quantization.dequantize/1`'s own `:runtime_call` operand) is never
  # specialized on, so it must not affect the cache key
  # either — otherwise every distinct quantized tensor identity would mint
  # its own permanently-unreused dispatch-cache entry for an
  # otherwise structurally-identical program.
  defp quant_signature(tensors, allowed_positions) do
    tensors
    |> Enum.with_index()
    |> Enum.reduce(%{}, fn {tensor, pos}, sig ->
      cond do
        not MapSet.member?(allowed_positions, pos) -> sig
        is_nil(tensor.data.quantization_config) -> sig
        true -> Map.put(sig, pos, tensor.data.quantization_config)
      end
    end)
  end

  defp input_refs(tensors) do
    Enum.map(tensors, fn %Nx.Tensor{data: %EMLX.Backend{ref: {_dev, ref}}} -> ref end)
  end

  # Builds the per-call eval closure for the flat (no-while) native path.
  # `output_expr` is only used *here*, up front, to derive `output_template`,
  # `real_output_count`, `quantizable_positions`, and `output_param_positions`
  # (each bounded by output/parameter count, not graph size) — the returned
  # closure itself never captures `output_expr` directly. The one exception
  # is the rare quantized-specialization branch below, which needs the full
  # expression to lower+compile a specialized program; it re-derives
  # `output_expr` on demand via `fun.(vars)` instead of keeping it resident
  # in every call's closure environment — see `get_or_compile_program/6`.
  # `base_key` is the structural dispatch key (`dispatch_key/3`), threaded
  # through so a quantized specialization is cached persistently too, not
  # just the plain program.
  # `output_template` (derived from `output_expr` above) serves as the
  # type/shape template for reconstructing output tensors after the NIF
  # returns raw resource refs. `plain_hooks`
  # (from `EMLX.Native.Expr.lower/2`'s `hooks` field, see its moduledoc
  # "Hooks" section) ride along as extra outputs after the real ones; the
  # corresponding Elixir callbacks fire here, once, right after the single
  # NIF call returns. `plain_runtime_calls` (from the `runtime_calls` field,
  # see its moduledoc "Runtime calls" section) is threaded down to
  # `await_worker/2`, which fires each one's real callback *during* the
  # single `eval_program` NIF call, as its `:emlx_runtime_call` requests
  # arrive.
  defp build_native_eval_fn(
         base_key,
         plain_resource,
         plain_hooks,
         plain_runtime_calls,
         output_expr,
         num_inputs,
         effective_device,
         fun,
         vars
       ) do
    output_template = Nx.Defn.Composite.traverse(output_expr, &Nx.to_template/1)
    real_output_count = [output_template] |> Nx.Defn.Composite.flatten_list() |> length()

    # See `quant_signature/2`'s doc for why this must be restricted to
    # positions actually consumed by a `:dot` — otherwise a quantized
    # tensor merely passed to e.g. an `EMLX.Fast.*` fused kernel would
    # fragment the dispatch cache with one dead-weight entry per
    # distinct tensor identity.
    quantizable_positions = EMLX.Native.Expr.quantizable_param_positions(output_expr)

    # Static (independent of quant_signature): which output leaves are a bare,
    # untouched pass-through of a parameter position — e.g. a loop-invariant
    # carry threaded across an Nx.Defn.Graph while-split stage boundary, never
    # consumed by any op in *this* stage. Flat, parallel to out_refs below
    # (both walk output_expr/output_template in the same Composite order).
    output_param_positions =
      [output_expr]
      |> Nx.Defn.Composite.flatten_list()
      |> Enum.map(fn
        %Nx.Tensor{data: %Nx.Defn.Expr{op: :parameter, args: [pos]}} -> pos
        _ -> nil
      end)

    fn [params] ->
      {worker, dev} = resolve_worker(effective_device)
      tensors = materialise_input_tensors(params, dev)
      quant_signature = quant_signature(tensors, quantizable_positions)

      # The plain (non-quantized) resource was already looked up/compiled
      # (and cached under `{base_key, %{}}`) by our caller, so the common
      # case skips a redundant cache round-trip; a quantized signature is
      # looked up (or compiled once and cached) here — see
      # get_or_compile_program/6. `fn -> fun.(vars) end` is only forced on
      # that cache's own miss (once per distinct `{base_key, quant_signature}`
      # for the process's lifetime) — never eagerly, so a cache *hit* here
      # costs nothing beyond the lookup itself.
      {program_resource, hooks, runtime_calls} =
        if quant_signature == %{} do
          {plain_resource, plain_hooks, plain_runtime_calls}
        else
          get_or_compile_program(
            base_key,
            quant_signature,
            fn -> fun.(vars) end,
            num_inputs,
            worker,
            dev
          )
        end

      job_ref =
        EMLX.NIF.eval_program(worker, program_resource, input_refs(tensors))
        |> unwrap!()

      all_refs = await_worker(job_ref, runtime_calls, tensors, dev)

      {out_refs, hook_refs} = Enum.split(all_refs, real_output_count)

      # Reconstruct output tensors: traverse the output expression template
      # (provides type/shape/names) and attach each returned resource ref as
      # a proper EMLX.Backend data value. Exception: a quantized-parameter
      # pass-through leaf (see output_param_positions above) — its Nx-visible
      # shape/type is a logical fiction (see quant_signature/2) that does not
      # match the physical (packed) array the NIF actually returns for an
      # untouched pass-through, so EMLX.Backend.to_nx/2 would raise a shape
      # mismatch. Substitute back the original bound tensor (quantization_config
      # and all) instead for those leaves. Checked against the tensor's own
      # `quantization_config` directly rather than `quant_signature` — the
      # latter is deliberately narrowed to positions a `:dot` node actually
      # consumes (see `quantizable_param_positions/1`'s doc), but a
      # loop-invariant quantized carry passed straight through untouched
      # (this leaf) need not be consumed by any `:dot` in this stage at all.
      {output_container, {[], []}} =
        Nx.Defn.Composite.traverse(
          output_template,
          {out_refs, output_param_positions},
          fn leaf, {[ref | refs_rest], [pos | pos_rest]} ->
            emlx_tensor =
              case pos && Enum.at(tensors, pos) do
                %Nx.Tensor{data: %EMLX.Backend{quantization_config: %EMLX.Quantization.Config{}}} =
                    t ->
                  t

                _ ->
                  EMLX.Backend.to_nx({dev, ref}, leaf)
              end

            {emlx_tensor, {refs_rest, pos_rest}}
          end
        )

      fire_hooks(hooks, hook_refs, dev)

      [output_container]
    end
  end

  # Looks up (or lazily lowers+compiles) the program for `{base_key,
  # quant_signature}` in the process-lifetime dispatch cache.
  # `base_key` (see `dispatch_key/3`) is a structural signature of the
  # `Expr` — stable across `Nx.Defn.Graph.run/3`'s per-call re-tracing and
  # shared across structurally-identical call sites (e.g. every one of
  # Qwen3's 28 attention layers's surrounding stages), not scoped to one
  # `build_native_eval_fn/6` closure the way this cache was pre-Stage-32.
  # First-compile-wins under concurrent calls with a never-before-seen key:
  # `:ets.insert_new/2` returning `false` means another caller raced us and
  # won — we discard our (still valid, just unused) compiled resource and
  # reuse theirs. Wasted work on that race, not a correctness hazard (no
  # shared mutable state is corrupted).
  #
  # `output_expr_thunk` (a 0-arity fun, not `output_expr` itself) is only
  # forced on this cache's own miss (the `[]` branch below) — callers that
  # already have `output_expr` in hand (the plain, non-quantized path) pay
  # nothing extra for the indirection, but the quantized-specialization
  # caller (`build_native_eval_fn/9`) that re-derives `output_expr` via
  # `fun.(vars)` only actually retraces when this cache misses, not on
  # every call.
  defp get_or_compile_program(
         base_key,
         quant_signature,
         output_expr_thunk,
         num_inputs,
         worker,
         dev
       ) do
    cache_key = {base_key, quant_signature}
    table = dispatch_cache_table()

    case :ets.lookup(table, cache_key) do
      [{_key, resource, hooks, runtime_calls}] ->
        {resource, hooks, runtime_calls}

      [] ->
        program = EMLX.Native.Expr.lower(output_expr_thunk.(), num_inputs, quant_signature)
        resource = compile_native_program(worker, dev, program)

        if :ets.insert_new(table, {cache_key, resource, program.hooks, program.runtime_calls}) do
          {resource, program.hooks, program.runtime_calls}
        else
          [{_key, winner_resource, winner_hooks, winner_runtime_calls}] =
            :ets.lookup(table, cache_key)

          {winner_resource, winner_hooks, winner_runtime_calls}
        end
    end
  end

  # Lazily creates (idempotently — races are resolved by `:ets.new/2` raising
  # `ArgumentError` on the loser, which we swallow) the named, public,
  # process-lifetime ETS table backing the dispatch cache. Named
  # (not `:persistent_term`-stashed) so no GC-triggering `:persistent_term`
  # writes are needed to publish it — the atom name is the handle.
  defp dispatch_cache_table, do: ensure_named_ets_table(@native_dispatch_cache_table)

  # Lazily creates (idempotently — races are resolved by `:ets.new/2` raising
  # `ArgumentError` on the loser, which we swallow) a named, public,
  # process-lifetime ETS table. Named (not `:persistent_term`-stashed) so no
  # GC-triggering `:persistent_term` writes are needed to publish it — the
  # atom name is the handle.
  defp ensure_named_ets_table(name) do
    case :ets.whereis(name) do
      :undefined ->
        try do
          :ets.new(name, [:named_table, :public, :set, read_concurrency: true])
        rescue
          ArgumentError -> :ok
        end

        name

      _tid ->
        name
    end
  end

  # A structural, id-independent signature of `output_expr` — the cache key
  # `get_or_compile_program/6` dispatches on (paired with a call-time
  # `quant_signature`). Two `Expr`s that are the *same shape of
  # computation* (same op sequence, shapes, dtypes, and static attrs) hash to
  # the same key even though `Nx.Defn.Expr` assigns each a fresh `id` per
  # trace — which is what lets Qwen3's 28 structurally-identical attention
  # layers (or the same layer across decode steps) share one compiled
  # program instead of recompiling per call. Coarser than a bit-identical
  # `Expr` hash (deliberately): tensor-valued args are replaced by their position in
  # `EMLX.Defn.Tree.post_order/2`'s dependency-first listing (itself
  # id-independent and structurally stable across identical call sites),
  # not by the referenced node's `id`.
  # `sanitize_key_term/2`'s opaque-scope fallback (below) recurses into a
  # `while`/`block`/`fun` sub-scope's own structural signature. Real models
  # reuse the *same* shared sub-expression (e.g. a RoPE frequency table, a
  # causal-mask block) across many call sites, so without memoization this
  # recomputes an identical signature once per reference — the same
  # unmemoized-shared-subexpression blowup `nx-graph-split-bugreport.md`'s
  # Bug 1 hit in `rewrite_subtree`. `@dispatch_key_memo` is a process-local
  # key => signature cache, live only for the duration of one `dispatch_key/3`
  # call (cleared in the `after`), so it's safe to reuse across unrelated
  # calls without stale entries leaking.
  @dispatch_key_memo_pdict_key {__MODULE__, :dispatch_key_memo}

  defp dispatch_key(output_expr, num_inputs, device) do
    id_key = {expr_id_fingerprint(output_expr), num_inputs, device}
    table = ensure_named_ets_table(@dispatch_key_by_id_table)

    case :ets.lookup(table, id_key) do
      [{_key, base_key}] ->
        base_key

      [] ->
        base_key = compute_dispatch_key(output_expr, num_inputs, device)
        :ets.insert(table, {id_key, base_key})
        base_key
    end
  end

  # Cheap (O(number of output leaves), not O(nodes)) identity fingerprint —
  # see `@dispatch_key_by_id_table`. Distinct traces never share an `Expr`
  # node id, so matching ids here guarantee the exact same graph.
  defp expr_id_fingerprint(output_expr) do
    [output_expr]
    |> Nx.Defn.Composite.flatten_list()
    |> Enum.map(fn %Nx.Tensor{data: %Nx.Defn.Expr{id: id}} -> id end)
  end

  defp compute_dispatch_key(output_expr, num_inputs, device) do
    Process.put(@dispatch_key_memo_pdict_key, %{})

    try do
      {node_sigs, output_sigs} = expr_structural_signature(output_expr)
      {node_sigs, output_sigs, num_inputs, device}
    after
      Process.delete(@dispatch_key_memo_pdict_key)
    end
  end

  # Computes `{node_sigs, output_sigs}` for one `EMLX.Defn.Tree.post_order/2`
  # scope. Split out from `dispatch_key/3` so `sanitize_key_term/2` can
  # recurse into an *opaque* scope root (a `while` condition/body, a
  # `block`'s `default_expr`, a `fun` body — see `EMLX.Defn.Tree.post_order/2`'s
  # "Sub-scope handling" doc) with a fresh, self-contained position map,
  # instead of assuming every tensor reachable from a visited node's `args`
  # was itself visited at this level.
  defp expr_structural_signature(expr_or_container) do
    nodes = EMLX.Defn.Tree.post_order(expr_or_container, &EMLX.Native.Expr.scope_dependencies/1)
    positions = nodes |> Enum.with_index() |> Map.new(fn {t, i} -> {t.data.id, i} end)

    node_sigs =
      Enum.map(nodes, fn
        %Nx.Tensor{data: %Nx.Defn.Expr{op: :metadata, args: [_inner, %{__EMLX__: emlx} = meta]}} =
            t
        when map_size(meta) == 1 ->
          # `_inner` (the plain-Nx reference formula) is deliberately excluded
          # from the signature, not just skipped like `EMLX.Defn.Tree.post_order/2`
          # (via `EMLX.Native.Expr.scope_dependencies/1`) does for lowering: `_inner`
          # closes over real upstream operands (e.g.
          # `x` in `rms_norm_reference/3`), so treating it like any other arg
          # here would make `sanitize_key_term/2` fall into its "opaque
          # sub-scope" fallback below and re-walk `_inner`'s *entire* transitive
          # dependency graph (i.e. every prior layer) from scratch — once per
          # fused-kernel call site. Signing only the real `__EMLX__` payload
          # (which is exactly what `EMLX.Native.Expr.lower/2` actually lowers)
          # keeps this O(1) per node instead of O(depth) per node.
          {:__emlx__, t.shape, t.type, sanitize_key_term(emlx, positions)}

        %Nx.Tensor{
          data: %Nx.Defn.Expr{op: :block, args: [struct, in_args, _default_expr, _fun] = args}
        } = t ->
          # See `EMLX.Native.Expr.native_lowerable_block?/2`'s doc: recognized
          # native blocks (e.g. `Nx.LinAlg.svd/1`'s `full_matrices?: true`)
          # never consult `default_expr` when lowering, so `default_expr` is
          # deliberately excluded from the signature too — walking/hashing it
          # (potentially a ~100-node Jacobi-rotation fallback graph, per call)
          # would be pure waste for a sub-scope that provably can't affect the
          # output. Anything not recognized falls back to signing the full
          # `args` (including `default_expr`, via the opaque-scope path in
          # `sanitize_key_term/2` below) exactly as before — the conservative
          # default.
          if EMLX.Native.Expr.native_lowerable_block?(struct, in_args) do
            {:block, t.shape, t.type, struct, sanitize_key_term(in_args, positions)}
          else
            {:block, t.shape, t.type, sanitize_key_term(args, positions)}
          end

        %Nx.Tensor{data: %Nx.Defn.Expr{op: op, args: args}} = t ->
          {op, t.shape, t.type, sanitize_key_term(args, positions)}
      end)

    output_sigs =
      [expr_or_container]
      |> Nx.Defn.Composite.flatten_list()
      |> Enum.map(fn %Nx.Tensor{data: %Nx.Defn.Expr{id: id}} -> Map.fetch!(positions, id) end)

    {node_sigs, output_sigs}
  end

  # Recursively strips an `Nx.Defn.Expr` node's `args` down to a hashable,
  # id-independent term for `dispatch_key/3`: an in-scope `%Nx.Tensor{}`
  # operand becomes its `post_order/1` position (see above); a tensor that
  # belongs to an opaque sub-scope (not in `positions` at all — a `while`
  # condition/body, a `block`'s `default_expr`, a `fun` body) recurses into
  # its own self-contained structural signature; an `Nx.TemplateBackend`-backed
  # tensor riding as a static arg (e.g. a `:runtime_call`'s `out_template`)
  # becomes its shape/type, since it has no `id`/scope of its own; a captured
  # function (e.g. a `:runtime_call`'s callback, or a hook's callback)
  # becomes its `{module, name, arity}` — identity without hashing its
  # closure environment. Everything else (numbers, atoms, strings, structs
  # used as plain option carriers) is kept as-is.
  #
  # An opaque sub-scope's signature is condensed to a `:crypto.hash/2` digest
  # (`{:scope, digest}`) rather than embedded verbatim: `@dispatch_key_memo_pdict_key`
  # only dedups *building* the raw signature once per `id`, but the same
  # (interned, i.e. shared-by-reference) large signature term can still be
  # *referenced* from many call sites — e.g. Axon's own `axon_layer:`
  # metadata wraps every layer, so a late layer's real (non-`__EMLX__`)
  # metadata `inner` recurses into an opaque scope covering that layer's
  # entire upstream history. `:ets.lookup/2`'s key hash walks every logical
  # occurrence of a subterm regardless of sharing, so embedding the raw
  # signature at O(occurrences) call sites made the final cache-key hash
  # O(occurrences × signature size) — cheap to *build* (reference reuse) but
  # catastrophically slow to *hash* on a real 28-layer model. Digesting once
  # per unique `id` and reusing the (small, fixed-size) digest everywhere
  # keeps both build and hash O(unique upstream work).
  defp sanitize_key_term(%Nx.Tensor{data: %Nx.Defn.Expr{id: id}} = t, positions) do
    case Map.fetch(positions, id) do
      {:ok, pos} ->
        {:ref, pos}

      :error ->
        memo = Process.get(@dispatch_key_memo_pdict_key, %{})

        case Map.fetch(memo, id) do
          {:ok, sig} ->
            sig

          :error ->
            digest = :crypto.hash(:sha256, :erlang.term_to_binary(expr_structural_signature(t)))
            sig = {:scope, digest}
            Process.put(@dispatch_key_memo_pdict_key, Map.put(memo, id, sig))
            sig
        end
    end
  end

  defp sanitize_key_term(%Nx.Tensor{} = t, _positions) do
    {:tensor_literal, t.shape, t.type}
  end

  defp sanitize_key_term(fun, _positions) when is_function(fun) do
    info = Function.info(fun)
    {:fun, info[:module], info[:name], info[:arity]}
  end

  defp sanitize_key_term(tuple, positions) when is_tuple(tuple) do
    tuple |> Tuple.to_list() |> sanitize_key_term(positions) |> List.to_tuple()
  end

  defp sanitize_key_term(list, positions) when is_list(list) do
    Enum.map(list, &sanitize_key_term(&1, positions))
  end

  defp sanitize_key_term(map, positions) when is_map(map) and not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, sanitize_key_term(v, positions)} end)
  end

  defp sanitize_key_term(%_struct{} = struct, positions) do
    {struct.__struct__, struct |> Map.from_struct() |> sanitize_key_term(positions)}
  end

  defp sanitize_key_term(other, _positions), do: other

  # Reconstructs each hook's value from its slice of `hook_refs` (in
  # `hooks` order, matching `EMLX.Native.Expr.to_native/1`'s flattening) and
  # invokes its callback for the side effect. Return value is discarded,
  # matching `Nx.Defn.Evaluator`'s hook semantics.
  defp fire_hooks([], [], _dev), do: :ok

  defp fire_hooks([%{template: template, refs: refs, callback: callback} | rest], hook_refs, dev) do
    {consumed, remaining} = Enum.split(hook_refs, length(refs))

    {value, []} =
      Nx.Defn.Composite.traverse(template, consumed, fn leaf, [ref | more] ->
        {EMLX.Backend.to_nx({dev, ref}, leaf), more}
      end)

    callback.(value)
    fire_hooks(rest, remaining, dev)
  end

  # Builds the eval closure for a `while` split point surrounded by other
  # computation. The expression is split on every `while` node and replayed
  # by `Nx.Defn.Graph`: `compiler: EMLX` makes every stage re-enter this
  # compiler, so straight-line stages compile flat (any `:runtime_call` in
  # them lowers in-graph — see `build_eval_fn/4`) and the isolated `while`
  # stage hits `build_while_base_eval_fn/2`. `device:` keeps stage
  # compilation on the same device; the command queue (if any) is
  # propagated through the process binding set by the outer wrapper.
  defp build_split_chain_eval_fn(output_expr, effective_device) do
    stages = Nx.Defn.Graph.split(output_expr, &if(split_point?(&1), do: :both, else: :none))

    fn [params] ->
      {_worker, dev} = resolve_worker(effective_device)
      inputs = Enum.map(params, &materialise_tensor(&1, dev))
      result = Nx.Defn.Graph.run(stages, inputs, compiler: __MODULE__, device: effective_device)
      [result]
    end
  end

  # Builds the eval closure for the base case: a bare tail `while` whose initial
  # carry is exactly the stage inputs (every output leaf is the `while` node or
  # an `:elem` of it). The condition and body are compiled by re-entering this
  # compiler via `Nx.Defn.jit/2`, so a nested `while` in the body recurses
  # through the same splitting machinery. The loop itself is driven host-side.
  defp build_while_base_eval_fn(output_expr, effective_device) do
    while_node = find_while_node(output_expr)
    [initial, _arg, cond_expr, body_expr] = while_node.data.args

    jit_opts = [compiler: __MODULE__, device: effective_device]
    cond_fn = Nx.Defn.jit(fn _ -> cond_expr end, jit_opts)
    body_fn = Nx.Defn.jit(fn _ -> body_expr end, jit_opts)

    # The stage inputs arrive in stage-argument order, which need not match the
    # carry (sub-scope parameter) order: each flattened `initial` leaf is the
    # parameter whose position picks the stage input feeding that carry slot.
    # Reorder inputs into carry order so the condition/body parameters bind to
    # the right tensors.
    initial_positions =
      [initial]
      |> Nx.Defn.Composite.flatten_list()
      |> Enum.map(fn %Nx.Tensor{data: %Nx.Defn.Expr{op: :parameter, args: [pos]}} -> pos end)

    while_id = while_node.data.id
    output_flat = Nx.Defn.Composite.flatten_list([output_expr])
    carry_indices = Enum.map(output_flat, &while_output_index(&1, while_id))
    output_template = Nx.Defn.Composite.traverse(output_expr, &Nx.to_template/1)

    fn [params] ->
      {_worker, dev} = resolve_worker(effective_device)
      inputs = Enum.map(params, &materialise_tensor(&1, dev))
      carry = Enum.map(initial_positions, &Enum.at(inputs, &1))
      final_carry = run_while_loop(carry, cond_fn, body_fn)

      output_tensors = Enum.map(carry_indices, &Enum.at(final_carry, &1))

      {output_container, []} =
        Nx.Defn.Composite.traverse(output_template, output_tensors, fn _leaf, [t | rest] ->
          {t, rest}
        end)

      [output_container]
    end
  end

  # Host-driven while loop. `carry` is the flat list of carry tensors; it is
  # passed to the compiled condition/body as a single tuple argument (mirroring
  # how `Nx.Defn.Graph.run` invokes a stage), which jit re-flattens to match the
  # carry parameters of the sub-scope. The body result is flattened back to a
  # flat list so a nested-container carry stays aligned with the leaf indices.
  defp run_while_loop(carry, cond_fn, body_fn) do
    if Nx.to_number(cond_fn.(List.to_tuple(carry))) == 0 do
      carry
    else
      new_carry = Nx.Defn.Composite.flatten_list([body_fn.(List.to_tuple(carry))])
      run_while_loop(new_carry, cond_fn, body_fn)
    end
  end

  defp find_while_node(output_expr) do
    output_expr
    |> EMLX.Defn.Tree.post_order(&EMLX.Native.Expr.scope_dependencies/1)
    |> Enum.find(&(&1.data.op == :while))
  end

  # True for the base case: the output projects exactly one `while` (each leaf
  # is the `while` node or an `:elem` of it) and that `while`'s initial carry is
  # made entirely of parameters — i.e. all pre-loop work has already been split
  # into an earlier stage, so the carry is the stage input as-is.
  defp bare_while?(output_expr) do
    leaves = Nx.Defn.Composite.flatten_list([output_expr])

    while_ids =
      leaves
      |> Enum.flat_map(fn
        %Nx.Tensor{data: %Nx.Defn.Expr{op: :while, id: id}} ->
          [id]

        %Nx.Tensor{
          data: %Nx.Defn.Expr{
            op: :elem,
            args: [%Nx.Tensor{data: %Nx.Defn.Expr{op: :while, id: id}}, _]
          }
        } ->
          [id]

        _ ->
          []
      end)
      |> Enum.uniq()

    case while_ids do
      [wid] ->
        all_project =
          Enum.all?(leaves, fn
            %Nx.Tensor{data: %Nx.Defn.Expr{op: :while, id: ^wid}} ->
              true

            %Nx.Tensor{
              data: %Nx.Defn.Expr{op: :elem, args: [%Nx.Tensor{data: %Nx.Defn.Expr{id: ^wid}}, _]}
            } ->
              true

            _ ->
              false
          end)

        all_project and while_initial_all_params?(find_while_node(output_expr))

      _ ->
        false
    end
  end

  defp while_initial_all_params?(%Nx.Tensor{data: %Nx.Defn.Expr{args: [initial | _]}}) do
    [initial]
    |> Nx.Defn.Composite.flatten_list()
    |> Enum.all?(&match?(%Nx.Tensor{data: %Nx.Defn.Expr{op: :parameter}}, &1))
  end

  # Maps a flat output leaf to the carry index it projects from `while_id`.
  defp while_output_index(%Nx.Tensor{data: %Nx.Defn.Expr{op: :while, id: id}}, while_id)
       when id == while_id,
       do: 0

  defp while_output_index(
         %Nx.Tensor{
           data: %Nx.Defn.Expr{op: :elem, args: [%Nx.Tensor{data: %Nx.Defn.Expr{id: id}}, i]}
         },
         while_id
       )
       when id == while_id,
       do: i

  # Materialises a defn input lazy ref to a concrete EMLX-backed tensor on `dev`
  # (for use as a `Nx.Defn.Graph.run` / jitted-stage argument).
  defp materialise_tensor(lazy, dev) do
    case lazy.() do
      %Nx.Tensor{data: %EMLX.Backend{}} = tensor -> tensor
      %Nx.Tensor{} = tensor -> Nx.backend_copy(tensor, {EMLX.Backend, device: dev})
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
  def clear_cache, do: EMLX.NIF.clear_cache() |> unwrap!()

  @doc """
  Resets the peak memory counter to zero.

  ## Examples

      EMLX.reset_peak_memory()
      #=> :ok
  """
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
