defmodule EMLX.Fast do
  @moduledoc """
  Single-kernel Metal shaders from `mlx::fast`, exposed as `deftransform`
  functions.

  Every function is defn-safe: call inside `defn`, `Nx.Defn.jit`, or from
  `Axon.rewrite_nodes/2` rewrite callbacks without restriction — **except
  `einsum/2`**, which is eager-only (see its docs).

  ## Two execution paths

  Each `deftransform` here dispatches on whether its tensor arguments are
  concrete (eager — a real `EMLX.Backend`-backed tensor) or expression-backed
  (traced — building an `Nx.Defn.Expr` graph):

  - **Eager**: the fused NIF (e.g. `EMLX.fast_rms_norm/3`) runs immediately.
  - **Traced**: the function returns an `Nx.Defn.Expr.metadata/2` node
    carrying a `:__EMLX__` key — `%{op: opcode, operands: [...], attrs: [...]}`
    — naming the native EMLX opcode, its operand tensors, and its int-encoded
    attributes (see `EMLX.Native.Expr.f64_bits/1`). `EMLX.Native.Expr`'s
    `:metadata` `expand_node` clause recognizes this key and lowers straight
    to the native op — no graph split. The metadata node wraps an
    `Nx.runtime_call/4` of the *same* eager callback used above (`inner`) —
    this compiler never evaluates it (it's discarded in favor of the
    `:__EMLX__` payload); it exists only so (a) the operand tensors are
    ordinary reachable dependencies for `EMLX.Defn.Tree.post_order/1` to
    visit, and (b) any other `Nx.Defn.Compiler` — notably the default
    `Nx.Defn.Evaluator` and `Nx.Defn.Grad` — still gets a correct fallback:
    `runtime_call` just runs the real NIF against concrete tensors, so it's
    both exact (not a slower plain-`Nx` approximation) and free to build
    (unlike a full composite reference formula, `Nx.runtime_call/4` is a
    single lightweight node — no per-op sub-expression tracing cost). The
    `:__EMLX__` node is itself wrapped once more in a
    `Nx.Defn.Kernel.custom_grad/3` annotation (`with_reference_grad/3`) so
    `Nx.Defn.grad` differentiates through each op's plain-`Nx` `*_reference/N`
    formula (VJP via `Nx.Defn.Grad.transform/3` — see `with_reference_grad/3`)
    instead of hitting the non-differentiable `runtime_call` forward pass.

  A bare `Nx.runtime_call` (anything *not* wrapped in `:__EMLX__` metadata)
  always forces a graph split — see `EMLX.split_point?/1`.

  ## Functions

  - `rms_norm/3` — fused RMS normalisation
  - `layer_norm/4` — fused layer normalisation (with bias)
  - `layer_norm/3` — fused layer normalisation (weight-only, no bias)
  - `rope/6` — fused RoPE with scalar integer offset
  - `rope_with_positions/6` — fused RoPE accepting a `position_ids` tensor
  - `rope_with_freqs/6` — fused RoPE with precomputed inv-frequency tensor (for `:llama3` scaling)
  - `scaled_dot_product_attention/4` — flash-attention SDPA (no mask)
  - `scaled_dot_product_attention/5` — flash-attention SDPA, either an additive/bool
    `mask` tensor, or an `opts` keyword list (`:sinks`) when there's no mask
  - `scaled_dot_product_attention/6` — flash-attention SDPA with `mask` and `opts` (`:sinks`)
  - `scaled_dot_product_attention_causal/4` — flash-attention SDPA with built-in causal mask
  - `scaled_dot_product_attention_causal/5` — same, plus `opts` (`:sinks`)
  - `scaled_dot_product_attention_causal_key_masked/5` — causal SDPA; checks key_mask at C++ level, fast-paths to pure causal when all-ones
  - `scaled_dot_product_attention_causal_key_masked/6` — same, plus `opts` (`:sinks`)
  - `swiglu/2` — fused SwiGLU: `silu(gate) * up`
  - `einsum/2` — variadic-operand Einstein summation (`mlx::core::einsum`);
    **eager-only, not defn-safe** (see its docs)

  ## Axon graph rewrite example

      Axon.rewrite_nodes(model, fn
        %Axon.Node{op: :rms_norm, opts: [eps: eps]} ->
          fn [x, weight], _output -> EMLX.Fast.rms_norm(x, weight, eps) end
        _ -> :skip
      end)
  """

  import Nx.Defn

  require EMLX.Debug
  import EMLX.Debug, only: [assert_no_nan_inf!: 2]

  alias EMLX.Native.Expr, as: NativeExpr

  # The `*_reference/N` plain-`Nx` formulas below (and their private helpers:
  # `position_freqs/6`, `rope_broadcast_shape/2`, `rope_rotate/5`,
  # `rope_split_half/5`, `rope_interleaved/5`, `repeat_kv_heads/2`,
  # `apply_causal_mask/4`, `apply_causal_key_mask/5`, `apply_generic_mask/2`,
  # `iota_bin/2`, `neg_inf_like/1`) are no longer used for the traced
  # *forward* dispatch path (see moduledoc) — each is instead called from a
  # `with_reference_grad/3` closure to compute `Nx.Defn.grad`'s backward pass.

  # ── Traced/eager dispatch helpers ───────────────────────────────────────────

  # True when any of `tensors` is expression-backed (i.e. we're being traced
  # by `Nx.Defn`, as opposed to running eagerly on concrete tensors).
  defp traced?(tensors) do
    Enum.any?(List.wrap(tensors), &match?(%Nx.Tensor{data: %Nx.Defn.Expr{}}, &1))
  end

  # Wraps `reference` (a plain-Nx formula, never evaluated by EMLX's
  # compiler) with the `:__EMLX__` metadata naming the real native opcode,
  # its operand tensors, and its int-encoded attrs.
  defp emlx_metadata(reference, opcode, operands, attrs) do
    Nx.Defn.Expr.metadata(reference, %{__EMLX__: %{op: opcode, operands: operands, attrs: attrs}})
  end

  # Wraps `fused` (an `emlx_metadata/4` node) with a `custom_grad/3`
  # annotation so `Nx.Defn.grad` differentiates through the op's plain-`Nx`
  # `*_reference/N` formula instead of hitting the opaque, non-differentiable
  # `Nx.runtime_call` forward pass. `reference_fn` is applied positionally to
  # `inputs` (same order as the `emlx_metadata/4` `operands` list) and must
  # recompute the same (single-tensor) value as `fused`.
  #
  # Reuses `Nx.Defn.Grad` itself rather than hand-deriving each op's backward
  # formula: for any (possibly multi-output-shaped) `y = reference_fn(inputs)`
  # and upstream cotangent `g` (same shape as `y`), the VJP w.r.t. `inputs` is
  # exactly `grad(inputs, sum(g * y))` — a standard trick for building a VJP
  # out of a scalar-output `grad`.
  defp with_reference_grad(fused, inputs, reference_fn) when is_function(reference_fn) do
    Nx.Defn.Kernel.custom_grad(fused, inputs, fn g ->
      {_value, grad} =
        Nx.Defn.Grad.transform(
          List.to_tuple(inputs),
          fn args -> args |> Tuple.to_list() |> then(&apply(reference_fn, &1)) |> Nx.multiply(g) |> Nx.sum() end,
          & &1
        )

      Tuple.to_list(grad)
    end)
  end

  # ── RMS Norm ────────────────────────────────────────────────────────────────

  @doc """
  Fused RMS normalisation (`mlx::fast::rms_norm`).

  - `x`      — input tensor; normalised over the last axis.
  - `weight` — `{hidden}` scale vector (same size as last axis of `x`).
  - `eps`    — numerical stability constant (e.g. `1.0e-6`).

  Output shape and type match `x`.
  """
  deftransform rms_norm(x, weight, eps) do
    if traced?([x, weight]) do
      fused =
        emlx_metadata(
          Nx.runtime_call(Nx.to_template(x), {x, weight}, [eps: eps], &rms_norm_callback/2),
          :fast_rms_norm,
          [x, weight],
          [NativeExpr.f64_bits(eps)]
        )

      with_reference_grad(fused, [x, weight], fn x, weight -> rms_norm_reference(x, weight, eps) end)
    else
      rms_norm_callback({x, weight}, eps: eps)
    end
  end

  defp rms_norm_reference(x, weight, eps) do
    x
    |> Nx.pow(2)
    |> Nx.mean(axes: [-1], keep_axes: true)
    |> Nx.add(eps)
    |> Nx.sqrt()
    |> then(&Nx.divide(x, &1))
    |> Nx.multiply(weight)
    |> Nx.as_type(Nx.type(x))
  end

  @doc false
  def rms_norm_callback({%Nx.Tensor{} = x, %Nx.Tensor{} = weight}, opts) do
    result_ref =
      EMLX.fast_rms_norm(EMLX.Backend.from_nx(x), EMLX.Backend.from_nx(weight), opts[:eps])

    assert_no_nan_inf!(result_ref, :rms_norm)
    EMLX.Backend.to_nx(result_ref)
  end

  # ── Layer Norm ───────────────────────────────────────────────────────────────

  @doc """
  Fused layer normalisation (`mlx::fast::layer_norm`).

  - `x`      — input tensor; normalised over the last axis.
  - `weight` — `{hidden}` scale vector (gamma).
  - `bias`   — `{hidden}` bias vector (beta).
  - `eps`    — numerical stability constant (e.g. `1.0e-5`).

  Output shape and type match `x`.
  """
  deftransform layer_norm(x, weight, bias, eps) do
    if traced?([x, weight, bias]) do
      fused =
        emlx_metadata(
          Nx.runtime_call(
            Nx.to_template(x),
            {x, weight, bias},
            [eps: eps],
            &layer_norm_callback/2
          ),
          :fast_layer_norm,
          [x, weight, bias],
          [NativeExpr.f64_bits(eps)]
        )

      with_reference_grad(fused, [x, weight, bias], fn x, weight, bias ->
        layer_norm_reference(x, weight, bias, eps)
      end)
    else
      layer_norm_callback({x, weight, bias}, eps: eps)
    end
  end

  defp layer_norm_reference(x, weight, bias, eps) do
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    centered = Nx.subtract(x, mean)
    variance = Nx.mean(Nx.pow(centered, 2), axes: [-1], keep_axes: true)

    centered
    |> Nx.divide(Nx.sqrt(Nx.add(variance, eps)))
    |> Nx.multiply(weight)
    |> Nx.add(bias)
    |> Nx.as_type(Nx.type(x))
  end

  @doc false
  def layer_norm_callback({%Nx.Tensor{} = x, %Nx.Tensor{} = weight, %Nx.Tensor{} = bias}, opts) do
    result_ref =
      EMLX.fast_layer_norm(
        EMLX.Backend.from_nx(x),
        EMLX.Backend.from_nx(weight),
        EMLX.Backend.from_nx(bias),
        opts[:eps]
      )

    assert_no_nan_inf!(result_ref, :layer_norm)
    EMLX.Backend.to_nx(result_ref)
  end

  @doc """
  Fused layer normalisation without bias (`mlx::fast::layer_norm`, weight-only variant).

  - `x`      — input tensor; normalised over the last axis.
  - `weight` — `{hidden}` scale vector (gamma).
  - `eps`    — numerical stability constant (e.g. `1.0e-5`).

  Output shape and type match `x`.
  """
  deftransform layer_norm(x, weight, eps) do
    if traced?([x, weight]) do
      fused =
        emlx_metadata(
          Nx.runtime_call(
            Nx.to_template(x),
            {x, weight},
            [eps: eps],
            &layer_norm_no_bias_callback/2
          ),
          :fast_layer_norm_no_bias,
          [x, weight],
          [NativeExpr.f64_bits(eps)]
        )

      with_reference_grad(fused, [x, weight], fn x, weight ->
        layer_norm_no_bias_reference(x, weight, eps)
      end)
    else
      layer_norm_no_bias_callback({x, weight}, eps: eps)
    end
  end

  defp layer_norm_no_bias_reference(x, weight, eps) do
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    centered = Nx.subtract(x, mean)
    variance = Nx.mean(Nx.pow(centered, 2), axes: [-1], keep_axes: true)

    centered
    |> Nx.divide(Nx.sqrt(Nx.add(variance, eps)))
    |> Nx.multiply(weight)
    |> Nx.as_type(Nx.type(x))
  end

  @doc false
  def layer_norm_no_bias_callback({%Nx.Tensor{} = x, %Nx.Tensor{} = weight}, opts) do
    result_ref =
      EMLX.fast_layer_norm_no_bias(
        EMLX.Backend.from_nx(x),
        EMLX.Backend.from_nx(weight),
        opts[:eps]
      )

    assert_no_nan_inf!(result_ref, :layer_norm)
    EMLX.Backend.to_nx(result_ref)
  end

  @doc """
  Causal SDPA with the key_mask check delegated to the C++ NIF (eager) or
  folded directly into the compiled graph (traced).

  When eager, the NIF evaluates `all(key_mask == 1)`:
  - **true** (no padding, e.g. single-sequence decode) → pure causal SDPA,
    no mask tensor allocated.
  - **false** (padded batch or multi-sequence) → builds a combined
    causal + key_mask additive mask and calls the masked SDPA kernel.

  This avoids the `Nx.cond` double-evaluation problem: the NIF forces eval
  of only the small `{B, T_kv}` key_mask subgraph, then branches in C++.

  When traced, the compiled `:fast_sdpa_causal_key_masked*` opcode always
  builds the combined causal+key_mask additive mask in-graph (a compiled
  program can't branch on a runtime `all(key_mask)` check).

  Input/output layout matches `scaled_dot_product_attention_causal/4`:
  - `q`        — `{B, N_q,  T_q,  D}`
  - `k`        — `{B, N_kv, T_kv, D}`
  - `v`        — `{B, N_kv, T_kv, D}`
  - `scale`    — pre-computed scalar
  - `key_mask` — `{B, T_kv}` boolean/int tensor (1 = attend, 0 = masked)
  - Output     — `{B, N_q, T_q, D}`, same dtype as `q`

  An optional 6th `opts` keyword list accepts `:sinks` (see
  `scaled_dot_product_attention/5`).
  """
  deftransform scaled_dot_product_attention_causal_key_masked(q, k, v, scale, key_mask) do
    scaled_dot_product_attention_causal_key_masked(q, k, v, scale, key_mask, [])
  end

  @doc false
  deftransform scaled_dot_product_attention_causal_key_masked(q, k, v, scale, key_mask, opts) do
    # Compute kv_offset at JIT/deftransform time: shapes are compile-time constants here.
    #  - Decode  (T_q = 1): single query attends to all prior positions;
    #    key_mask filters which positions are valid.  kv_offset = T_kv - 1.
    #  - Prefill (T_q > 1): lower-triangular causal mask.  kv_offset = 0.
    #    (T_kv - T_q would be wrong for left-padded prefill with a pre-alloc cache.)
    t_q = elem(Nx.shape(q), 2)
    t_kv = elem(Nx.shape(k), 2)
    kv_offset = if t_q == 1, do: t_kv - 1, else: 0
    sinks = Keyword.get(opts, :sinks)

    if traced?([q, k, v, key_mask, sinks]) do
      out = Nx.to_template(q)

      if sinks do
        inner =
          Nx.runtime_call(
            out,
            {q, k, v, key_mask, sinks},
            [scale: scale, kv_offset: kv_offset],
            &sdpa_causal_key_masked_sinks_callback/2
          )

        fused =
          emlx_metadata(
            inner,
            :fast_sdpa_causal_key_masked_sinks,
            [q, k, v, key_mask, sinks],
            [NativeExpr.f64_bits(scale), kv_offset]
          )

        with_reference_grad(fused, [q, k, v, key_mask, sinks], fn q, k, v, key_mask, sinks ->
          sdpa_reference(q, k, v, scale,
            causal: true,
            key_mask: key_mask,
            kv_offset: kv_offset,
            sinks: sinks
          )
        end)
      else
        inner =
          Nx.runtime_call(
            out,
            {q, k, v, key_mask},
            [scale: scale, kv_offset: kv_offset],
            &sdpa_causal_key_masked_callback/2
          )

        fused =
          emlx_metadata(
            inner,
            :fast_sdpa_causal_key_masked,
            [q, k, v, key_mask],
            [NativeExpr.f64_bits(scale), kv_offset]
          )

        with_reference_grad(fused, [q, k, v, key_mask], fn q, k, v, key_mask ->
          sdpa_reference(q, k, v, scale, causal: true, key_mask: key_mask, kv_offset: kv_offset)
        end)
      end
    else
      if sinks do
        sdpa_causal_key_masked_sinks_callback({q, k, v, key_mask, sinks}, scale: scale, kv_offset: kv_offset)
      else
        sdpa_causal_key_masked_callback({q, k, v, key_mask}, scale: scale, kv_offset: kv_offset)
      end
    end
  end

  @doc false
  def sdpa_causal_key_masked_callback(
        {%Nx.Tensor{} = q, %Nx.Tensor{} = k, %Nx.Tensor{} = v, %Nx.Tensor{} = key_mask},
        opts
      ) do
    # Q/K/V arrive in {B, N, T, D} layout (transposed by the outer Axon layer).
    # kv_offset was computed at deftransform time from the static shapes.
    # MLX's fast SDPA handles GQA natively (N_q may differ from N_kv) — no
    # explicit head expansion needed here.
    scale = opts[:scale]
    kv_offset = opts[:kv_offset]

    result_ref =
      EMLX.fast_sdpa_causal_key_masked(
        EMLX.Backend.from_nx(q),
        EMLX.Backend.from_nx(k),
        EMLX.Backend.from_nx(v),
        scale,
        EMLX.Backend.from_nx(key_mask),
        kv_offset
      )

    assert_no_nan_inf!(result_ref, :sdpa)
    out = EMLX.Backend.to_nx(result_ref)

    dtype = Nx.type(q)
    if Nx.type(out) != dtype, do: Nx.as_type(out, dtype), else: out
  end

  @doc false
  def sdpa_causal_key_masked_sinks_callback(
        {%Nx.Tensor{} = q, %Nx.Tensor{} = k, %Nx.Tensor{} = v, %Nx.Tensor{} = key_mask,
         %Nx.Tensor{} = sinks},
        opts
      ) do
    scale = opts[:scale]
    kv_offset = opts[:kv_offset]

    result_ref =
      EMLX.fast_sdpa_causal_key_masked(
        EMLX.Backend.from_nx(q),
        EMLX.Backend.from_nx(k),
        EMLX.Backend.from_nx(v),
        scale,
        EMLX.Backend.from_nx(key_mask),
        kv_offset,
        EMLX.Backend.from_nx(sinks)
      )

    assert_no_nan_inf!(result_ref, :sdpa)
    out = EMLX.Backend.to_nx(result_ref)

    dtype = Nx.type(q)
    if Nx.type(out) != dtype, do: Nx.as_type(out, dtype), else: out
  end

  # ── RoPE ────────────────────────────────────────────────────────────────────

  @doc """
  Fused rotary position embedding (`mlx::fast::rope`).

  - `a`           — input `{B, ..., T, D}`; `...` dims are passed through.
  - `dims`        — number of feature dims to rotate (≤ last-axis size, must be even).
  - `traditional` — `false` for split-half (Qwen3); `true` for interleaved.
  - `base`        — angular frequency base (e.g. `10_000` or `1_000_000`).
  - `scale`       — position scale (`1.0` unless using NTK-aware scaling).
  - `offset`      — integer position offset (tokens already in the KV cache).

  **`traditional` must match the model checkpoint's convention.**
  For Qwen3 (split-half): `traditional: false`.

  Output shape and type match `a`.
  """
  deftransform rope(a, dims, traditional, base, scale, offset) do
    if traced?(a) do
      traditional_int = if traditional, do: 1, else: 0
      opts = [dims: dims, traditional: traditional, base: base, scale: scale, offset: offset]

      fused =
        emlx_metadata(
          Nx.runtime_call(Nx.to_template(a), a, opts, &rope_callback/2),
          :fast_rope,
          [a],
          [dims, traditional_int, NativeExpr.f64_bits(base), NativeExpr.f64_bits(scale), offset]
        )

      with_reference_grad(fused, [a], fn a ->
        rope_reference(a, dims, traditional, base, scale, offset)
      end)
    else
      rope_callback(a, dims: dims, traditional: traditional, base: base, scale: scale, offset: offset)
    end
  end

  @doc false
  def rope_callback(%Nx.Tensor{} = a, opts) do
    EMLX.fast_rope(
      EMLX.Backend.from_nx(a),
      opts[:dims],
      opts[:traditional],
      opts[:base],
      opts[:scale],
      opts[:offset]
    )
    |> EMLX.Backend.to_nx()
  end

  @doc """
  Fused RoPE accepting a `position_ids` tensor (`mlx::fast::rope`, array-offset overload).

  Use this variant when the calling convention provides `position_ids` as a tensor
  (e.g. from Bumblebee's rotary embedding layer) rather than a scalar integer offset.

  - `a`            — input `{B, T, ..., D}` (Bumblebee convention: heads NOT yet transposed)
  - `position_ids` — `{B, T}` integer tensor; each row holds the token positions for
                     one batch example. **Positions must be sequential within each row**
                     (standard causal LM). The starting offset for batch item `b` is
                     taken as `position_ids[b, 0]`; subsequent positions are inferred
                     by MLX as `offset + 0, offset + 1, ...`.
  - `dims`         — number of feature dims to rotate.
  - `traditional`  — `false` for split-half (Bumblebee / Qwen3); `true` for interleaved.
  - `base`         — angular frequency base (e.g. `10_000`).
  - `scale`        — position scale (`1.0` unless using NTK-aware scaling).

  Output shape and type match `a`.

  > ### Sequential positions only (fast T=1 path) {: .warning}
  > For **decode** with `T = 1` and `base` below about `1.0e5`, the `fast_rope_ids`
  > opcode is used; it assumes sequential positions from `position_ids[b, 0]`. For
  > **larger** `base` (e.g. Qwen3 `rope_theta` 1M) or **prefill** (`T > 1`), the
  > per-token `fast_rope_positions` opcode is used, matching Bumblebee for
  > arbitrary per-token `position_ids`.
  """
  deftransform rope_with_positions(a, position_ids, dims, traditional, base, scale) do
    # Branch at JIT/deftransform time on T (index 1 in Bumblebee {B, T, N, D} layout).
    # T is a compile-time constant when Bumblebee uses static sequence_length compilation.
    #  - Decode  (T = 1): sequential positions — fast_rope_ids (1 Metal dispatch).
    #  - Prefill (T > 1): arbitrary per-token positions — fast_rope_positions.
    t = elem(Nx.shape(a), 1)
    base = base * 1.0
    # `fast_rope_ids` uses a scalar `base` in `mlx::fast::rope` that does not
    # match Bumblebee's per-row `take(cos, position_ids)` table for very large
    # `rope_theta` (Qwen2/3 use 1e6+). `fast_rope_positions` matches
    # Bumblebee; use it for T=1 when `base` is in that regime (A13).
    t1_use_fast? = t == 1 and base < 1.0e5
    traditional_int = if traditional, do: 1, else: 0

    if traced?([a, position_ids]) do
      out = Nx.to_template(a)
      opts = [dims: dims, traditional: traditional, base: base, scale: scale]

      reference_fn = fn a, position_ids ->
        rope_with_positions_reference(a, position_ids, dims, traditional, base, scale)
      end

      if t1_use_fast? do
        inner =
          Nx.runtime_call(out, {a, position_ids}, opts, &rope_with_positions_fast_callback/2)

        fused =
          emlx_metadata(
            inner,
            :fast_rope_ids,
            [a, position_ids],
            [dims, traditional_int, NativeExpr.f64_bits(base), NativeExpr.f64_bits(scale)]
          )

        with_reference_grad(fused, [a, position_ids], reference_fn)
      else
        inner = Nx.runtime_call(out, {a, position_ids}, opts, &rope_with_positions_callback/2)

        fused =
          emlx_metadata(
            inner,
            :fast_rope_positions,
            [a, position_ids],
            [dims, traditional_int, NativeExpr.f64_bits(base), NativeExpr.f64_bits(scale)]
          )

        with_reference_grad(fused, [a, position_ids], reference_fn)
      end
    else
      if t1_use_fast? do
        rope_with_positions_fast_callback({a, position_ids},
          dims: dims,
          traditional: traditional,
          base: base,
          scale: scale
        )
      else
        rope_with_positions_callback({a, position_ids},
          dims: dims,
          traditional: traditional,
          base: base,
          scale: scale
        )
      end
    end
  end

  @doc false
  # Fast decode path: uses MLX fast_rope_ids NIF (1 Metal dispatch per call).
  # position_ids = {B, 1} — take position_ids[b, 0] as the offset for each batch.
  def rope_with_positions_fast_callback({%Nx.Tensor{} = a, %Nx.Tensor{} = position_ids}, opts) do
    batch_size = elem(Nx.shape(position_ids), 0)
    offsets = position_ids[[.., 0]] |> Nx.reshape({batch_size})

    EMLX.fast_rope_ids(
      EMLX.Backend.from_nx(a),
      opts[:dims],
      opts[:traditional],
      opts[:base] * 1.0,
      opts[:scale] * 1.0,
      EMLX.Backend.from_nx(offsets)
    )
    |> EMLX.Backend.to_nx()
  end

  @doc false
  # Per-token-position RoPE fallback path, handled in native C++.
  # Correct for arbitrary (non-sequential) position IDs and high rope_theta.
  def rope_with_positions_callback({%Nx.Tensor{} = a, %Nx.Tensor{} = position_ids}, opts) do
    EMLX.fast_rope_positions(
      EMLX.Backend.from_nx(a),
      opts[:dims],
      opts[:traditional],
      opts[:base] * 1.0,
      opts[:scale] * 1.0,
      EMLX.Backend.from_nx(position_ids)
    )
    |> EMLX.Backend.to_nx()
  end

  @doc """
  Fused RoPE with precomputed inverse-frequency vector (`mlx::fast::rope`, freqs overload).

  Use this variant when the model's RoPE scaling strategy produces a fixed
  `{dims/2}` inv-frequency tensor that can be baked at graph-rewrite time
  (e.g. `:llama3` smooth-interpolation). Strategies that are seq-len conditional
  or require cos/sin post-multiply (`:linear`, `:dynamic`, `:longrope`) should
  use `rope_with_positions/6` instead.

  - `a`            — input `{B, T, ..., D}` (Bumblebee convention: heads NOT yet transposed)
  - `position_ids` — `{B, T}` integer tensor. For **decode** (`T = 1`) the fast path uses
    `position_ids[b,0]` as the per-batch offset into `freqs` (same contract as
    `mlx::fast::rope` with a scalar offset per batch). For **prefill** (`T > 1`) a
    per-token path runs so arbitrary positions (e.g. left-padded
    `[0,…,0,1,2,…]`) are correct; the offset-only entry point cannot represent
    that.
  - `dims`         — number of feature dims to rotate.
  - `traditional`  — `false` for split-half (Bumblebee / Qwen3); `true` for interleaved.
  - `scale`        — position scale (`1.0` for most strategies with precomputed freqs).
  - `freqs`        — `{dims/2}` tensor of precomputed inverse frequencies.

  Output shape and type match `a`.
  """
  deftransform rope_with_freqs(a, position_ids, dims, traditional, scale, freqs) do
    t = elem(Nx.shape(a), 1)
    traditional_int = if traditional, do: 1, else: 0

    if traced?([a, position_ids, freqs]) do
      out = Nx.to_template(a)
      opts = [dims: dims, traditional: traditional, scale: scale]

      reference_fn = fn a, position_ids, freqs ->
        rope_with_freqs_reference(a, position_ids, dims, traditional, scale, freqs)
      end

      if t == 1 do
        inner =
          Nx.runtime_call(out, {a, position_ids, freqs}, opts, &rope_with_freqs_fast_callback/2)

        fused =
          emlx_metadata(
            inner,
            :fast_rope_with_freqs,
            [a, position_ids, freqs],
            [dims, traditional_int, NativeExpr.f64_bits(scale)]
          )

        with_reference_grad(fused, [a, position_ids, freqs], reference_fn)
      else
        inner = Nx.runtime_call(out, {a, position_ids, freqs}, opts, &rope_with_freqs_callback/2)

        fused =
          emlx_metadata(
            inner,
            :fast_rope_with_freqs_positions,
            [a, position_ids, freqs],
            [dims, traditional_int, NativeExpr.f64_bits(scale)]
          )

        with_reference_grad(fused, [a, position_ids, freqs], reference_fn)
      end
    else
      if t == 1 do
        rope_with_freqs_fast_callback({a, position_ids, freqs},
          dims: dims,
          traditional: traditional,
          scale: scale
        )
      else
        rope_with_freqs_callback({a, position_ids, freqs},
          dims: dims,
          traditional: traditional,
          scale: scale
        )
      end
    end
  end

  @doc false
  def rope_with_freqs_fast_callback(
        {%Nx.Tensor{} = a, %Nx.Tensor{} = position_ids, %Nx.Tensor{} = freqs},
        opts
      ) do
    batch_size = elem(Nx.shape(position_ids), 0)
    offsets = position_ids[[.., 0]] |> Nx.reshape({batch_size})

    EMLX.fast_rope_with_freqs(
      EMLX.Backend.from_nx(a),
      opts[:dims],
      opts[:traditional],
      opts[:scale],
      EMLX.Backend.from_nx(offsets),
      EMLX.Backend.from_nx(freqs)
    )
    |> EMLX.Backend.to_nx()
  end

  @doc false
  # Prefill: one `fast_rope_with_freqs` call per time step (same kernel as T=1 decode) so
  # per-token positions (e.g. left-pad zeros then real) match MLX — an element-wise Nx
  # formula can diverge from the fused NIF. Prefill is once per generation.
  def rope_with_freqs_callback(
        {%Nx.Tensor{} = a, %Nx.Tensor{} = position_ids, %Nx.Tensor{} = freqs},
        opts
      ) do
    t = elem(Nx.shape(a), 1)

    if t == 0 do
      a
    else
      parts =
        for ti <- 0..(t - 1) do
          a_t = a[[.., ti..ti, .., ..]]
          pos_t = position_ids[[.., ti..ti]]
          rope_with_freqs_fast_callback({a_t, pos_t, freqs}, opts)
        end

      Nx.concatenate(parts, axis: 1)
    end
  end

  # ── RoPE reference formulas ─────────────────────────────────────────────────
  #
  # Plain-Nx formulas used only from a `with_reference_grad/3` closure (see
  # moduledoc) to compute `Nx.Defn.grad`'s backward pass — the fused forward
  # pass never evaluates them.

  # `a` is `{B, ..., T, D}` (T second-to-last axis) — matches `rope/6`.
  defp rope_reference(a, dims, traditional, base, scale, offset) do
    rank = Nx.rank(a)
    t_axis = rank - 2
    d_axis = rank - 1
    t = elem(Nx.shape(a), t_axis)
    half = div(dims, 2)

    inv_freq =
      Nx.iota({half}, type: :f32, backend: Nx.BinaryBackend)
      |> Nx.multiply(2.0 / dims)
      |> then(&Nx.pow(base * 1.0, &1))
      |> then(&Nx.divide(1.0, &1))

    positions =
      Nx.iota({t}, type: :f32, backend: Nx.BinaryBackend)
      |> Nx.add(offset * 1.0)
      |> Nx.multiply(scale * 1.0)

    freqs = Nx.outer(positions, inv_freq)
    freqs_bcast = Nx.reshape(freqs, rope_broadcast_shape(rank, [{t_axis, t}, {d_axis, half}]))

    rope_rotate(a, dims, traditional, freqs_bcast, d_axis)
  end

  # `a` is `{B, T, ..., D}` (Bumblebee convention, T at axis 1) — matches
  # `rope_with_positions/6`. `position_ids` is `{B, T}`.
  defp rope_with_positions_reference(a, position_ids, dims, traditional, base, scale) do
    rank = Nx.rank(a)
    d_axis = rank - 1
    half = div(dims, 2)
    {b, t} = {elem(Nx.shape(position_ids), 0), elem(Nx.shape(position_ids), 1)}

    inv_freq =
      Nx.iota({half}, type: :f32, backend: Nx.BinaryBackend)
      |> Nx.multiply(2.0 / dims)
      |> then(&Nx.pow(base * 1.0, &1))
      |> then(&Nx.divide(1.0, &1))

    freqs_bt = position_freqs(position_ids, inv_freq, scale, b, t, half)
    freqs_bcast = Nx.reshape(freqs_bt, rope_broadcast_shape(rank, [{0, b}, {1, t}, {d_axis, half}]))

    rope_rotate(a, dims, traditional, freqs_bcast, d_axis)
  end

  # Same layout as `rope_with_positions_reference/6`, but `inv_freq` is
  # supplied directly as `reciprocal(freqs)` (matches `mlx::fast::rope`'s
  # freqs overload — see the `fast_rope_with_freqs*` opcodes).
  defp rope_with_freqs_reference(a, position_ids, dims, traditional, scale, freqs) do
    rank = Nx.rank(a)
    d_axis = rank - 1
    half = div(dims, 2)
    {b, t} = {elem(Nx.shape(position_ids), 0), elem(Nx.shape(position_ids), 1)}

    inv_freq = Nx.divide(1.0, Nx.as_type(freqs, :f32))

    freqs_bt = position_freqs(position_ids, inv_freq, scale, b, t, half)
    freqs_bcast = Nx.reshape(freqs_bt, rope_broadcast_shape(rank, [{0, b}, {1, t}, {d_axis, half}]))

    rope_rotate(a, dims, traditional, freqs_bcast, d_axis)
  end

  defp position_freqs(position_ids, inv_freq, scale, b, t, half) do
    pos_bt1 =
      position_ids
      |> Nx.as_type(:f32)
      |> Nx.multiply(scale * 1.0)
      |> Nx.reshape({b, t, 1})

    Nx.multiply(pos_bt1, Nx.reshape(inv_freq, {1, 1, half}))
  end

  # Builds a reshape target with `1`s everywhere except the given
  # `{axis, size}` pairs — used to broadcast a `{..., half}`-shaped
  # cos/sin table against `a`'s full rank without touching axes it
  # passes through untouched (e.g. the heads axis).
  defp rope_broadcast_shape(rank, axis_sizes) do
    for i <- 0..(rank - 1) do
      case List.keyfind(axis_sizes, i, 0) do
        {^i, size} -> size
        nil -> 1
      end
    end
    |> List.to_tuple()
  end

  # Rotates the first `dims` elements of `a`'s last axis by `freqs_bcast`
  # (already reshaped to broadcast against `a`), passing the remainder
  # through unchanged. `traditional` selects interleaved-pair vs
  # split-half rotation (must match the model checkpoint's convention).
  defp rope_rotate(a, dims, traditional, freqs_bcast, d_axis) do
    full_d = elem(Nx.shape(a), d_axis)
    half = div(dims, 2)
    rotate_part = Nx.slice_along_axis(a, 0, dims, axis: d_axis)
    cos = Nx.cos(freqs_bcast)
    sin = Nx.sin(freqs_bcast)

    rotated =
      if traditional do
        rope_interleaved(rotate_part, cos, sin, d_axis, half)
      else
        rope_split_half(rotate_part, cos, sin, d_axis, half)
      end

    result =
      if dims == full_d do
        rotated
      else
        pass_part = Nx.slice_along_axis(a, dims, full_d - dims, axis: d_axis)
        Nx.concatenate([rotated, pass_part], axis: d_axis)
      end

    Nx.as_type(result, Nx.type(a))
  end

  defp rope_split_half(rotate_part, cos, sin, d_axis, half) do
    x1 = Nx.slice_along_axis(rotate_part, 0, half, axis: d_axis)
    x2 = Nx.slice_along_axis(rotate_part, half, half, axis: d_axis)
    rotated_half = Nx.concatenate([Nx.negate(x2), x1], axis: d_axis)
    cos2 = Nx.concatenate([cos, cos], axis: d_axis)
    sin2 = Nx.concatenate([sin, sin], axis: d_axis)
    Nx.add(Nx.multiply(rotate_part, cos2), Nx.multiply(rotated_half, sin2))
  end

  defp rope_interleaved(rotate_part, cos, sin, d_axis, half) do
    front = rotate_part |> Nx.shape() |> Tuple.to_list() |> Enum.take(d_axis)
    paired = Nx.reshape(rotate_part, List.to_tuple(front ++ [half, 2]))

    x1 = paired |> Nx.slice_along_axis(0, 1, axis: d_axis + 1) |> Nx.squeeze(axes: [d_axis + 1])
    x2 = paired |> Nx.slice_along_axis(1, 1, axis: d_axis + 1) |> Nx.squeeze(axes: [d_axis + 1])

    out1 = Nx.subtract(Nx.multiply(x1, cos), Nx.multiply(x2, sin))
    out2 = Nx.add(Nx.multiply(x1, sin), Nx.multiply(x2, cos))

    Nx.stack([out1, out2], axis: d_axis + 1) |> Nx.reshape(Nx.shape(rotate_part))
  end

  # ── SwiGLU ──────────────────────────────────────────────────────────────────

  @doc """
  Fused SwiGLU activation: `silu(gate) * up` where `silu(x) = x * sigmoid(x)`.

  Eliminates the two-op `silu(gate_proj) * up_proj` pattern that appears in
  Qwen3's FFN layers (28× per decode step).

  - `gate` — gate-projection output; silu is applied element-wise.
  - `up`   — up-projection output; same shape as `gate`.

  Output has the same shape and dtype as `gate`.
  """
  deftransform swiglu(gate, up) do
    if traced?([gate, up]) do
      inner = Nx.runtime_call(Nx.to_template(gate), {gate, up}, [], &swiglu_callback/2)
      fused = emlx_metadata(inner, :fast_swiglu, [gate, up], [])
      with_reference_grad(fused, [gate, up], &swiglu_reference/2)
    else
      swiglu_callback({gate, up}, [])
    end
  end

  defp swiglu_reference(gate, up) do
    gate
    |> Nx.multiply(Nx.sigmoid(gate))
    |> Nx.multiply(up)
    |> Nx.as_type(Nx.type(gate))
  end

  @doc false
  def swiglu_callback({%Nx.Tensor{} = gate, %Nx.Tensor{} = up}, _opts) do
    EMLX.fast_swiglu(EMLX.Backend.from_nx(gate), EMLX.Backend.from_nx(up))
    |> EMLX.Backend.to_nx()
  end

  # ── Scaled Dot-Product Attention ─────────────────────────────────────────────

  @doc """
  Flash-attention SDPA, no mask (`mlx::fast::scaled_dot_product_attention`).

  GQA-native: `k`/`v` may have fewer heads than `q` — no pre-tiling required.

  - `q`     — `{B, N_q,  T_q,  D}`
  - `k`     — `{B, N_kv, T_kv, D}`
  - `v`     — `{B, N_kv, T_kv, D}`
  - `scale` — scalar (typically `1 / sqrt(D)`)

  Output: `{B, N_q, T_q, D}` — same dtype as `q`.
  Softmax accumulates in float32 internally regardless of input dtype.
  """
  deftransform scaled_dot_product_attention(q, k, v, scale) do
    scaled_dot_product_attention(q, k, v, scale, [])
  end

  @doc """
  Flash-attention SDPA — either an additive/boolean `mask` tensor (5th arg),
  or, with no mask, an `opts` keyword list (disambiguated by `is_list/1`):

  * `:sinks` — optional learned per-head attention-sink logits tensor,
    appended as an extra key/value pair the softmax normalises against (see
    `mlx::fast::scaled_dot_product_attention`'s `sinks` parameter). Shape
    must broadcast against `{B, N_q}` (typically `{N_q}`).

  `mask` must be broadcast-compatible with `{B, N_q, T_q, T_kv}`.
  Boolean `false` entries are masked out (`-∞`); float entries are added to
  the pre-softmax scores.

  For causal masking in decode (single query token), prefer the no-mask arity
  since `T_q=1` is always trivially causal.
  """
  deftransform scaled_dot_product_attention(q, k, v, scale, opts) when is_list(opts) do
    sinks = Keyword.get(opts, :sinks)

    if traced?([q, k, v, sinks]) do
      out = Nx.to_template(q)

      if sinks do
        inner = Nx.runtime_call(out, {q, k, v, sinks}, [scale: scale], &sdpa_sinks_callback/2)
        fused = emlx_metadata(inner, :fast_sdpa_sinks, [q, k, v, sinks], [NativeExpr.f64_bits(scale)])

        with_reference_grad(fused, [q, k, v, sinks], fn q, k, v, sinks ->
          sdpa_reference(q, k, v, scale, sinks: sinks)
        end)
      else
        inner = Nx.runtime_call(out, {q, k, v}, [scale: scale], &sdpa_callback/2)
        fused = emlx_metadata(inner, :fast_sdpa, [q, k, v], [NativeExpr.f64_bits(scale)])

        with_reference_grad(fused, [q, k, v], fn q, k, v ->
          sdpa_reference(q, k, v, scale, [])
        end)
      end
    else
      if sinks do
        sdpa_sinks_callback({q, k, v, sinks}, scale: scale)
      else
        sdpa_callback({q, k, v}, scale: scale)
      end
    end
  end

  deftransform scaled_dot_product_attention(q, k, v, scale, mask) do
    scaled_dot_product_attention(q, k, v, scale, mask, [])
  end

  @doc """
  Flash-attention SDPA with an additive/boolean `mask` and `opts` (`:sinks`
  — see `scaled_dot_product_attention/5`).
  """
  deftransform scaled_dot_product_attention(q, k, v, scale, mask, opts) when is_list(opts) do
    sinks = Keyword.get(opts, :sinks)

    if traced?([q, k, v, mask, sinks]) do
      out = Nx.to_template(q)

      if sinks do
        inner =
          Nx.runtime_call(out, {q, k, v, mask, sinks}, [scale: scale], &sdpa_masked_sinks_callback/2)

        fused =
          emlx_metadata(inner, :fast_sdpa_masked_sinks, [q, k, v, mask, sinks], [
            NativeExpr.f64_bits(scale)
          ])

        with_reference_grad(fused, [q, k, v, mask, sinks], fn q, k, v, mask, sinks ->
          sdpa_reference(q, k, v, scale, mask: mask, sinks: sinks)
        end)
      else
        inner = Nx.runtime_call(out, {q, k, v, mask}, [scale: scale], &sdpa_masked_callback/2)
        fused = emlx_metadata(inner, :fast_sdpa_masked, [q, k, v, mask], [NativeExpr.f64_bits(scale)])

        with_reference_grad(fused, [q, k, v, mask], fn q, k, v, mask ->
          sdpa_reference(q, k, v, scale, mask: mask)
        end)
      end
    else
      if sinks do
        sdpa_masked_sinks_callback({q, k, v, mask, sinks}, scale: scale)
      else
        sdpa_masked_callback({q, k, v, mask}, scale: scale)
      end
    end
  end

  @doc false
  def sdpa_callback({%Nx.Tensor{} = q, %Nx.Tensor{} = k, %Nx.Tensor{} = v}, opts) do
    result_ref =
      EMLX.fast_sdpa(
        EMLX.Backend.from_nx(q),
        EMLX.Backend.from_nx(k),
        EMLX.Backend.from_nx(v),
        opts[:scale]
      )

    assert_no_nan_inf!(result_ref, :sdpa)
    out = EMLX.Backend.to_nx(result_ref)

    # mlx::fast::sdpa may upcast to f32 internally; cast back to q's dtype
    if Nx.type(out) != Nx.type(q), do: Nx.as_type(out, Nx.type(q)), else: out
  end

  @doc false
  def sdpa_sinks_callback(
        {%Nx.Tensor{} = q, %Nx.Tensor{} = k, %Nx.Tensor{} = v, %Nx.Tensor{} = sinks},
        opts
      ) do
    result_ref =
      EMLX.fast_sdpa(
        EMLX.Backend.from_nx(q),
        EMLX.Backend.from_nx(k),
        EMLX.Backend.from_nx(v),
        opts[:scale],
        EMLX.Backend.from_nx(sinks)
      )

    assert_no_nan_inf!(result_ref, :sdpa)
    out = EMLX.Backend.to_nx(result_ref)

    if Nx.type(out) != Nx.type(q), do: Nx.as_type(out, Nx.type(q)), else: out
  end

  @doc false
  def sdpa_masked_callback(
        {%Nx.Tensor{} = q, %Nx.Tensor{} = k, %Nx.Tensor{} = v, %Nx.Tensor{} = mask},
        opts
      ) do
    result_ref =
      EMLX.fast_sdpa_masked(
        EMLX.Backend.from_nx(q),
        EMLX.Backend.from_nx(k),
        EMLX.Backend.from_nx(v),
        EMLX.Backend.from_nx(mask),
        opts[:scale]
      )

    assert_no_nan_inf!(result_ref, :sdpa)
    out = EMLX.Backend.to_nx(result_ref)

    if Nx.type(out) != Nx.type(q), do: Nx.as_type(out, Nx.type(q)), else: out
  end

  @doc false
  def sdpa_masked_sinks_callback(
        {%Nx.Tensor{} = q, %Nx.Tensor{} = k, %Nx.Tensor{} = v, %Nx.Tensor{} = mask,
         %Nx.Tensor{} = sinks},
        opts
      ) do
    result_ref =
      EMLX.fast_sdpa_masked(
        EMLX.Backend.from_nx(q),
        EMLX.Backend.from_nx(k),
        EMLX.Backend.from_nx(v),
        EMLX.Backend.from_nx(mask),
        opts[:scale],
        EMLX.Backend.from_nx(sinks)
      )

    assert_no_nan_inf!(result_ref, :sdpa)
    out = EMLX.Backend.to_nx(result_ref)

    if Nx.type(out) != Nx.type(q), do: Nx.as_type(out, Nx.type(q)), else: out
  end

  @doc """
  Flash-attention SDPA with a built-in causal mask (`mlx::fast::scaled_dot_product_attention`,
  `mask_mode="causal"`).

  MLX constructs the upper-triangular causal mask internally without materialising it,
  making this equivalent to `scaled_dot_product_attention/5` with a causal boolean mask
  but cheaper: no mask tensor allocation, and the mask is fused into the Metal kernel.

  GQA-native: `k`/`v` may have fewer heads than `q` — no pre-tiling required.

  Input/output layout matches `scaled_dot_product_attention/4`:
  - `q`     — `{B, N_q,  T_q,  D}`
  - `k`     — `{B, N_kv, T_kv, D}`
  - `v`     — `{B, N_kv, T_kv, D}`
  - `scale` — pre-computed scalar (typically `1 / sqrt(D)`)
  - Output  — `{B, N_q, T_q, D}`, same dtype as `q`
  """
  deftransform scaled_dot_product_attention_causal(q, k, v, scale) do
    scaled_dot_product_attention_causal(q, k, v, scale, [])
  end

  @doc """
  Causal flash-attention SDPA with `opts` (`:sinks` — see
  `scaled_dot_product_attention/5`).
  """
  deftransform scaled_dot_product_attention_causal(q, k, v, scale, opts) when is_list(opts) do
    sinks = Keyword.get(opts, :sinks)

    if traced?([q, k, v, sinks]) do
      out = Nx.to_template(q)

      if sinks do
        inner =
          Nx.runtime_call(out, {q, k, v, sinks}, [scale: scale], &sdpa_causal_sinks_callback/2)

        fused =
          emlx_metadata(inner, :fast_sdpa_causal_sinks, [q, k, v, sinks], [
            NativeExpr.f64_bits(scale)
          ])

        with_reference_grad(fused, [q, k, v, sinks], fn q, k, v, sinks ->
          sdpa_reference(q, k, v, scale, causal: true, sinks: sinks)
        end)
      else
        inner = Nx.runtime_call(out, {q, k, v}, [scale: scale], &sdpa_causal_callback/2)
        fused = emlx_metadata(inner, :fast_sdpa_causal, [q, k, v], [NativeExpr.f64_bits(scale)])

        with_reference_grad(fused, [q, k, v], fn q, k, v ->
          sdpa_reference(q, k, v, scale, causal: true)
        end)
      end
    else
      if sinks do
        sdpa_causal_sinks_callback({q, k, v, sinks}, scale: scale)
      else
        sdpa_causal_callback({q, k, v}, scale: scale)
      end
    end
  end

  @doc false
  def sdpa_causal_callback({%Nx.Tensor{} = q, %Nx.Tensor{} = k, %Nx.Tensor{} = v}, opts) do
    result_ref =
      EMLX.fast_sdpa_causal(
        EMLX.Backend.from_nx(q),
        EMLX.Backend.from_nx(k),
        EMLX.Backend.from_nx(v),
        opts[:scale]
      )

    assert_no_nan_inf!(result_ref, :sdpa)
    out = EMLX.Backend.to_nx(result_ref)

    if Nx.type(out) != Nx.type(q), do: Nx.as_type(out, Nx.type(q)), else: out
  end

  @doc false
  def sdpa_causal_sinks_callback(
        {%Nx.Tensor{} = q, %Nx.Tensor{} = k, %Nx.Tensor{} = v, %Nx.Tensor{} = sinks},
        opts
      ) do
    result_ref =
      EMLX.fast_sdpa_causal(
        EMLX.Backend.from_nx(q),
        EMLX.Backend.from_nx(k),
        EMLX.Backend.from_nx(v),
        opts[:scale],
        EMLX.Backend.from_nx(sinks)
      )

    assert_no_nan_inf!(result_ref, :sdpa)
    out = EMLX.Backend.to_nx(result_ref)

    if Nx.type(out) != Nx.type(q), do: Nx.as_type(out, Nx.type(q)), else: out
  end

  # ── SDPA reference formula ───────────────────────────────────────────────────
  #
  # Plain-Nx formula used only from a `with_reference_grad/3` closure (see
  # moduledoc) to compute `Nx.Defn.grad`'s backward pass. q/k/v are
  # `{B, N, T, D}`; GQA-repeats k/v to N_q heads before a broadcasted
  # dot-product (no batched-matmul axis juggling).

  defp sdpa_reference(q, k, v, scale, opts) do
    causal = Keyword.get(opts, :causal, false)
    mask = Keyword.get(opts, :mask)
    key_mask = Keyword.get(opts, :key_mask)
    sinks = Keyword.get(opts, :sinks)
    kv_offset = Keyword.get(opts, :kv_offset, 0)

    {_b, n_q, t_q, _d} = Nx.shape(q)
    {_b, n_kv, t_kv, _d} = Nx.shape(k)
    groups = div(n_q, n_kv)

    k_g = repeat_kv_heads(k, groups)
    v_g = repeat_kv_heads(v, groups)

    # Batched contraction over D (no explicit {B,N,T_q,T_kv,D} intermediate —
    # keeps this reference cheap to *trace* even at prefill sequence lengths,
    # since it's rebuilt (never evaluated) on every EMLX-compiled call).
    scores =
      q
      |> Nx.dot([3], [0, 1], k_g, [3], [0, 1])
      |> Nx.multiply(scale * 1.0)

    scores =
      cond do
        causal and key_mask != nil -> apply_causal_key_mask(scores, kv_offset, t_q, t_kv, key_mask)
        causal -> apply_causal_mask(scores, kv_offset, t_q, t_kv)
        mask != nil -> apply_generic_mask(scores, mask)
        true -> scores
      end

    scores =
      if sinks do
        b = elem(Nx.shape(scores), 0)

        sinks_col =
          sinks
          |> Nx.as_type(Nx.type(scores))
          |> Nx.reshape({1, n_q, 1, 1})
          |> Nx.broadcast({b, n_q, t_q, 1})

        Nx.concatenate([scores, sinks_col], axis: 3)
      else
        scores
      end

    probs =
      scores
      |> Nx.subtract(Nx.reduce_max(scores, axes: [3], keep_axes: true))
      |> Nx.exp()

    probs = Nx.divide(probs, Nx.sum(probs, axes: [3], keep_axes: true))
    probs = Nx.slice_along_axis(probs, 0, t_kv, axis: 3)

    probs
    |> Nx.dot([3], [0, 1], v_g, [2], [0, 1])
    |> Nx.as_type(Nx.type(q))
  end

  defp repeat_kv_heads(kv, 1), do: kv

  defp repeat_kv_heads(kv, groups) do
    {b, n_kv, t, d} = Nx.shape(kv)

    kv
    |> Nx.new_axis(2)
    |> Nx.broadcast({b, n_kv, groups, t, d})
    |> Nx.reshape({b, n_kv * groups, t, d})
  end

  # `Nx.select/3` uses its *predicate*'s shape as the output shape verbatim
  # (it does not pick the pairwise-broadcast-compatible union like `add`/
  # `multiply` do), so `mask`/`keep` must already be pre-broadcast to
  # `Nx.shape(scores)` here. That broadcast runs on `Nx.iota`'s constant
  # backend (see `iota_bin/2`'s note) — `Nx.BinaryBackend`, plain CPU memory —
  # rather than on `scores`'s own (traced) backend, so it never touches a
  # real GPU/MLX allocation.
  defp apply_causal_mask(scores, kv_offset, t_q, t_kv) do
    query_positions = iota_bin({t_q}, :s32) |> Nx.add(kv_offset) |> Nx.reshape({1, 1, t_q, 1})
    key_positions = iota_bin({t_kv}, :s32) |> Nx.reshape({1, 1, 1, t_kv})

    mask =
      key_positions |> Nx.less_equal(query_positions) |> Nx.broadcast(Nx.shape(scores))

    Nx.select(mask, scores, neg_inf_like(scores))
  end

  defp apply_causal_key_mask(scores, kv_offset, t_q, t_kv, key_mask) do
    b = elem(Nx.shape(key_mask), 0)
    km = key_mask |> Nx.not_equal(0) |> Nx.reshape({b, 1, 1, t_kv})

    query_positions = iota_bin({t_q}, :s32) |> Nx.add(kv_offset) |> Nx.reshape({1, 1, t_q, 1})
    key_positions = iota_bin({t_kv}, :s32) |> Nx.reshape({1, 1, 1, t_kv})
    causal_bool = Nx.less_equal(key_positions, query_positions)

    keep = km |> Nx.logical_and(causal_bool) |> Nx.broadcast(Nx.shape(scores))
    Nx.select(keep, scores, neg_inf_like(scores))
  end

  defp apply_generic_mask(scores, mask) do
    if Nx.Type.integer?(Nx.type(mask)) do
      bool_mask = mask |> Nx.not_equal(0) |> Nx.broadcast(Nx.shape(scores))
      Nx.select(bool_mask, scores, neg_inf_like(scores))
    else
      Nx.add(scores, mask)
    end
  end

  # Pinned to `Nx.BinaryBackend` (cheap, CPU-only) rather than
  # `Nx.default_backend/0` — `deftransform` code (unlike `defn` bodies) isn't
  # auto-rewritten into `Nx.Defn.Expr` construction, so plain `Nx.iota/2` and
  # `Nx.tensor/2` calls here would otherwise allocate *real* tensors on
  # whatever backend the caller has configured (typically `EMLX.Backend`,
  # i.e. real, uncomputed MLX graph nodes) merely to be embedded as constants
  # in this never-evaluated reference formula.
  defp iota_bin(shape, type), do: Nx.iota(shape, type: type, backend: Nx.BinaryBackend)

  defp neg_inf_like(scores), do: Nx.tensor(-1.0e9, type: Nx.type(scores), backend: Nx.BinaryBackend)

  # ── Einsum ────────────────────────────────────────────────────────────────

  @doc """
  Variadic-operand einsum computed by MLX's path-optimised
  `mlx::core::einsum` kernel (Emily `Emily.Fast.einsum/2` parity).

  `subscripts` is a standard Einstein-summation equation (e.g.
  `"ij,jk->ik"`, `"bij,bjk->bik"`, `"bhid,bhjd->bhij"`,
  `"ij,jk,kl->il"`). `operands` is the corresponding list of 2+ tensors.

  ## Eager-only, not defn-callable

  Unlike the other helpers in this module, `einsum/2` does **not** emit an
  `Nx.Defn.Expr` node — it takes refs directly off `EMLX.Backend`-backed
  tensors and calls the NIF eagerly, in the same "direct-call helper" style
  as `EMLX.quantized_matmul/2`. Every operand must live on `EMLX.Backend`;
  anything else raises `ArgumentError`.

  ## Examples

      iex> a = Nx.iota({2, 3}, backend: EMLX.Backend, type: :f32)
      iex> b = Nx.iota({3, 4}, backend: EMLX.Backend, type: :f32)
      iex> y = EMLX.Fast.einsum("ij,jk->ik", [a, b])
      iex> Nx.shape(y)
      {2, 4}

  """
  @spec einsum(String.t(), [Nx.Tensor.t()]) :: Nx.Tensor.t()
  def einsum(subscripts, operands) when is_binary(subscripts) and is_list(operands) do
    refs = Enum.map(operands, &einsum_operand_ref!/1)

    EMLX.einsum(refs, subscripts)
    |> EMLX.Backend.to_nx()
  end

  defp einsum_operand_ref!(%Nx.Tensor{data: %EMLX.Backend{ref: ref}}), do: ref

  defp einsum_operand_ref!(%Nx.Tensor{data: %other_backend{}}) do
    raise ArgumentError,
          "EMLX.Fast.einsum/2: every operand must live on EMLX.Backend, got a " <>
            "#{inspect(other_backend)}-backed tensor. Transfer with " <>
            "Nx.backend_transfer/2 first."
  end
end
