#include "emlx_nif_shared.hpp"

// mlx::fast ops — single fused Metal shaders
// ============================================================================

// fast_rms_norm — fused RMS normalisation
// MLX: mlx::fast::rms_norm(x, weight, eps, stream) → array, same shape as x
// weight is optional<array> in the C++ API; TENSOR_PARAM gives array* which
// implicitly converts to optional<array>.
NIF(fast_rms_norm) {
  TENSOR_PARAM(0, x);
  TENSOR_PARAM(1, weight);
  PARAM(2, double, eps);
  DEVICE_PARAM(3, device);

  TENSOR(fast::rms_norm(*x, *weight, (float)eps, device));
}
ASYNC_NIF(fast_rms_norm)

// fast_rope — fused rotary position embedding (scalar offset arity)
// MLX: mlx::fast::rope(x, dims, traditional, base, scale, offset, freqs, stream)
// base is optional<float>; we always supply it.
// traditional=false → split-half (Qwen3 convention).
NIF(fast_rope) {
  TENSOR_PARAM(0, a);
  PARAM(1, int, dims);
  PARAM(2, bool, traditional);
  PARAM(3, double, base);
  PARAM(4, double, scale);
  PARAM(5, int, offset);
  DEVICE_PARAM(6, device);

  TENSOR(fast::rope(*a, dims, traditional, (float)base, (float)scale,
                   offset, std::nullopt, device));
}
ASYNC_NIF(fast_rope)

// fast_sdpa — flash-attention SDPA, no mask
// MLX: mlx::fast::scaled_dot_product_attention(q, k, v, scale, mask_mode, mask_arr, sinks, stream)
// GQA-native: k/v may have fewer heads than q.
NIF(fast_sdpa) {
  TENSOR_PARAM(0, q);
  TENSOR_PARAM(1, k);
  TENSOR_PARAM(2, v);
  PARAM(3, double, scale);
  DEVICE_PARAM(4, device);

  TENSOR(fast::scaled_dot_product_attention(
      *q, *k, *v, (float)scale, "", std::nullopt, std::nullopt, device));
}
ASYNC_NIF(fast_sdpa)

// fast_sdpa_masked — flash-attention SDPA with additive or boolean mask
// mask_mode="array" tells MLX to treat mask_arr as an additive bias/bool mask.
NIF(fast_sdpa_masked) {
  TENSOR_PARAM(0, q);
  TENSOR_PARAM(1, k);
  TENSOR_PARAM(2, v);
  PARAM(3, double, scale);
  TENSOR_PARAM(4, mask);
  DEVICE_PARAM(5, device);

  TENSOR(fast::scaled_dot_product_attention(
      *q, *k, *v, (float)scale, "array", *mask, std::nullopt, device));
}
ASYNC_NIF(fast_sdpa_masked)

// fast_layer_norm — fused layer normalisation
// MLX: mlx::fast::layer_norm(x, weight?, bias?, eps, stream)
// weight and bias are optional; we always provide both.
NIF(fast_layer_norm) {
  TENSOR_PARAM(0, x);
  TENSOR_PARAM(1, weight);
  TENSOR_PARAM(2, bias);
  PARAM(3, double, eps);
  DEVICE_PARAM(4, device);

  TENSOR(fast::layer_norm(*x, *weight, *bias, (float)eps, device));
}
ASYNC_NIF(fast_layer_norm)

// fast_layer_norm_no_bias — fused layer norm without bias (weight-only variant)
NIF(fast_layer_norm_no_bias) {
  TENSOR_PARAM(0, x);
  TENSOR_PARAM(1, weight);
  PARAM(2, double, eps);
  DEVICE_PARAM(3, device);

  TENSOR(fast::layer_norm(*x, *weight, std::nullopt, (float)eps, device));
}
ASYNC_NIF(fast_layer_norm_no_bias)

// fast_rope_ids — fused RoPE with per-batch offset array (position_ids)
// Calls the array-offset overload of mlx::fast::rope.
// offset must be shape {B} — one starting position per batch example.
// Assumes positions are sequential within each example: [offset[b], offset[b]+1, ..., offset[b]+T-1].
NIF(fast_rope_ids) {
  TENSOR_PARAM(0, a);
  PARAM(1, int, dims);
  PARAM(2, bool, traditional);
  PARAM(3, double, base);
  PARAM(4, double, scale);
  TENSOR_PARAM(5, offset);
  DEVICE_PARAM(6, device);

  TENSOR(fast::rope(*a, dims, traditional, (float)base, (float)scale,
                   *offset, std::nullopt, device));
}
ASYNC_NIF(fast_rope_ids)

// fast_rope_with_freqs — fused RoPE with precomputed inv-frequency vector
// Calls the freqs overload of mlx::fast::rope (base=nullopt, freqs supplied).
// offset must be shape {B} — one starting position per batch example.
// freqs must be shape {dims/2} — precomputed inverse frequencies.
NIF(fast_rope_with_freqs) {
  TENSOR_PARAM(0, a);
  PARAM(1, int, dims);
  PARAM(2, bool, traditional);
  PARAM(3, double, scale);
  TENSOR_PARAM(4, offset);
  TENSOR_PARAM(5, freqs);
  DEVICE_PARAM(6, device);

  TENSOR(fast::rope(*a, dims, traditional, std::nullopt, (float)scale,
                   *offset, *freqs, device));
}
ASYNC_NIF(fast_rope_with_freqs)

// fast_rope_positions — RoPE for arbitrary per-token position_ids.
//
// Unlike fast_rope_ids, this does not assume positions are sequential within a
// row. It mirrors Bumblebee's apply_rotary_embedding:
//   inv_freq = 1 / base^(2i/dims)
//   angle    = position_ids * inv_freq
//   out      = a * cos(angle) + rotate_half(a) * sin(angle)
//
// Input:
//   a            {B, T, H, D}
//   position_ids {B, T}
//
// Notes:
// - `traditional=true` is currently unsupported in this fallback NIF.
// - Intended for high-base / padded paths where fast_rope_ids is not correct.
NIF(fast_rope_positions) {
  TENSOR_PARAM(0, a);
  PARAM(1, int, dims);
  PARAM(2, bool, traditional);
  PARAM(3, double, base);
  PARAM(4, double, scale);
  TENSOR_PARAM(5, position_ids);
  DEVICE_PARAM(6, device);

  try {
    if (a->ndim() != 4) {
      return nx::nif::error(env, "fast_rope_positions expects a rank-4 tensor a {B,T,H,D}");
    }
    if (position_ids->ndim() != 2) {
      return nx::nif::error(env, "fast_rope_positions expects rank-2 position_ids {B,T}");
    }

    int B = a->shape(0);
    int T = a->shape(1);
    int H = a->shape(2);
    int D = a->shape(3);

    if (position_ids->shape(0) != B || position_ids->shape(1) != T) {
      return nx::nif::error(env, "fast_rope_positions: position_ids shape must match {B,T} from a");
    }
    if (dims <= 0 || dims > D || (dims % 2) != 0) {
      return nx::nif::error(env, "fast_rope_positions: dims must be even and <= last dimension");
    }
    if (traditional) {
      return nx::nif::error(env, "fast_rope_positions: traditional=true not supported");
    }

    int half = dims / 2;

    std::vector<float> inv_freq_host(half);
    float base_f = static_cast<float>(base);
    for (int i = 0; i < half; ++i) {
      float expo = (2.0f * static_cast<float>(i)) / static_cast<float>(dims);
      inv_freq_host[i] = 1.0f / std::pow(base_f, expo);
    }

    auto inv_freq = array(inv_freq_host.begin(), {half}, float32);

    auto pos_f = astype(*position_ids, float32, device);
    auto pos_bt1 = reshape(pos_f, {B, T, 1}, device);
    auto inv_11h = reshape(inv_freq, {1, 1, half}, device);
    auto scale_arr = array(static_cast<float>(scale), float32);
    auto angles = multiply(multiply(pos_bt1, inv_11h, device), scale_arr, device);

    auto cos_bt1h = astype(reshape(cos(angles, device), {B, T, 1, half}, device), a->dtype(), device);
    auto sin_bt1h = astype(reshape(sin(angles, device), {B, T, 1, half}, device), a->dtype(), device);

    auto cos_full = concatenate(std::vector<array>{cos_bt1h, cos_bt1h}, 3, device);
    auto sin_full = concatenate(std::vector<array>{sin_bt1h, sin_bt1h}, 3, device);

    auto x1 = slice(*a, to_shape({0, 0, 0, 0}), to_shape({B, T, H, half}), device);
    auto x2 = slice(*a, to_shape({0, 0, 0, half}), to_shape({B, T, H, dims}), device);
    auto rotated = concatenate(std::vector<array>{negative(x2, device), x1}, 3, device);

    auto a_head = slice(*a, to_shape({0, 0, 0, 0}), to_shape({B, T, H, dims}), device);
    auto rope_head = add(multiply(a_head, cos_full, device), multiply(rotated, sin_full, device), device);

    if (dims == D) {
      TENSOR(rope_head);
    } else {
      auto tail = slice(*a, to_shape({0, 0, 0, dims}), to_shape({B, T, H, D}), device);
      TENSOR(concatenate(std::vector<array>{rope_head, tail}, 3, device));
    }
  }
  CATCH()
}
ASYNC_NIF(fast_rope_positions)

// fast_sdpa_causal_key_masked — causal SDPA that checks key_mask at C++ level.
// key_mask shape: {B, T_kv} — 1 = attend, 0 = padding.
// If all values are 1 (no padding), dispatches to fast causal SDPA (no mask alloc).
// Otherwise builds a combined causal + key_mask additive float mask and uses
// masked SDPA. The all-ones check forces eval of only the small key_mask subgraph.
NIF(fast_sdpa_causal_key_masked) {
  TENSOR_PARAM(0, q);        // {B, N_q,  T_q,  D}
  TENSOR_PARAM(1, k);        // {B, N_kv, T_kv, D}
  TENSOR_PARAM(2, v);        // {B, N_kv, T_kv, D}
  PARAM(3, double, scale);
  TENSOR_PARAM(4, key_mask); // {B, T_kv} boolean / int
  PARAM(5, int, kv_offset);  // caller-controlled: decode → T_kv-1, prefill → 0
  DEVICE_PARAM(6, device);

  // key_mask values are 0/1 (int or bool); astype→bool_ then all() checks all non-zero.
  bool trivial = all(astype(*key_mask, bool_)).item<bool>();

  if (trivial) {
    TENSOR(fast::scaled_dot_product_attention(
        *q, *k, *v, (float)scale, "causal", std::nullopt, std::nullopt, device));
  } else {
    // Expand key_mask {B, T_kv} → {B, 1, 1, T_kv} for broadcasting.
    auto km = reshape(*key_mask, {key_mask->shape(0), 1, 1, key_mask->shape(1)});
    int T_q  = q->shape(2);
    int T_kv = k->shape(2);

    // Causal boolean: position j is visible from query row i if j <= i + kv_offset.
    auto row = reshape(arange(T_q, int32), {1, 1, T_q, 1});
    auto col = reshape(arange(T_kv, int32), {1, 1, 1, T_kv});
    auto causal_bool = less_equal(col, add(row, array(kv_offset, int32)));

    // Combined: attend where key_mask=1 AND position is causally reachable.
    auto keep = logical_and(km, causal_bool);

    // Additive mask: 0.0 = attend, -inf = masked out.
    // Use Q's dtype so the mask does not promote the SDPA output above Q's type.
    // (e.g. float32 mask + bfloat16 Q/K/V would force float32 output — rejected by MLX.)
    auto mask_dtype = q->dtype();
    auto zero_val = zeros({}, mask_dtype);
    auto neginf_val = full({}, -std::numeric_limits<float>::infinity(), mask_dtype);
    auto additive = where(keep, zero_val, neginf_val);

    TENSOR(fast::scaled_dot_product_attention(
        *q, *k, *v, (float)scale, "array", additive, std::nullopt, device));
  }
}
ASYNC_NIF(fast_sdpa_causal_key_masked)

// fast_swiglu — fused SwiGLU: silu(gate) * up
// SiLU (Sigmoid Linear Unit): silu(x) = x * sigmoid(x).
// gate and up must have the same shape; output has the same shape and dtype.
NIF(fast_swiglu) {
  TENSOR_PARAM(0, gate);
  TENSOR_PARAM(1, up);
  DEVICE_PARAM(2, device);

  // silu(gate) * up where silu(x) = x * sigmoid(x).
  // MLX's lazy graph evaluation fuses these into a single kernel dispatch.
  TENSOR(multiply(multiply(*gate, sigmoid(*gate, device), device), *up, device));
}
ASYNC_NIF(fast_swiglu)

NIF(fast_sdpa_causal) {
  TENSOR_PARAM(0, q);
  TENSOR_PARAM(1, k);
  TENSOR_PARAM(2, v);
  PARAM(3, double, scale);
  DEVICE_PARAM(4, device);

  TENSOR(fast::scaled_dot_product_attention(
      *q, *k, *v, (float)scale, "causal", std::nullopt, std::nullopt, device));
}
ASYNC_NIF(fast_sdpa_causal)

// kv_cache_attention — fused KV cache update + variable-length SDPA in one Metal pass.
//
// Inputs (Bumblebee {B, T, N, D} convention — no pre-transpose needed from caller):
//   q        — {B, T_q,   N_q,  D}  post-RoPE query
//   new_k    — {B, T_new, N_kv, D}  post-RoPE key for new token(s)
//   new_v    — {B, T_new, N_kv, D}  value for new token(s)
//   k_cache  — {B, T_max, N_kv, D}  pre-allocated key buffer (ETS side-channel)
//   v_cache  — {B, T_max, N_kv, D}  pre-allocated value buffer
//   offset   — int   number of valid positions already in cache
//   scale    — float 1/sqrt(head_dim)
//
// Returns 3-tuple {attn_out, k_upd, v_upd}:
//   attn_out — {B, T_q, N_q, D}   (Bumblebee format, ready for flatten_trailing + o_proj)
//   k_upd    — {B, T_max, N_kv, D}  full updated key buffer (for ETS insert)
//   v_upd    — {B, T_max, N_kv, D}  full updated value buffer (for ETS insert)
//
// All three are eval'd in a single MLX command buffer — slice_update → valid_slice
// → SDPA form one fused Metal encoder submission.
NIF(kv_cache_attention) {
  TENSOR_PARAM(0, q);
  TENSOR_PARAM(1, new_k);
  TENSOR_PARAM(2, new_v);
  TENSOR_PARAM(3, k_cache);
  TENSOR_PARAM(4, v_cache);
  PARAM(5, int, offset);
  PARAM(6, double, scale);
  DEVICE_PARAM(7, device);

  try {
    int B     = q->shape(0);
    int T_q   = q->shape(1);
    int D     = q->shape(3);
    int N_kv  = new_k->shape(2);
    int T_new = new_k->shape(1);
    int valid_len = offset + T_new;

    // Donate the cache buffers so MLX can reuse them in-place.
    // After std::move, the ENIF resource blocks hold moved-from arrays.
    // k_cache_owned / v_cache_owned are the sole shared_ptr owners until
    // slice_update copies them into its inputs list (count rises to 2).
    // When this function returns, those locals destruct → count drops to 1
    // → SliceUpdate::eval_gpu detects is_donatable() → no new 4 MB buffer.
    auto k_cache_owned = std::move(*k_cache);
    auto v_cache_owned = std::move(*v_cache);

    // 1. Insert new K/V at cache position `offset`.
    //    Output has the same shape as k_cache: {B, T_max, N_kv, D}.
    auto k_upd = mlx::core::slice_update(
        k_cache_owned, *new_k,
        to_shape({0, offset, 0, 0}),
        to_shape({B, valid_len, N_kv, D}),
        device);
    auto v_upd = mlx::core::slice_update(
        v_cache_owned, *new_v,
        to_shape({0, offset, 0, 0}),
        to_shape({B, valid_len, N_kv, D}),
        device);

    // 2. Slice to valid portion: {B, valid_len, N_kv, D}.
    auto k_valid = mlx::core::slice(
        k_upd, to_shape({0, 0, 0, 0}), to_shape({B, valid_len, N_kv, D}), device);
    auto v_valid = mlx::core::slice(
        v_upd, to_shape({0, 0, 0, 0}), to_shape({B, valid_len, N_kv, D}), device);

    // 3. Transpose from Bumblebee {B, T, N, D} to MLX SDPA {B, N, T, D}.
    auto q_t      = mlx::core::transpose(*q,      {0, 2, 1, 3}, device);
    auto k_valid_t = mlx::core::transpose(k_valid, {0, 2, 1, 3}, device);
    auto v_valid_t = mlx::core::transpose(v_valid, {0, 2, 1, 3}, device);

    // 4. SDPA over the valid slice.
    //    T_new == 1 (decode): no mask — single query is trivially causal.
    //    T_new > 1  (prefill): build an additive causal mask in q's dtype to avoid
    //    any float32 promotion that would mismatch BF16 Q/K/V tensors.
    auto build_prefill_mask = [&]() -> mlx::core::array {
      auto mask_dtype = q->dtype();
      auto zero_val   = mlx::core::zeros({}, mask_dtype, device);
      auto neginf_val = mlx::core::full({}, -std::numeric_limits<float>::infinity(), mask_dtype, device);

      // Causal: query position i can attend to key position j iff j <= i + kv_offset.
      int kv_offset = valid_len - T_q;
      auto row = mlx::core::reshape(mlx::core::arange(T_q,   mlx::core::int32, device), {1, 1, T_q, 1},   device);
      auto col = mlx::core::reshape(mlx::core::arange(valid_len, mlx::core::int32, device), {1, 1, 1, valid_len}, device);
      auto causal_bool = mlx::core::less_equal(
          col, mlx::core::add(row, mlx::core::array(kv_offset, mlx::core::int32), device), device);

      return mlx::core::where(causal_bool, zero_val, neginf_val, device);
    };

    auto attn_t = (T_new == 1)
      ? mlx::core::fast::scaled_dot_product_attention(
            q_t, k_valid_t, v_valid_t, (float)scale, "", std::nullopt, std::nullopt, device)
      : mlx::core::fast::scaled_dot_product_attention(
            q_t, k_valid_t, v_valid_t, (float)scale, "array", build_prefill_mask(), std::nullopt, device);

    // 5. Transpose output back: {B, N_q, T_q, D} → {B, T_q, N_q, D} (Bumblebee format).
    auto attn_out = mlx::core::transpose(attn_t, {0, 2, 1, 3}, device);

    ERL_NIF_TERM result_tuple[3];
    result_tuple[0] = create_tensor_resource(env, attn_out);
    result_tuple[1] = create_tensor_resource(env, k_upd);
    result_tuple[2] = create_tensor_resource(env, v_upd);

    return nx::nif::ok(env, enif_make_tuple3(env, result_tuple[0], result_tuple[1], result_tuple[2]));
  }
  CATCH()
}
ASYNC_NIF(kv_cache_attention)

// kv_cache_attention_masked — same as kv_cache_attention but applies a key_mask
// in addition to the causal mask. This is required for padded prefill where the
// first `padding_len` key positions contain garbage and must be excluded from
// attention in both the prefill pass and subsequent decode steps.
//
// Extra parameter vs kv_cache_attention:
//   key_mask — {B, T_kv} with 1 = attend, 0 = padding-skip
//
// The combined additive mask: 0 where (key_mask=1 AND causal), -inf otherwise.
// This is built directly without calling `all()` to avoid triggering Metal sort
// kernels that may not be available for all tensor shapes.
//
// Returns same 3-tuple as kv_cache_attention: {attn_out, k_upd, v_upd}.
NIF(kv_cache_attention_masked) {
  TENSOR_PARAM(0, q);         // {B, T_q,   N_q,  D}  Bumblebee format
  TENSOR_PARAM(1, new_k);     // {B, T_new, N_kv, D}
  TENSOR_PARAM(2, new_v);     // {B, T_new, N_kv, D}
  TENSOR_PARAM(3, k_cache);   // {B, T_max, N_kv, D}
  TENSOR_PARAM(4, v_cache);   // {B, T_max, N_kv, D}
  PARAM(5, int, offset);
  PARAM(6, double, scale);
  TENSOR_PARAM(7, key_mask);  // {B, T_kv} — 1=attend, 0=skip
  DEVICE_PARAM(8, device);

  try {
    int B     = q->shape(0);
    int T_q   = q->shape(1);
    int D     = q->shape(3);
    int N_kv  = new_k->shape(2);
    int T_new = new_k->shape(1);
    int valid_len = offset + T_new;

    // Donate cache buffers (same pattern as kv_cache_attention).
    auto k_cache_owned = std::move(*k_cache);
    auto v_cache_owned = std::move(*v_cache);

    // 1. Insert new K/V at cache position `offset`.
    auto k_upd = mlx::core::slice_update(
        k_cache_owned, *new_k,
        to_shape({0, offset, 0, 0}),
        to_shape({B, valid_len, N_kv, D}),
        device);
    auto v_upd = mlx::core::slice_update(
        v_cache_owned, *new_v,
        to_shape({0, offset, 0, 0}),
        to_shape({B, valid_len, N_kv, D}),
        device);

    // 2. Slice to valid portion.
    auto k_valid = mlx::core::slice(
        k_upd, to_shape({0, 0, 0, 0}), to_shape({B, valid_len, N_kv, D}), device);
    auto v_valid = mlx::core::slice(
        v_upd, to_shape({0, 0, 0, 0}), to_shape({B, valid_len, N_kv, D}), device);

    // 3. Transpose from Bumblebee {B, T, N, D} to MLX SDPA {B, N, T, D}.
    auto q_t       = mlx::core::transpose(*q,       {0, 2, 1, 3}, device);
    auto k_valid_t = mlx::core::transpose(k_valid,  {0, 2, 1, 3}, device);
    auto v_valid_t = mlx::core::transpose(v_valid,  {0, 2, 1, 3}, device);

    // 4. Compute attention output.
    //
    //    For decode (T_q == 1), the causal constraint is trivially satisfied: a single
    //    query position always sees all valid keys. We skip the arange/reshape/less_equal
    //    construction and dispatch to the cheapest SDPA variant:
    //      - trivial key_mask (all-ones, non-padded batch): pure causal Metal kernel.
    //      - non-trivial key_mask (padded batch): key_mask-only additive mask.
    //    The all().item<bool>() sync is negligible at {B, valid_len} (1–256 elements).
    //
    //    For prefill (T_q > 1), build the full causal + key_mask combined additive mask.
    auto mask_dtype = q->dtype();
    auto zero_val   = mlx::core::zeros({}, mask_dtype, device);
    auto neginf_val = mlx::core::full({}, -std::numeric_limits<float>::infinity(), mask_dtype, device);

    mlx::core::array attn_t = [&]() -> mlx::core::array {
      if (T_q == 1) {
        // For single-query decode, the causal constraint is trivially satisfied
        // (one query always attends to all preceding keys). Build a key_mask-only
        // additive mask — 3 GPU ops vs 8 in the original full causal+mask path.
        //
        // We intentionally skip the all().item<bool>() trivial check here: that
        // check forces a GPU→CPU sync on every layer call (28×/step), whose
        // latency cost exceeds the savings from choosing "causal" mode. For
        // non-padded inference the additive mask is all-zeros, which is
        // functionally identical to pure causal mode.
        auto km = mlx::core::reshape(
            mlx::core::astype(*key_mask, mlx::core::bool_, device),
            {key_mask->shape(0), 1, 1, key_mask->shape(1)},
            device);
        auto additive = mlx::core::where(km, zero_val, neginf_val, device);
        return mlx::core::fast::scaled_dot_product_attention(
            q_t, k_valid_t, v_valid_t, (float)scale, "array",
            additive, std::nullopt, device);
      } else {
        // Prefill (T_q > 1): full causal + key_mask combined additive mask.
        //   kv_offset: valid_len - T_q (= 0 for a fresh prefill of length T_q).
        int kv_offset = valid_len - T_q;
        auto row = mlx::core::reshape(
            mlx::core::arange(T_q, mlx::core::int32, device), {1, 1, T_q, 1}, device);
        auto col = mlx::core::reshape(
            mlx::core::arange(valid_len, mlx::core::int32, device), {1, 1, 1, valid_len}, device);
        auto causal_bool = mlx::core::less_equal(
            col, mlx::core::add(row, mlx::core::array(kv_offset, mlx::core::int32), device), device);

        auto km = mlx::core::reshape(
            mlx::core::astype(*key_mask, mlx::core::bool_, device),
            {key_mask->shape(0), 1, 1, key_mask->shape(1)},
            device);
        auto keep = mlx::core::logical_and(km, causal_bool, device);
        auto additive = mlx::core::where(keep, zero_val, neginf_val, device);

        return mlx::core::fast::scaled_dot_product_attention(
            q_t, k_valid_t, v_valid_t, (float)scale, "array",
            additive, std::nullopt, device);
      }
    }();

    // Replace NaN with 0 for all-masked rows (softmax(-inf,...,-inf) = NaN in
    // Flash-Attention when seq_len >= Metal tile size, but semantically = 0).
    auto attn_safe = mlx::core::where(
        mlx::core::isnan(attn_t),
        mlx::core::zeros_like(attn_t, device),
        attn_t, device);

    // 6. Transpose output back: {B, N_q, T_q, D} → {B, T_q, N_q, D}.
    auto attn_out = mlx::core::transpose(attn_safe, {0, 2, 1, 3}, device);

    ERL_NIF_TERM result_tuple[3];
    result_tuple[0] = create_tensor_resource(env, attn_out);
    result_tuple[1] = create_tensor_resource(env, k_upd);
    result_tuple[2] = create_tensor_resource(env, v_upd);

    return nx::nif::ok(env, enif_make_tuple3(env, result_tuple[0], result_tuple[1], result_tuple[2]));
  }
  CATCH()
}
ASYNC_NIF(kv_cache_attention_masked)

// kv_cache_sdpa_update — fused donation-optimised KV cache update + SDPA
// for the native NIF loop (BNHD layout: {B, N, T, D}).
//
// The Bumblebee path uses {B, T, N, D} and goes through kv_cache_attention.
// This variant accepts q / new_k / new_v already transposed to {B, N, T, D}
// (as done by the native loop before put_slice), with the cache stored in the
// same {B, N_kv, T_max, D} layout.
//
// DONATION SEMANTICS: k_cache and v_cache are move-extracted from their ENIF
// resource blocks before slice_update.  When this function returns those
// locals destruct, dropping their shared_ptr refs from 2 → 1.  Only
// slice_update's inputs list retains the reference → SliceUpdate::eval_gpu
// detects is_donatable() → the existing 4 MB Metal buffer is reused in-place,
// no new allocation needed.
//
// Inputs (argv indices inside the sync NIF body, after worker is stripped):
//   q        — {B, N_q,  T_q,   D}  post-RoPE query  (native BNHD)
//   new_k    — {B, N_kv, T_new, D}  post-RoPE key
//   new_v    — {B, N_kv, T_new, D}  value
//   k_cache  — {B, N_kv, T_max, D}  pre-allocated key buffer
//   v_cache  — {B, N_kv, T_max, D}  pre-allocated value buffer
//   offset   — int   tokens already in cache
//   scale    — float 1/sqrt(head_dim)
//   device   — atom
//
// Returns {attn_out, k_upd, v_upd}:
//   attn_out — {B, N_q, T_q, D}   (native BNHD, caller transposes/reshapes)
//   k_upd    — {B, N_kv, T_max, D}  same Metal buffer as k_cache, updated
//   v_upd    — {B, N_kv, T_max, D}  same Metal buffer as v_cache, updated
NIF(kv_cache_sdpa_update) {
  TENSOR_PARAM(0, q);
  TENSOR_PARAM(1, new_k);
  TENSOR_PARAM(2, new_v);
  TENSOR_PARAM(3, k_cache);
  TENSOR_PARAM(4, v_cache);
  PARAM(5, int, offset);
  PARAM(6, double, scale);
  DEVICE_PARAM(7, device);

  try {
    int B       = q->shape(0);
    int N_kv    = new_k->shape(1);
    int T_new   = new_k->shape(2);
    int D       = q->shape(3);
    int valid_len = offset + T_new;

    // Move-extract: after this the ENIF resource blocks hold moved-from arrays.
    // k_cache_owned / v_cache_owned are the sole owners (use_count == 1) until
    // slice_update copies them into its inputs (use_count rises to 2).
    // On function return both locals destruct → use_count drops to 1 again
    // → is_donatable() is true at eval time → no new 4 MB Metal buffer.
    auto k_cache_owned = std::move(*k_cache);
    auto v_cache_owned = std::move(*v_cache);

    // 1. Insert new_k / new_v at cache position `offset` along axis 2 (T axis).
    //    Layout: {B, N_kv, T_max, D} — offset along dimension 2.
    auto k_upd = mlx::core::slice_update(
        k_cache_owned, *new_k,
        to_shape({0, 0, offset, 0}),
        to_shape({B, N_kv, valid_len, D}),
        device);
    auto v_upd = mlx::core::slice_update(
        v_cache_owned, *new_v,
        to_shape({0, 0, offset, 0}),
        to_shape({B, N_kv, valid_len, D}),
        device);

    // 2. Slice valid prefix: k_upd[0:B, 0:N_kv, 0:valid_len, 0:D].
    auto k_valid = mlx::core::slice(
        k_upd, to_shape({0, 0, 0, 0}), to_shape({B, N_kv, valid_len, D}), device);
    auto v_valid = mlx::core::slice(
        v_upd, to_shape({0, 0, 0, 0}), to_shape({B, N_kv, valid_len, D}), device);

    // 3. SDPA: q / k_valid / v_valid are already in {B, N, T, D} format.
    //    Decode (T_new == 1): no mask — single query is trivially causal.
    //    Prefill (T_new > 1): additive causal mask in q's dtype.
    auto build_prefill_mask = [&]() -> mlx::core::array {
      auto mask_dtype = q->dtype();
      auto zero_val   = mlx::core::zeros({}, mask_dtype, device);
      auto neginf_val = mlx::core::full({}, -std::numeric_limits<float>::infinity(), mask_dtype, device);
      int kv_offset = valid_len - T_new;
      auto row = mlx::core::reshape(
          mlx::core::arange(T_new,     mlx::core::int32, device), {1, 1, T_new, 1},     device);
      auto col = mlx::core::reshape(
          mlx::core::arange(valid_len, mlx::core::int32, device), {1, 1, 1, valid_len}, device);
      auto causal_bool = mlx::core::less_equal(
          col, mlx::core::add(row, mlx::core::array(kv_offset, mlx::core::int32), device), device);
      return mlx::core::where(causal_bool, zero_val, neginf_val, device);
    };

    auto attn_out = (T_new == 1)
      ? mlx::core::fast::scaled_dot_product_attention(
            *q, k_valid, v_valid, (float)scale, "", std::nullopt, std::nullopt, device)
      : mlx::core::fast::scaled_dot_product_attention(
            *q, k_valid, v_valid, (float)scale, "array", build_prefill_mask(), std::nullopt, device);
    // k_cache_owned and v_cache_owned destruct here → use_count 2 → 1.

    ERL_NIF_TERM result_tuple[3];
    result_tuple[0] = create_tensor_resource(env, attn_out);
    result_tuple[1] = create_tensor_resource(env, k_upd);
    result_tuple[2] = create_tensor_resource(env, v_upd);

    return nx::nif::ok(env, enif_make_tuple3(env, result_tuple[0], result_tuple[1], result_tuple[2]));
  }
  CATCH()
}
ASYNC_NIF(kv_cache_sdpa_update)

