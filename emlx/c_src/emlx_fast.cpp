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

static ERL_NIF_TERM qwen3_error(ErlNifEnv *env, const std::string &message) {
  return nx::nif::error(env, message.c_str());
}

static bool qwen3_check_rank(
    const mlx::core::array &tensor,
    int expected,
    const char *name,
    std::string &error) {
  if (tensor.ndim() != expected) {
    std::ostringstream msg;
    msg << name << " expects rank " << expected << ", got rank " << tensor.ndim();
    error = msg.str();
    return false;
  }

  return true;
}

static bool qwen3_check_positive(int value, const char *name, std::string &error) {
  if (value <= 0) {
    std::ostringstream msg;
    msg << name << " must be positive";
    error = msg.str();
    return false;
  }

  return true;
}

static bool qwen3_check_non_negative(int value, const char *name, std::string &error) {
  if (value < 0) {
    std::ostringstream msg;
    msg << name << " must be non-negative";
    error = msg.str();
    return false;
  }

  return true;
}

static bool qwen3_check_dim(
    const mlx::core::array &tensor,
    int axis,
    int expected,
    const char *name,
    const char *dim_name,
    std::string &error) {
  if (tensor.shape(axis) != expected) {
    std::ostringstream msg;
    msg << name << " " << dim_name << " must be " << expected
        << ", got " << tensor.shape(axis);
    error = msg.str();
    return false;
  }

  return true;
}

static bool qwen3_check_rank4_positive(
    const mlx::core::array &tensor,
    const char *name,
    std::string &error) {
  if (!qwen3_check_rank(tensor, 4, name, error)) {
    return false;
  }

  for (int axis = 0; axis < 4; ++axis) {
    if (tensor.shape(axis) <= 0) {
      std::ostringstream msg;
      msg << name << " dimensions must be positive";
      error = msg.str();
      return false;
    }
  }

  return true;
}

static bool qwen3_check_rank3_positive(
    const mlx::core::array &tensor,
    const char *name,
    std::string &error) {
  if (!qwen3_check_rank(tensor, 3, name, error)) {
    return false;
  }

  for (int axis = 0; axis < 3; ++axis) {
    if (tensor.shape(axis) <= 0) {
      std::ostringstream msg;
      msg << name << " dimensions must be positive";
      error = msg.str();
      return false;
    }
  }

  return true;
}

static bool qwen3_check_rank2_positive(
    const mlx::core::array &tensor,
    const char *name,
    std::string &error) {
  if (!qwen3_check_rank(tensor, 2, name, error)) {
    return false;
  }

  for (int axis = 0; axis < 2; ++axis) {
    if (tensor.shape(axis) <= 0) {
      std::ostringstream msg;
      msg << name << " dimensions must be positive";
      error = msg.str();
      return false;
    }
  }

  return true;
}

static bool qwen3_check_rank1_dim(
    const mlx::core::array &tensor,
    int expected,
    const char *name,
    std::string &error) {
  if (!qwen3_check_rank(tensor, 1, name, error)) {
    return false;
  }

  return qwen3_check_dim(tensor, 0, expected, name, "size", error);
}

static bool qwen3_validate_projection_width(
    const mlx::core::array &projection,
    int input_width,
    int head_dim,
    const char *name,
    std::string &error) {
  if (!qwen3_check_rank2_positive(projection, name, error)) {
    return false;
  }
  if (!qwen3_check_dim(projection, 0, input_width, name, "input width", error)) {
    return false;
  }
  if ((projection.shape(1) % head_dim) != 0) {
    std::ostringstream msg;
    msg << name << " output width must be divisible by head_dim";
    error = msg.str();
    return false;
  }

  return true;
}

static bool qwen3_validate_kv_cache_bn(
    const mlx::core::array &k_cache,
    const mlx::core::array &v_cache,
    int batch,
    int num_kv_heads,
    int offset,
    int token_count,
    int head_dim,
    std::string &error) {
  if (!qwen3_check_rank4_positive(k_cache, "k_cache", error) ||
      !qwen3_check_rank4_positive(v_cache, "v_cache", error)) {
    return false;
  }

  if (!qwen3_check_dim(k_cache, 0, batch, "k_cache", "batch", error) ||
      !qwen3_check_dim(v_cache, 0, batch, "v_cache", "batch", error) ||
      !qwen3_check_dim(k_cache, 1, num_kv_heads, "k_cache", "heads", error) ||
      !qwen3_check_dim(v_cache, 1, num_kv_heads, "v_cache", "heads", error) ||
      !qwen3_check_dim(k_cache, 3, head_dim, "k_cache", "head_dim", error) ||
      !qwen3_check_dim(v_cache, 3, head_dim, "v_cache", "head_dim", error)) {
    return false;
  }

  if (v_cache.shape(2) != k_cache.shape(2)) {
    error = "k_cache and v_cache capacity must match";
    return false;
  }

  int64_t required_len = static_cast<int64_t>(offset) + static_cast<int64_t>(token_count);
  int capacity = k_cache.shape(2);

  if (required_len > capacity) {
    std::ostringstream msg;
    msg << "KV cache capacity " << capacity
        << " is smaller than required length " << required_len;
    error = msg.str();
    return false;
  }

  return true;
}

static bool qwen3_validate_qkv_cache_attention(
    const mlx::core::array &q,
    const mlx::core::array &new_k,
    const mlx::core::array &new_v,
    const mlx::core::array &k_cache,
    const mlx::core::array &v_cache,
    int offset,
    int head_dim,
    std::string &error) {
  if (!qwen3_check_rank4_positive(q, "q", error) ||
      !qwen3_check_rank4_positive(new_k, "new_k", error) ||
      !qwen3_check_rank4_positive(new_v, "new_v", error) ||
      !qwen3_check_non_negative(offset, "offset", error) ||
      !qwen3_check_positive(head_dim, "head_dim", error)) {
    return false;
  }

  int B = q.shape(0);
  int T_new = q.shape(1);
  int N_q = q.shape(2);
  int D = q.shape(3);
  int N_kv = new_k.shape(2);

  if (D != head_dim) {
    error = "q last dimension must match head_dim";
    return false;
  }
  if ((N_q % N_kv) != 0) {
    error = "query heads must be divisible by key/value heads";
    return false;
  }
  if (!qwen3_check_dim(new_k, 0, B, "new_k", "batch", error) ||
      !qwen3_check_dim(new_v, 0, B, "new_v", "batch", error) ||
      !qwen3_check_dim(new_k, 1, T_new, "new_k", "sequence length", error) ||
      !qwen3_check_dim(new_v, 1, T_new, "new_v", "sequence length", error) ||
      !qwen3_check_dim(new_v, 2, N_kv, "new_v", "heads", error) ||
      !qwen3_check_dim(new_k, 3, D, "new_k", "head_dim", error) ||
      !qwen3_check_dim(new_v, 3, D, "new_v", "head_dim", error)) {
    return false;
  }

  return qwen3_validate_kv_cache_bn(k_cache, v_cache, B, N_kv, offset, T_new, D, error);
}

// qwen3_kv_cache_attention — Qwen3 fused RoPE + KV update + SDPA.
//
// Inputs:
//   q        — {B, T_new, N_q,  D}  Q projection after Q norm
//   new_k    — {B, T_new, N_kv, D}  K projection after K norm
//   new_v    — {B, T_new, N_kv, D}  V projection
//   k_cache  — {B, N_kv, T_max, D}  preallocated key buffer
//   v_cache  — {B, N_kv, T_max, D}  preallocated value buffer
//   offset   — int                  tokens already in cache
//   scale    — float                1/sqrt(head_dim)
//   head_dim — int                  RoPE dimensions
//   theta    — float                RoPE base
//   device   — atom
//
// Returns {attn_out, k_upd, v_upd}:
//   attn_out — {B, T_new, N_q * D}  ready for projection BTH layout
//   k_upd    — {B, N_kv, T_max, D}
//   v_upd    — {B, N_kv, T_max, D}
NIF(qwen3_kv_cache_attention) {
  TENSOR_PARAM(0, q);
  TENSOR_PARAM(1, new_k);
  TENSOR_PARAM(2, new_v);
  TENSOR_PARAM(3, k_cache);
  TENSOR_PARAM(4, v_cache);
  PARAM(5, int, offset);
  PARAM(6, double, scale);
  PARAM(7, int, head_dim);
  PARAM(8, double, theta);
  DEVICE_PARAM(9, device);

  try {
    std::string error;
    if (!qwen3_validate_qkv_cache_attention(
            *q, *new_k, *new_v, *k_cache, *v_cache, offset, head_dim, error)) {
      return qwen3_error(env, error);
    }

    int B       = q->shape(0);
    int T_new   = q->shape(1);
    int N_q     = q->shape(2);
    int D       = q->shape(3);
    int N_kv    = new_k->shape(2);
    int valid_len = offset + T_new;

    auto q_bn = mlx::core::transpose(*q, {0, 2, 1, 3}, device);
    auto k_bn = mlx::core::transpose(*new_k, {0, 2, 1, 3}, device);
    auto v_bn = mlx::core::transpose(*new_v, {0, 2, 1, 3}, device);

    auto q_rope = mlx::core::fast::rope(
        q_bn, head_dim, false, (float)theta, 1.0f, offset, std::nullopt, device);
    auto k_rope = mlx::core::fast::rope(
        k_bn, head_dim, false, (float)theta, 1.0f, offset, std::nullopt, device);

    auto k_cache_owned = std::move(*k_cache);
    auto v_cache_owned = std::move(*v_cache);

    auto k_upd = mlx::core::slice_update(
        k_cache_owned, k_rope,
        to_shape({0, 0, offset, 0}),
        to_shape({B, N_kv, valid_len, D}),
        device);
    auto v_upd = mlx::core::slice_update(
        v_cache_owned, v_bn,
        to_shape({0, 0, offset, 0}),
        to_shape({B, N_kv, valid_len, D}),
        device);

    auto k_valid = mlx::core::slice(
        k_upd, to_shape({0, 0, 0, 0}), to_shape({B, N_kv, valid_len, D}), device);
    auto v_valid = mlx::core::slice(
        v_upd, to_shape({0, 0, 0, 0}), to_shape({B, N_kv, valid_len, D}), device);

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

    auto attn_out_bn = (T_new == 1)
      ? mlx::core::fast::scaled_dot_product_attention(
            q_rope, k_valid, v_valid, (float)scale, "", std::nullopt, std::nullopt, device)
      : mlx::core::fast::scaled_dot_product_attention(
            q_rope, k_valid, v_valid, (float)scale, "array", build_prefill_mask(), std::nullopt, device);
    auto attn_out_bthd = mlx::core::transpose(attn_out_bn, {0, 2, 1, 3}, device);
    auto attn_out = mlx::core::reshape(attn_out_bthd, {B, T_new, N_q * D}, device);

    ERL_NIF_TERM result_tuple[3];
    result_tuple[0] = create_tensor_resource(env, attn_out);
    result_tuple[1] = create_tensor_resource(env, k_upd);
    result_tuple[2] = create_tensor_resource(env, v_upd);

    return nx::nif::ok(env, enif_make_tuple3(env, result_tuple[0], result_tuple[1], result_tuple[2]));
  }
  CATCH()
}
ASYNC_NIF(qwen3_kv_cache_attention)

static mlx::core::array qwen3_linear_in_out(
    const mlx::core::array &x,
    const mlx::core::array &weight,
    const mlx::core::Device &device) {
  if (x.ndim() == 3 && x.shape(1) == 1) {
    auto x_2d = mlx::core::reshape(x, {x.shape(0), x.shape(2)}, device);
    auto out = mlx::core::matmul(x_2d, weight, device);
    return mlx::core::reshape(out, {x.shape(0), 1, weight.shape(1)}, device);
  }

  return mlx::core::matmul(x, weight, device);
}

static mlx::core::array qwen3_linear_out_in(
    const mlx::core::array &x,
    const mlx::core::array &weight,
    const mlx::core::Device &device) {
  return mlx::core::tensordot(
      x, weight, std::vector<int>{static_cast<int>(x.ndim()) - 1}, std::vector<int>{1}, device);
}

static int64_t qwen3_token_to_int64(mlx::core::array &token) {
  mlx::core::eval(token);
  auto dtype = token.dtype();

  if (dtype == mlx::core::uint8) {
    return static_cast<int64_t>(token.item<uint8_t>());
  } else if (dtype == mlx::core::uint16) {
    return static_cast<int64_t>(token.item<uint16_t>());
  } else if (dtype == mlx::core::uint32) {
    return static_cast<int64_t>(token.item<uint32_t>());
  } else if (dtype == mlx::core::uint64) {
    return static_cast<int64_t>(token.item<uint64_t>());
  } else if (dtype == mlx::core::int8) {
    return static_cast<int64_t>(token.item<int8_t>());
  } else if (dtype == mlx::core::int16) {
    return static_cast<int64_t>(token.item<int16_t>());
  } else if (dtype == mlx::core::int32) {
    return static_cast<int64_t>(token.item<int32_t>());
  } else {
    return token.item<int64_t>();
  }
}

// qwen3_mlp — dense Qwen3 MLP block: RMSNorm + gate/up + SwiGLU + down + residual.
//
// Inputs:
//   hidden    — {B, T, H}
//   norm      — {H}
//   gate_proj — {H, I}
//   up_proj   — {H, I}
//   down_proj — {I, H}
//   eps       — RMSNorm epsilon
//
// Returns hidden + out {B, T, H}.
NIF(qwen3_mlp) {
  TENSOR_PARAM(0, hidden);
  TENSOR_PARAM(1, norm);
  TENSOR_PARAM(2, gate_proj);
  TENSOR_PARAM(3, up_proj);
  TENSOR_PARAM(4, down_proj);
  PARAM(5, double, eps);
  DEVICE_PARAM(6, device);

  try {
    std::string error;
    if (!qwen3_check_rank3_positive(*hidden, "hidden", error)) {
      return qwen3_error(env, error);
    }

    int H = hidden->shape(2);
    if (!qwen3_check_rank1_dim(*norm, H, "norm", error) ||
        !qwen3_check_rank2_positive(*gate_proj, "gate_proj", error) ||
        !qwen3_check_rank2_positive(*up_proj, "up_proj", error) ||
        !qwen3_check_rank2_positive(*down_proj, "down_proj", error) ||
        !qwen3_check_dim(*gate_proj, 0, H, "gate_proj", "input width", error) ||
        !qwen3_check_dim(*up_proj, 0, H, "up_proj", "input width", error) ||
        !qwen3_check_dim(*up_proj, 1, gate_proj->shape(1), "up_proj", "output width", error) ||
        !qwen3_check_dim(*down_proj, 0, gate_proj->shape(1), "down_proj", "input width", error) ||
        !qwen3_check_dim(*down_proj, 1, H, "down_proj", "output width", error)) {
      return qwen3_error(env, error);
    }

    auto xn = mlx::core::fast::rms_norm(*hidden, *norm, (float)eps, device);
    auto gate = qwen3_linear_in_out(xn, *gate_proj, device);
    auto up = qwen3_linear_in_out(xn, *up_proj, device);
    auto mlp = mlx::core::multiply(
        mlx::core::multiply(gate, mlx::core::sigmoid(gate, device), device),
        up,
        device);
    auto out = qwen3_linear_in_out(mlp, *down_proj, device);
    auto residual = mlx::core::add(*hidden, out, device);

    TENSOR(residual);
  }
  CATCH()
}
ASYNC_NIF(qwen3_mlp)

// qwen3_layer — dense Qwen3 transformer layer:
// attention input RMSNorm + dense attention block + RMSNorm after attention
// + dense MLP + residual add.
//
// Inputs:
//   hidden    — {B, T_new, H}
//   norm1     — {H}
//   q_proj    — {H, N_q * D}
//   k_proj    — {H, N_kv * D}
//   v_proj    — {H, N_kv * D}
//   o_proj    — {N_q * D, H}
//   q_norm    — {D}
//   k_norm    — {D}
//   k_cache   — {B, N_kv, T_max, D}
//   v_cache   — {B, N_kv, T_max, D}
//   norm2     — {H}
//   gate_proj — {H, I}
//   up_proj   — {H, I}
//   down_proj — {I, H}
//   offset    — int
//   scale     — float
//   head_dim  — int
//   theta     — float
//   eps       — RMSNorm epsilon
//
// Returns {hidden_out, k_upd, v_upd}.
NIF(qwen3_layer) {
  TENSOR_PARAM(0, hidden);
  TENSOR_PARAM(1, norm1);
  TENSOR_PARAM(2, q_proj);
  TENSOR_PARAM(3, k_proj);
  TENSOR_PARAM(4, v_proj);
  TENSOR_PARAM(5, o_proj);
  TENSOR_PARAM(6, q_norm);
  TENSOR_PARAM(7, k_norm);
  TENSOR_PARAM(8, k_cache);
  TENSOR_PARAM(9, v_cache);
  TENSOR_PARAM(10, norm2);
  TENSOR_PARAM(11, gate_proj);
  TENSOR_PARAM(12, up_proj);
  TENSOR_PARAM(13, down_proj);
  PARAM(14, int, offset);
  PARAM(15, double, scale);
  PARAM(16, int, head_dim);
  PARAM(17, double, theta);
  PARAM(18, double, eps);
  DEVICE_PARAM(19, device);

  try {
    std::string error;
    if (!qwen3_check_rank3_positive(*hidden, "hidden", error) ||
        !qwen3_check_non_negative(offset, "offset", error) ||
        !qwen3_check_positive(head_dim, "head_dim", error)) {
      return qwen3_error(env, error);
    }

    int B       = hidden->shape(0);
    int T_new   = hidden->shape(1);
    int H       = hidden->shape(2);
    int D       = head_dim;
    if (!qwen3_check_rank1_dim(*norm1, H, "norm1", error) ||
        !qwen3_check_rank1_dim(*norm2, H, "norm2", error) ||
        !qwen3_validate_projection_width(*q_proj, H, D, "q_proj", error) ||
        !qwen3_validate_projection_width(*k_proj, H, D, "k_proj", error) ||
        !qwen3_validate_projection_width(*v_proj, H, D, "v_proj", error) ||
        !qwen3_check_dim(*v_proj, 1, k_proj->shape(1), "v_proj", "output width", error) ||
        !qwen3_check_rank1_dim(*q_norm, D, "q_norm", error) ||
        !qwen3_check_rank1_dim(*k_norm, D, "k_norm", error)) {
      return qwen3_error(env, error);
    }

    int N_q     = q_proj->shape(1) / D;
    int N_kv    = k_proj->shape(1) / D;
    int attn_width = N_q * D;

    if ((N_q % N_kv) != 0) {
      return qwen3_error(env, "query heads must be divisible by key/value heads");
    }
    if (!qwen3_check_rank2_positive(*o_proj, "o_proj", error) ||
        !qwen3_check_dim(*o_proj, 0, attn_width, "o_proj", "input width", error) ||
        !qwen3_check_dim(*o_proj, 1, H, "o_proj", "output width", error) ||
        !qwen3_validate_kv_cache_bn(*k_cache, *v_cache, B, N_kv, offset, T_new, D, error) ||
        !qwen3_check_rank2_positive(*gate_proj, "gate_proj", error) ||
        !qwen3_check_rank2_positive(*up_proj, "up_proj", error) ||
        !qwen3_check_rank2_positive(*down_proj, "down_proj", error) ||
        !qwen3_check_dim(*gate_proj, 0, H, "gate_proj", "input width", error) ||
        !qwen3_check_dim(*up_proj, 0, H, "up_proj", "input width", error) ||
        !qwen3_check_dim(*up_proj, 1, gate_proj->shape(1), "up_proj", "output width", error) ||
        !qwen3_check_dim(*down_proj, 0, gate_proj->shape(1), "down_proj", "input width", error) ||
        !qwen3_check_dim(*down_proj, 1, H, "down_proj", "output width", error)) {
      return qwen3_error(env, error);
    }
    int valid_len = offset + T_new;

    auto xn = mlx::core::fast::rms_norm(*hidden, *norm1, (float)eps, device);
    auto q_flat = qwen3_linear_in_out(xn, *q_proj, device);
    auto k_flat = qwen3_linear_in_out(xn, *k_proj, device);
    auto v_flat = qwen3_linear_in_out(xn, *v_proj, device);

    auto q = mlx::core::reshape(q_flat, {B, T_new, N_q, D}, device);
    auto k = mlx::core::reshape(k_flat, {B, T_new, N_kv, D}, device);
    auto v = mlx::core::reshape(v_flat, {B, T_new, N_kv, D}, device);

    q = mlx::core::fast::rms_norm(q, *q_norm, (float)eps, device);
    k = mlx::core::fast::rms_norm(k, *k_norm, (float)eps, device);

    auto q_bn = mlx::core::transpose(q, {0, 2, 1, 3}, device);
    auto k_bn = mlx::core::transpose(k, {0, 2, 1, 3}, device);
    auto v_bn = mlx::core::transpose(v, {0, 2, 1, 3}, device);

    auto q_rope = mlx::core::fast::rope(
        q_bn, D, false, (float)theta, 1.0f, offset, std::nullopt, device);
    auto k_rope = mlx::core::fast::rope(
        k_bn, D, false, (float)theta, 1.0f, offset, std::nullopt, device);

    auto k_cache_owned = std::move(*k_cache);
    auto v_cache_owned = std::move(*v_cache);

    auto k_upd = mlx::core::slice_update(
        k_cache_owned, k_rope,
        to_shape({0, 0, offset, 0}),
        to_shape({B, N_kv, valid_len, D}),
        device);
    auto v_upd = mlx::core::slice_update(
        v_cache_owned, v_bn,
        to_shape({0, 0, offset, 0}),
        to_shape({B, N_kv, valid_len, D}),
        device);

    auto k_valid = mlx::core::slice(
        k_upd, to_shape({0, 0, 0, 0}), to_shape({B, N_kv, valid_len, D}), device);
    auto v_valid = mlx::core::slice(
        v_upd, to_shape({0, 0, 0, 0}), to_shape({B, N_kv, valid_len, D}), device);

    auto build_prefill_mask = [&]() -> mlx::core::array {
      auto mask_dtype = q.dtype();
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

    auto attn_out_bn = (T_new == 1)
      ? mlx::core::fast::scaled_dot_product_attention(
            q_rope, k_valid, v_valid, (float)scale, "", std::nullopt, std::nullopt, device)
      : mlx::core::fast::scaled_dot_product_attention(
            q_rope, k_valid, v_valid, (float)scale, "array", build_prefill_mask(), std::nullopt, device);
    auto attn_out_bthd = mlx::core::transpose(attn_out_bn, {0, 2, 1, 3}, device);
    auto attn_out = mlx::core::reshape(attn_out_bthd, {B, T_new, attn_width}, device);
    auto attn_projected = qwen3_linear_in_out(attn_out, *o_proj, device);
    auto attn_hidden = mlx::core::add(*hidden, attn_projected, device);

    auto xn2 = mlx::core::fast::rms_norm(attn_hidden, *norm2, (float)eps, device);
    auto gate = qwen3_linear_in_out(xn2, *gate_proj, device);
    auto up = qwen3_linear_in_out(xn2, *up_proj, device);
    auto mlp = mlx::core::multiply(
        mlx::core::multiply(gate, mlx::core::sigmoid(gate, device), device),
        up,
        device);
    auto mlp_out = qwen3_linear_in_out(mlp, *down_proj, device);
    auto out = mlx::core::add(attn_hidden, mlp_out, device);

    ERL_NIF_TERM result_tuple[3];
    result_tuple[0] = create_tensor_resource(env, out);
    result_tuple[1] = create_tensor_resource(env, k_upd);
    result_tuple[2] = create_tensor_resource(env, v_upd);

    return nx::nif::ok(env, enif_make_tuple3(env, result_tuple[0], result_tuple[1], result_tuple[2]));
  }
  CATCH()
}
ASYNC_NIF(qwen3_layer)

struct Qwen3LayerParams {
  mlx::core::array *norm1;
  mlx::core::array *norm2;
  mlx::core::array *q_norm;
  mlx::core::array *k_norm;
  mlx::core::array *q_proj;
  mlx::core::array *k_proj;
  mlx::core::array *v_proj;
  mlx::core::array *o_proj;
  mlx::core::array *gate_proj;
  mlx::core::array *up_proj;
  mlx::core::array *down_proj;
};

struct Qwen3KVCache {
  mlx::core::array *k;
  mlx::core::array *v;
};

class Qwen3TensorHandle {
public:
  explicit Qwen3TensorHandle(mlx::core::array *ptr) : ptr_(ptr) {
    refcount_ = reinterpret_cast<std::atomic<int> *>(ptr_ + 1);

    if (refcount_->load() == 0) {
      ptr_ = nullptr;
      return;
    }

    ++(*refcount_);
  }

  ~Qwen3TensorHandle() {
    if (is_valid()) {
      if (refcount_->fetch_sub(1) == 0) {
        ptr_->~array();
      }
    }
  }

  bool is_valid() const { return ptr_ != nullptr; }
  mlx::core::array *data() const { return ptr_; }

private:
  mlx::core::array *ptr_;
  std::atomic<int> *refcount_;
};

using Qwen3TensorHandles = std::vector<std::unique_ptr<Qwen3TensorHandle>>;

static bool qwen3_get_tensor(
    ErlNifEnv *env,
    ERL_NIF_TERM term,
    mlx::core::array **out,
    Qwen3TensorHandles &handles,
    ERL_NIF_TERM *error) {
  mlx::core::array *raw = nullptr;

  if (!enif_get_resource(
          env, term, resource_object<mlx::core::array>::type,
          reinterpret_cast<void **>(&raw))) {
    return false;
  }

  auto handle = std::make_unique<Qwen3TensorHandle>(raw);
  if (!handle->is_valid()) {
    *error = nx::nif::error(env, "Tensor has been deallocated");
    return false;
  }

  *out = handle->data();
  handles.push_back(std::move(handle));
  return true;
}

static bool qwen3_get_tensor_or_device_ref(
    ErlNifEnv *env,
    ERL_NIF_TERM term,
    mlx::core::array **out,
    Qwen3TensorHandles &handles,
    ERL_NIF_TERM *error) {
  if (qwen3_get_tensor(env, term, out, handles, error)) {
    return true;
  }
  if (*error != 0) {
    return false;
  }

  int arity = 0;
  const ERL_NIF_TERM *items = nullptr;
  if (!enif_get_tuple(env, term, &arity, &items) || arity != 2) {
    return false;
  }

  return qwen3_get_tensor(env, items[1], out, handles, error);
}

static bool qwen3_get_layer(
    ErlNifEnv *env,
    ERL_NIF_TERM term,
    Qwen3LayerParams &layer,
    Qwen3TensorHandles &handles,
    ERL_NIF_TERM *error) {
  int arity = 0;
  const ERL_NIF_TERM *items = nullptr;
  if (!enif_get_tuple(env, term, &arity, &items) || arity != 11) {
    return false;
  }

  return qwen3_get_tensor(env, items[0], &layer.norm1, handles, error) &&
         qwen3_get_tensor(env, items[1], &layer.norm2, handles, error) &&
         qwen3_get_tensor(env, items[2], &layer.q_norm, handles, error) &&
         qwen3_get_tensor(env, items[3], &layer.k_norm, handles, error) &&
         qwen3_get_tensor(env, items[4], &layer.q_proj, handles, error) &&
         qwen3_get_tensor(env, items[5], &layer.k_proj, handles, error) &&
         qwen3_get_tensor(env, items[6], &layer.v_proj, handles, error) &&
         qwen3_get_tensor(env, items[7], &layer.o_proj, handles, error) &&
         qwen3_get_tensor(env, items[8], &layer.gate_proj, handles, error) &&
         qwen3_get_tensor(env, items[9], &layer.up_proj, handles, error) &&
         qwen3_get_tensor(env, items[10], &layer.down_proj, handles, error);
}

static bool qwen3_get_kv(
    ErlNifEnv *env,
    ERL_NIF_TERM term,
    Qwen3KVCache &kv,
    Qwen3TensorHandles &handles,
    ERL_NIF_TERM *error) {
  int arity = 0;
  const ERL_NIF_TERM *items = nullptr;
  if (!enif_get_tuple(env, term, &arity, &items) || arity != 2) {
    return false;
  }

  return qwen3_get_tensor_or_device_ref(env, items[0], &kv.k, handles, error) &&
         qwen3_get_tensor_or_device_ref(env, items[1], &kv.v, handles, error);
}

static ERL_NIF_TERM qwen3_ref_error_or(
    ErlNifEnv *env, ERL_NIF_TERM error, const char *fallback) {
  if (error != 0) {
    return error;
  }

  return nx::nif::error(env, fallback);
}

static bool qwen3_validate_dense_layer(
    const mlx::core::array &hidden,
    const Qwen3LayerParams &layer,
    const Qwen3KVCache &kv,
    int offset,
    int head_dim,
    std::string &error) {
  if (!qwen3_check_rank3_positive(hidden, "hidden", error) ||
      !qwen3_check_non_negative(offset, "offset", error) ||
      !qwen3_check_positive(head_dim, "head_dim", error)) {
    return false;
  }

  int B = hidden.shape(0);
  int T_new = hidden.shape(1);
  int H = hidden.shape(2);
  int D = head_dim;

  if (!qwen3_check_rank1_dim(*layer.norm1, H, "norm1", error) ||
      !qwen3_check_rank1_dim(*layer.norm2, H, "norm2", error) ||
      !qwen3_validate_projection_width(*layer.q_proj, H, D, "q_proj", error) ||
      !qwen3_validate_projection_width(*layer.k_proj, H, D, "k_proj", error) ||
      !qwen3_validate_projection_width(*layer.v_proj, H, D, "v_proj", error) ||
      !qwen3_check_dim(*layer.v_proj, 1, layer.k_proj->shape(1), "v_proj", "output width", error) ||
      !qwen3_check_rank1_dim(*layer.q_norm, D, "q_norm", error) ||
      !qwen3_check_rank1_dim(*layer.k_norm, D, "k_norm", error)) {
    return false;
  }

  int N_q = layer.q_proj->shape(1) / D;
  int N_kv = layer.k_proj->shape(1) / D;
  int attn_width = N_q * D;

  if ((N_q % N_kv) != 0) {
    error = "query heads must be divisible by key/value heads";
    return false;
  }
  if (!qwen3_check_rank2_positive(*layer.o_proj, "o_proj", error) ||
      !qwen3_check_dim(*layer.o_proj, 0, attn_width, "o_proj", "input width", error) ||
      !qwen3_check_dim(*layer.o_proj, 1, H, "o_proj", "output width", error) ||
      !qwen3_validate_kv_cache_bn(*kv.k, *kv.v, B, N_kv, offset, T_new, D, error) ||
      !qwen3_check_rank2_positive(*layer.gate_proj, "gate_proj", error) ||
      !qwen3_check_rank2_positive(*layer.up_proj, "up_proj", error) ||
      !qwen3_check_rank2_positive(*layer.down_proj, "down_proj", error) ||
      !qwen3_check_dim(*layer.gate_proj, 0, H, "gate_proj", "input width", error) ||
      !qwen3_check_dim(*layer.up_proj, 0, H, "up_proj", "input width", error) ||
      !qwen3_check_dim(*layer.up_proj, 1, layer.gate_proj->shape(1), "up_proj", "output width", error) ||
      !qwen3_check_dim(*layer.down_proj, 0, layer.gate_proj->shape(1), "down_proj", "input width", error) ||
      !qwen3_check_dim(*layer.down_proj, 1, H, "down_proj", "output width", error)) {
    return false;
  }

  return true;
}

static mlx::core::array qwen3_dense_layer_impl(
    const mlx::core::array &hidden,
    const Qwen3LayerParams &layer,
    Qwen3KVCache &kv,
    int offset,
    float scale,
    int head_dim,
    float theta,
    float eps,
    const mlx::core::Device &device,
    ERL_NIF_TERM &k_term,
    ERL_NIF_TERM &v_term,
    ErlNifEnv *env,
    mlx::core::array *k_out = nullptr,
    mlx::core::array *v_out = nullptr) {
  int B       = hidden.shape(0);
  int T_new   = hidden.shape(1);
  int D       = head_dim;
  int N_q     = layer.q_proj->shape(1) / D;
  int N_kv    = layer.k_proj->shape(1) / D;
  int attn_width = N_q * D;
  int valid_len = offset + T_new;

  auto xn = mlx::core::fast::rms_norm(hidden, *layer.norm1, eps, device);
  auto q_flat = qwen3_linear_in_out(xn, *layer.q_proj, device);
  auto k_flat = qwen3_linear_in_out(xn, *layer.k_proj, device);
  auto v_flat = qwen3_linear_in_out(xn, *layer.v_proj, device);

  auto q = mlx::core::reshape(q_flat, {B, T_new, N_q, D}, device);
  auto k = mlx::core::reshape(k_flat, {B, T_new, N_kv, D}, device);
  auto v = mlx::core::reshape(v_flat, {B, T_new, N_kv, D}, device);

  q = mlx::core::fast::rms_norm(q, *layer.q_norm, eps, device);
  k = mlx::core::fast::rms_norm(k, *layer.k_norm, eps, device);

  auto q_bn = mlx::core::transpose(q, {0, 2, 1, 3}, device);
  auto k_bn = mlx::core::transpose(k, {0, 2, 1, 3}, device);
  auto v_bn = mlx::core::transpose(v, {0, 2, 1, 3}, device);

  auto q_rope = mlx::core::fast::rope(
      q_bn, D, false, theta, 1.0f, offset, std::nullopt, device);
  auto k_rope = mlx::core::fast::rope(
      k_bn, D, false, theta, 1.0f, offset, std::nullopt, device);

  auto k_cache_owned = std::move(*kv.k);
  auto v_cache_owned = std::move(*kv.v);

  auto k_upd = mlx::core::slice_update(
      k_cache_owned, k_rope,
      to_shape({0, 0, offset, 0}),
      to_shape({B, N_kv, valid_len, D}),
      device);
  auto v_upd = mlx::core::slice_update(
      v_cache_owned, v_bn,
      to_shape({0, 0, offset, 0}),
      to_shape({B, N_kv, valid_len, D}),
      device);

  auto k_valid = mlx::core::slice(
      k_upd, to_shape({0, 0, 0, 0}), to_shape({B, N_kv, valid_len, D}), device);
  auto v_valid = mlx::core::slice(
      v_upd, to_shape({0, 0, 0, 0}), to_shape({B, N_kv, valid_len, D}), device);

  auto build_prefill_mask = [&]() -> mlx::core::array {
    auto mask_dtype = q.dtype();
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

  auto attn_out_bn = (T_new == 1)
      ? mlx::core::fast::scaled_dot_product_attention(
            q_rope, k_valid, v_valid, scale, "", std::nullopt, std::nullopt, device)
      : mlx::core::fast::scaled_dot_product_attention(
            q_rope, k_valid, v_valid, scale, "array", build_prefill_mask(), std::nullopt, device);
  auto attn_out_bthd = mlx::core::transpose(attn_out_bn, {0, 2, 1, 3}, device);
  auto attn_out = mlx::core::reshape(attn_out_bthd, {B, T_new, attn_width}, device);
  auto attn_projected = qwen3_linear_in_out(attn_out, *layer.o_proj, device);
  auto attn_hidden = mlx::core::add(hidden, attn_projected, device);

  auto xn2 = mlx::core::fast::rms_norm(attn_hidden, *layer.norm2, eps, device);
  auto gate = qwen3_linear_in_out(xn2, *layer.gate_proj, device);
  auto up = qwen3_linear_in_out(xn2, *layer.up_proj, device);
  auto mlp = mlx::core::multiply(
      mlx::core::multiply(gate, mlx::core::sigmoid(gate, device), device),
      up,
      device);
  auto mlp_out = qwen3_linear_in_out(mlp, *layer.down_proj, device);

  if (k_out != nullptr) {
    *k_out = k_upd;
  }
  if (v_out != nullptr) {
    *v_out = v_upd;
  }

  if (env != nullptr) {
    k_term = create_tensor_resource(env, k_upd);
    v_term = create_tensor_resource(env, v_upd);
  }

  return mlx::core::add(attn_hidden, mlp_out, device);
}

static ERL_NIF_TERM qwen3_forward_greedy_from_hidden(
    ErlNifEnv *env,
    const mlx::core::array &hidden,
    ERL_NIF_TERM layers_arg,
    ERL_NIF_TERM kv_cache_arg,
    mlx::core::array *norm,
    mlx::core::array *lm_head,
    int offset,
    double scale,
    int head_dim,
    double theta,
    double eps,
    bool return_token_id,
    const mlx::core::Device &device) {
  unsigned int layer_count = 0;
  unsigned int kv_count = 0;

  if (!enif_get_list_length(env, layers_arg, &layer_count)) {
    return nx::nif::error(env, "Qwen3 greedy forward expects layers to be a list");
  }

  if (!enif_get_list_length(env, kv_cache_arg, &kv_count)) {
    return nx::nif::error(env, "Qwen3 greedy forward expects kv_cache to be a list");
  }

  if (layer_count != kv_count) {
    return nx::nif::error(env, "Qwen3 greedy forward layers and kv_cache length mismatch");
  }
  std::string input_error;
  if (!qwen3_check_rank3_positive(hidden, "hidden", input_error) ||
      !qwen3_check_non_negative(offset, "offset", input_error) ||
      !qwen3_check_positive(head_dim, "head_dim", input_error)) {
    return qwen3_error(env, input_error);
  }

  auto current = hidden;
  std::vector<ERL_NIF_TERM> kv_terms;
  std::vector<mlx::core::array> eval_arrays;
  kv_terms.reserve(layer_count);
  eval_arrays.reserve((layer_count * 2) + 1);

  ERL_NIF_TERM layer_head, layer_tail = layers_arg;
  ERL_NIF_TERM kv_head, kv_tail = kv_cache_arg;
  Qwen3TensorHandles handles;
  ERL_NIF_TERM ref_error = 0;

  while (enif_get_list_cell(env, layer_tail, &layer_head, &layer_tail) &&
         enif_get_list_cell(env, kv_tail, &kv_head, &kv_tail)) {
    Qwen3LayerParams layer;
    Qwen3KVCache kv;

    if (!qwen3_get_layer(env, layer_head, layer, handles, &ref_error)) {
      return qwen3_ref_error_or(
          env, ref_error, "Qwen3 greedy forward got invalid layer tuple");
    }
    if (!qwen3_get_kv(env, kv_head, kv, handles, &ref_error)) {
      return qwen3_ref_error_or(
          env, ref_error, "Qwen3 greedy forward got invalid kv_cache tuple");
    }
    std::string error;
    if (!qwen3_validate_dense_layer(current, layer, kv, offset, head_dim, error)) {
      return qwen3_error(env, error);
    }

    auto k_new = *kv.k;
    auto v_new = *kv.v;
    ERL_NIF_TERM unused_k_term;
    ERL_NIF_TERM unused_v_term;

    current = qwen3_dense_layer_impl(
        current, layer, kv, offset, static_cast<float>(scale), head_dim,
        static_cast<float>(theta), static_cast<float>(eps), device,
        unused_k_term, unused_v_term, nullptr, &k_new, &v_new);

    eval_arrays.push_back(k_new);
    eval_arrays.push_back(v_new);
    kv_terms.push_back(
        enif_make_tuple2(
            env,
            create_tensor_resource(env, k_new),
            create_tensor_resource(env, v_new)));
  }

  int B = current.shape(0);
  int T = current.shape(1);
  int H = current.shape(2);
  if (return_token_id && B != 1) {
    return qwen3_error(env, "token_id return paths require batch size 1");
  }
  std::string final_error;
  if (!qwen3_check_rank1_dim(*norm, H, "norm", final_error) ||
      !qwen3_check_rank2_positive(*lm_head, "lm_head", final_error) ||
      !qwen3_check_dim(*lm_head, 1, H, "lm_head", "hidden width", final_error)) {
    return qwen3_error(env, final_error);
  }

  auto last = (T == 1)
      ? mlx::core::reshape(current, {B, H}, device)
      : mlx::core::reshape(
          mlx::core::slice(current, to_shape({0, T - 1, 0}), to_shape({B, T, H}), device),
          {B, H},
          device);

  auto normed = mlx::core::fast::rms_norm(last, *norm, static_cast<float>(eps), device);
  auto logits = qwen3_linear_out_in(normed, *lm_head, device);
  auto token = mlx::core::argmax(logits, 1, false, device);

  eval_arrays.push_back(token);
  mlx::core::async_eval(eval_arrays);

  ERL_NIF_TERM token_term =
      return_token_id
          ? nx::nif::make(env, qwen3_token_to_int64(token))
          : create_tensor_resource(env, token);
  ERL_NIF_TERM kv_list = enif_make_list_from_array(env, kv_terms.data(), kv_terms.size());

  return nx::nif::ok(env, enif_make_tuple2(env, token_term, kv_list));
}

// qwen3_forward_greedy_ids — embedding lookup + dense forward through all layers +
// final greedy token. Returns {token_ids, kv_cache} where token_ids has
// shape {B}.
NIF(qwen3_forward_greedy_ids) {
  TENSOR_PARAM(0, input_ids);
  TENSOR_PARAM(1, embed_tokens);
  PARAM(6, int, offset);
  PARAM(7, double, scale);
  PARAM(8, int, head_dim);
  PARAM(9, double, theta);
  PARAM(10, double, eps);
  DEVICE_PARAM(11, device);

  try {
    Qwen3TensorHandles handles;
    ERL_NIF_TERM ref_error = 0;
    mlx::core::array *norm = nullptr;
    mlx::core::array *lm_head = nullptr;
    if (!qwen3_get_tensor(env, argv[4], &norm, handles, &ref_error)) {
      return qwen3_ref_error_or(
          env, ref_error, "qwen3_forward_greedy_ids expects norm tensor ref");
    }
    if (!qwen3_get_tensor(env, argv[5], &lm_head, handles, &ref_error)) {
      return qwen3_ref_error_or(
          env, ref_error, "qwen3_forward_greedy_ids expects lm_head tensor ref");
    }

    std::string error;
    if (!qwen3_check_rank2_positive(*input_ids, "input_ids", error) ||
        !qwen3_check_rank2_positive(*embed_tokens, "embed_tokens", error) ||
        !qwen3_check_non_negative(offset, "offset", error) ||
        !qwen3_check_positive(head_dim, "head_dim", error)) {
      return qwen3_error(env, error);
    }

    int B = input_ids->shape(0);
    int T = input_ids->shape(1);

    auto ids = mlx::core::reshape(*input_ids, {B * T}, device);

    auto embedded = mlx::core::reshape(
        mlx::core::take(*embed_tokens, ids, 0, device),
        {B, T, embed_tokens->shape(1)},
        device);

    return qwen3_forward_greedy_from_hidden(
        env, embedded, argv[2], argv[3], norm, lm_head, offset, scale, head_dim,
        theta, eps, false, device);
  }
  CATCH()
}
ASYNC_NIF(qwen3_forward_greedy_ids)

// qwen3_forward_greedy_ids_chunk — repeatedly decode greedy tokens from a
// single token id tensor without returning to Elixir between decode steps.
// Returns {token_id_refs, kv_cache}, where token_id_refs is a list of token
// tensors in generation order and kv_cache is the final updated raw cache.
NIF(qwen3_forward_greedy_ids_chunk) {
  TENSOR_PARAM(0, input_ids);
  TENSOR_PARAM(1, embed_tokens);
  PARAM(6, int, offset);
  PARAM(7, int, count);
  PARAM(8, double, scale);
  PARAM(9, int, head_dim);
  PARAM(10, double, theta);
  PARAM(11, double, eps);
  DEVICE_PARAM(12, device);

  try {
    if (count <= 0) {
      return nx::nif::error(env, "qwen3_forward_greedy_ids_chunk expects positive count");
    }

    Qwen3TensorHandles handles;
    ERL_NIF_TERM ref_error = 0;
    mlx::core::array *norm = nullptr;
    mlx::core::array *lm_head = nullptr;
    if (!qwen3_get_tensor(env, argv[4], &norm, handles, &ref_error)) {
      return qwen3_ref_error_or(
          env, ref_error, "qwen3_forward_greedy_ids_chunk expects norm tensor ref");
    }
    if (!qwen3_get_tensor(env, argv[5], &lm_head, handles, &ref_error)) {
      return qwen3_ref_error_or(
          env, ref_error, "qwen3_forward_greedy_ids_chunk expects lm_head tensor ref");
    }

    unsigned int layer_count = 0;
    unsigned int kv_count = 0;

    if (!enif_get_list_length(env, argv[2], &layer_count)) {
      return nx::nif::error(env, "qwen3_forward_greedy_ids_chunk expects layers to be a list");
    }
    if (!enif_get_list_length(env, argv[3], &kv_count)) {
      return nx::nif::error(env, "qwen3_forward_greedy_ids_chunk expects kv_cache to be a list");
    }
    if (layer_count != kv_count) {
      return nx::nif::error(env, "qwen3_forward_greedy_ids_chunk layers and kv_cache length mismatch");
    }

    std::vector<Qwen3LayerParams> layers;
    layers.reserve(layer_count);

    ERL_NIF_TERM layer_head;
    ERL_NIF_TERM layer_tail = argv[2];
    while (enif_get_list_cell(env, layer_tail, &layer_head, &layer_tail)) {
      Qwen3LayerParams layer;
      if (!qwen3_get_layer(env, layer_head, layer, handles, &ref_error)) {
        return qwen3_ref_error_or(
            env, ref_error, "qwen3_forward_greedy_ids_chunk got invalid layer tuple");
      }
      layers.push_back(layer);
    }

    std::vector<Qwen3KVCache> initial_kv;
    initial_kv.reserve(kv_count);

    ERL_NIF_TERM kv_head;
    ERL_NIF_TERM kv_tail = argv[3];
    while (enif_get_list_cell(env, kv_tail, &kv_head, &kv_tail)) {
      Qwen3KVCache kv;
      if (!qwen3_get_kv(env, kv_head, kv, handles, &ref_error)) {
        return qwen3_ref_error_or(
            env, ref_error, "qwen3_forward_greedy_ids_chunk got invalid kv_cache tuple");
      }
      initial_kv.push_back(kv);
    }

    std::string error;
    if (!qwen3_check_rank2_positive(*input_ids, "input_ids", error) ||
        !qwen3_check_rank2_positive(*embed_tokens, "embed_tokens", error) ||
        !qwen3_check_non_negative(offset, "offset", error) ||
        !qwen3_check_positive(head_dim, "head_dim", error) ||
        !qwen3_check_rank1_dim(*norm, embed_tokens->shape(1), "norm", error) ||
        !qwen3_check_rank2_positive(*lm_head, "lm_head", error) ||
        !qwen3_check_dim(*lm_head, 1, embed_tokens->shape(1), "lm_head", "hidden width", error)) {
      return qwen3_error(env, error);
    }
    if (input_ids->shape(0) != 1) {
      return qwen3_error(env, "qwen3_forward_greedy_ids_chunk requires batch size 1");
    }
    if (input_ids->shape(1) != 1) {
      return qwen3_error(env, "qwen3_forward_greedy_ids_chunk requires sequence length 1");
    }

    std::vector<mlx::core::array> k_cache;
    std::vector<mlx::core::array> v_cache;
    std::vector<mlx::core::array> next_k_cache;
    std::vector<mlx::core::array> next_v_cache;
    std::vector<ERL_NIF_TERM> token_terms;
    std::vector<mlx::core::array> token_arrays;
    k_cache.reserve(layer_count);
    v_cache.reserve(layer_count);
    next_k_cache.reserve(layer_count);
    next_v_cache.reserve(layer_count);
    token_terms.reserve(count);
    token_arrays.reserve(count);

    auto current_ids = *input_ids;
    int current_offset = offset;

    for (int step = 0; step < count; ++step) {
      int B = current_ids.shape(0);
      int T = current_ids.shape(1);

      auto ids = mlx::core::reshape(current_ids, {B * T}, device);
      auto current = mlx::core::reshape(
          mlx::core::take(*embed_tokens, ids, 0, device),
          {B, T, embed_tokens->shape(1)},
          device);

      next_k_cache.clear();
      next_v_cache.clear();

      for (unsigned int layer_idx = 0; layer_idx < layer_count; ++layer_idx) {
        Qwen3KVCache kv =
            (step == 0)
                ? initial_kv[layer_idx]
                : Qwen3KVCache{&k_cache[layer_idx], &v_cache[layer_idx]};

        std::string layer_error;
        if (!qwen3_validate_dense_layer(
                current, layers[layer_idx], kv, current_offset, head_dim, layer_error)) {
          return qwen3_error(env, layer_error);
        }

        ERL_NIF_TERM unused_k_term;
        ERL_NIF_TERM unused_v_term;
        auto k_new = *kv.k;
        auto v_new = *kv.v;

        current = qwen3_dense_layer_impl(
            current,
            layers[layer_idx],
            kv,
            current_offset,
            static_cast<float>(scale),
            head_dim,
            static_cast<float>(theta),
            static_cast<float>(eps),
            device,
            unused_k_term,
            unused_v_term,
            nullptr,
            &k_new,
            &v_new);

        next_k_cache.push_back(k_new);
        next_v_cache.push_back(v_new);
      }

      int B_out = current.shape(0);
      int T_out = current.shape(1);
      int H_out = current.shape(2);

      auto last = (T_out == 1)
          ? mlx::core::reshape(current, {B_out, H_out}, device)
          : mlx::core::reshape(
              mlx::core::slice(
                  current,
                  to_shape({0, T_out - 1, 0}),
                  to_shape({B_out, T_out, H_out}),
                  device),
              {B_out, H_out},
              device);

      auto normed = mlx::core::fast::rms_norm(last, *norm, static_cast<float>(eps), device);
      auto logits = qwen3_linear_out_in(normed, *lm_head, device);
      auto token = mlx::core::argmax(logits, 1, false, device);

      token_arrays.push_back(token);
      token_terms.push_back(create_tensor_resource(env, token));
      current_ids = mlx::core::reshape(token, {B_out, 1}, device);
      k_cache.swap(next_k_cache);
      v_cache.swap(next_v_cache);
      current_offset += 1;
    }

    // Queue token materialisation and final cache ownership before Elixir
    // starts copying/decoding tokens or returns the cache to the next chunk.
    std::vector<mlx::core::array> eval_arrays;
    eval_arrays.reserve(token_arrays.size() + (layer_count * 2));
    eval_arrays.insert(eval_arrays.end(), token_arrays.begin(), token_arrays.end());

    for (unsigned int layer_idx = 0; layer_idx < layer_count; ++layer_idx) {
      eval_arrays.push_back(k_cache[layer_idx]);
      eval_arrays.push_back(v_cache[layer_idx]);
    }

    mlx::core::async_eval(eval_arrays);

    std::vector<ERL_NIF_TERM> kv_terms;
    kv_terms.reserve(layer_count);

    for (unsigned int layer_idx = 0; layer_idx < layer_count; ++layer_idx) {
      kv_terms.push_back(
          enif_make_tuple2(
              env,
              create_tensor_resource(env, k_cache[layer_idx]),
              create_tensor_resource(env, v_cache[layer_idx])));
    }

    ERL_NIF_TERM token_list =
        enif_make_list_from_array(env, token_terms.data(), token_terms.size());
    ERL_NIF_TERM kv_list =
        enif_make_list_from_array(env, kv_terms.data(), kv_terms.size());

    return nx::nif::ok(env, enif_make_tuple2(env, token_list, kv_list));
  }
  CATCH()
}
ASYNC_NIF(qwen3_forward_greedy_ids_chunk)

// qwen3_forward_greedy_ids_token_id — same as qwen3_forward_greedy_ids, but
// returns the sampled token id as a BEAM integer. This is for decode paths that
// already need each token on the host for streaming/callbacks.
NIF(qwen3_forward_greedy_ids_token_id) {
  TENSOR_PARAM(0, input_ids);
  TENSOR_PARAM(1, embed_tokens);
  PARAM(6, int, offset);
  PARAM(7, double, scale);
  PARAM(8, int, head_dim);
  PARAM(9, double, theta);
  PARAM(10, double, eps);
  DEVICE_PARAM(11, device);

  try {
    Qwen3TensorHandles handles;
    ERL_NIF_TERM ref_error = 0;
    mlx::core::array *norm = nullptr;
    mlx::core::array *lm_head = nullptr;
    if (!qwen3_get_tensor(env, argv[4], &norm, handles, &ref_error)) {
      return qwen3_ref_error_or(
          env, ref_error, "qwen3_forward_greedy_ids_token_id expects norm tensor ref");
    }
    if (!qwen3_get_tensor(env, argv[5], &lm_head, handles, &ref_error)) {
      return qwen3_ref_error_or(
          env, ref_error, "qwen3_forward_greedy_ids_token_id expects lm_head tensor ref");
    }

    std::string error;
    if (!qwen3_check_rank2_positive(*input_ids, "input_ids", error) ||
        !qwen3_check_rank2_positive(*embed_tokens, "embed_tokens", error) ||
        !qwen3_check_non_negative(offset, "offset", error) ||
        !qwen3_check_positive(head_dim, "head_dim", error)) {
      return qwen3_error(env, error);
    }

    int B = input_ids->shape(0);
    int T = input_ids->shape(1);

    auto ids = mlx::core::reshape(*input_ids, {B * T}, device);

    auto embedded = mlx::core::reshape(
        mlx::core::take(*embed_tokens, ids, 0, device),
        {B, T, embed_tokens->shape(1)},
        device);

    return qwen3_forward_greedy_from_hidden(
        env, embedded, argv[2], argv[3], norm, lm_head, offset, scale, head_dim,
        theta, eps, true, device);
  }
  CATCH()
}
ASYNC_NIF(qwen3_forward_greedy_ids_token_id)

// qwen3_forward_greedy_token_id — decode variant that accepts the previous
// token as a BEAM integer, avoiding host Nx tensor construction and backend
// transfer for the single token greedy decode hot path.
NIF(qwen3_forward_greedy_token_id) {
  PARAM(0, int64_t, token_id);
  TENSOR_PARAM(1, embed_tokens);
  PARAM(6, int, offset);
  PARAM(7, double, scale);
  PARAM(8, int, head_dim);
  PARAM(9, double, theta);
  PARAM(10, double, eps);
  DEVICE_PARAM(11, device);

  try {
    Qwen3TensorHandles handles;
    ERL_NIF_TERM ref_error = 0;
    mlx::core::array *norm = nullptr;
    mlx::core::array *lm_head = nullptr;
    if (!qwen3_get_tensor(env, argv[4], &norm, handles, &ref_error)) {
      return qwen3_ref_error_or(
          env, ref_error, "qwen3_forward_greedy_token_id expects norm tensor ref");
    }
    if (!qwen3_get_tensor(env, argv[5], &lm_head, handles, &ref_error)) {
      return qwen3_ref_error_or(
          env, ref_error, "qwen3_forward_greedy_token_id expects lm_head tensor ref");
    }
    std::string error;
    if (!qwen3_check_rank2_positive(*embed_tokens, "embed_tokens", error) ||
        !qwen3_check_non_negative(offset, "offset", error) ||
        !qwen3_check_positive(head_dim, "head_dim", error)) {
      return qwen3_error(env, error);
    }
    if (token_id < 0 || token_id >= embed_tokens->shape(0)) {
      return qwen3_error(env, "token_id is outside the embedding vocabulary");
    }

    auto ids = mlx::core::array(token_id, mlx::core::int64);
    auto embedded = mlx::core::reshape(
        mlx::core::take(*embed_tokens, ids, 0, device),
        {1, 1, embed_tokens->shape(1)},
        device);

    return qwen3_forward_greedy_from_hidden(
        env, embedded, argv[2], argv[3], norm, lm_head, offset, scale, head_dim,
        theta, eps, true, device);
  }
  CATCH()
}
ASYNC_NIF(qwen3_forward_greedy_token_id)

// qwen3_final_greedy — final RMSNorm + dense lm_head + argmax for greedy decode.
//
// Inputs:
//   hidden  — {B, T, H}
//   norm    — {H}
//   lm_head — {V, H}
//   eps     — RMSNorm epsilon
//
// Returns token ids with shape {B}.
NIF(qwen3_final_greedy) {
  TENSOR_PARAM(0, hidden);
  TENSOR_PARAM(1, norm);
  TENSOR_PARAM(2, lm_head);
  PARAM(3, double, eps);
  DEVICE_PARAM(4, device);

  try {
    std::string error;
    if (!qwen3_check_rank3_positive(*hidden, "hidden", error)) {
      return qwen3_error(env, error);
    }

    int B = hidden->shape(0);
    int T = hidden->shape(1);
    int H = hidden->shape(2);
    if (!qwen3_check_rank1_dim(*norm, H, "norm", error) ||
        !qwen3_check_rank2_positive(*lm_head, "lm_head", error) ||
        !qwen3_check_dim(*lm_head, 1, H, "lm_head", "hidden width", error)) {
      return qwen3_error(env, error);
    }

    auto last = (T == 1)
      ? mlx::core::reshape(*hidden, {B, H}, device)
      : mlx::core::reshape(
          mlx::core::slice(*hidden, to_shape({0, T - 1, 0}), to_shape({B, T, H}), device),
          {B, H},
          device);

    auto normed = mlx::core::fast::rms_norm(last, *norm, (float)eps, device);
    auto logits = mlx::core::tensordot(normed, *lm_head, std::vector<int>{1}, std::vector<int>{1}, device);

    TENSOR(mlx::core::argmax(logits, 1, false, device));
  }
  CATCH()
}
ASYNC_NIF(qwen3_final_greedy)

// qwen3_attention_residual — dense attention output projection + residual add.
//
// Inputs:
//   hidden   — {B, T, H}
//   attn_out — {B, T, I}
//   o_proj   — {I, H}
//
// Returns hidden + attn_out @ o_proj as {B, T, H}.
NIF(qwen3_attention_residual) {
  TENSOR_PARAM(0, hidden);
  TENSOR_PARAM(1, attn_out);
  TENSOR_PARAM(2, o_proj);
  DEVICE_PARAM(3, device);

  try {
    std::string error;
    if (!qwen3_check_rank3_positive(*hidden, "hidden", error) ||
        !qwen3_check_rank3_positive(*attn_out, "attn_out", error)) {
      return qwen3_error(env, error);
    }

    int B = hidden->shape(0);
    int T = hidden->shape(1);
    int H = hidden->shape(2);
    if (!qwen3_check_dim(*attn_out, 0, B, "attn_out", "batch", error) ||
        !qwen3_check_dim(*attn_out, 1, T, "attn_out", "sequence length", error) ||
        !qwen3_check_rank2_positive(*o_proj, "o_proj", error) ||
        !qwen3_check_dim(*o_proj, 0, attn_out->shape(2), "o_proj", "input width", error) ||
        !qwen3_check_dim(*o_proj, 1, H, "o_proj", "output width", error)) {
      return qwen3_error(env, error);
    }

    auto projected = qwen3_linear_in_out(*attn_out, *o_proj, device);
    auto residual = mlx::core::add(*hidden, projected, device);

    TENSOR(residual);
  }
  CATCH()
}
ASYNC_NIF(qwen3_attention_residual)

// qwen3_attention_block — dense Qwen3 attention block:
// input RMSNorm + Q/K/V projections + Q/K RMSNorm + RoPE + KV update + SDPA
// + output projection + residual add.
//
// Inputs:
//   hidden   — {B, T_new, H}       residual hidden state before attention
//   norm     — {H}                 input RMSNorm weight
//   q_proj   — {H, N_q * D}
//   k_proj   — {H, N_kv * D}
//   v_proj   — {H, N_kv * D}
//   o_proj   — {N_q * D, H}
//   q_norm   — {D}
//   k_norm   — {D}
//   k_cache  — {B, N_kv, T_max, D}
//   v_cache  — {B, N_kv, T_max, D}
//   offset   — int
//   scale    — float
//   head_dim — int
//   theta    — float
//   eps      — RMSNorm epsilon
//
// Returns {hidden_out, k_upd, v_upd}.
NIF(qwen3_attention_block) {
  TENSOR_PARAM(0, hidden);
  TENSOR_PARAM(1, norm);
  TENSOR_PARAM(2, q_proj);
  TENSOR_PARAM(3, k_proj);
  TENSOR_PARAM(4, v_proj);
  TENSOR_PARAM(5, o_proj);
  TENSOR_PARAM(6, q_norm);
  TENSOR_PARAM(7, k_norm);
  TENSOR_PARAM(8, k_cache);
  TENSOR_PARAM(9, v_cache);
  PARAM(10, int, offset);
  PARAM(11, double, scale);
  PARAM(12, int, head_dim);
  PARAM(13, double, theta);
  PARAM(14, double, eps);
  DEVICE_PARAM(15, device);

  try {
    std::string error;
    if (!qwen3_check_rank3_positive(*hidden, "hidden", error) ||
        !qwen3_check_non_negative(offset, "offset", error) ||
        !qwen3_check_positive(head_dim, "head_dim", error)) {
      return qwen3_error(env, error);
    }

    int B       = hidden->shape(0);
    int T_new   = hidden->shape(1);
    int H       = hidden->shape(2);
    int D       = head_dim;
    if (!qwen3_check_rank1_dim(*norm, H, "norm", error) ||
        !qwen3_check_rank1_dim(*q_norm, D, "q_norm", error) ||
        !qwen3_check_rank1_dim(*k_norm, D, "k_norm", error) ||
        !qwen3_check_rank2_positive(*q_proj, "q_proj", error) ||
        !qwen3_check_rank2_positive(*k_proj, "k_proj", error) ||
        !qwen3_check_rank2_positive(*v_proj, "v_proj", error) ||
        !qwen3_check_dim(*q_proj, 0, H, "q_proj", "input width", error) ||
        !qwen3_check_dim(*k_proj, 0, H, "k_proj", "input width", error) ||
        !qwen3_check_dim(*v_proj, 0, H, "v_proj", "input width", error)) {
      return qwen3_error(env, error);
    }

    if ((q_proj->shape(1) % D) != 0 || (k_proj->shape(1) % D) != 0) {
      return qwen3_error(env, "projection output widths must be divisible by head_dim");
    }
    if (v_proj->shape(1) != k_proj->shape(1)) {
      return qwen3_error(env, "v_proj output width must match k_proj output width");
    }

    int N_q     = q_proj->shape(1) / D;
    int N_kv    = k_proj->shape(1) / D;
    int attn_width = N_q * D;

    if ((N_q % N_kv) != 0) {
      return qwen3_error(env, "query heads must be divisible by key/value heads");
    }
    if (!qwen3_check_rank2_positive(*o_proj, "o_proj", error) ||
        !qwen3_check_dim(*o_proj, 0, attn_width, "o_proj", "input width", error) ||
        !qwen3_check_dim(*o_proj, 1, H, "o_proj", "output width", error) ||
        !qwen3_validate_kv_cache_bn(*k_cache, *v_cache, B, N_kv, offset, T_new, D, error)) {
      return qwen3_error(env, error);
    }
    int valid_len = offset + T_new;

    auto xn = mlx::core::fast::rms_norm(*hidden, *norm, (float)eps, device);
    auto q_flat = qwen3_linear_in_out(xn, *q_proj, device);
    auto k_flat = qwen3_linear_in_out(xn, *k_proj, device);
    auto v_flat = qwen3_linear_in_out(xn, *v_proj, device);

    auto q = mlx::core::reshape(q_flat, {B, T_new, N_q, D}, device);
    auto k = mlx::core::reshape(k_flat, {B, T_new, N_kv, D}, device);
    auto v = mlx::core::reshape(v_flat, {B, T_new, N_kv, D}, device);

    q = mlx::core::fast::rms_norm(q, *q_norm, (float)eps, device);
    k = mlx::core::fast::rms_norm(k, *k_norm, (float)eps, device);

    auto q_bn = mlx::core::transpose(q, {0, 2, 1, 3}, device);
    auto k_bn = mlx::core::transpose(k, {0, 2, 1, 3}, device);
    auto v_bn = mlx::core::transpose(v, {0, 2, 1, 3}, device);

    auto q_rope = mlx::core::fast::rope(
        q_bn, D, false, (float)theta, 1.0f, offset, std::nullopt, device);
    auto k_rope = mlx::core::fast::rope(
        k_bn, D, false, (float)theta, 1.0f, offset, std::nullopt, device);

    auto k_cache_owned = std::move(*k_cache);
    auto v_cache_owned = std::move(*v_cache);

    auto k_upd = mlx::core::slice_update(
        k_cache_owned, k_rope,
        to_shape({0, 0, offset, 0}),
        to_shape({B, N_kv, valid_len, D}),
        device);
    auto v_upd = mlx::core::slice_update(
        v_cache_owned, v_bn,
        to_shape({0, 0, offset, 0}),
        to_shape({B, N_kv, valid_len, D}),
        device);

    auto k_valid = mlx::core::slice(
        k_upd, to_shape({0, 0, 0, 0}), to_shape({B, N_kv, valid_len, D}), device);
    auto v_valid = mlx::core::slice(
        v_upd, to_shape({0, 0, 0, 0}), to_shape({B, N_kv, valid_len, D}), device);

    auto build_prefill_mask = [&]() -> mlx::core::array {
      auto mask_dtype = q.dtype();
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

    auto attn_out_bn = (T_new == 1)
      ? mlx::core::fast::scaled_dot_product_attention(
            q_rope, k_valid, v_valid, (float)scale, "", std::nullopt, std::nullopt, device)
      : mlx::core::fast::scaled_dot_product_attention(
            q_rope, k_valid, v_valid, (float)scale, "array", build_prefill_mask(), std::nullopt, device);
    auto attn_out_bthd = mlx::core::transpose(attn_out_bn, {0, 2, 1, 3}, device);
    auto attn_out = mlx::core::reshape(attn_out_bthd, {B, T_new, attn_width}, device);
    auto projected = qwen3_linear_in_out(attn_out, *o_proj, device);
    auto out = mlx::core::add(*hidden, projected, device);

    ERL_NIF_TERM result_tuple[3];
    result_tuple[0] = create_tensor_resource(env, out);
    result_tuple[1] = create_tensor_resource(env, k_upd);
    result_tuple[2] = create_tensor_resource(env, v_upd);

    return nx::nif::ok(env, enif_make_tuple3(env, result_tuple[0], result_tuple[1], result_tuple[2]));
  }
  CATCH()
}
ASYNC_NIF(qwen3_attention_block)
