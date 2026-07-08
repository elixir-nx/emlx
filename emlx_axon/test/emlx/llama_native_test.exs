defmodule EMLXAxon.LlamaNativeTest do
  use ExUnit.Case, async: true

  @moduletag :metal

  @backend {EMLX.Backend, device: :gpu}

  test "llama_layer matches Nx reference for one token" do
    hidden = gpu_tensor([[[0.2, -0.4]]], :f32)
    norm1 = gpu_tensor([1.0, 0.5], :f32)
    norm2 = gpu_tensor([0.75, 1.25], :f32)
    q_proj = gpu_tensor([[0.1, -0.2], [0.3, 0.4]], :f32)
    k_proj = gpu_tensor([[0.2, 0.1], [-0.3, 0.5]], :f32)
    v_proj = gpu_tensor([[0.6, -0.1], [0.2, 0.3]], :f32)
    o_proj = gpu_tensor([[0.4, -0.2], [0.1, 0.5]], :f32)
    gate_proj = gpu_tensor([[0.2, -0.1, 0.3], [0.4, 0.5, -0.2]], :f32)
    up_proj = gpu_tensor([[0.1, 0.6, -0.3], [-0.4, 0.2, 0.7]], :f32)
    down_proj = gpu_tensor([[0.3, -0.2], [0.5, 0.1], [-0.4, 0.6]], :f32)
    k_cache = EMLX.full(0.0, {1, 1, 4, 2}, :float32, :gpu)
    v_cache = EMLX.full(0.0, {1, 1, 4, 2}, :float32, :gpu)
    rope_freqs = gpu_tensor([1.0], :f32)

    {out, _k_updated, _v_updated} =
      EMLX.Native.Llama.layer(
        EMLX.Backend.from_nx(hidden),
        EMLX.Backend.from_nx(norm1),
        EMLX.Backend.from_nx(q_proj),
        EMLX.Backend.from_nx(k_proj),
        EMLX.Backend.from_nx(v_proj),
        EMLX.Backend.from_nx(o_proj),
        k_cache,
        v_cache,
        EMLX.Backend.from_nx(norm2),
        EMLX.Backend.from_nx(gate_proj),
        EMLX.Backend.from_nx(up_proj),
        EMLX.Backend.from_nx(down_proj),
        0,
        1.0 / :math.sqrt(2),
        2,
        EMLX.Backend.from_nx(rope_freqs),
        1.0e-5
      )

    expected =
      llama_layer_reference(
        hidden,
        norm1,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        norm2,
        gate_proj,
        up_proj,
        down_proj,
        1.0e-5
      )

    assert_all_close(to_binary(out), expected, atol: 1.0e-4)
  end

  test "llama_layer matches Nx reference for GQA prefill with nonzero offset" do
    {hidden, norm1, norm2, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, k_cache,
     v_cache, rope_freqs} =
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        {
          Nx.iota({1, 2, 4}, type: :f32) |> Nx.add(1) |> Nx.divide(10),
          Nx.tensor([1.0, 0.5, 1.5, 0.75], type: :f32),
          Nx.tensor([0.75, 1.25, 0.5, 1.0], type: :f32),
          Nx.iota({4, 8}, type: :f32) |> Nx.add(1) |> Nx.divide(100),
          Nx.iota({4, 4}, type: :f32) |> Nx.add(11) |> Nx.divide(110),
          Nx.iota({4, 4}, type: :f32) |> Nx.add(17) |> Nx.divide(120),
          Nx.iota({8, 4}, type: :f32) |> Nx.add(23) |> Nx.divide(130),
          Nx.iota({4, 6}, type: :f32) |> Nx.add(29) |> Nx.divide(140),
          Nx.iota({4, 6}, type: :f32) |> Nx.add(31) |> Nx.divide(150),
          Nx.iota({6, 4}, type: :f32) |> Nx.add(37) |> Nx.divide(160),
          Nx.iota({1, 1, 5, 4}, type: :f32) |> Nx.divide(1_000),
          Nx.iota({1, 1, 5, 4}, type: :f32) |> Nx.add(50) |> Nx.divide(1_000),
          Nx.tensor([1.0, 0.25], type: :f32)
        }
      end)

    offset = 1
    head_dim = 4
    scale = 1.0 / :math.sqrt(head_dim)
    eps = 1.0e-5

    {out, k_updated, v_updated} =
      EMLX.Native.Llama.layer(
        EMLX.Backend.from_nx(gpu_tensor(hidden)),
        EMLX.Backend.from_nx(gpu_tensor(norm1)),
        EMLX.Backend.from_nx(gpu_tensor(q_proj)),
        EMLX.Backend.from_nx(gpu_tensor(k_proj)),
        EMLX.Backend.from_nx(gpu_tensor(v_proj)),
        EMLX.Backend.from_nx(gpu_tensor(o_proj)),
        EMLX.Backend.from_nx(gpu_tensor(k_cache)),
        EMLX.Backend.from_nx(gpu_tensor(v_cache)),
        EMLX.Backend.from_nx(gpu_tensor(norm2)),
        EMLX.Backend.from_nx(gpu_tensor(gate_proj)),
        EMLX.Backend.from_nx(gpu_tensor(up_proj)),
        EMLX.Backend.from_nx(gpu_tensor(down_proj)),
        offset,
        scale,
        head_dim,
        EMLX.Backend.from_nx(gpu_tensor(rope_freqs)),
        eps
      )

    {expected, expected_k, expected_v} =
      llama_layer_reference(
        hidden,
        norm1,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        k_cache,
        v_cache,
        norm2,
        gate_proj,
        up_proj,
        down_proj,
        offset,
        scale,
        head_dim,
        rope_freqs,
        eps
      )

    assert_all_close(to_binary(out), expected, atol: 5.0e-4)
    assert_all_close(to_binary(k_updated), expected_k, atol: 1.0e-4)
    assert_all_close(to_binary(v_updated), expected_v, atol: 1.0e-4)
  end

  test "llama_final_greedy matches Nx reference on last hidden position" do
    hidden =
      gpu_tensor(
        [
          [
            [0.2, -0.1, 0.4, 0.3],
            [-0.5, 0.6, 0.1, 0.7]
          ],
          [
            [0.8, -0.2, 0.5, -0.4],
            [0.3, 0.9, -0.6, 0.2]
          ]
        ],
        :f32
      )

    norm = gpu_tensor([1.0, 0.5, 1.5, 0.75], :f32)

    lm_head =
      gpu_tensor(
        [
          [0.2, -0.1, 0.4, 0.3],
          [-0.5, 0.6, 0.1, 0.7],
          [0.8, -0.2, 0.5, -0.4],
          [0.3, 0.9, -0.6, 0.2],
          [-0.7, 0.4, 0.2, 0.5]
        ],
        :f32
      )

    out =
      hidden
      |> EMLX.Backend.from_nx()
      |> EMLX.Native.Llama.final_greedy(
        EMLX.Backend.from_nx(norm),
        EMLX.Backend.from_nx(lm_head),
        1.0e-5
      )
      |> EMLX.Backend.to_nx()
      |> Nx.backend_transfer(Nx.BinaryBackend)

    expected = llama_final_greedy_reference(hidden, norm, lm_head, 1.0e-5)

    assert Nx.to_flat_list(out) == Nx.to_flat_list(expected)
  end

  defp llama_mlp_reference(hidden, norm, gate_proj, up_proj, down_proj, eps) do
    hidden = Nx.backend_transfer(hidden, Nx.BinaryBackend)
    norm = Nx.backend_transfer(norm, Nx.BinaryBackend)
    gate_proj = Nx.backend_transfer(gate_proj, Nx.BinaryBackend)
    up_proj = Nx.backend_transfer(up_proj, Nx.BinaryBackend)
    down_proj = Nx.backend_transfer(down_proj, Nx.BinaryBackend)

    variance = hidden |> Nx.pow(2) |> Nx.mean(axes: [-1], keep_axes: true)

    xn =
      hidden
      |> Nx.multiply(Nx.rsqrt(Nx.add(variance, eps)))
      |> Nx.multiply(norm)

    gate = Nx.dot(xn, [2], gate_proj, [0])
    up = Nx.dot(xn, [2], up_proj, [0])

    mlp =
      gate
      |> Nx.multiply(Nx.sigmoid(gate))
      |> Nx.multiply(up)

    Nx.add(hidden, Nx.dot(mlp, [2], down_proj, [0]))
  end

  defp llama_kv_cache_attention_reference(
         q,
         new_k,
         new_v,
         k_cache,
         v_cache,
         offset,
         scale,
         head_dim,
         rope_freqs
       ) do
    {batch, seq_len, q_heads, _head_dim} = Nx.shape(q)
    {_batch, _seq_len, kv_heads, _head_dim} = Nx.shape(new_k)

    q_bn =
      q
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> llama_rope_reference(head_dim, offset, rope_freqs)
      |> Nx.backend_transfer(Nx.BinaryBackend)

    k_bn =
      new_k
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> llama_rope_reference(head_dim, offset, rope_freqs)
      |> Nx.backend_transfer(Nx.BinaryBackend)

    v_bn = new_v |> Nx.transpose(axes: [0, 2, 1, 3]) |> Nx.backend_transfer(Nx.BinaryBackend)

    k_cache = Nx.put_slice(k_cache, [0, 0, offset, 0], k_bn)
    v_cache = Nx.put_slice(v_cache, [0, 0, offset, 0], v_bn)

    valid_len = offset + seq_len
    k_valid = Nx.slice_along_axis(k_cache, 0, valid_len, axis: 2)
    v_valid = Nx.slice_along_axis(v_cache, 0, valid_len, axis: 2)

    groups = div(q_heads, kv_heads)
    k_repeated = repeat_kv_heads_bn(k_valid, groups)
    v_repeated = repeat_kv_heads_bn(v_valid, groups)

    scores =
      q_bn
      |> Nx.new_axis(3)
      |> Nx.multiply(Nx.new_axis(k_repeated, 2))
      |> Nx.sum(axes: [4])
      |> Nx.multiply(scale)
      |> apply_causal_mask(offset, seq_len, valid_len)

    weights = softmax_reference(scores, 3)

    attn =
      weights
      |> Nx.new_axis(4)
      |> Nx.multiply(Nx.new_axis(v_repeated, 2))
      |> Nx.sum(axes: [3])

    attn_out =
      attn
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Nx.reshape({batch, seq_len, q_heads * head_dim})

    {attn_out, k_cache, v_cache}
  end

  defp llama_rope_reference(tensor, dims, offset, rope_freqs) do
    {_batch, _heads, seq_len, _dims} = Nx.shape(tensor)
    half = div(dims, 2)

    positions =
      Nx.iota({seq_len}, type: :f32)
      |> Nx.add(offset)
      |> Nx.reshape({seq_len, 1})

    inv_freqs =
      rope_freqs
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> then(&Nx.divide(1.0, &1))
      |> Nx.reshape({1, half})

    freqs = Nx.multiply(positions, inv_freqs)

    cos =
      Nx.concatenate([Nx.cos(freqs), Nx.cos(freqs)], axis: 1)
      |> Nx.reshape({1, 1, seq_len, dims})

    sin =
      Nx.concatenate([Nx.sin(freqs), Nx.sin(freqs)], axis: 1)
      |> Nx.reshape({1, 1, seq_len, dims})

    first = tensor[[.., .., .., 0..(half - 1)//1]]
    second = tensor[[.., .., .., half..(dims - 1)//1]]
    rotated = Nx.concatenate([Nx.negate(second), first], axis: 3)

    Nx.add(Nx.multiply(tensor, cos), Nx.multiply(rotated, sin))
  end

  defp repeat_kv_heads_bn(tensor, 1), do: tensor

  defp repeat_kv_heads_bn(tensor, groups) do
    {batch, kv_heads, seq_len, head_dim} = Nx.shape(tensor)

    tensor
    |> Nx.new_axis(2)
    |> Nx.broadcast({batch, kv_heads, groups, seq_len, head_dim})
    |> Nx.reshape({batch, kv_heads * groups, seq_len, head_dim})
  end

  defp apply_causal_mask(scores, offset, seq_len, valid_len) do
    query_positions =
      Nx.iota({seq_len}, type: :s32, backend: Nx.BinaryBackend)
      |> Nx.add(offset)
      |> Nx.reshape({1, 1, seq_len, 1})

    key_positions =
      Nx.iota({valid_len}, type: :s32, backend: Nx.BinaryBackend)
      |> Nx.reshape({1, 1, 1, valid_len})

    mask =
      key_positions
      |> Nx.less_equal(query_positions)
      |> Nx.broadcast(Nx.shape(scores))

    Nx.select(mask, scores, Nx.broadcast(-1.0e9, Nx.shape(scores)))
  end

  defp softmax_reference(tensor, axis) do
    shifted = Nx.subtract(tensor, Nx.reduce_max(tensor, axes: [axis], keep_axes: true))
    exp = Nx.exp(shifted)
    Nx.divide(exp, Nx.sum(exp, axes: [axis], keep_axes: true))
  end

  defp llama_layer_reference(
         hidden,
         norm1,
         _q_proj,
         _k_proj,
         v_proj,
         o_proj,
         norm2,
         gate_proj,
         up_proj,
         down_proj,
         eps
       ) do
    hidden = Nx.backend_transfer(hidden, Nx.BinaryBackend)
    norm1 = Nx.backend_transfer(norm1, Nx.BinaryBackend)
    v_proj = Nx.backend_transfer(v_proj, Nx.BinaryBackend)
    o_proj = Nx.backend_transfer(o_proj, Nx.BinaryBackend)

    xn = rms_norm(hidden, norm1, eps)

    # With a single token at offset 0 there is exactly one key, so causal SDPA
    # has probability 1.0 and the attention output is the projected value.
    attn_out = Nx.dot(xn, [2], v_proj, [0])
    attn_hidden = Nx.add(hidden, Nx.dot(attn_out, [2], o_proj, [0]))

    llama_mlp_reference(attn_hidden, norm2, gate_proj, up_proj, down_proj, eps)
  end

  defp llama_layer_reference(
         hidden,
         norm1,
         q_proj,
         k_proj,
         v_proj,
         o_proj,
         k_cache,
         v_cache,
         norm2,
         gate_proj,
         up_proj,
         down_proj,
         offset,
         scale,
         head_dim,
         rope_freqs,
         eps
       ) do
    hidden = Nx.backend_transfer(hidden, Nx.BinaryBackend)
    norm1 = Nx.backend_transfer(norm1, Nx.BinaryBackend)
    q_proj = Nx.backend_transfer(q_proj, Nx.BinaryBackend)
    k_proj = Nx.backend_transfer(k_proj, Nx.BinaryBackend)
    v_proj = Nx.backend_transfer(v_proj, Nx.BinaryBackend)
    o_proj = Nx.backend_transfer(o_proj, Nx.BinaryBackend)

    hidden_norm = rms_norm(hidden, norm1, eps)
    q = hidden_norm |> Nx.dot([2], q_proj, [0]) |> reshape_projection(head_dim)
    k = hidden_norm |> Nx.dot([2], k_proj, [0]) |> reshape_projection(head_dim)
    v = hidden_norm |> Nx.dot([2], v_proj, [0]) |> reshape_projection(head_dim)

    {attn_out, k_updated, v_updated} =
      llama_kv_cache_attention_reference(
        q,
        k,
        v,
        k_cache,
        v_cache,
        offset,
        scale,
        head_dim,
        rope_freqs
      )

    attn_hidden = Nx.add(hidden, Nx.dot(attn_out, [2], o_proj, [0]))

    {
      llama_mlp_reference(attn_hidden, norm2, gate_proj, up_proj, down_proj, eps),
      k_updated,
      v_updated
    }
  end

  defp reshape_projection(tensor, head_dim) do
    {batch, seq_len, width} = Nx.shape(tensor)
    Nx.reshape(tensor, {batch, seq_len, div(width, head_dim), head_dim})
  end

  defp llama_final_greedy_reference(hidden, norm, lm_head, eps) do
    hidden = Nx.backend_transfer(hidden, Nx.BinaryBackend)
    {_batch, seq_len, _hidden_size} = Nx.shape(hidden)
    last_hidden = hidden[[.., (seq_len - 1)..(seq_len - 1)//1, ..]] |> Nx.squeeze(axes: [1])
    normed = rms_norm(last_hidden, norm, eps)
    logits = Nx.dot(normed, [1], Nx.backend_transfer(lm_head, Nx.BinaryBackend), [1])
    Nx.argmax(logits, axis: 1)
  end

  defp rms_norm(hidden, norm, eps) do
    variance = hidden |> Nx.pow(2) |> Nx.mean(axes: [-1], keep_axes: true)

    hidden
    |> Nx.multiply(Nx.rsqrt(Nx.add(variance, eps)))
    |> Nx.multiply(Nx.backend_transfer(norm, Nx.BinaryBackend))
  end

  defp gpu_tensor(%Nx.Tensor{} = tensor), do: Nx.backend_transfer(tensor, @backend)
  defp gpu_tensor(data, type), do: data |> Nx.tensor(type: type) |> Nx.backend_transfer(@backend)

  defp to_binary(ref), do: ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend)

  defp assert_all_close(left, right, opts) do
    diff =
      left
      |> Nx.subtract(right)
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    assert diff <= Keyword.fetch!(opts, :atol)
  end
end
