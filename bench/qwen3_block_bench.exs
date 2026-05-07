# bench/qwen3_block_bench.exs
#
# Synthetic-weight Qwen3 transformer-block benchmark.
# Stage A2 of the emlx#108 quantization perf investigation.
#
# Run with:
#   mix run bench/qwen3_block_bench.exs
#
# Measures one full Qwen3-0.6B block (RMSNorm + GQA + RoPE + KV-cache + SwiGLU)
# with synthetic quantized weights. No safetensors, no Bumblebee, no checkpoint.
#
# Three groups:
#   Group 1 — single block, varying past_len (0 / 64 / 512 / 2048)
#   Group 2 — 28-block chain at past_len=512 (per-token latency headline)
#   Group 3 — single block at past_len=512, quantized vs f16 dense

Nx.default_backend({EMLX.Backend, device: :gpu})

# ── Model dimensions (Qwen3-0.6B) ────────────────────────────────────────────

hidden       = 1024
num_heads    = 16
num_kv_heads = 8
head_dim     = 128
q_per_kv     = div(num_heads, num_kv_heads)   # 2
intermediate = 3072
num_layers   = 28
eps          = 1.0e-6

# ── GPU-sync helper ───────────────────────────────────────────────────────────

defmodule Bench.Sync do
  @moduledoc false
  def eval!(t), do: EMLX.eval(EMLX.Backend.from_nx(t))
end

# ── Ops ───────────────────────────────────────────────────────────────────────

defmodule Bench.Ops do
  @moduledoc false

  # RMSNorm: x / rms(x) * weight. weight: {hidden}.
  def rms_norm(x, weight, eps) do
    # x: {1, 1, hidden}
    rms = x |> Nx.pow(2) |> Nx.mean(axes: [-1], keep_axes: true) |> Nx.add(eps) |> Nx.rsqrt()
    Nx.multiply(Nx.multiply(x, rms), weight)
  end

  # Quantized linear: act {1,1,in} × qw {out,in} → {1,1,out}.
  def qlinear(act, qw) do
    Nx.dot(act, [2], qw, [1])
  end

  # Dense f16 linear (same axes, for Group 3 baseline).
  def dlinear(act, w) do
    Nx.dot(act, [2], w, [1])
  end

  # Rotate the second half of head_dim to the first, negating the first half.
  # Used in RoPE: rotate_half([x1, x2]) = [-x2, x1].
  def rotate_half(x, head_dim) do
    half = div(head_dim, 2)
    x1 = x[[.., .., .., 0..(half - 1)//1]]
    x2 = x[[.., .., .., half..(head_dim - 1)//1]]
    Nx.concatenate([Nx.negate(x2), x1], axis: 3)
  end

  # Apply RoPE. x: {batch, seq, heads, head_dim}. cos/sin: {1, seq, 1, head_dim}.
  def apply_rope(x, cos, sin, head_dim) do
    Nx.add(Nx.multiply(x, cos), Nx.multiply(rotate_half(x, head_dim), sin))
  end

  # GQA attention. Returns attn_out: {1, 1, num_heads * head_dim}.
  # q:       {1, 1,        num_heads,    head_dim}
  # k_cache: {1, past_len, num_kv_heads, head_dim}
  # v_cache: {1, past_len, num_kv_heads, head_dim}
  def gqa_attention(q, k_cache, v_cache, num_kv_heads, q_per_kv, head_dim) do
    [batch, _seq_q, _num_heads, _hd] = Nx.shape(q) |> Tuple.to_list()
    [_batch, _new_len, _kv_heads, _hd] = Nx.shape(k_cache) |> Tuple.to_list()

    # Reshape q: {1, 1, num_heads, head_dim} → {1, num_kv_heads, q_per_kv, head_dim}
    q_g = Nx.reshape(q, {batch, num_kv_heads, q_per_kv, head_dim})

    # k/v cache: {1, seq_k, num_kv_heads, head_dim} → {1, num_kv_heads, seq_k, head_dim}
    k_t = Nx.transpose(k_cache, axes: [0, 2, 1, 3])
    v_t = Nx.transpose(v_cache, axes: [0, 2, 1, 3])

    # k needs to be transposed to {1, num_kv_heads, head_dim, seq_k} for matmul
    k_t2 = Nx.transpose(k_t, axes: [0, 1, 3, 2])

    # scores: {1, num_kv_heads, q_per_kv, seq_k}
    # contract: axis 3 of q_g (head_dim) with axis 2 of k_t2 (head_dim)
    # batch over: axes [0, 1] (batch, num_kv_heads)
    scores =
      Nx.dot(q_g, [3], [0, 1], k_t2, [2], [0, 1])
      |> Nx.divide(:math.sqrt(head_dim))

    # Numerically stable softmax over seq_k axis (3).
    # Nx.softmax/1 doesn't accept an axis keyword in this Nx version.
    max_s = Nx.reduce_max(scores, axes: [3], keep_axes: true)
    exp_s = Nx.exp(Nx.subtract(scores, max_s))
    attn_weights = Nx.divide(exp_s, Nx.sum(exp_s, axes: [3], keep_axes: true))

    # attn_out: {1, num_kv_heads, q_per_kv, head_dim}
    # contract: axis 3 of attn_weights (seq_k) with axis 2 of v_t (seq_k)
    attn_out = Nx.dot(attn_weights, [3], [0, 1], v_t, [2], [0, 1])

    # reshape to {1, 1, num_heads * head_dim}
    Nx.reshape(attn_out, {batch, 1, num_kv_heads * q_per_kv * head_dim})
  end

  # SwiGLU: silu(gate) * up. gate/up: {1,1,intermediate}.
  def swiglu(gate, up) do
    # silu(x) = x * sigmoid(x)
    Nx.multiply(Nx.multiply(gate, Nx.sigmoid(gate)), up)
  end
end

# ── Block forward ─────────────────────────────────────────────────────────────

defmodule Bench.Block do
  @moduledoc false

  # One full Qwen3 transformer block.
  # k_cache, v_cache: {1, past_len, num_kv_heads, head_dim} (pre-built outside closure)
  # cos, sin: {1, past_len+1, 1, head_dim} — RoPE for the new position
  # Returns {x_out, k_cache_new, v_cache_new}.
  def forward(x, k_cache, v_cache, weights, cos, sin, opts) do
    head_dim     = Keyword.fetch!(opts, :head_dim)
    num_kv_heads = Keyword.fetch!(opts, :num_kv_heads)
    q_per_kv     = Keyword.fetch!(opts, :q_per_kv)
    num_heads    = Keyword.fetch!(opts, :num_heads)
    eps          = Keyword.fetch!(opts, :eps)

    %{
      norm1:      norm1,
      q_proj:     q_proj,
      k_proj:     k_proj,
      v_proj:     v_proj,
      o_proj:     o_proj,
      norm2:      norm2,
      gate_proj:  gate_proj,
      up_proj:    up_proj,
      down_proj:  down_proj
    } = weights

    # 1. Pre-attention RMSNorm
    xn = Bench.Ops.rms_norm(x, norm1, eps)

    # 2. Projections (quantized linears → {1,1,out_features})
    q = Bench.Ops.qlinear(xn, q_proj) |> Nx.reshape({1, 1, num_heads, head_dim})
    k = Bench.Ops.qlinear(xn, k_proj) |> Nx.reshape({1, 1, num_kv_heads, head_dim})
    v = Bench.Ops.qlinear(xn, v_proj) |> Nx.reshape({1, 1, num_kv_heads, head_dim})

    # 3. RoPE on the new token's position (cos/sin sliced to just the last position)
    pos_cos = cos[[.., -1..-1//1, .., ..]]
    pos_sin = sin[[.., -1..-1//1, .., ..]]
    q = Bench.Ops.apply_rope(q, pos_cos, pos_sin, head_dim)
    k = Bench.Ops.apply_rope(k, pos_cos, pos_sin, head_dim)

    # 4. KV cache append (inside closure — this is the real decode overhead)
    k_cache_new = Nx.concatenate([k_cache, k], axis: 1)
    v_cache_new = Nx.concatenate([v_cache, v], axis: 1)

    # 5. GQA attention
    attn_out = Bench.Ops.gqa_attention(q, k_cache_new, v_cache_new, num_kv_heads, q_per_kv, head_dim)

    # 6. o_proj + residual
    x = Nx.add(x, Bench.Ops.qlinear(attn_out, o_proj))

    # 7. Post-attention RMSNorm
    xn2 = Bench.Ops.rms_norm(x, norm2, eps)

    # 8. SwiGLU MLP
    gate = Bench.Ops.qlinear(xn2, gate_proj)
    up   = Bench.Ops.qlinear(xn2, up_proj)
    mlp  = Bench.Ops.swiglu(gate, up)
    x    = Nx.add(x, Bench.Ops.qlinear(mlp, down_proj))

    {x, k_cache_new, v_cache_new}
  end

  # Same block but with dense f16 weights (for Group 3 comparison).
  def forward_dense(x, k_cache, v_cache, weights, cos, sin, opts) do
    head_dim     = Keyword.fetch!(opts, :head_dim)
    num_kv_heads = Keyword.fetch!(opts, :num_kv_heads)
    q_per_kv     = Keyword.fetch!(opts, :q_per_kv)
    num_heads    = Keyword.fetch!(opts, :num_heads)
    eps          = Keyword.fetch!(opts, :eps)

    %{
      norm1:     norm1,
      q_proj:    q_proj,
      k_proj:    k_proj,
      v_proj:    v_proj,
      o_proj:    o_proj,
      norm2:     norm2,
      gate_proj: gate_proj,
      up_proj:   up_proj,
      down_proj: down_proj
    } = weights

    xn = Bench.Ops.rms_norm(x, norm1, eps)

    q = Bench.Ops.dlinear(xn, q_proj) |> Nx.reshape({1, 1, num_heads, head_dim})
    k = Bench.Ops.dlinear(xn, k_proj) |> Nx.reshape({1, 1, num_kv_heads, head_dim})
    v = Bench.Ops.dlinear(xn, v_proj) |> Nx.reshape({1, 1, num_kv_heads, head_dim})

    pos_cos = cos[[.., -1..-1//1, .., ..]]
    pos_sin = sin[[.., -1..-1//1, .., ..]]
    q = Bench.Ops.apply_rope(q, pos_cos, pos_sin, head_dim)
    k = Bench.Ops.apply_rope(k, pos_cos, pos_sin, head_dim)

    k_cache_new = Nx.concatenate([k_cache, k], axis: 1)
    v_cache_new = Nx.concatenate([v_cache, v], axis: 1)

    attn_out = Bench.Ops.gqa_attention(q, k_cache_new, v_cache_new, num_kv_heads, q_per_kv, head_dim)

    x = Nx.add(x, Bench.Ops.dlinear(attn_out, o_proj))

    xn2  = Bench.Ops.rms_norm(x, norm2, eps)
    gate = Bench.Ops.dlinear(xn2, gate_proj)
    up   = Bench.Ops.dlinear(xn2, up_proj)
    mlp  = Bench.Ops.swiglu(gate, up)
    x    = Nx.add(x, Bench.Ops.dlinear(mlp, down_proj))

    {x, k_cache_new, v_cache_new}
  end
end

# ── Setup: pre-quantize weights ───────────────────────────────────────────────

IO.puts("==> Pre-quantizing weights for one block...")

to_gpu = fn t -> Nx.backend_transfer(t, {EMLX.Backend, device: :gpu}) end

make_f16 = fn shape ->
  Nx.broadcast(Nx.tensor(0.01, type: :f16), shape) |> to_gpu.()
end

make_quant = fn shape, opts ->
  make_f16.(shape) |> EMLX.Quantization.quantize(opts)
end

q_opts = [type: {:s, 4}, group_size: 64]

quant_weights = %{
  norm1:     make_f16.({hidden}),
  q_proj:    make_quant.({num_heads * head_dim, hidden}, q_opts),
  k_proj:    make_quant.({num_kv_heads * head_dim, hidden}, q_opts),
  v_proj:    make_quant.({num_kv_heads * head_dim, hidden}, q_opts),
  o_proj:    make_quant.({hidden, num_heads * head_dim}, q_opts),
  norm2:     make_f16.({hidden}),
  gate_proj: make_quant.({intermediate, hidden}, q_opts),
  up_proj:   make_quant.({intermediate, hidden}, q_opts),
  down_proj: make_quant.({hidden, intermediate}, q_opts)
}

dense_weights = %{
  norm1:     make_f16.({hidden}),
  q_proj:    make_f16.({num_heads * head_dim, hidden}),
  k_proj:    make_f16.({num_kv_heads * head_dim, hidden}),
  v_proj:    make_f16.({num_kv_heads * head_dim, hidden}),
  o_proj:    make_f16.({hidden, num_heads * head_dim}),
  norm2:     make_f16.({hidden}),
  gate_proj: make_f16.({intermediate, hidden}),
  up_proj:   make_f16.({intermediate, hidden}),
  down_proj: make_f16.({hidden, intermediate})
}

IO.puts("==> Weights ready. Pre-computing RoPE frequencies...")

# Pre-compute RoPE cos/sin for the maximum sequence length needed (2049 positions)
max_pos = 2048 + 1

# theta: {1, 1, 1, head_dim/2}
theta =
  Nx.iota({div(head_dim, 2)}, type: :f32)
  |> then(fn i -> Nx.pow(10000.0, Nx.divide(Nx.multiply(-2.0, i), head_dim)) end)
  |> Nx.reshape({1, 1, 1, div(head_dim, 2)})
  |> to_gpu.()

# positions: {1, max_pos, 1, 1}
positions =
  Nx.iota({max_pos}, type: :f32)
  |> Nx.reshape({1, max_pos, 1, 1})
  |> to_gpu.()

# freqs: {1, max_pos, 1, head_dim/2}
freqs = Nx.multiply(positions, theta)

# cos/sin: {1, max_pos, 1, head_dim} (duplicate halves)
half_cos = Nx.cos(freqs)
half_sin = Nx.sin(freqs)
rope_cos  = Nx.concatenate([half_cos, half_cos], axis: 3)
rope_sin  = Nx.concatenate([half_sin, half_sin], axis: 3)

IO.puts("==> RoPE ready. Setting up KV caches...\n")

block_opts = [
  head_dim:     head_dim,
  num_kv_heads: num_kv_heads,
  q_per_kv:     q_per_kv,
  num_heads:    num_heads,
  eps:          eps
]

act = make_f16.({1, 1, hidden})

# ── Helper: build KV cache for a given past_len ───────────────────────────────

make_kv_cache = fn past_len ->
  k_cache = Nx.broadcast(Nx.tensor(0.01, type: :f16), {1, past_len, num_kv_heads, head_dim}) |> to_gpu.()
  v_cache = Nx.broadcast(Nx.tensor(0.01, type: :f16), {1, past_len, num_kv_heads, head_dim}) |> to_gpu.()
  {k_cache, v_cache}
end

# Slice RoPE for a given past_len (the "new" token is at position past_len)
rope_for = fn past_len ->
  len = past_len + 1
  {rope_cos[[.., 0..(len - 1)//1, .., ..]], rope_sin[[.., 0..(len - 1)//1, .., ..]]}
end

# ── Group 1 — single block, varying past_len ─────────────────────────────────

IO.puts("=== Group 1: single block, varying past_len ===\n")

g1_jobs =
  for past_len <- [1, 64, 512, 2048], into: %{} do
    {k_cache, v_cache} = make_kv_cache.(past_len)
    {cos, sin}         = rope_for.(past_len)

    key = "block/4bit_g64/past=#{past_len}"

    job = fn ->
      {x_out, _, _} = Bench.Block.forward(act, k_cache, v_cache, quant_weights, cos, sin, block_opts)
      Bench.Sync.eval!(x_out)
    end

    {key, job}
  end

Benchee.run(g1_jobs, warmup: 2, time: 5, memory_time: 0,
  formatters: [Benchee.Formatters.Console])

# ── Group 2 — 28-block chain at past_len=512 ─────────────────────────────────

IO.puts("\n=== Group 2: 28-block chain at past_len=512 ===\n")

# Pre-build one KV cache per layer (each layer has its own cache).
# All start at past_len=512 (simulating steady-state decode at that position).
g2_caches =
  for _i <- 1..num_layers do
    make_kv_cache.(512)
  end

{g2_cos, g2_sin} = rope_for.(512)

g2_jobs = %{
  "28_blocks/4bit_g64/past=512" => fn ->
    # Thread x and each layer's KV cache through 28 blocks in sequence.
    {x_out, _} =
      Enum.reduce(1..num_layers, {act, g2_caches}, fn i, {x_in, caches} ->
        {k_c, v_c} = Enum.at(caches, i - 1)
        {x_out, k_c_new, v_c_new} = Bench.Block.forward(x_in, k_c, v_c, quant_weights, g2_cos, g2_sin, block_opts)
        updated = List.replace_at(caches, i - 1, {k_c_new, v_c_new})
        {x_out, updated}
      end)

    Bench.Sync.eval!(x_out)
  end
}

Benchee.run(g2_jobs, warmup: 2, time: 10, memory_time: 0,
  formatters: [Benchee.Formatters.Console])

# ── Group 3 — quantized vs dense at past_len=512 ─────────────────────────────

IO.puts("\n=== Group 3: quantized vs f16 dense at past_len=512 ===\n")

{k512, v512}   = make_kv_cache.(512)
{cos512, sin512} = rope_for.(512)

g3_jobs = %{
  "block/4bit_g64/past=512" => fn ->
    {x_out, _, _} = Bench.Block.forward(act, k512, v512, quant_weights, cos512, sin512, block_opts)
    Bench.Sync.eval!(x_out)
  end,
  "block/f16_dense/past=512" => fn ->
    {x_out, _, _} = Bench.Block.forward_dense(act, k512, v512, dense_weights, cos512, sin512, block_opts)
    Bench.Sync.eval!(x_out)
  end
}

Benchee.run(g3_jobs, warmup: 2, time: 5, memory_time: 0,
  formatters: [Benchee.Formatters.Console])

# ── Decision gate ─────────────────────────────────────────────────────────────

IO.puts("""

=== Decision gate — compute manually from the tables above ===

  kernel_sum_ms_per_step (A1, per-layer, 7 linears excl. lm_head) = 1.043 ms  [serial]
  A0 greedy wall-clock per-block (14.23 ms / 28)                  = 0.508 ms  [parallel]

  block_sum_ms_per_step = Group2 median ms / 28

  overhead_ratio (vs serial A1) = block_sum / 1.043
  overhead_ratio (vs A0 wall)   = block_sum / 0.508  ← advisor-recommended comparison

  Decision table (using vs-A0-wall):
    < 1.2  → overhead negligible, A3 only needs sampler swap
    1.2–2× → some overhead (likely KV concat), preallocate KV cache in A3
    ≥ 2×   → significant overhead, investigate dispatch before A3

  Ceiling tok/s (ignoring sampler) = 1000 / (Group2_total_ms)
    e.g. if Group2 = 14 ms → ceiling = 71 tok/s (greedy, no sampler)
""")

# ── Write results file ────────────────────────────────────────────────────────

date_str = Date.utc_today() |> Date.to_string() |> String.replace("-", "")
{:ok, hostname} = :inet.gethostname()
hostname_str = List.to_string(hostname)
results_dir  = Path.join([__DIR__, "results"])
results_file = Path.join(results_dir, "qwen3_block_#{date_str}_#{hostname_str}.txt")

File.mkdir_p!(results_dir)
IO.puts("==> Results written to #{results_file}")
IO.puts("    (Re-run to capture stdout to file; pipe with: mix run bench/qwen3_block_bench.exs | tee #{results_file})")
