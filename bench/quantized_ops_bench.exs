# bench/quantized_ops_bench.exs
#
# Op-level microbenchmark for emlx quantized matmul kernels.
# Stage A1 of the emlx#108 quantization perf investigation.
#
# Run with:
#   mix run bench/quantized_ops_bench.exs
#
# Results are written to bench/results/quantized_ops_<date>_<hostname>.txt.
# The bench/results/ directory is version-controlled — commit the .txt file
# in PRs that touch the quantization kernels (regression gate).

Nx.default_backend({EMLX.Backend, device: :gpu})

# ── Helpers ──────────────────────────────────────────────────────────────────

defmodule Bench.Sync do
  @moduledoc false
  # Force GPU evaluation without a CPU round-trip.
  # EMLX.eval/1 expects {device, ref}; EMLX.Backend.from_nx/1 returns exactly that.
  def eval!(result), do: EMLX.eval(EMLX.Backend.from_nx(result))
end

# ── Qwen3-0.6B dimensions ────────────────────────────────────────────────────

hidden       = 1024
intermediate = 3072
num_heads    = 16
num_kv_heads = 8
head_dim     = 128
vocab_size   = 151_936

# Weight shapes: {out_features, in_features} (row-major, as MLX expects).
# Activation shape per linear is {1, 1, in_features} (decode step, seq_len=1).
linears = [
  {:q_proj,    {num_heads * head_dim, hidden}},        # {2048, 1024}  act {1,1,1024}
  {:k_proj,    {num_kv_heads * head_dim, hidden}},     # {1024, 1024}  act {1,1,1024}
  {:v_proj,    {num_kv_heads * head_dim, hidden}},     # {1024, 1024}  act {1,1,1024}
  {:o_proj,    {hidden, num_heads * head_dim}},        # {1024, 2048}  act {1,1,2048}
  {:gate_proj, {intermediate, hidden}},                # {3072, 1024}  act {1,1,1024}
  {:up_proj,   {intermediate, hidden}},                # {3072, 1024}  act {1,1,1024}
  {:down_proj, {hidden, intermediate}},                # {1024, 3072}  act {1,1,3072}
  {:lm_head,   {vocab_size, hidden}},                  # {151_936, 1024} act {1,1,1024}
]

# ── Quantization variants ─────────────────────────────────────────────────────
# 4 quantized + 1 f16 dense = 5 variants per linear = 40 scenarios total.

quant_variants = [
  {"4bit_g64",  [type: {:s, 4}, group_size: 64]},
  {"4bit_g32",  [type: {:s, 4}, group_size: 32]},
  {"4bit_g128", [type: {:s, 4}, group_size: 128]},
  {"8bit_g64",  [type: {:s, 8}, group_size: 64]},
]

# ── Pre-build all weights ─────────────────────────────────────────────────────
# Quantization happens outside the benchmarked closure to avoid measuring
# quantize overhead alongside matmul.

IO.puts("==> Pre-quantizing weights (this may take ~20 s for lm_head)...")

weights =
  for {name, {out_f, in_f} = w_shape} <- linears do
    dense_f16 =
      Nx.broadcast(Nx.tensor(0.01, type: :f16), w_shape)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    quant_weights =
      for {variant, opts} <- quant_variants do
        qw = EMLX.Quantization.quantize(dense_f16, opts)
        {variant, qw}
      end

    # Pre-build the activation tensor for this linear's input dimension
    act =
      Nx.broadcast(Nx.tensor(0.01, type: :f16), {1, 1, in_f})
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    {name, out_f, dense_f16, quant_weights, act}
  end

IO.puts("==> Weights ready. Starting benchmarks...\n")

# ── Build Benchee job map ─────────────────────────────────────────────────────

jobs =
  for {name, _out_f, dense_f16, quant_weights, act} <- weights,
      reduce: %{} do
    acc ->
      # Dense f16 baseline
      dense_key = "#{name}/f16"

      dense_job = fn ->
        result = Nx.dot(act, [2], dense_f16, [1])
        Bench.Sync.eval!(result)
      end

      acc = Map.put(acc, dense_key, dense_job)

      # Quantized variants
      for {variant, qw} <- quant_weights, reduce: acc do
        inner_acc ->
          key = "#{name}/#{variant}"

          job = fn ->
            result = Nx.dot(act, [2], qw, [1])
            Bench.Sync.eval!(result)
          end

          Map.put(inner_acc, key, job)
      end
  end

# ── Result file ──────────────────────────────────────────────────────────────

date_str = Date.utc_today() |> Date.to_string() |> String.replace("-", "")
{:ok, hostname} = :inet.gethostname()
hostname_str = List.to_string(hostname)
results_dir = Path.join([__DIR__, "results"])
results_file = Path.join(results_dir, "quantized_ops_#{date_str}_#{hostname_str}.txt")

File.mkdir_p!(results_dir)

# Tee output to both console and file
{:ok, file} = File.open(results_file, [:write, :utf8])
header = """
# emlx quantized ops benchmark
# Date: #{Date.utc_today()}
# Host: #{hostname_str}
# Hardware: see system_info below
# Stage: A1 (emlx#108 perf investigation)
"""
IO.write(file, header)
IO.puts(header)

# ── Run ───────────────────────────────────────────────────────────────────────

Benchee.run(
  jobs,
  warmup: 2,
  time: 5,
  memory_time: 0,
  formatters: [
    Benchee.Formatters.Console,
    {Benchee.Formatters.Console, file: file}
  ]
)

File.close(file)
IO.puts("\n==> Results written to #{results_file}")

# ── Per-step kernel sum (4-bit, group=64) vs A0 baseline ─────────────────────
# Collect median times for all linears at the primary variant and sum them.
# This is the "kernel headroom" check from the A1 acceptance criteria.

IO.puts("""

=== Kernel-sum vs A0 baseline ===

Compute this manually from the Benchee table above:
  1. Sum the median (ms) for all 8 linears at "4bit_g64" variant.
  2. Multiply by 28 layers → estimated kernel cost per decode step.
  3. Compare to A0 greedy median: 14.23 ms/tok (0.6B, M4 Max).

If kernel-sum-per-step ≈ per-token time → kernels are the bottleneck.
If kernel-sum-per-step << per-token time → overhead (sampler, dispatch,
  KV concat, etc.) dominates — consistent with what A0 found.
""")
