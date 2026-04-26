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

# ── A7-1: Transpose hypothesis ────────────────────────────────────────────────
#
# Hypothesis: transpose=false (Bumblebee {in,out} layout) is slower than
# transpose=true ({out,in} layout) on the Metal quantized_matmul kernel.
# If confirmed, A7-2 pre-transposes Bumblebee kernels to fix Gap A.
#
# Uses q_proj at Qwen3-0.6B decode shape (act {1,1,1024}, out 2048).

IO.puts("\n=== A7-1 Transpose hypothesis (q_proj, 4-bit g64, decode shape) ===\n")

{_, _, q_w_native, q_quant_weights, q_act_3d} =
  Enum.find(weights, fn {n, _, _, _, _} -> n == :q_proj end)

# Native path: {out, in} = {2048, 1024}, contract on axis [1] (last) → transpose=true
{_, q_qw_true} = Enum.find(q_quant_weights, fn {v, _} -> v == "4bit_g64" end)

# Bumblebee path: {in, out} = {1024, 2048}, contract on axis [0] (first) → transpose=false
q_w_in_out  = Nx.transpose(q_w_native)
q_qw_false  = EMLX.Quantization.quantize(q_w_in_out, type: {:s, 4}, group_size: 64)

hyp_runs = 500

warmup_hyp = fn f ->
  for _ <- 1..50, do: Bench.Sync.eval!(f.())
end

time_hyp = fn f ->
  t0 = System.monotonic_time(:microsecond)
  for _ <- 1..hyp_runs, do: f.()
  Bench.Sync.eval!(f.())
  t1 = System.monotonic_time(:microsecond)
  Float.round((t1 - t0) / (hyp_runs + 1), 2)
end

fn_true  = fn -> Nx.dot(q_act_3d, [2], q_qw_true,  [1]) end
fn_false = fn -> Nx.dot(q_act_3d, [2], q_qw_false, [0]) end

IO.puts("Warming up both paths...")
warmup_hyp.(fn_true)
warmup_hyp.(fn_false)

IO.puts("Timing transpose=true  ({out,in} layout, axis [last])...")
us_true = time_hyp.(fn_true)

IO.puts("Timing transpose=false ({in,out} layout, axis [0])...")
us_false = time_hyp.(fn_false)

ratio = Float.round(us_false / max(us_true, 0.001), 2)

IO.puts("""

q_proj 4-bit g64 (decode shape, #{hyp_runs} iters):
  transpose=true  (native {out,in}):    #{us_true} µs/op
  transpose=false (bumblebee {in,out}): #{us_false} µs/op
  slowdown ratio (false/true):          #{ratio}×

=> #{if ratio > 1.3, do: "CONFIRMED: transpose=false is #{ratio}× slower. A7-2 is high priority.", else: "NOT CONFIRMED: format is not the bottleneck (ratio #{ratio}×). Profile before A7-2."}
""")

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
