# bench/nif_overhead_microbench.exs
#
# Cross-branch NIF dispatch-overhead microbenchmark.
#
# Isolates pure per-call NIF dispatch cost from the model/benchmark harness,
# to confirm or falsify the hypothesis (see native_mlx4bit_perf_investigation.md)
# that the `FINE_ASYNC_NIF` bridge introduced on pv-feat/lowering-compiler adds
# per-call overhead vs `main`'s hand-rolled `ASYNC_NIF` macros.
#
# Three legs are benchmarked on a small fixed-size tensor:
#   - `EMLX.Fast.rms_norm/3`   — moved to `FINE_ASYNC_NIF` on this branch.
#   - `EMLX.quantized_matmul/2` — still old-style `ASYNC_NIF`, unchanged; the
#                                  control used to distinguish "universal
#                                  CallerPidGuard overhead" from "fine-bridge
#                                  specific overhead" (both wrap through
#                                  `async_dispatch<OP>` in emlx_async.hpp).
#   - `EMLX.eval/1` alone (re-evaling an already-computed tensor) — an
#     **eval-only control**. `NIF(eval)` itself differs between branches
#     (`pv-feat/lowering-compiler` added a `mlx::core::synchronize()` call
#     after `mlx::core::eval()`; `main` does not have it), and every timed
#     iteration below calls `EMLX.eval` to force dispatch — so any Δ in
#     `eval()` itself confounds the rms_norm/quantized_matmul comparison.
#     This leg isolates that cost so it can be subtracted out.
#
# Methodology:
#   - ~100 warmup iterations per NIF (separate, since they hit different
#     Metal kernels/pipeline-state caches).
#   - N = 10,000 timed iterations per NIF.
#   - `EMLX.eval/1` is called after every timed call to force MLX (which is
#     lazy) to actually dispatch + compute, not just enqueue.
#   - Per-call timings are recorded individually (not just total/N), so we
#     can report median (p50) instead of mean to dodge GC/scheduler jitter.
#
# Run:
#   cd emlx_axon
#   mix run bench/nif_overhead_microbench.exs
#
# Run on both branches (e.g. `main` and `pv-feat/lowering-compiler`) and
# compare the reported µs/call numbers.

Nx.default_backend({EMLX.Backend, device: :gpu})

warmup_n = 100
n = 10_000

# ── Fixture tensors ───────────────────────────────────────────────────────────

hidden = 1024
x = Nx.Random.key(0) |> Nx.Random.normal(shape: {1, hidden}) |> elem(0)
weight = Nx.Random.key(1) |> Nx.Random.normal(shape: {hidden}) |> elem(0)
eps = 1.0e-6

activation = Nx.Random.key(2) |> Nx.Random.normal(shape: {1, hidden}) |> elem(0)
dense_w = Nx.Random.key(3) |> Nx.Random.normal(shape: {hidden, hidden}) |> elem(0)
qw = EMLX.quantize(dense_w, type: {:s, 4}, group_size: 64)

EMLX.eval(EMLX.Backend.from_nx(x))
EMLX.eval(EMLX.Backend.from_nx(weight))
EMLX.eval(EMLX.Backend.from_nx(activation))

# ── Timing helper ─────────────────────────────────────────────────────────────

defmodule Bench do
  def time_calls(n, fun) do
    for _ <- 1..n do
      start = System.monotonic_time(:microsecond)
      fun.()
      System.monotonic_time(:microsecond) - start
    end
  end

  def median(samples) do
    sorted = Enum.sort(samples)
    len = length(sorted)
    Enum.at(sorted, div(len, 2))
  end

  def mean(samples), do: Enum.sum(samples) / length(samples)

  def report(label, samples) do
    IO.puts(
      "#{label}: median=#{median(samples)}us mean=#{Float.round(mean(samples), 2)}us " <>
        "min=#{Enum.min(samples)}us max=#{Enum.max(samples)}us (N=#{length(samples)})"
    )
  end
end

# ── rms_norm ───────────────────────────────────────────────────────────────────

IO.puts("==> Warming up EMLX.Fast.rms_norm/3 (#{warmup_n} iterations) ...")

for _ <- 1..warmup_n do
  EMLX.Fast.rms_norm(x, weight, eps) |> EMLX.Backend.from_nx() |> EMLX.eval()
end

IO.puts("==> Timing EMLX.Fast.rms_norm/3 (#{n} iterations) ...")

rms_norm_samples =
  Bench.time_calls(n, fn ->
    EMLX.Fast.rms_norm(x, weight, eps) |> EMLX.Backend.from_nx() |> EMLX.eval()
  end)

Bench.report("rms_norm (FINE_ASYNC_NIF)", rms_norm_samples)

# ── quantized_matmul (control) ─────────────────────────────────────────────────

IO.puts("==> Warming up EMLX.quantized_matmul/2 (#{warmup_n} iterations) ...")

for _ <- 1..warmup_n do
  EMLX.quantized_matmul(activation, qw) |> EMLX.Backend.from_nx() |> EMLX.eval()
end

IO.puts("==> Timing EMLX.quantized_matmul/2 (#{n} iterations) ...")

quantized_matmul_samples =
  Bench.time_calls(n, fn ->
    EMLX.quantized_matmul(activation, qw) |> EMLX.Backend.from_nx() |> EMLX.eval()
  end)

Bench.report("quantized_matmul (ASYNC_NIF, control)", quantized_matmul_samples)

# ── eval-only (isolates NIF(eval)'s own cost, e.g. added `synchronize()`) ──────

already_evaled = EMLX.Backend.from_nx(activation)
EMLX.eval(already_evaled)

IO.puts("==> Warming up EMLX.eval/1 alone (#{warmup_n} iterations) ...")

for _ <- 1..warmup_n do
  EMLX.eval(already_evaled)
end

IO.puts("==> Timing EMLX.eval/1 alone (#{n} iterations) ...")

eval_only_samples = Bench.time_calls(n, fn -> EMLX.eval(already_evaled) end)

Bench.report("eval-only (re-eval of already-computed tensor)", eval_only_samples)

IO.puts("")
IO.puts("==> Summary (µs/call, median):")
IO.puts("    rms_norm:          #{Bench.median(rms_norm_samples)}")
IO.puts("    quantized_matmul:  #{Bench.median(quantized_matmul_samples)}")
IO.puts("    eval-only:         #{Bench.median(eval_only_samples)}")
