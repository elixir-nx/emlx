# bench/sync_boundary_timing.exs
#
# Measures the wall-clock cost of the per-token sampler-boundary
# `EMLX.eval` call (see `EMLXAxon.Qwen3.Model`'s moduledoc: "EMLX.eval is
# called once per token, at the sampler boundary, so the full lazy MLX
# graph spans all 28 layers before any CPU sync") for BOTH the dense and
# MLX-4bit quantized checkpoints, on this branch.
#
# Motivation (see native_mlx4bit_perf_investigation.md "Update" section):
# `NIF(eval)` gained a `mlx::core::synchronize()` call on this branch (absent
# on `main`). A near-empty-queue microbenchmark measured this at ~28µs/call,
# which is negligible if the cost is fixed — but `synchronize()`'s real cost
# could scale with the amount of queued async GPU work, which is far larger
# and more decomposed for quantized decode (~364 NIF calls queued per token)
# than dense (~28 calls queued per token). This script measures the eval
# call's wall time at REAL queue depth for both lanes to test that.
#
# Run:
#   cd emlx_axon
#   mix run bench/sync_boundary_timing.exs
#
# Optional env vars:
#   EMLX_QWEN3_MODEL_PATH        — MLX-4bit checkpoint (default: ~/models/Qwen3-0.6B-MLX-4bit)
#   EMLX_QWEN3_DENSE_MODEL_PATH  — dense checkpoint (default: ~/models/Qwen3-0.6B)

Nx.default_backend({EMLX.Backend, device: :gpu})

alias EMLXAxon.Qwen3.{Loader, DenseLoader, Model, Sampler}

quantized_path =
  System.get_env("EMLX_QWEN3_MODEL_PATH") || Path.expand("~/models/Qwen3-0.6B-MLX-4bit")

dense_path =
  System.get_env("EMLX_QWEN3_DENSE_MODEL_PATH") || Path.expand("~/models/Qwen3-0.6B")

prompt = "Write a short story about a robot who learns to love."
max_len = 2048
n_warmup = 5
n_timed = 30

decode_loop = fn state, input_ids, label ->
  kv_cache = Model.init_kv_cache(state, max_len)
  {logits, kv_cache} = Model.forward(input_ids, kv_cache, 0, state)
  first_token = Sampler.greedy(logits)
  EMLX.eval(EMLX.Backend.from_nx(first_token))

  [seq_len] = Nx.shape(input_ids) |> Tuple.to_list() |> tl()

  {last_id, kv_cache, cur} =
    Enum.reduce(1..n_warmup, {Nx.to_number(first_token), kv_cache, seq_len}, fn
      _step, {last_id, kv, cur} ->
        next_input =
          Nx.tensor([[last_id]], type: :s64) |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

        {logits, kv_new} = Model.forward(next_input, kv, cur, state)
        next_tok = Sampler.greedy(logits)
        EMLX.eval(EMLX.Backend.from_nx(next_tok))
        {Nx.to_number(next_tok), kv_new, cur + 1}
    end)

  IO.puts("==> [#{label}] Timing #{n_timed} decode-step eval() calls at the sampler boundary ...")

  {_last_id, _kv, _cur, samples} =
    Enum.reduce(1..n_timed, {last_id, kv_cache, cur, []}, fn
      _step, {last_id, kv, cur, acc} ->
        next_input =
          Nx.tensor([[last_id]], type: :s64) |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

        {logits, kv_new} = Model.forward(next_input, kv, cur, state)
        next_tok_ref = EMLX.Backend.from_nx(Sampler.greedy(logits))

        start = System.monotonic_time(:microsecond)
        EMLX.eval(next_tok_ref)
        elapsed = System.monotonic_time(:microsecond) - start

        next_id = next_tok_ref |> EMLX.Backend.to_nx() |> Nx.to_number()
        {next_id, kv_new, cur + 1, [elapsed | acc]}
    end)

  Enum.reverse(samples)
end

defmodule Bench do
  def median(samples) do
    sorted = Enum.sort(samples)
    Enum.at(sorted, div(length(sorted), 2))
  end

  def mean(samples), do: Enum.sum(samples) / length(samples)

  def report(label, samples) do
    IO.puts(
      "#{label}: median=#{median(samples)}us mean=#{Float.round(mean(samples), 2)}us " <>
        "min=#{Enum.min(samples)}us max=#{Enum.max(samples)}us (N=#{length(samples)})"
    )
  end
end

IO.puts("==> Loading quantized (MLX-4bit) model from #{quantized_path} ...")
{:ok, quantized_state} = Loader.load(quantized_path)
{:ok, tokenizer} = Bumblebee.load_tokenizer({:local, quantized_path})
%{"input_ids" => q_input_ids} = Bumblebee.apply_tokenizer(tokenizer, prompt)
q_input_ids = Nx.backend_transfer(q_input_ids, {EMLX.Backend, device: :gpu})

quantized_samples = decode_loop.(quantized_state, q_input_ids, "quantized")
Bench.report("quantized eval()-at-sampler-boundary", quantized_samples)

IO.puts("")
IO.puts("==> Loading dense model from #{dense_path} ...")
{:ok, dense_model_info} = Bumblebee.load_model({:local, dense_path}, type: :bf16)
{:ok, dense_state} = DenseLoader.from_model_info(dense_model_info)
%{"input_ids" => d_input_ids} = Bumblebee.apply_tokenizer(tokenizer, prompt)
d_input_ids = Nx.backend_transfer(d_input_ids, {EMLX.Backend, device: :gpu})

dense_samples = decode_loop.(dense_state, d_input_ids, "dense")
Bench.report("dense eval()-at-sampler-boundary", dense_samples)

IO.puts("")
IO.puts("==> Summary (µs/call, median):")
IO.puts("    quantized: #{Bench.median(quantized_samples)}")
IO.puts("    dense:     #{Bench.median(dense_samples)}")
IO.puts("    Δ (quantized - dense): #{Bench.median(quantized_samples) - Bench.median(dense_samples)}us")
