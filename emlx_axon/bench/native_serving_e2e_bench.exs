# bench/native_serving_e2e_bench.exs
#
# Faithfully reproduces the ORIGINAL reported `:native` lane regression by
# exercising the actual production code path — `EMLXAxon.TextGeneration
# .from_mlx4bit/3` + `Nx.Serving.run/2` — rather than calling `Generate
# .generate/3` directly (which defaults to `host_sync: :per_token`, unlike
# `serving/3`'s default `host_sync: {:chunk, min(max_new, 31)}`).
#
# This distinction matters: `sync_boundary_timing.exs` and
# `nif_overhead_microbench.exs` both forced an `EMLX.eval`/host-sync after
# every single token/op, but the real chunked default lets the lazy MLX
# graph grow across up to 31 tokens (~11,000+ queued NIF calls for the
# quantized path) before one host sync — a much larger queue depth than
# either of those microbenchmarks tested. This script measures at the real
# queue depth to establish whether the regression is even still
# reproducible, before further isolating its cause.
#
# Run:
#   cd emlx_axon
#   mix run bench/native_serving_e2e_bench.exs
#
# Optional env vars:
#   EMLX_QWEN3_MODEL_PATH — MLX-4bit checkpoint (default: ~/models/Qwen3-0.6B-MLX-4bit)

Nx.default_backend({EMLX.Backend, device: :gpu})

alias EMLXAxon.Qwen3.Loader
alias EMLXAxon.TextGeneration

model_path =
  System.get_env("EMLX_QWEN3_MODEL_PATH") || Path.expand("~/models/Qwen3-0.6B-MLX-4bit")

prompt = "Write a short story about a robot who learns to love."
max_new_tokens = 60
n_runs = 6

IO.puts("==> Loading model + tokenizer from #{model_path} ...")
{:ok, state} = Loader.load(model_path)
{:ok, tokenizer} = Bumblebee.load_tokenizer({:local, model_path})

# `TextGeneration.run/4` uses the same `default_host_sync/1` ({:chunk, N})
# as `serving/3`/`from_mlx4bit/3`, without the Nx.Serving/Bumblebee-batch
# preprocessing overhead — same decode-loop code path, cleaner timing.
run = fn ->
  TextGeneration.run(tokenizer, state, prompt,
    max_new_tokens: max_new_tokens,
    sampler: :greedy,
    return_timing: true
  )
end

IO.puts("==> Warming up (2 runs) ...")
_ = run.()
_ = run.()

IO.puts("==> Timing #{n_runs} runs of #{max_new_tokens} tokens via TextGeneration.run/4 ...")

results =
  for i <- 1..n_runs do
    result = run.()
    timing = Map.fetch!(result, :timing)
    tok_s = Float.round(max_new_tokens / timing.total_ms * 1000, 1)

    IO.puts(
      "  run #{i}: total_ms=#{timing.total_ms} tok/s=#{tok_s} prefill_ms=#{timing.prefill_ms}"
    )

    tok_s
  end

IO.puts("")
IO.puts("==> Summary: tok/s per run: #{inspect(results)}")
IO.puts("    median tok/s: #{Enum.sort(results) |> Enum.at(div(length(results), 2))}")
