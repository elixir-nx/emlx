# bench/qwen3_e2e_bench.exs
#
# End-to-end throughput benchmark for the emlx-native Qwen3 quantized inference.
# Stage A3 of the emlx#108 quantization perf investigation.
#
# Run with:
#   cd emlx_axon
#   mix run bench/qwen3_e2e_bench.exs
#
# For 8B:
#   EMLX_QWEN3_MODEL_PATH=~/models/Qwen3-8B-MLX-4bit mix run bench/qwen3_e2e_bench.exs

Nx.default_backend({EMLX.Backend, device: :gpu})

alias EMLXAxon.Qwen3.{Loader, Generate}

model_path =
  System.get_env("EMLX_QWEN3_MODEL_PATH") ||
    Path.expand("~/models/Qwen3-0.6B-MLX-4bit")

IO.puts("==> Loading model from #{model_path} ...")
t_load_start = System.monotonic_time(:millisecond)
{:ok, state} = Loader.load(model_path)
t_load_end   = System.monotonic_time(:millisecond)
IO.puts("    Model loaded in #{t_load_end - t_load_start} ms")

IO.puts("==> Loading tokenizer ...")
{:ok, tokenizer} = Bumblebee.load_tokenizer({:local, Path.expand(model_path)})

prompt = "Write a short story about a robot who learns to love."
%{"input_ids" => input_ids} = Bumblebee.apply_tokenizer(tokenizer, prompt)
input_ids = Nx.backend_transfer(input_ids, {EMLX.Backend, device: :gpu})

max_new = 100

IO.puts("==> Warming up (20 tokens greedy) ...")
{_, _} = Generate.generate(input_ids, state, max_new_tokens: 20, sampler: :greedy)

IO.puts("==> Benchmarking #{max_new} tokens per sampler ...\n")

bench_sampler = fn sampler, label ->
  {tokens, %{timing: t}} =
    Generate.generate(input_ids, state, max_new_tokens: max_new, sampler: sampler)

  sorted_ms  = Enum.sort(t.per_token_ms)
  n          = length(sorted_ms)
  median_ms  = Enum.at(sorted_ms, div(n, 2))
  p95_ms     = Enum.at(sorted_ms, floor(n * 0.95))
  e2e_tok_s  = if t.total_ms > 0, do: Float.round(length(tokens) / t.total_ms * 1000, 1), else: 0.0
  kernel_tok_s = if median_ms > 0, do: Float.round(1000.0 / median_ms, 1), else: 0.0

  IO.puts("""
  #{label}
    prefill:       #{t.prefill_ms} ms
    median tok:    #{median_ms} ms  (#{kernel_tok_s} tok/s)
    p95 tok:       #{p95_ms} ms
    e2e tok/s:     #{e2e_tok_s} tok/s  (#{length(tokens)} tokens / #{t.total_ms} ms)
  """)

  %{
    sampler:       label,
    prefill_ms:    t.prefill_ms,
    median_ms:     median_ms,
    kernel_tok_s:  kernel_tok_s,
    e2e_tok_s:     e2e_tok_s,
    total_ms:      t.total_ms,
    n_tokens:      length(tokens)
  }
end

results = [
  bench_sampler.(:greedy,      "greedy"),
  bench_sampler.(:top_p_cpu,   "top_p_cpu (CPU sampler — matches bobby_posts)"),
  bench_sampler.(:top_p_gpu,   "top_p_gpu (defn Gumbel-max, GPU sampler)")
]

IO.puts("""
=== Summary ===

| sampler        | prefill ms | median tok ms | kernel tok/s | e2e tok/s |
|----------------|-----------|---------------|--------------|-----------|
""" <>
  Enum.map_join(results, "\n", fn r ->
    "| #{String.pad_trailing(r.sampler, 14)} | #{String.pad_leading(to_string(r.prefill_ms), 9)} | #{String.pad_leading(to_string(r.median_ms), 13)} | #{String.pad_leading(to_string(r.kernel_tok_s), 12)} | #{String.pad_leading(to_string(r.e2e_tok_s), 9)} |"
  end))

IO.puts("""

A0 reference (bobby_posts, M4 Max 64 GB):
  0.6B greedy:   14.23 ms/tok,  69.7 e2e tok/s
  0.6B top_p_cpu: 56.19 ms/tok, 17.3 e2e tok/s  (+42 ms from CPU sampler)
""")
