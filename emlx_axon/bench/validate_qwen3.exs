# emlx_axon/bench/validate_qwen3.exs
#
# Throughput benchmark for EMLX.Axon.rewrite/2 on Qwen3-0.6B-MLX-4bit.
# Measures e2e tok/s for:
#   - Bumblebee + stock Axon graph (no EMLX.Axon.rewrite)
#   - Bumblebee + full rewrite (default: all rewrites, including :native_attention)
#   - EMLX.Native.TextGeneration (native bypass)
#
# Run from the emlx_axon directory:
#
#   EMLX_QWEN3_MODEL_PATH=~/models/Qwen3-0.6B-MLX-4bit \
#     mix run bench/validate_qwen3.exs
#
# Environment variables (all optional):
#   EMLX_QWEN3_MODEL_PATH  — local model dir or HF repo id (default: ~/models/Qwen3-0.6B-MLX-4bit)
#   EMLX_QWEN3_MAX_NEW             — max new tokens per run (default: 60)
#   EMLX_QWEN3_BENCH_RUNS          — number of timed runs (default: 5)
#   EMLX_QWEN3_WARMUP_RUNS         — number of warmup runs before timing (default: 2)
#   EMLX_QWEN3_SEQUENCE_LENGTH     — sequence_length for Bumblebee compile (default: 1024)
#   EMLX_QWEN3_NATIVE_PROFILE_TIMING — set to "0" to disable per-token monotonic_time in native decode (default: on)

Nx.default_backend({EMLX.Backend, device: :gpu})

# ── Config ────────────────────────────────────────────────────────────────────

model_path_raw =
  System.get_env("EMLX_QWEN3_MODEL_PATH") || "~/models/Qwen3-0.6B-MLX-4bit"

model_source =
  if File.exists?(Path.expand(model_path_raw)) do
    {:local, Path.expand(model_path_raw)}
  else
    {:hf, model_path_raw}
  end

max_new      = String.to_integer(System.get_env("EMLX_QWEN3_MAX_NEW",         "60"))
bench_runs   = String.to_integer(System.get_env("EMLX_QWEN3_BENCH_RUNS",      "5"))
warmup_runs  = String.to_integer(System.get_env("EMLX_QWEN3_WARMUP_RUNS",     "2"))
seq_len      = String.to_integer(System.get_env("EMLX_QWEN3_SEQUENCE_LENGTH", "1024"))

native_profile_timing? = System.get_env("EMLX_QWEN3_NATIVE_PROFILE_TIMING") != "0"

# Qwen3 instruct chat template — long enough that EOS won't hit within max_new tokens.
prompt =
  "<|im_start|>user\nList twenty programming languages with a one-line description each.<|im_end|>\n<|im_start|>assistant\n"

IO.puts("""
=== emlx_axon/bench/validate_qwen3.exs ===
model:    #{inspect(model_source)}
max_new:  #{max_new}  bench_runs: #{bench_runs}  warmup_runs: #{warmup_runs}
seq_len:  #{seq_len}
""")

# ── Load ──────────────────────────────────────────────────────────────────────

IO.puts("==> Loading model structure ...")
t0 = System.monotonic_time(:millisecond)
# Load model spec/structure only — params are loaded separately below because
# the MLX-4bit checkpoint stores weights in a packed uint32 format that
# Bumblebee's safetensors loader cannot dequantize on its own.
{:ok, model_info} = Bumblebee.load_model(model_source,
  backend: {EMLX.Backend, device: :gpu},
  type: :bf16)
t1 = System.monotonic_time(:millisecond)
IO.puts("    model structure loaded in #{t1 - t0} ms")

IO.puts("==> Loading MLX-4bit params (dequantize → Bumblebee layout → re-quantize) ...")
t2a = System.monotonic_time(:millisecond)
params = EMLX.Axon.MLX4BitParams.load(Path.expand(model_path_raw))
params = EMLX.Axon.QuantizeParams.quantize(params)
model_info = %{model_info | params: params}
t2b = System.monotonic_time(:millisecond)
IO.puts("    params loaded in #{t2b - t2a} ms")

IO.puts("==> Loading tokenizer ...")
{:ok, tokenizer} = Bumblebee.load_tokenizer(model_source)

IO.puts("==> Loading generation config ...")
{:ok, generation_config} = Bumblebee.load_generation_config(model_source)
generation_config = Bumblebee.configure(generation_config,
  max_new_tokens: max_new,
  strategy: %{type: :greedy_search}
)

# ── Rewrite ───────────────────────────────────────────────────────────────────

model_base = model_info.model

IO.puts("==> Applying EMLX.Axon.rewrite/2 (all rewrites — best performing path) ...")
t3 = System.monotonic_time(:millisecond)
model_rewritten = EMLX.Axon.rewrite(model_base)
t4 = System.monotonic_time(:millisecond)
IO.puts("    rewrite done in #{t4 - t3} ms")

# ── Build serving ─────────────────────────────────────────────────────────────

IO.puts("==> Building Bumblebee.Text.generation serving (base — no rewrite) ...")
serving_base =
  Bumblebee.Text.generation(
    %{model_info | model: model_base},
    tokenizer,
    generation_config,
    compile: [batch_size: 1, sequence_length: seq_len],
    defn_options: [compiler: EMLX]
  )

IO.puts("==> Building Bumblebee.Text.generation serving (rewritten) ...")
serving_rewrite =
  Bumblebee.Text.generation(
    %{model_info | model: model_rewritten},
    tokenizer,
    generation_config,
    compile: [batch_size: 1, sequence_length: seq_len],
    defn_options: [compiler: EMLX]
  )

# ── Benchmark helpers ─────────────────────────────────────────────────────────

defmodule Bench do
  def run_serving(serving, prompt, extract_fn) do
    t_start = System.monotonic_time(:millisecond)
    result  = Nx.Serving.run(serving, prompt)
    t_end   = System.monotonic_time(:millisecond)
    {text, n_tokens} = extract_fn.(result)
    elapsed_ms = t_end - t_start
    tok_s = if elapsed_ms > 0, do: Float.round(n_tokens / elapsed_ms * 1000.0, 1), else: 0.0
    {elapsed_ms, n_tokens, tok_s, text}
  end

  def warmup(label, serving, prompt, extract_fn, runs) do
    IO.puts("\n==> [#{label}] Warmup (#{runs} run(s), not timed) ...")
    for i <- 1..runs do
      {ms, n, _, text} = run_serving(serving, prompt, extract_fn)
      preview = text |> String.slice(0, 60) |> String.replace("\n", " ")
      IO.puts("    warmup #{i}: #{n} tokens / #{ms} ms  \"#{preview}...\"")
    end
  end

  def bench(label, serving, prompt, extract_fn, runs) do
    IO.puts("\n==> [#{label}] Benchmark (#{runs} run(s)) ...")
    for i <- 1..runs do
      {ms, n, tok_s, _} = run_serving(serving, prompt, extract_fn)
      IO.puts("    run #{i}: #{n} tokens / #{ms} ms = #{tok_s} tok/s")
      %{elapsed_ms: ms, n_tokens: n, tok_s: tok_s}
    end
  end

  def stats(label, results) do
    toks_list = Enum.map(results, & &1.tok_s)
    sorted    = Enum.sort(toks_list)
    n         = length(sorted)
    median    = Enum.at(sorted, div(n, 2))
    mean      = Float.round(Enum.sum(toks_list) / n, 1)
    min_v     = List.first(sorted)
    max_v     = List.last(sorted)
    variance  = Enum.reduce(toks_list, 0.0, fn x, acc -> acc + (x - mean) * (x - mean) end) / n
    stddev    = Float.round(:math.sqrt(variance), 1)
    IO.puts("    #{label}: median=#{median}  mean=#{mean}±#{stddev}  min/max=#{min_v}/#{max_v} tok/s")
    %{label: label, median: median, mean: mean, stddev: stddev, min: min_v, max: max_v}
  end
end

# ── Bumblebee paths (base vs rewrite) ───────────────────────────────────────

# Bumblebee.Text.generation returns %{results: [%{text: ..., token_summary: ...}]}
bb_extract = fn %{results: [%{text: text, token_summary: summary}]} ->
  {text, summary.output}
end

Bench.warmup("bb base", serving_base, prompt, bb_extract, warmup_runs)
base_bb_results = Bench.bench("bb base", serving_base, prompt, bb_extract, bench_runs)

Bench.warmup("bb+rewrite", serving_rewrite, prompt, bb_extract, warmup_runs)
bb_results = Bench.bench("bb+rewrite", serving_rewrite, prompt, bb_extract, bench_runs)

# ── Native TextGeneration path ────────────────────────────────────────────────

IO.puts("\n==> Loading EMLX.Native.TextGeneration (native 28-layer bypass) ...")
t_native = System.monotonic_time(:millisecond)
native_serving =
  EMLX.Native.TextGeneration.from_mlx4bit(
    Path.expand(model_path_raw),
    tokenizer,
    max_new_tokens: max_new,
    sampler: :greedy,
    profile_timing: native_profile_timing?
  )
IO.puts("    loaded in #{System.monotonic_time(:millisecond) - t_native} ms")

# Native serving includes :num_tokens (tensor seq dim); avoids tokenizer round-trip for bench throughput.
native_extract = fn
  %{results: [%{generated_text: text, num_tokens: n}]} ->
    {text, n}

  %{results: [%{generated_text: text}]} ->
    %{"input_ids" => ids} =
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        Bumblebee.apply_tokenizer(tokenizer, text)
      end)

    {text, elem(Nx.shape(ids), 1)}
end

Bench.warmup("native", native_serving, prompt, native_extract, warmup_runs)
native_results = Bench.bench("native", native_serving, prompt, native_extract, bench_runs)

# ── Summary ───────────────────────────────────────────────────────────────────

IO.puts("""

=== Summary (#{bench_runs} runs, #{max_new} max_new_tokens, Qwen3-0.6B-MLX-4bit) ===
""")

base_stats   = Bench.stats("bb base    (Bumblebee, no rewrite)           ", base_bb_results)
bb_stats     = Bench.stats("bb+rewrite (Bumblebee + EMLX.Axon.rewrite)   ", bb_results)
native_stats = Bench.stats("native     (EMLX.Native.TextGeneration)      ", native_results)

rewrite_vs_base =
  if base_stats.median > 0,
    do: Float.round(bb_stats.median / base_stats.median, 2),
    else: :n_a

native_vs_base =
  if base_stats.median > 0,
    do: Float.round(native_stats.median / base_stats.median, 2),
    else: :n_a

native_vs_rewrite =
  if bb_stats.median > 0,
    do: Float.round(native_stats.median / bb_stats.median, 2),
    else: :n_a

IO.puts("""
  bb+rewrite / bb base:  #{rewrite_vs_base}×
  native / bb base:      #{native_vs_base}×
  native / bb+rewrite:   #{native_vs_rewrite}×
""")
