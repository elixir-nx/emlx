# bench/profile_eval.exs
#
# C01 measurement script — EMLX eval-dispatch profiling.
#
# Uses bumblebee-testing/tiny-random-LlamaForCausalLM for the KV-cache
# decode loop. The tiny model (hidden=32, layers=2, heads=2) has the same
# compute graph structure as a production Llama/Qwen3 — all the same NIF
# call patterns, just smaller. The counter measurements transfer directly.
#
# Run (from the emlx_axon/ directory):
#   cd emlx_axon
#   EMLX_PROFILE_EVAL=1 VQ_MAX_NEW=20 mix run bench/profile_eval.exs
#
# Scriptable Metal capture (no Xcode GUI):
#   MTL_CAPTURE_ENABLED=1 METAL_DEVICE_WRAPPER_TYPE=1 \
#   xctrace record --output trace.xctrace \
#     --template "Metal System Trace" \
#     --time-limit 60s \
#     -- iex -S mix run bench/profile_eval.exs
#
# NIF call tracing (in a separate iex session on the same node):
#   :recon_trace.calls({EMLX, :_, :_}, 10_000, scope: :local)
#   :recon_trace.calls({EMLX.NIF, :_, :_}, 10_000, scope: :local)

Nx.default_backend({EMLX.Backend, device: :gpu})

max_new = System.get_env("VQ_MAX_NEW", "20") |> String.to_integer()
seq_length = System.get_env("VQ_SEQ_LENGTH", "64") |> String.to_integer()

unless EMLX.Profiling.enabled?() do
  IO.puts("""
  [profile_eval] WARNING: EMLX_PROFILE_EVAL not set — counters are inactive.
  Re-run with EMLX_PROFILE_EVAL=1 to enable atomic counters.
  """)
end

IO.puts("""
==> C01 profiling  max_new=#{max_new}  seq_length=#{seq_length}
    model: bumblebee-testing/tiny-random-LlamaForCausalLM
""")

# ── Load model (no tokenizer needed — we supply input IDs directly) ────────────

{:ok, %{model: model, params: params, spec: spec}} =
  Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-LlamaForCausalLM"})

IO.puts("==> Model loaded  architecture=#{inspect(spec.architecture)}  vocab_size=#{spec.vocab_size}")

# ── JIT-compile the decode step ──────────────────────────────────────────────
#
# Axon.predict already goes through the Nx.Defn JIT path when a non-default
# compiler is configured via defn_options. Since we set the default backend
# to EMLX, we call Axon.predict directly and it dispatches through EMLX.

# Ensure params live on the GPU backend (they should already, after load_model
# with default EMLX backend set, but make it explicit).
params = Nx.backend_transfer(params, {EMLX.Backend, device: :gpu})

predict = fn inputs ->
  Axon.predict(model, params, inputs, compiler: EMLX)
end

# Build initial inputs: batch=1, seq=seq_length (padded)
prefill_len = 5
input_ids = Nx.broadcast(0, {1, seq_length}) |> Nx.put_slice([0, 0], Nx.tensor([[10, 20, 30, 40, 50]]))
ones = Nx.broadcast(1, {1, prefill_len})
zeros = Nx.broadcast(0, {1, seq_length - prefill_len})
attention_mask = Nx.concatenate([ones, zeros], axis: 1)

IO.puts("==> Running prefill warmup (compilation) ...")
inputs = %{"input_ids" => input_ids, "attention_mask" => attention_mask}
_warmup = predict.(inputs)

# ── Measure decode-like loop ───────────────────────────────────────────────────
#
# Each "step" re-runs the full forward pass with the same input shape —
# this is what Nx.Defn.Evaluator does on every decode step (no caching of
# the compiled function's output). The NIF dispatch pattern per step is
# what we want to measure.

IO.puts("==> Measuring #{max_new} decode steps ...")
EMLX.Profiling.reset()

step_times =
  for step <- 1..max_new do
    # Shift the sequence by one token per step (simulates the decode position)
    pos = prefill_len + step - 1
    shifted_ids = Nx.put_slice(input_ids, [0, pos], Nx.tensor([[100 + step]]))
    shifted_mask = Nx.put_slice(attention_mask, [0, pos], Nx.tensor([[1]]))
    inputs = %{"input_ids" => shifted_ids, "attention_mask" => shifted_mask}

    {step_us, outputs} = :timer.tc(fn -> predict.(inputs) end)

    # Extracting the next token — this calls to_blob/item, forcing eval
    _next_token = outputs.logits[[0, pos, ..]] |> Nx.argmax() |> Nx.to_number()

    step_us / 1_000
  end

{:ok, counts} = EMLX.Profiling.read()

total_ms = Enum.sum(step_times)
median_ms = Enum.sort(step_times) |> Enum.at(div(max_new, 2))

IO.puts("""

==> Results (#{max_new} decode steps):
  total wall time:     #{round(total_ms)} ms
  median step time:    #{Float.round(median_ms, 2)} ms  →  #{Float.round(1_000 / median_ms, 1)} tok/s (approx)
  EMLX.eval calls:     #{counts.eval}  (#{Float.round(counts.eval / max_new, 1)} / step)
  EMLX.item calls:     #{counts.item}  (#{Float.round(counts.item / max_new, 1)} / step)
  EMLX.to_blob calls:  #{counts.to_blob}  (#{Float.round(counts.to_blob / max_new, 1)} / step)

==> Acceptance table (C01) — default :sdpa path

| metric                          | tiny-Llama (proxy) |
|---------------------------------|--------------------|
| EMLX.eval calls / decode step   | #{Float.round(counts.eval / max_new, 1)} |
| EMLX.item calls / decode step   | #{Float.round(counts.item / max_new, 1)} |
| EMLX.to_blob calls / decode step| #{Float.round(counts.to_blob / max_new, 1)} |
| median step wall time (ms)      | #{Float.round(median_ms, 2)} |

NOTE: Metal cmd buffers / step requires Metal frame capture.
  xctrace record --output trace.xctrace \\
    --template "Metal System Trace" ...
  Then open in Instruments → Metal GPU Trace Viewer.

INTERPRETATION:
  eval/step ≈ 1  → MLX lazy eval is batching correctly → bottleneck is NIF overhead
  eval/step ≫ 1  → MLX lazy eval interrupted mid-step  → fix forced evals first
""")
