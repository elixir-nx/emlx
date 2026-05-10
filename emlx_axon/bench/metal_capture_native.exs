# bench/metal_capture_native.exs
#
# Captures a Metal GPU trace of the native emlx-validation Qwen3 decode path.
# The resulting .gputrace file can be opened in Xcode's GPU Debugger
# (File → Open) to inspect per-kernel timing and Metal command counts.
#
# Run:
#   cd emlx_axon
#   mix run bench/metal_capture_native.exs
#
# Optional env vars:
#   EMLX_QWEN3_MODEL_PATH  — path to Qwen3-0.6B-MLX-4bit checkout
#                            (default: ~/models/Qwen3-0.6B-MLX-4bit)
#   METAL_CAPTURE_PATH     — output .gputrace path
#                            (default: /tmp/native_decode.gputrace)

Nx.default_backend({EMLX.Backend, device: :gpu})

alias EMLXAxon.Qwen3.{Loader, Model, Sampler}

model_path =
  System.get_env("EMLX_QWEN3_MODEL_PATH") ||
    Path.expand("~/models/Qwen3-0.6B-MLX-4bit")

capture_path =
  System.get_env("METAL_CAPTURE_PATH") ||
    "/tmp/native_decode.gputrace"

IO.puts("==> Loading model from #{model_path} ...")
{:ok, state} = Loader.load(model_path)
IO.puts("==> Loading tokenizer ...")
{:ok, tokenizer} = Bumblebee.load_tokenizer({:local, model_path})

prompt = "Write a short story about a robot who learns to love."
%{"input_ids" => input_ids} = Bumblebee.apply_tokenizer(tokenizer, prompt)
input_ids = Nx.backend_transfer(input_ids, {EMLX.Backend, device: :gpu})

max_len = 2048
kv_cache = Model.init_kv_cache(state, max_len)

# ── Prefill (produces first token and primes the JIT) ────────────────────────

IO.puts("==> Prefill ...")
{logits, kv_cache} = Model.forward(input_ids, kv_cache, 0, state)
first_token = Sampler.greedy(logits)
EMLX.eval(EMLX.Backend.from_nx(first_token))

[seq_len] = Nx.shape(input_ids) |> Tuple.to_list() |> tl()
current_len = seq_len

# ── Warmup decode (5 tokens — ensures all JIT paths are compiled) ─────────────

IO.puts("==> Warming up 5 decode tokens ...")
{_tokens, _cache, _current_len} =
  Enum.reduce(1..5, {[Nx.to_number(first_token)], kv_cache, current_len}, fn
    _step, {[last_id | _] = acc, kv, cur} ->
      next_input =
        Nx.tensor([[last_id]], type: :s64)
        |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

      {logits, kv_new} = Model.forward(next_input, kv, cur, state)
      next_tok = Sampler.greedy(logits)
      EMLX.eval(EMLX.Backend.from_nx(next_tok))
      {[Nx.to_number(next_tok) | acc], kv_new, cur + 1}
  end)

# Re-read the warmed-up cache and last token from the state after warmup.
# Redo from scratch with the same warmup so we have a consistent state.
# (Simplest approach: re-prefill and re-warmup, discarding timing.)
kv_cache = Model.init_kv_cache(state, max_len)
{logits, kv_cache} = Model.forward(input_ids, kv_cache, 0, state)
first_token = Sampler.greedy(logits)
EMLX.eval(EMLX.Backend.from_nx(first_token))
current_len = seq_len

{_tokens, kv_cache, current_len} =
  Enum.reduce(1..5, {[Nx.to_number(first_token)], kv_cache, current_len}, fn
    _step, {[last_id | _] = acc, kv, cur} ->
      next_input =
        Nx.tensor([[last_id]], type: :s64)
        |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

      {logits, kv_new} = Model.forward(next_input, kv, cur, state)
      next_tok = Sampler.greedy(logits)
      EMLX.eval(EMLX.Backend.from_nx(next_tok))
      {[Nx.to_number(next_tok) | acc], kv_new, cur + 1}
  end)

last_token_id = Nx.to_number(first_token)

# ── Metal capture: 3 decode tokens ────────────────────────────────────────────

IO.puts("==> Starting Metal capture → #{capture_path}")
EMLX.metal_start_capture(capture_path)

n_capture = 3
{_toks, _kv, _cur} =
  Enum.reduce(1..n_capture, {[last_token_id], kv_cache, current_len}, fn
    step, {[last_id | _] = acc, kv, cur} ->
      next_input =
        Nx.tensor([[last_id]], type: :s64)
        |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

      {logits, kv_new} = Model.forward(next_input, kv, cur, state)
      next_tok = Sampler.greedy(logits)
      EMLX.eval(EMLX.Backend.from_nx(next_tok))

      IO.puts("    captured token #{step}/#{n_capture}: #{Nx.to_number(next_tok)}")
      {[Nx.to_number(next_tok) | acc], kv_new, cur + 1}
  end)

EMLX.metal_stop_capture()
IO.puts("==> Metal capture saved to: #{capture_path}")
IO.puts("    Open in Xcode → File → Open to inspect GPU timeline.")
