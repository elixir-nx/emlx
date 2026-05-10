defmodule EMLXAxon.Qwen3.Generate do
  @moduledoc """
  Autoregressive token generation loop.

  `EMLX.eval` is called once per token, at the sampler boundary, so the
  full 28-layer forward runs as a single lazy MLX graph before any CPU
  synchronisation.

  ## Usage

      {:ok, state}     = Loader.load(model_path)
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-0.6B"})

      encoded   = Bumblebee.apply_tokenizer(tokenizer, "Hello")
      input_ids = encoded["input_ids"]

      {tokens, timing} = Generate.generate(input_ids, state,
        max_new_tokens: 100,
        sampler: :greedy
      )
  """

  alias EMLXAxon.Qwen3.{Model, Sampler}

  @cpu_backend Nx.BinaryBackend

  @default_max_new_tokens 100
  @default_max_len 2048

  @doc """
  Run the autoregressive generation loop.

  ## Options

  - `:max_new_tokens` — max number of tokens to generate (default 100)
  - `:max_len`        — KV cache allocation size (default 2048); ignored when `:kv_cache` is given
  - `:kv_cache`       — pre-allocated KV cache from `Model.init_kv_cache/2`; if provided,
                        `Model.init_kv_cache/2` is skipped. The cache is used as-is — callers
                        are responsible for ensuring it is clean (stale K/V beyond `current_len`
                        is never read because `Model.forward/4` slices to the valid prefix).
  - `:sampler`        — `:greedy | :top_p_cpu | :top_p_gpu` (default `:greedy`)
  - `:temperature`    — float, passed to samplers that use it (default 0.95)
  - `:top_p`          — float, passed to nucleus samplers (default 0.9)
  - `:rng_key`        — `Nx.Random.key/1`, used by `:top_p_gpu` (split each step via
                        `Nx.Random.split/2`; avoids host time + transfer per token)
  - `:profile_timing` — when `true` (default), record `per_token_ms` decode samples via
                        `System.monotonic_time/1` each step; set `false` to skip that overhead
                        (prefill/total wall time is still measured)

  ## Returns

  `{generated_token_ids, %{timing: timing_map}}` where `timing_map` has:
  - `:prefill_ms`     — first-token time in milliseconds
  - `:per_token_ms`   — list of per-token decode times (median ≈ steady-state)
  - `:total_ms`       — wall time for the whole call
  """
  @spec generate(Nx.Tensor.t(), Model.State.t(), keyword()) ::
          {[non_neg_integer()], map()}
  def generate(input_ids, %Model.State{} = state, opts \\ []) do
    max_new = Keyword.get(opts, :max_new_tokens, @default_max_new_tokens)
    max_len = Keyword.get(opts, :max_len, @default_max_len)
    sampler = Keyword.get(opts, :sampler, :greedy)
    temp = Keyword.get(opts, :temperature, 0.95)
    top_p = Keyword.get(opts, :top_p, 0.9)
    rng_key = Keyword.get(opts, :rng_key, Nx.Random.key(42))
    profile_timing? = Keyword.get(opts, :profile_timing, true)

    kv_cache =
      case Keyword.fetch(opts, :kv_cache) do
        {:ok, prealloc} -> prealloc
        :error -> Model.init_kv_cache(state, max_len)
      end

    gpu = {EMLX.Backend, device: :gpu}

    # Reused {1,1} token buffer for decode steps (avoids reallocating the slot each token).
    decode_slot =
      if max_new > 1 do
        Nx.broadcast(Nx.tensor(0, type: :s64, backend: @cpu_backend), {1, 1})
        |> Nx.backend_transfer(gpu)
      else
        nil
      end

    # Prefill: pass the full prompt in one forward
    t0 = System.monotonic_time(:millisecond)

    {rng_key, first_gpu_key} = advance_rng_for_gpu(sampler, rng_key)

    {logits, kv_cache} = Model.forward(input_ids, kv_cache, 0, state)
    first_token = sample(logits, sampler, temp, top_p, first_gpu_key)
    # Sync to GPU here (once per token)
    EMLX.eval(EMLX.Backend.from_nx(first_token))

    t1 = System.monotonic_time(:millisecond)
    prefill_ms = t1 - t0

    # Native path uses `{1, prompt_len}`; KV length tracks sequence dimension.
    current_len = elem(Nx.shape(input_ids), 1)
    eos_id = eos_token_id(state)

    decode_ctx = {eos_id, sampler, temp, top_p, state, profile_timing?}

    # Tail-recursive decode: max_new - 1 steps after prefill (no Enum range / closure).
    remaining_decode = max(max_new - 1, 0)

    {tokens, per_token_ms, _kv, _cur, _rng, _decode_slot} =
      decode_tokens(
        remaining_decode,
        [Nx.to_number(first_token)],
        [],
        kv_cache,
        current_len,
        rng_key,
        decode_slot,
        decode_ctx
      )

    t_end = System.monotonic_time(:millisecond)

    timing = %{
      prefill_ms: prefill_ms,
      per_token_ms: :lists.reverse(per_token_ms),
      total_ms: t_end - t0
    }

    {:lists.reverse(tokens), %{timing: timing}}
  end

  # ── Helpers ───────────────────────────────────────────────────────────────

  defp decode_tokens(0, acc_tokens, acc_times, kv, cur, rng_key, decode_slot, _ctx) do
    {acc_tokens, acc_times, kv, cur, rng_key, decode_slot}
  end

  defp decode_tokens(n, acc_tokens, acc_times, kv, cur, rng_key, decode_slot, ctx)
       when is_integer(n) and n > 0 do
    {eos_id, _, _, _, _, _} = ctx
    [last_id | _] = acc_tokens

    if last_id == eos_id do
      {acc_tokens, acc_times, kv, cur, rng_key, decode_slot}
    else
      decode_step_timed(n, acc_tokens, acc_times, kv, cur, rng_key, decode_slot, ctx, last_id)
    end
  end

  defp decode_step_timed(n, acc_tokens, acc_times, kv, cur, rng_key, decode_slot, ctx, last_id) do
    {_eos_id, sampler, temp, top_p, state, profile_timing?} = ctx

    ts = if profile_timing?, do: System.monotonic_time(:millisecond)

    # Host {1,1} slice; put_slice uploads into the GPU decode_slot (no separate transfer).
    id_patch = Nx.tensor([[last_id]], type: :s64, backend: @cpu_backend)

    decode_slot = Nx.put_slice(decode_slot, [0, 0], id_patch)

    {logits, kv_new} = Model.forward(decode_slot, kv, cur, state)

    {rng_key, gpu_key} = advance_rng_for_gpu(sampler, rng_key)
    next_token = sample(logits, sampler, temp, top_p, gpu_key)
    EMLX.eval(EMLX.Backend.from_nx(next_token))

    acc_times =
      if profile_timing? do
        te = System.monotonic_time(:millisecond)
        [te - ts | acc_times]
      else
        acc_times
      end

    decode_tokens(
      n - 1,
      [Nx.to_number(next_token) | acc_tokens],
      acc_times,
      kv_new,
      cur + 1,
      rng_key,
      decode_slot,
      ctx
    )
  end

  defp sample(logits, :greedy, _temp, _top_p, _key), do: Sampler.greedy(logits)

  defp sample(logits, :top_p_cpu, temp, top_p, _key),
    do: Sampler.top_p_cpu(logits, temp, top_p)

  defp sample(logits, :top_p_gpu, temp, _top_p, gpu_key) do
    Sampler.top_p_gpu(logits, gpu_key, temperature: temp)
  end

  defp advance_rng_for_gpu(:top_p_gpu, key) do
    keys = Nx.Random.split(key, parts: 2)

    k_this =
      keys[[0..0, ..]]
      |> Nx.squeeze(axes: [0])
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    k_next = keys[[1..1, ..]] |> Nx.squeeze(axes: [0])
    {k_next, k_this}
  end

  defp advance_rng_for_gpu(_sampler, key), do: {key, nil}

  defp eos_token_id(%Model.State{config: cfg}) do
    cfg[:eos_token_id] || 151_645
  end
end
