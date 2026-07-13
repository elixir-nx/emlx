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

  alias EMLXAxon.Qwen3.{Model, NativeChunkRunner, Sampler}

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
  - `:profile_timing` — when `true`, record `per_token_ms` decode samples using
                        microsecond monotonic time; defaults to `false`
                        to keep the generation hot path free of timing for each token
                        overhead
                        (prefill/total wall time is still measured)
  - `:token_callback` — optional function of arity 1 called with each generated token id
                        after the token has been evaluated and copied to the host
  - `:chunk_callback` — optional function of arity 1 called with a list of generated
                        token ids whenever a deferred host sync chunk is copied to
                        the host. This is only used with `host_sync: {:chunk, n}`.
  - `:chunk_callback_first_token` — when `true`, emit the prefill token through
                        `:chunk_callback` immediately and then chunk the remaining
                        decode tokens. Defaults to `false` to preserve generic
                        chunk callback semantics.
  - `:return_kv_cache` — when `true`, include the final KV cache in the returned
                         metadata so serving layers can reuse the owned cache on
                         the next request
  - `:host_sync`      — `:per_token | :end | {:chunk, pos_integer()}` (default
                        `:per_token`). `:end` keeps greedy/top-p GPU sampled tokens
                        on the EMLX backend until generation finishes, then copies
                        token ids to the host once as a stacked tensor. `{:chunk, n}`
                        copies stacked token chunks every `n` generated tokens so it
                        can stop soon after EOS. Deferred host sync modes cannot be
                        used with `:token_callback`.

  ## Returns

  `{generated_token_ids, %{timing: timing_map}}` where `timing_map` has:
  - `:prefill_ms`     — first-token time in milliseconds, with microsecond resolution
  - `:per_token_ms`   — list of per-token decode times, with microsecond resolution
                        (median ≈ steady-state)
  - `:total_ms`       — wall time for the whole call, with microsecond resolution
  """
  def generate(input_ids, %Model.State{} = state, opts \\ []) do
    generate_with_native_chunk_runner(
      input_ids,
      state,
      opts,
      &Model.forward_greedy_chunk/5
    )
  end

  @doc false
  def generate_with_native_chunk_runner(
        input_ids,
        %Model.State{} = state,
        opts,
        native_chunk_runner
      )
      when is_function(native_chunk_runner, 5) do
    ensure_single_batch!(input_ids)

    requested_max_new =
      opts
      |> Keyword.get(:max_new_tokens, @default_max_new_tokens)
      |> validate_max_new_tokens!()

    configured_max_len = Keyword.get(opts, :max_len, @default_max_len)
    sampler = Keyword.get(opts, :sampler, :greedy)
    temp = Keyword.get(opts, :temperature, 0.95)
    top_p = Keyword.get(opts, :top_p, 0.9)
    input_length = elem(Nx.shape(input_ids), 1)

    kv_cache =
      case Keyword.fetch(opts, :kv_cache) do
        {:ok, prealloc} -> prealloc
        :error -> Model.init_kv_cache(state, configured_max_len)
      end

    max_len = effective_max_len(configured_max_len, kv_cache)
    max_new = safe_max_new_tokens!(requested_max_new, max_len, input_length)

    rng_key =
      if sampler == :top_p_gpu do
        Keyword.get(opts, :rng_key, Nx.Random.key(42))
      else
        nil
      end

    profile_timing? = Keyword.get(opts, :profile_timing, false)
    token_callback = Keyword.get(opts, :token_callback)
    chunk_callback = Keyword.get(opts, :chunk_callback)
    chunk_callback_first_token? = Keyword.get(opts, :chunk_callback_first_token, false)
    return_kv_cache? = Keyword.get(opts, :return_kv_cache, false)

    host_sync =
      host_sync_mode(Keyword.get(opts, :host_sync, :per_token), sampler, token_callback)

    # Prefill: pass the full prompt in one forward
    t0 = timing_now_us()

    {rng_key, first_gpu_key} = advance_rng_for_gpu(sampler, rng_key)

    {first_token, kv_cache} =
      forward_sample(input_ids, kv_cache, 0, sampler, temp, top_p, first_gpu_key, state)

    t1 = timing_now_us()
    prefill_ms = elapsed_ms(t0, t1)

    # Native path uses `{1, prompt_len}`; KV length tracks sequence dimension.
    current_len = input_length
    eos_id = eos_token_id(state)

    decode_ctx =
      {eos_id, sampler, temp, top_p, state, profile_timing?, token_callback, chunk_callback,
       native_chunk_runner}

    # Tail-recursive decode: max_new - 1 steps after prefill (no Enum range / closure).
    remaining_decode = max(max_new - 1, 0)

    {tokens, per_token_ms, kv_cache, _cur, _rng} =
      case host_sync do
        :per_token ->
          first_token_id = Nx.to_number(first_token)
          emit_token(token_callback, first_token_id)

          decode_tokens(
            remaining_decode,
            [first_token_id],
            [],
            kv_cache,
            current_len,
            rng_key,
            decode_ctx
          )

        :end ->
          decode_tensors(
            remaining_decode,
            [first_token],
            [],
            kv_cache,
            current_len,
            rng_key,
            decode_ctx
          )

        {:chunk, chunk_size} ->
          decode_tensor_chunks_start(
            remaining_decode,
            first_token,
            kv_cache,
            current_len,
            rng_key,
            decode_ctx,
            chunk_size,
            chunk_callback_first_token?
          )
      end

    t_end = timing_now_us()

    timing = %{
      prefill_ms: prefill_ms,
      per_token_ms: :lists.reverse(per_token_ms),
      total_ms: elapsed_ms(t0, t_end)
    }

    metadata =
      %{timing: timing}
      |> Map.put(:finish_reason, finish_reason(tokens, eos_id, requested_max_new, max_new))
      |> maybe_put_kv_cache(return_kv_cache?, kv_cache)

    {:lists.reverse(tokens), metadata}
  end

  # ── Helpers ───────────────────────────────────────────────────────────────

  defp validate_max_new_tokens!(max_new) when is_integer(max_new) and max_new > 0,
    do: max_new

  defp validate_max_new_tokens!(max_new) do
    raise ArgumentError,
          "expected :max_new_tokens to be a positive integer, got: #{inspect(max_new)}"
  end

  defp timing_now_us, do: System.monotonic_time(:microsecond)
  defp elapsed_ms(start_us, end_us), do: (end_us - start_us) / 1000

  defp safe_max_new_tokens!(requested_max_new, max_len, input_length) do
    safe_max_new = max_len - input_length + 1

    cond do
      input_length > max_len ->
        raise ArgumentError,
              "expected :max_len to be greater than or equal to input length, " <>
                "got max_len=#{inspect(max_len)} and input_length=#{inspect(input_length)}"

      safe_max_new < requested_max_new ->
        safe_max_new

      true ->
        requested_max_new
    end
  end

  defp finish_reason(tokens, eos_id, _requested_max_new, _max_new) do
    if Enum.any?(tokens, &eos_token?(&1, eos_id)), do: :stop, else: :length
  end

  defp maybe_put_kv_cache(metadata, true, kv_cache), do: Map.put(metadata, :kv_cache, kv_cache)
  defp maybe_put_kv_cache(metadata, false, _kv_cache), do: metadata

  defp effective_max_len(configured_max_len, kv_cache) do
    case kv_cache_capacity(kv_cache) do
      nil -> configured_max_len
      capacity -> min(configured_max_len, capacity)
    end
  end

  defp kv_cache_capacity([{k_cache, _v_cache} | _rest]) do
    case cache_shape(k_cache) do
      {_batch, _heads, max_len, _head_dim} -> max_len
      _other -> nil
    end
  end

  defp kv_cache_capacity(_kv_cache), do: nil

  defp cache_shape(%Nx.Tensor{} = tensor), do: Nx.shape(tensor)

  defp cache_shape({device, ref}) when is_atom(device) and is_reference(ref),
    do: EMLX.shape({device, ref})

  defp cache_shape(_cache), do: nil

  defp host_sync_mode(:per_token, _sampler, _token_callback), do: :per_token

  defp host_sync_mode(:end, sampler, nil) when sampler in [:greedy, :top_p_gpu], do: :end

  defp host_sync_mode({:chunk, chunk_size}, sampler, nil)
       when sampler in [:greedy, :top_p_gpu] and is_integer(chunk_size) and chunk_size > 0,
       do: {:chunk, chunk_size}

  defp host_sync_mode({:chunk, chunk_size}, _sampler, _token_callback)
       when not is_integer(chunk_size) or chunk_size <= 0 do
    raise ArgumentError,
          "expected :host_sync chunk size to be a positive integer, got: #{inspect(chunk_size)}"
  end

  defp host_sync_mode(:end, _sampler, token_callback) when is_function(token_callback, 1),
    do: :per_token

  defp host_sync_mode({:chunk, _chunk_size}, _sampler, token_callback)
       when is_function(token_callback, 1),
       do: :per_token

  defp host_sync_mode(:end, _sampler, _token_callback), do: :per_token
  defp host_sync_mode({:chunk, _chunk_size}, _sampler, _token_callback), do: :per_token

  defp host_sync_mode(other, _sampler, _token_callback) do
    raise ArgumentError,
          "expected :host_sync to be :per_token, :end, or {:chunk, positive_integer}, got: #{inspect(other)}"
  end

  defp decode_tokens(0, acc_tokens, acc_times, kv, cur, rng_key, _ctx) do
    {acc_tokens, acc_times, kv, cur, rng_key}
  end

  defp decode_tokens(n, acc_tokens, acc_times, kv, cur, rng_key, ctx)
       when is_integer(n) and n > 0 do
    {eos_id, _, _, _, _, _, _, _, _} = ctx
    [last_id | _] = acc_tokens

    if eos_token?(last_id, eos_id) do
      {acc_tokens, acc_times, kv, cur, rng_key}
    else
      decode_step_timed(n, acc_tokens, acc_times, kv, cur, rng_key, ctx, last_id)
    end
  end

  defp decode_step_timed(n, acc_tokens, acc_times, kv, cur, rng_key, ctx, last_id) do
    {_eos_id, sampler, temp, top_p, state, profile_timing?, token_callback, _chunk_callback,
     _native_chunk_runner} = ctx

    ts = if profile_timing?, do: timing_now_us()

    {rng_key, gpu_key} = advance_rng_for_gpu(sampler, rng_key)

    {next_token_id, kv_new} =
      forward_sample_token_id(last_id, kv, cur, sampler, temp, top_p, gpu_key, state)

    emit_token(token_callback, next_token_id)

    acc_times =
      if profile_timing? do
        te = timing_now_us()
        [elapsed_ms(ts, te) | acc_times]
      else
        acc_times
      end

    decode_tokens(
      n - 1,
      [next_token_id | acc_tokens],
      acc_times,
      kv_new,
      cur + 1,
      rng_key,
      ctx
    )
  end

  defp decode_tensors(0, acc_tokens, acc_times, kv, cur, rng_key, ctx) do
    {eos_id, _sampler, _temp, _top_p, _state, _profile_timing?, _token_callback, _chunk_callback,
     _native_chunk_runner} = ctx

    tokens =
      acc_tokens
      |> Enum.reverse()
      |> tokens_to_host()
      |> truncate_at_eos(eos_id)

    {Enum.reverse(tokens), Enum.reverse(acc_times), kv, cur, rng_key}
  end

  defp decode_tensors(n, acc_tokens, acc_times, kv, cur, rng_key, ctx)
       when is_integer(n) and n > 0 do
    {_eos_id, sampler, _temp, _top_p, state, profile_timing?, _token_callback, _chunk_callback,
     native_chunk_runner} = ctx

    if sampler == :greedy and not profile_timing? and fused_end_sync?(state) do
      [last_token | _] = acc_tokens

      {next_tokens, kv_new, next_cur} =
        run_native_chunks(last_token, kv, cur, n, state, native_chunk_runner)

      decode_tensors(
        0,
        Enum.reduce(next_tokens, acc_tokens, fn token, acc -> [token | acc] end),
        acc_times,
        kv_new,
        next_cur,
        rng_key,
        ctx
      )
    else
      decode_tensors_per_token(n, acc_tokens, acc_times, kv, cur, rng_key, ctx)
    end
  end

  defp decode_tensors_per_token(n, acc_tokens, acc_times, kv, cur, rng_key, ctx) do
    {_eos_id, sampler, temp, top_p, state, profile_timing?, _token_callback, _chunk_callback,
     _native_chunk_runner} = ctx

    [last_token | _] = acc_tokens

    ts = if profile_timing?, do: timing_now_us()
    decode_input = Nx.reshape(last_token, {1, 1})

    {rng_key, gpu_key} = advance_rng_for_gpu(sampler, rng_key)

    {next_token, kv_new} =
      forward_sample(decode_input, kv, cur, sampler, temp, top_p, gpu_key, state)

    acc_times =
      if profile_timing? do
        te = timing_now_us()
        [elapsed_ms(ts, te) | acc_times]
      else
        acc_times
      end

    decode_tensors(
      n - 1,
      [next_token | acc_tokens],
      acc_times,
      kv_new,
      cur + 1,
      rng_key,
      ctx
    )
  end

  defp fused_end_sync?(%Model.State{config: config, lm_head: lm_head}) do
    config[:dense_layers?] != true or EMLX.Quantization.quantized?(lm_head)
  end

  defp decode_tensor_chunks_start(
         remaining_decode,
         first_token,
         kv_cache,
         current_len,
         rng_key,
         ctx,
         chunk_size,
         false
       ) do
    decode_tensor_chunks(
      remaining_decode,
      first_token,
      [first_token],
      1,
      [],
      [],
      kv_cache,
      current_len,
      rng_key,
      ctx,
      chunk_size
    )
  end

  defp decode_tensor_chunks_start(
         remaining_decode,
         first_token,
         kv_cache,
         current_len,
         rng_key,
         {eos_id, _sampler, _temp, _top_p, _state, _profile_timing?, _token_callback,
          chunk_callback, _native_chunk_runner} = ctx,
         chunk_size,
         true
       )
       when is_function(chunk_callback, 1) do
    first_token_id = Nx.to_number(first_token)
    emit_chunk(chunk_callback, [first_token_id])

    if eos_token?(first_token_id, eos_id) do
      {[first_token_id], [], kv_cache, current_len, rng_key}
    else
      decode_tensor_chunks(
        remaining_decode,
        first_token,
        [],
        0,
        [first_token_id],
        [],
        kv_cache,
        current_len,
        rng_key,
        ctx,
        chunk_size
      )
    end
  end

  defp decode_tensor_chunks(
         0,
         _last_token,
         pending_tokens,
         _pending_count,
         host_tokens,
         acc_times,
         kv,
         cur,
         rng_key,
         ctx,
         _chunk_size
       ) do
    {eos_id, _sampler, _temp, _top_p, _state, _profile_timing?, _token_callback, chunk_callback,
     _native_chunk_runner} = ctx

    tokens = flush_chunk(host_tokens, pending_tokens, eos_id, chunk_callback)
    {:lists.reverse(tokens), :lists.reverse(acc_times), kv, cur, rng_key}
  end

  defp decode_tensor_chunks(
         n,
         last_token,
         pending_tokens,
         pending_count,
         host_tokens,
         acc_times,
         kv,
         cur,
         rng_key,
         ctx,
         chunk_size
       )
       when is_integer(n) and n > 0 do
    {eos_id, sampler, _temp, _top_p, state, profile_timing?, _token_callback, chunk_callback,
     native_chunk_runner} = ctx

    case maybe_flush_chunk(
           host_tokens,
           pending_tokens,
           pending_count,
           eos_id,
           chunk_size,
           chunk_callback
         ) do
      {:halt, tokens} ->
        {Enum.reverse(tokens), Enum.reverse(acc_times), kv, cur, rng_key}

      {:cont, host_tokens, pending_tokens, pending_count} ->
        ts = if profile_timing?, do: timing_now_us()

        if sampler == :greedy and not profile_timing? do
          chunk_count = min(n, chunk_size - pending_count)

          {next_tokens, kv_new, next_cur} =
            run_native_chunks(
              last_token,
              kv,
              cur,
              chunk_count,
              state,
              native_chunk_runner
            )

          {last_token, pending_tokens} =
            append_reversed_with_last(next_tokens, pending_tokens)

          decode_tensor_chunks(
            n - chunk_count,
            last_token,
            pending_tokens,
            pending_count + chunk_count,
            host_tokens,
            acc_times,
            kv_new,
            next_cur,
            rng_key,
            ctx,
            chunk_size
          )
        else
          decode_input = Nx.reshape(last_token, {1, 1})

          decode_tensor_chunk_step(
            n,
            decode_input,
            pending_tokens,
            pending_count,
            host_tokens,
            acc_times,
            kv,
            cur,
            rng_key,
            ctx,
            ts,
            chunk_size
          )
        end
    end
  end

  defp append_reversed_with_last([token | rest], pending_tokens) do
    append_reversed_with_last(rest, token, [token | pending_tokens])
  end

  defp append_reversed_with_last([], _pending_tokens) do
    raise ArgumentError, "expected at least one generated token in a greedy chunk"
  end

  defp append_reversed_with_last([token | rest], _last_token, pending_tokens) do
    append_reversed_with_last(rest, token, [token | pending_tokens])
  end

  defp append_reversed_with_last([], last_token, pending_tokens), do: {last_token, pending_tokens}

  defp decode_tensor_chunk_step(
         n,
         decode_input,
         pending_tokens,
         pending_count,
         host_tokens,
         acc_times,
         kv,
         cur,
         rng_key,
         ctx,
         ts,
         chunk_size
       ) do
    {_eos_id, sampler, temp, top_p, state, profile_timing?, _token_callback, _chunk_callback,
     _native_chunk_runner} = ctx

    {rng_key, gpu_key} = advance_rng_for_gpu(sampler, rng_key)

    {next_token, kv_new} =
      forward_sample(decode_input, kv, cur, sampler, temp, top_p, gpu_key, state)

    acc_times =
      if profile_timing? do
        te = timing_now_us()
        [elapsed_ms(ts, te) | acc_times]
      else
        acc_times
      end

    decode_tensor_chunks(
      n - 1,
      next_token,
      [next_token | pending_tokens],
      pending_count + 1,
      host_tokens,
      acc_times,
      kv_new,
      cur + 1,
      rng_key,
      ctx,
      chunk_size
    )
  end

  defp forward_sample(input_ids, kv, cur, :greedy, _temp, _top_p, _key, state) do
    Model.forward_greedy(input_ids, kv, cur, state)
  end

  defp forward_sample(input_ids, kv, cur, sampler, temp, top_p, key, state) do
    {logits, kv_new} = Model.forward(input_ids, kv, cur, state)
    {sample(logits, sampler, temp, top_p, key), kv_new}
  end

  defp forward_sample_token_id(token_id, kv, cur, :greedy, _temp, _top_p, _key, state)
       when is_integer(token_id) do
    Model.forward_greedy_decode_token_id(token_id, kv, cur, state)
  end

  defp forward_sample_token_id(token_id, kv, cur, sampler, temp, top_p, key, state)
       when is_integer(token_id) do
    input_ids = Nx.tensor([[token_id]], type: :s64, backend: @cpu_backend)
    {next_token, kv_new} = forward_sample(input_ids, kv, cur, sampler, temp, top_p, key, state)
    {Nx.to_number(next_token), kv_new}
  end

  defp run_native_chunks(last_token, kv_cache, current_len, count, state, native_chunk_runner) do
    NativeChunkRunner.run(
      last_token,
      kv_cache,
      current_len,
      count,
      fn token, kv_cache, offset, chunk_count ->
        input_ids = Nx.reshape(token, {1, 1})
        native_chunk_runner.(input_ids, kv_cache, offset, chunk_count, state)
      end
    )
  end

  defp sample(logits, :top_p_cpu, temp, top_p, _key),
    do: Sampler.top_p_cpu(logits, temp, top_p)

  defp sample(logits, :top_p_gpu, temp, _top_p, gpu_key) do
    Sampler.top_p_gpu(logits, gpu_key, temperature: temp)
  end

  defp emit_token(nil, _token_id), do: :ok

  defp emit_token(callback, token_id) when is_function(callback, 1) do
    callback.(token_id)
    :ok
  end

  defp ensure_single_batch!(input_ids) do
    case Nx.shape(input_ids) do
      {1, _seq_len} ->
        :ok

      {batch_size, _seq_len} ->
        raise ArgumentError,
              "native Qwen3 generation currently supports batch size 1, got batch size #{batch_size}"

      shape ->
        raise ArgumentError,
              "expected input_ids to have shape {1, sequence_length}, got: #{inspect(shape)}"
    end
  end

  defp emit_chunk(nil, _token_ids), do: :ok
  defp emit_chunk(_callback, []), do: :ok

  defp emit_chunk(callback, token_ids) when is_function(callback, 1) do
    callback.(token_ids)
    :ok
  end

  defp truncate_at_eos(tokens, eos_id) do
    case Enum.split_while(tokens, &(not eos_token?(&1, eos_id))) do
      {prefix, []} -> prefix
      {prefix, [eos_token | _rest]} -> prefix ++ [eos_token]
    end
  end

  defp maybe_flush_chunk(host_tokens, pending_tokens, pending_count, eos_id, chunk_size, callback)
       when pending_count >= chunk_size do
    tokens = flush_chunk(host_tokens, pending_tokens, eos_id, callback)

    if eos_seen?(tokens, eos_id) do
      {:halt, tokens}
    else
      {:cont, tokens, [], 0}
    end
  end

  defp maybe_flush_chunk(
         host_tokens,
         pending_tokens,
         pending_count,
         _eos_id,
         _chunk_size,
         _callback
       ),
       do: {:cont, host_tokens, pending_tokens, pending_count}

  defp flush_chunk(host_tokens, [] = _pending_tokens, _eos_id, _callback), do: host_tokens

  defp flush_chunk(host_tokens, pending_tokens, eos_id, callback) do
    emitted =
      pending_tokens
      |> :lists.reverse()
      |> tokens_to_host()
      |> truncate_at_eos(eos_id)

    emit_chunk(callback, emitted)
    host_tokens ++ emitted
  end

  defp eos_seen?(tokens, eos_id), do: Enum.any?(tokens, &eos_token?(&1, eos_id))

  defp eos_token?(token_id, eos_ids) when is_list(eos_ids), do: token_id in eos_ids
  defp eos_token?(token_id, eos_id), do: token_id == eos_id

  defp tokens_to_host(tokens) do
    Enum.flat_map(tokens, &Nx.to_flat_list/1)
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
