defmodule EMLXAxon.TextGeneration do
  @moduledoc """
  A `Nx.Serving`-compatible wrapper around native Qwen3 generation.

  Bypasses the Axon graph entirely — the 28-layer forward pass runs as a single
  `mlx::eval` per token (via `EMLXAxon.Qwen3.Generate`), avoiding
  the 28 separate Metal command buffer submissions that the Bumblebee + Axon path
  incurs.

  Only Bumblebee tokenization is used from upstream Bumblebee. No Bumblebee model
  function or Axon graph is involved in the decode forward pass.

  ## MLX-4bit usage

      {:ok, tokenizer} = Bumblebee.load_tokenizer({:local, "~/models/Qwen3-0.6B-MLX-4bit"})
      serving = EMLXAxon.TextGeneration.from_mlx4bit(
        "~/models/Qwen3-0.6B-MLX-4bit",
        tokenizer,
        max_new_tokens: 100,
        sampler: :greedy
      )

      result = Nx.Serving.run(serving, "Write a short story about a robot who learns to love.")
      IO.puts(result.results |> hd() |> Map.fetch!(:generated_text))

  ## Dense safetensors usage

      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-0.6B"})
      {:ok, state} =
        EMLXAxon.Qwen3.DenseLoader.from_safetensors_dir(
          "~/models/Qwen3-0.6B",
          type: :f16
        )

      serving = EMLXAxon.TextGeneration.serving(tokenizer, state,
        max_new_tokens: 100,
        sampler: :greedy
      )

      result = Nx.Serving.run(serving, "Write a short story about a robot who learns to love.")
      IO.puts(result.results |> hd() |> Map.fetch!(:generated_text))
  """

  alias EMLXAxon.Qwen3.{Model, Generate, Loader}

  @cpu_backend Nx.BinaryBackend
  @decode_cache_table __MODULE__.DecodeCache
  @default_host_sync_chunk_cap 31
  @default_stream_host_sync {:chunk, 5}

  @doc """
  Runs native Qwen3 text generation directly, without wrapping the call in
  `Nx.Serving`.

  This is useful for host applications that already own a loaded
  `EMLXAxon.Qwen3.Model.State` and want to avoid serving preprocessing
  overhead. The return shape matches `serving/3` for the same `:output_format`.
  """
  @spec run(
          Bumblebee.Tokenizer.t(),
          Model.State.t(),
          binary() | %{text: binary()},
          keyword()
        ) ::
          map()
  def run(tokenizer, %Model.State{} = state, input, opts \\ []) do
    max_new = max_new_tokens!(opts, 100)
    configured_max_len = Keyword.get(opts, :max_len, 2048)
    sampler = Keyword.get(opts, :sampler, :greedy)
    profile_timing = Keyword.get(opts, :profile_timing, false)
    output_format = Keyword.get(opts, :output_format, :native)
    temperature = Keyword.get(opts, :temperature, 0.95)
    top_p = Keyword.get(opts, :top_p, 0.9)
    return_kv_cache? = Keyword.get(opts, :return_kv_cache, false)
    return_timing? = Keyword.get(opts, :return_timing, false)
    host_sync = Keyword.get_lazy(opts, :host_sync, fn -> default_host_sync(max_new) end)

    %{text: text} = normalize_input(input)

    encoded =
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        Bumblebee.apply_tokenizer(tokenizer, [text])
      end)

    input_ids = Nx.backend_transfer(encoded["input_ids"], {EMLX.Backend, device: :gpu})

    input_length = Nx.axis_size(input_ids, 1)

    {kv_cache, max_len} =
      kv_cache_for_request(opts, state, configured_max_len, input_length, max_new, sampler)

    {tokens, metadata} =
      Generate.generate(input_ids, state,
        kv_cache: kv_cache,
        max_new_tokens: max_new,
        max_len: max_len,
        sampler: sampler,
        profile_timing: profile_timing,
        temperature: temperature,
        top_p: top_p,
        host_sync: host_sync,
        return_kv_cache: return_kv_cache?
      )

    num_tokens = length(tokens)
    decoded = Bumblebee.Tokenizer.decode(tokenizer, tokens)

    output = %{results: [result(decoded_text(decoded), input_length, num_tokens, output_format)]}

    output
    |> maybe_put_kv_cache(return_kv_cache?, metadata)
    |> maybe_put_timing(return_timing?, metadata)
    |> maybe_put_finish_reason(metadata)
  end

  defp maybe_put_kv_cache(output, true, %{kv_cache: kv_cache}),
    do: Map.put(output, :kv_cache, kv_cache)

  defp maybe_put_kv_cache(output, _return?, _metadata), do: output

  defp maybe_put_timing(output, true, %{timing: timing}), do: Map.put(output, :timing, timing)
  defp maybe_put_timing(output, _return?, _metadata), do: output

  defp maybe_put_finish_reason(output, %{finish_reason: finish_reason}),
    do: Map.put(output, :finish_reason, finish_reason)

  defp maybe_put_finish_reason(output, _metadata), do: output

  defp default_host_sync(max_new) when is_integer(max_new) and max_new > 0,
    do: {:chunk, min(max_new, @default_host_sync_chunk_cap)}

  defp default_host_sync(_max_new), do: {:chunk, @default_host_sync_chunk_cap}

  defp kv_cache_for_request(opts, state, configured_max_len, input_length, max_new, sampler) do
    required_max_len = adaptive_max_len(configured_max_len, input_length, max_new)

    case Keyword.fetch(opts, :kv_cache) do
      {:ok, kv_cache} ->
        case kv_cache_capacity(kv_cache) do
          capacity when is_integer(capacity) and capacity >= required_max_len ->
            {kv_cache, min(configured_max_len, capacity)}

          _too_small_or_unknown ->
            {init_kv_cache_for_sampler(state, required_max_len, sampler), required_max_len}
        end

      :error ->
        {init_kv_cache_for_sampler(state, required_max_len, sampler), required_max_len}
    end
  end

  defp init_kv_cache_for_sampler(state, max_len, :greedy),
    do: Model.init_native_kv_cache(state, max_len)

  defp init_kv_cache_for_sampler(state, max_len, _sampler),
    do: Model.init_kv_cache(state, max_len)

  defp adaptive_max_len(configured_max_len, input_length, max_new) do
    min(configured_max_len, input_length + max(max_new, 1) - 1)
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

  defp stream_host_sync_mode(:per_token, _sampler), do: :per_token

  defp stream_host_sync_mode({:chunk, chunk_size} = mode, sampler)
       when sampler in [:greedy, :top_p_gpu] and is_integer(chunk_size) and chunk_size > 0,
       do: mode

  defp stream_host_sync_mode({:chunk, chunk_size}, _sampler)
       when is_integer(chunk_size) and chunk_size > 0,
       do: :per_token

  defp stream_host_sync_mode(other, _sampler), do: other

  @doc """
  Streams native Qwen3 text generation through `emit_fun`.

  `emit_fun` is called with decoded text chunks as they become available. The
  function returns a map with `:token_summary`, matching the shape used by
  `run/4` and `serving/3`.

  ## Options

  Accepts the same generation options as `run/4`, plus:

  - `:stream_chunking` — `:stable | :token` (default `:stable`). Stable chunking
    buffers decoded text until a whitespace/newline boundary. Token chunking
    emits each decoded token immediately, which reduces time to first token for
    server streaming clients that can handle token fragments.
  - `:stream_host_sync` — `:per_token | {:chunk, pos_integer()}` (default
    `#{inspect(@default_stream_host_sync)}`). Chunked mode copies generated token ids back to the host and
    emits text every N tokens, which reduces host synchronizations and BEAM
    messages. Streaming chunked mode emits the prefill token immediately, then
    chunks the remaining decode tokens, so server clients keep low
    time to first token while still avoiding host sync after each token for the rest of
    the response. Samplers that cannot use deferred host sync fall back to
    emitting each token.
  """
  @spec stream(
          Bumblebee.Tokenizer.t(),
          Model.State.t(),
          binary() | map(),
          (binary() -> term()),
          keyword()
        ) :: %{token_summary: map()}
  def stream(tokenizer, %Model.State{} = state, input, emit_fun, opts \\ [])
      when is_function(emit_fun, 1) do
    max_new = max_new_tokens!(opts, 100)
    configured_max_len = Keyword.get(opts, :max_len, 2048)
    sampler = Keyword.get(opts, :sampler, :greedy)
    profile_timing = Keyword.get(opts, :profile_timing, false)
    temperature = Keyword.get(opts, :temperature, 0.95)
    top_p = Keyword.get(opts, :top_p, 0.9)
    return_kv_cache? = Keyword.get(opts, :return_kv_cache, false)
    return_timing? = Keyword.get(opts, :return_timing, false)
    stream_chunking = Keyword.get(opts, :stream_chunking, :stable)

    stream_host_sync =
      opts
      |> Keyword.get(:stream_host_sync, @default_stream_host_sync)
      |> stream_host_sync_mode(sampler)

    %{text: text} = normalize_input(input)

    encoded =
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        Bumblebee.apply_tokenizer(tokenizer, [text])
      end)

    input_ids =
      encoded["input_ids"]
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    input_length = Nx.axis_size(input_ids, 1)

    {kv_cache, max_len} =
      kv_cache_for_request(opts, state, configured_max_len, input_length, max_new, sampler)

    stream_key = {__MODULE__, :stream_state, make_ref()}
    Process.put(stream_key, %{pending_text: ""})

    try do
      {tokens, metadata} =
        generate_stream_tokens(
          tokenizer,
          state,
          input_ids,
          kv_cache,
          stream_key,
          emit_fun,
          stream_chunking,
          stream_host_sync,
          max_new: max_new,
          max_len: max_len,
          sampler: sampler,
          profile_timing: profile_timing,
          temperature: temperature,
          top_p: top_p,
          return_kv_cache?: return_kv_cache?
        )

      flush_stream_chunks(tokenizer, emit_fun, stream_key, stream_chunking)

      %{
        token_summary: %{
          input: input_length,
          output: length(tokens),
          padding: 0
        }
      }
      |> maybe_put_kv_cache(return_kv_cache?, metadata)
      |> maybe_put_timing(return_timing?, metadata)
      |> maybe_put_finish_reason(metadata)
    after
      Process.delete(stream_key)
    end
  end

  @doc """
  Builds an `Nx.Serving` wrapping a native Qwen3 model state.

  The state may come from the MLX-4bit loader or from
  `EMLXAxon.Qwen3.DenseLoader` for standard dense Hugging Face safetensors.

  Accepts the same text-string input format as `Bumblebee.Text.generation/4`:
  a plain binary or `%{text: binary()}`. Returns `%{results: [%{generated_text: binary(),
  num_tokens: pos_integer()}]}`. Native Qwen3 generation currently supports
  one input at a time; list/batch inputs raise `ArgumentError`.

  ## Options

  - `:max_new_tokens` — max tokens to generate per request (default 100)
  - `:max_len`        — KV cache preallocated token budget (default 2048)
  - `:sampler`        — `:greedy | :top_p_cpu | :top_p_gpu` (default `:greedy`)
  - `:temperature`    — sampler temperature for samplers other than greedy (default 0.95)
  - `:top_p`          — nucleus cutoff for `:top_p_cpu` (default 0.9);
                        `:top_p_gpu` currently uses temperature sampling only
  - `:profile_timing` — forwarded to `Generate.generate/3`; when `true`, records
                        timing for each token with `System.monotonic_time` in the decode
                        loop (default `false`)
  - `:host_sync`      — `:per_token | :end | {:chunk, pos_integer()}`; deferred
                        modes avoid synchronising every generated token back to the
                        host for greedy generation without streaming. `:end` copies a
                        stacked token tensor once after generation; `{:chunk, n}`
                        copies token chunks every `n` tokens so generation can stop
                        soon after EOS (default `{:chunk, min(max_new_tokens, 31)}`)
  - `:output_format`  — `:native` returns `%{generated_text: text, num_tokens: n}`;
                        `:bumblebee` returns `%{text: text, token_summary: summary}`
                        compatible with `Bumblebee.Text.generation/4` (default `:native`)
  """
  @spec serving(Bumblebee.Tokenizer.t(), Model.State.t(), keyword()) :: Nx.Serving.t()
  def serving(tokenizer, state, opts \\ []) do
    max_new = max_new_tokens!(opts, 100)
    configured_max_len = Keyword.get(opts, :max_len, 2048)
    sampler = Keyword.get(opts, :sampler, :greedy)
    temperature = Keyword.get(opts, :temperature, 0.95)
    top_p = Keyword.get(opts, :top_p, 0.9)
    profile_timing = Keyword.get(opts, :profile_timing, false)
    output_format = Keyword.get(opts, :output_format, :native)
    host_sync = Keyword.get_lazy(opts, :host_sync, fn -> default_host_sync(max_new) end)

    Nx.Serving.new(fn _init_opts ->
      fn batch ->
        # `Nx.Batch` stays lazy until a defn/jit entry; `jit_apply(identity)` is the
        # supported way to concatenate stacked entries into concrete tensors.
        %{"input_ids" => input_ids} = Nx.Defn.jit_apply(&Function.identity/1, [batch], opts[:defn_options])
        ensure_single_batch!(input_ids)

        # Pre-alloc KV cache per call. The cache is safe to reuse because
        # Model.forward/4 always slices to 0..current_len-1, so stale entries
        # beyond current_len are never read (advisor confirmed: no zero-fill needed).
        input_length = Nx.axis_size(input_ids, 1)
        max_len = adaptive_max_len(configured_max_len, input_length, max_new)
        kv_cache = init_kv_cache_for_sampler(state, max_len, sampler)

        {tokens, _timing} =
          Generate.generate(input_ids, state,
            kv_cache: kv_cache,
            max_new_tokens: max_new,
            max_len: max_len,
            sampler: sampler,
            temperature: temperature,
            top_p: top_p,
            host_sync: host_sync,
            profile_timing: profile_timing
          )

        # Return as 2-D tensor {1, num_tokens} matching the shape that
        # Bumblebee.Tokenizer.decode/2 expects for batch input.
        tokens_to_batch_tensor(tokens)
      end
    end)
    |> Nx.Serving.client_preprocessing(fn input ->
      %{text: text} = normalize_input(input)

      # Tokenize on CPU backend (variable length — no compile-time padding needed
      # since the forward pass is a plain Elixir function, not a defn).
      encoded =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          Bumblebee.apply_tokenizer(tokenizer, [text])
        end)

      input_ids =
        encoded["input_ids"]
        |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

      input_length = Nx.axis_size(input_ids, 1)

      {Nx.Batch.concatenate([%{"input_ids" => input_ids}]), input_length}
    end)
    |> Nx.Serving.client_postprocessing(fn {token_ids, _metadata}, input_length ->
      # token_ids: {1, num_generated_tokens} on Nx.BinaryBackend
      num_tokens = elem(Nx.shape(token_ids), 1)
      decoded = Bumblebee.Tokenizer.decode(tokenizer, token_ids)

      %{results: [result(decoded_text(decoded), input_length, num_tokens, output_format)]}
    end)
  end

  @doc """
  Convenience: load `%State{}` from an MLX-4bit checkpoint directory and build a serving.

  The tokenizer is expected to come from the same directory (same `tokenizer.json`).
  Loading both from the same directory avoids chat-template / BOS-token divergence.
  """
  @spec from_mlx4bit(Path.t(), Bumblebee.Tokenizer.t(), keyword()) :: Nx.Serving.t()
  def from_mlx4bit(checkpoint_path, tokenizer, opts \\ []) do
    {:ok, state} = Loader.load(checkpoint_path)
    serving(tokenizer, state, opts)
  end

  # ── Input normalisation ──────────────────────────────────────────────────────

  defp normalize_input(text) when is_binary(text), do: %{text: text}
  defp normalize_input(%{text: text} = item) when is_binary(text), do: item

  defp normalize_input(inputs) when is_list(inputs) do
    raise ArgumentError,
          "native Qwen3 text generation currently supports one input at a time, got a list"
  end

  defp normalize_input(input) do
    raise ArgumentError,
          "expected input to be a binary or %{text: binary}, got: #{inspect(input)}"
  end

  defp max_new_tokens!(opts, default) do
    opts
    |> Keyword.get(:max_new_tokens, default)
    |> validate_max_new_tokens!()
  end

  defp validate_max_new_tokens!(max_new) when is_integer(max_new) and max_new > 0,
    do: max_new

  defp validate_max_new_tokens!(max_new) do
    raise ArgumentError,
          "expected :max_new_tokens to be a positive integer, got: #{inspect(max_new)}"
  end

  defp ensure_single_batch!(input_ids) do
    case Nx.shape(input_ids) do
      {1, _seq_len} ->
        :ok

      {batch_size, _seq_len} ->
        raise ArgumentError,
              "native Qwen3 text generation currently supports batch size 1, got batch size #{batch_size}"

      shape ->
        raise ArgumentError,
              "expected tokenized input_ids to have shape {1, sequence_length}, got: #{inspect(shape)}"
    end
  end

  defp decoded_text([text]) when is_binary(text), do: text
  defp decoded_text(text) when is_binary(text), do: text

  defp result(text, input_length, num_tokens, :bumblebee) do
    %{
      text: text,
      token_summary: %{input: input_length, output: num_tokens, padding: 0}
    }
  end

  defp result(text, _input_length, num_tokens, :native) do
    %{generated_text: text, num_tokens: num_tokens}
  end

  defp result(_text, _input_length, _num_tokens, output_format) do
    raise ArgumentError,
          "expected :output_format to be :native or :bumblebee, got: #{inspect(output_format)}"
  end

  defp generate_stream_tokens(
         tokenizer,
         state,
         input_ids,
         kv_cache,
         stream_key,
         emit_fun,
         stream_chunking,
         :per_token,
         opts
       ) do
    Generate.generate(input_ids, state,
      kv_cache: kv_cache,
      max_new_tokens: Keyword.fetch!(opts, :max_new),
      max_len: Keyword.fetch!(opts, :max_len),
      sampler: Keyword.fetch!(opts, :sampler),
      profile_timing: Keyword.fetch!(opts, :profile_timing),
      temperature: Keyword.fetch!(opts, :temperature),
      top_p: Keyword.fetch!(opts, :top_p),
      host_sync: :per_token,
      return_kv_cache: Keyword.fetch!(opts, :return_kv_cache?),
      token_callback: fn token_id ->
        emit_stream_chunk(tokenizer, emit_fun, stream_key, token_id, stream_chunking)
      end
    )
  end

  defp generate_stream_tokens(
         tokenizer,
         state,
         input_ids,
         kv_cache,
         stream_key,
         emit_fun,
         stream_chunking,
         {:chunk, chunk_size},
         opts
       )
       when is_integer(chunk_size) and chunk_size > 0 do
    Generate.generate(input_ids, state,
      kv_cache: kv_cache,
      max_new_tokens: Keyword.fetch!(opts, :max_new),
      max_len: Keyword.fetch!(opts, :max_len),
      sampler: Keyword.fetch!(opts, :sampler),
      profile_timing: Keyword.fetch!(opts, :profile_timing),
      temperature: Keyword.fetch!(opts, :temperature),
      top_p: Keyword.fetch!(opts, :top_p),
      host_sync: {:chunk, chunk_size},
      return_kv_cache: Keyword.fetch!(opts, :return_kv_cache?),
      chunk_callback_first_token: true,
      chunk_callback: fn token_ids ->
        emit_stream_token_chunk(tokenizer, emit_fun, stream_key, token_ids, stream_chunking)
      end
    )
  end

  defp generate_stream_tokens(
         _tokenizer,
         _state,
         _input_ids,
         _kv_cache,
         _stream_key,
         _emit_fun,
         _stream_chunking,
         stream_host_sync,
         _opts
       ) do
    raise ArgumentError,
          "expected :stream_host_sync to be :per_token or {:chunk, positive_integer}, got: #{inspect(stream_host_sync)}"
  end

  defp emit_stream_chunk(tokenizer, emit_fun, _stream_key, token_id, :token) do
    emit_fun.(decode_token(tokenizer, token_id))
    :ok
  end

  defp emit_stream_chunk(tokenizer, emit_fun, stream_key, token_id, :stable) do
    state = Process.get(stream_key)
    state = %{state | pending_text: state.pending_text <> decode_token(tokenizer, token_id)}
    chunk = state.pending_text

    state =
      cond do
        chunk == "" ->
          state

        String.ends_with?(chunk, "\n") ->
          emit_fun.(chunk)
          %{pending_text: ""}

        space_idx = find_last_occurrence(chunk, " ") ->
          if space_idx > 0 do
            chunk = binary_slice(chunk, 0, space_idx)
            emit_fun.(chunk)
            %{pending_text: binary_slice(state.pending_text, byte_size(chunk)..-1//1)}
          else
            state
          end

        true ->
          state
      end

    Process.put(stream_key, state)
    :ok
  end

  defp emit_stream_chunk(_tokenizer, _emit_fun, _stream_key, _token_id, stream_chunking) do
    raise ArgumentError,
          "expected :stream_chunking to be :stable or :token, got: #{inspect(stream_chunking)}"
  end

  defp emit_stream_token_chunk(_tokenizer, _emit_fun, _stream_key, [], _stream_chunking), do: :ok

  defp emit_stream_token_chunk(tokenizer, emit_fun, _stream_key, token_ids, :token) do
    text = decode_tokens(token_ids, tokenizer)
    emit_fun.(text)
    :ok
  end

  defp emit_stream_token_chunk(tokenizer, emit_fun, stream_key, token_ids, :stable) do
    Enum.each(token_ids, fn token_id ->
      emit_stream_chunk(tokenizer, emit_fun, stream_key, token_id, :stable)
    end)
  end

  defp emit_stream_token_chunk(_tokenizer, _emit_fun, _stream_key, _token_ids, stream_chunking) do
    raise ArgumentError,
          "expected :stream_chunking to be :stable or :token, got: #{inspect(stream_chunking)}"
  end

  defp flush_stream_chunks(_tokenizer, _emit_fun, _stream_key, :token), do: :ok

  defp flush_stream_chunks(_tokenizer, emit_fun, stream_key, :stable) do
    chunk = Process.get(stream_key).pending_text
    if chunk != "", do: emit_fun.(chunk)
    :ok
  end

  defp decode_token(tokenizer, token_id) do
    table = decode_cache_table()
    key = {tokenizer_cache_key(tokenizer), token_id}

    case :ets.lookup(table, key) do
      [{^key, text}] ->
        text

      [] ->
        text = Bumblebee.Tokenizer.decode(tokenizer, [token_id])
        :ets.insert(table, {key, text})
        text
    end
  end

  defp decode_tokens(token_ids, tokenizer) do
    Bumblebee.Tokenizer.decode(tokenizer, token_ids)
  end

  defp decode_cache_table do
    case :ets.whereis(@decode_cache_table) do
      :undefined ->
        try do
          :ets.new(@decode_cache_table, [
            :named_table,
            :public,
            :set,
            read_concurrency: true,
            write_concurrency: true
          ])
        rescue
          ArgumentError -> @decode_cache_table
        end

      table ->
        table
    end
  end

  defp tokenizer_cache_key(%{__struct__: module, native_tokenizer: native_tokenizer, type: type}) do
    {module, type, :erlang.phash2(native_tokenizer)}
  end

  defp tokenizer_cache_key(tokenizer), do: {tokenizer.__struct__, :erlang.phash2(tokenizer)}

  defp find_last_occurrence(string, pattern) do
    case :binary.matches(string, pattern) do
      [] -> nil
      matches -> matches |> List.last() |> elem(0)
    end
  end

  defp tokens_to_batch_tensor(tokens) when is_list(tokens) and tokens != [] do
    tokens
    |> Nx.tensor(type: :s64, backend: @cpu_backend)
    |> Nx.reshape({1, :auto})
  end
end
