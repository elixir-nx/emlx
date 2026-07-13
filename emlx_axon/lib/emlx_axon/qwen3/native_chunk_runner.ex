defmodule EMLXAxon.Qwen3.NativeChunkRunner do
  @moduledoc false

  # The native Qwen3 callbacks bound one invocation to keep callback policies
  # and temporary token graphs finite. Larger generation requests are split
  # here without synchronizing token tensors to the host between invocations.
  @max_chunk_size 4096

  @doc false
  def max_chunk_size, do: @max_chunk_size

  @doc false
  def run(last_token, kv_cache, current_len, count, forward)
      when is_integer(count) and count > 0 and is_function(forward, 4) do
    do_run(last_token, kv_cache, current_len, count, forward, [])
  end

  def run(_last_token, _kv_cache, _current_len, count, _forward) do
    raise ArgumentError,
          "expected native Qwen3 chunk count to be a positive integer, got: #{inspect(count)}"
  end

  defp do_run(_last_token, kv_cache, current_len, 0, _forward, tokens) do
    {:lists.reverse(tokens), kv_cache, current_len}
  end

  defp do_run(last_token, kv_cache, current_len, remaining, forward, tokens) do
    count = min(remaining, @max_chunk_size)
    {next_tokens, kv_cache} = forward.(last_token, kv_cache, current_len, count)

    unless is_list(next_tokens) and length(next_tokens) == count do
      raise ArgumentError,
            "native Qwen3 chunk returned #{chunk_length(next_tokens)} tokens, expected #{count}"
    end

    {last_token, tokens} = prepend_tokens(next_tokens, tokens)

    do_run(
      last_token,
      kv_cache,
      current_len + count,
      remaining - count,
      forward,
      tokens
    )
  end

  defp prepend_tokens([token | rest], tokens) do
    prepend_tokens(rest, token, [token | tokens])
  end

  defp prepend_tokens([], _tokens) do
    raise ArgumentError, "native Qwen3 chunk returned no tokens"
  end

  defp prepend_tokens([token | rest], _last_token, tokens) do
    prepend_tokens(rest, token, [token | tokens])
  end

  defp prepend_tokens([], last_token, tokens), do: {last_token, tokens}

  defp chunk_length(tokens) when is_list(tokens), do: length(tokens)
  defp chunk_length(_tokens), do: "a non-list value"
end
