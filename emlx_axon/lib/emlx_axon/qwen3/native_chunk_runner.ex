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
    case tokens do
      [last_chunk | _] ->
        last_token = last_chunk[-1]
        {:lists.reverse(tokens), last_token, kv_cache, current_len}

      [] ->
        raise ArgumentError, "native Qwen3 chunk returned no tokens"
    end
  end

  defp do_run(last_token, kv_cache, current_len, remaining, forward, tokens) do
    count = min(remaining, @max_chunk_size)
    {next_tokens, kv_cache} = forward.(last_token, kv_cache, current_len, count)

    unless match?(%Nx.Tensor{}, next_tokens) and Nx.shape(next_tokens) == {count} do
      raise ArgumentError,
            "native Qwen3 chunk returned #{chunk_length(next_tokens)} tokens, expected #{count}"
    end

    last_token = next_tokens[-1]

    do_run(
      last_token,
      kv_cache,
      current_len + count,
      remaining - count,
      forward,
      [next_tokens | tokens]
    )
  end

  defp chunk_length(%Nx.Tensor{} = tokens), do: "tensor shape #{inspect(Nx.shape(tokens))}"
  defp chunk_length(_tokens), do: "a non-tensor value"
end
