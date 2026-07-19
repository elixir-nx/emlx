defmodule EMLXAxon.Llama.NativeChunkRunner do
  @moduledoc false

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
          "expected native Llama chunk count to be a positive integer, got: #{inspect(count)}"
  end

  defp do_run(_last_token, kv_cache, current_len, 0, _forward, tokens) do
    case tokens do
      [last_chunk | _] ->
        last_token = last_chunk[-1]
        {Enum.reverse(tokens), last_token, kv_cache, current_len}

      [] ->
        raise ArgumentError, "native Llama chunk returned no tokens"
    end
  end

  defp do_run(last_token, kv_cache, current_len, remaining, forward, tokens) do
    count = min(remaining, @max_chunk_size)
    {next_tokens, kv_cache} = forward.(last_token, kv_cache, current_len, count)

    unless match?(%Nx.Tensor{}, next_tokens) and Nx.shape(next_tokens) == {count} do
      raise ArgumentError,
            "native Llama chunk returned #{chunk_length(next_tokens)} tokens, expected #{count}"
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
