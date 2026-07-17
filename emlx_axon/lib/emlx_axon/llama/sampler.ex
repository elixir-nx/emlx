defmodule EMLXAxon.Llama.Sampler do
  @moduledoc """
  Three sampling strategies for autoregressive token generation.

  All functions accept `logits` of shape `{1, vocab_size}` and return a
  scalar integer tensor (the sampled token id).

  - `greedy/1`     — deterministic argmax, fastest
  - `top_p_cpu/3`  — nucleus sampling on the host after copying logits from the backend
  - `top_p_gpu/2`  — `defn`-compiled temperature sampling on the GPU; avoids
    the host transfer
  """

  import Nx.Defn

  @doc "Greedy decoding: return the token with the highest logit as a scalar tensor."
  def greedy(logits) do
    logits |> Nx.squeeze(axes: [0]) |> Nx.argmax(axis: 0)
  end

  @doc """
  CPU top-p (nucleus) sampler.

  Transfers the full vocabulary logits to the host, applies temperature and
  nucleus filtering, then samples from the retained probability distribution.
  """
  def top_p_cpu(logits, temperature \\ 0.95, top_p \\ 0.9) do
    validate_temperature!(temperature)
    validate_top_p!(top_p)

    shape = Nx.shape(logits)
    vocab_size = elem(shape, tuple_size(shape) - 1)

    # Scale by temperature then softmax on the host
    scaled = Nx.divide(logits, temperature) |> Nx.to_flat_list()

    max_l = Enum.max(scaled)
    exps = Enum.map(scaled, fn l -> :math.exp(l - max_l) end)
    sum_e = Enum.sum(exps)
    probs = Enum.map(exps, &(&1 / sum_e))

    # Sort descending by probability
    indexed = Enum.zip(0..(vocab_size - 1), probs)
    sorted = Enum.sort_by(indexed, fn {_i, p} -> p end, :desc)

    # Nucleus cutoff
    {chosen, _cum} =
      Enum.reduce_while(sorted, {[], 0.0}, fn {idx, p}, {acc, cum} ->
        new_cum = cum + p

        if new_cum - p >= top_p do
          {:halt, {acc, cum}}
        else
          {:cont, {[{idx, p} | acc], new_cum}}
        end
      end)

    candidates = Enum.reverse(chosen)

    # Re-normalise and sample
    total = Enum.sum(Enum.map(candidates, fn {_i, p} -> p end))
    normed = Enum.map(candidates, fn {i, p} -> {i, p / total} end)

    u = :rand.uniform()
    token = sample_from_probs(normed, u, 0.0)

    Nx.tensor(token, type: :s64)
  end

  defp sample_from_probs([], _u, _acc), do: 0

  defp sample_from_probs([{idx, p} | rest], u, acc) do
    new_acc = acc + p
    if new_acc >= u, do: idx, else: sample_from_probs(rest, u, new_acc)
  end

  @doc """
  GPU temperature sampler compiled with `defn`.

  Uses the Gumbel-max trick: `argmax(logits/temp + Gumbel_noise)` draws a
  sample proportional to `softmax(logits/temp)`. This is mathematically
  equivalent to categorical sampling without any vocabulary-wide sort, which
  avoids the MLX argsort kernel limitation at vocab_size=151936.

  Note: this implements temperature sampling without nucleus (top-p) cutoff.
  The top-p filtering requires argsort of a 151k-element tensor, which hits
  an unsupported MLX Metal kernel for this vocab size. Temperature sampling
  achieves similar randomisation and benchmarks the GPU sampler path.

  `key` must be a `Nx.Random.key/1` value passed as a positional argument
  (not through opts) so defn can trace it correctly.

  ## Options
  - `:temperature` — float, default 0.95
  """
  def top_p_gpu(logits, key, opts \\ []) do
    temperature = Keyword.get(opts, :temperature, 0.95)
    validate_temperature!(temperature)
    top_p_gpu_defn(logits, key, temperature)
  end

  defn top_p_gpu_defn(logits, key, temperature) do
    logits_1d = Nx.squeeze(logits)

    # Gumbel noise: G = -log(-log(U)) where U ~ Uniform(0, 1)
    {u, _key} = Nx.Random.uniform(key, shape: Nx.shape(logits_1d), type: :f32)
    gumbel = Nx.negate(Nx.log(Nx.negate(Nx.log(u + 1.0e-10)) + 1.0e-10))

    # argmax(logits/temp + Gumbel) ~ Categorical(softmax(logits/temp))
    Nx.argmax(logits_1d / temperature + gumbel, axis: 0)
  end

  defp validate_temperature!(temperature) when is_number(temperature) and temperature > 0,
    do: :ok

  defp validate_temperature!(temperature) do
    raise ArgumentError,
          "expected temperature to be a positive number, got: #{inspect(temperature)}"
  end

  defp validate_top_p!(top_p) when is_number(top_p) and top_p > 0 and top_p <= 1,
    do: :ok

  defp validate_top_p!(top_p) do
    raise ArgumentError,
          "expected top_p to be a number greater than 0.0 and less than or equal to 1.0, got: #{inspect(top_p)}"
  end
end
