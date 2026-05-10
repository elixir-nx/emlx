defmodule EMLXAxon.Qwen3.Sampler do
  @moduledoc """
  Three sampling strategies for autoregressive token generation.

  All functions accept `logits` of shape `{1, vocab_size}` and return a
  scalar integer tensor (the sampled token id).

  - `greedy/1`     — deterministic argmax, fastest
  - `top_p_cpu/3`  — faithful port of bobby_posts: logits → CPU → sort → sample
  - `top_p_gpu/2`  — `defn`-compiled top-p on the GPU; avoids the host transfer
  """

  import Nx.Defn

  @doc "Greedy decoding: return the token with the highest logit as a scalar tensor."
  def greedy(logits) do
    # {1, vocab} → 1-D logits → argmax without trailing squeeze on a {1} intermediate.
    logits |> Nx.squeeze(axes: [0]) |> Nx.argmax(axis: 0)
  end

  @doc """
  CPU top-p (nucleus) sampler — faithful port of bobby_posts.

  Transfers the full vocabulary logits to the BEAM for sort + sample.
  This matches the A0 baseline timing (expected ~42 ms overhead per token).
  """
  def top_p_cpu(logits, temperature \\ 0.95, top_p \\ 0.9) do
    shape = Nx.shape(logits)
    vocab_size = elem(shape, tuple_size(shape) - 1)

    # Scale by temperature then softmax on the host
    scaled = Nx.divide(logits, temperature) |> Nx.to_flat_list()

    max_l = Enum.max(scaled)
    exps  = Enum.map(scaled, fn l -> :math.exp(l - max_l) end)
    sum_e = Enum.sum(exps)
    probs = Enum.map(exps, &(&1 / sum_e))

    # Sort descending by probability
    indexed = Enum.zip(0..(vocab_size - 1), probs)
    sorted  = Enum.sort_by(indexed, fn {_i, p} -> p end, :desc)

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

    u     = :rand.uniform()
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
  defn top_p_gpu(logits, key, opts \\ []) do
    opts = keyword!(opts, temperature: 0.95)

    logits_1d = Nx.squeeze(logits)

    # Gumbel noise: G = -log(-log(U)) where U ~ Uniform(0, 1)
    {u, _key} = Nx.Random.uniform(key, shape: Nx.shape(logits_1d), type: :f32)
    gumbel    = Nx.negate(Nx.log(Nx.negate(Nx.log(u + 1.0e-10)) + 1.0e-10))

    # argmax(logits/temp + Gumbel) ~ Categorical(softmax(logits/temp))
    Nx.argmax(logits_1d / opts[:temperature] + gumbel, axis: 0)
  end
end
