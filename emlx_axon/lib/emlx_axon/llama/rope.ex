defmodule EMLXAxon.Llama.Rope do
  @moduledoc false

  @gpu {EMLX.Backend, device: :gpu}

  @spec freqs_from_config!(map()) :: Nx.Tensor.t()
  def freqs_from_config!(config) do
    head_dim = Map.fetch!(config, :head_dim)
    theta = Map.get(config, :rope_theta, 10_000.0)
    scaling = Map.get(config, :rope_scaling)

    freqs =
      case scaling do
        nil ->
          head_dim
          |> base_inv_freqs(theta)
          |> mlx_freqs_from_inv_freqs()

        %{type: :llama3} = opts ->
          opts
          |> llama3_inv_freqs(head_dim, theta)
          |> mlx_freqs_from_inv_freqs()

        other ->
          raise ArgumentError,
                "native dense Llama generation supports nil or :llama3 rope scaling, got: #{inspect(other)}"
      end

    Nx.backend_transfer(freqs, @gpu)
  end

  defp base_inv_freqs(head_dim, theta) do
    dims = div(head_dim, 2)

    range =
      Nx.iota({dims}, type: :f32)
      |> Nx.multiply(2.0)
      |> Nx.divide(head_dim)

    Nx.divide(1.0, Nx.pow(theta, range))
  end

  defp llama3_inv_freqs(opts, head_dim, theta) do
    factor = Map.get(opts, :factor, 8.0)
    low_frequency_factor = Map.get(opts, :low_frequency_factor, 1.0)
    high_frequency_factor = Map.get(opts, :high_frequency_factor, 4.0)
    original_max_positions = Map.get(opts, :original_max_positions, 8_192)

    inv_freq = base_inv_freqs(head_dim, theta)
    wavelength = Nx.multiply(2.0 * :math.pi(), Nx.divide(1.0, inv_freq))

    low_frequency_wavelength = original_max_positions / low_frequency_factor
    high_frequency_wavelength = original_max_positions / high_frequency_factor

    smooth_factor =
      original_max_positions
      |> Nx.divide(wavelength)
      |> Nx.subtract(low_frequency_factor)
      |> Nx.divide(high_frequency_factor - low_frequency_factor)

    smoothed =
      Nx.add(
        Nx.multiply(Nx.subtract(1.0, smooth_factor), Nx.divide(inv_freq, factor)),
        Nx.multiply(smooth_factor, inv_freq)
      )

    Nx.select(
      Nx.less(wavelength, high_frequency_wavelength),
      inv_freq,
      Nx.select(
        Nx.greater(wavelength, low_frequency_wavelength),
        Nx.divide(inv_freq, factor),
        smoothed
      )
    )
  end

  defp mlx_freqs_from_inv_freqs(inv_freqs), do: Nx.divide(1.0, inv_freqs)
end
