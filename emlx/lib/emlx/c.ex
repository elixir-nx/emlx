defmodule EMLX.C do
  @moduledoc """
  Config-selectable op dispatch: `:c` uses the official mlx-c API,
  `:cpp` uses direct mlx::core C++ calls (the reference).

      config :emlx, :mlx_api, :c
  """

  @doc """
  The MLX-facing layer selected by configuration (`:c` or `:cpp`).
  """
  def lane, do: Application.get_env(:emlx, :mlx_api, :cpp)

  defp impl, do: if(lane() == :c, do: EMLX.C.NIF, else: EMLX.C.RefNIF)

  def device_check, do: impl().device_check()
  def copy(bin, dev, n, dtype), do: impl().copy(bin, dev, n, dtype)
  def matmul(a, b, dtype, dev, m, k, n), do: impl().matmul(a, b, dtype, dev, m, k, n)
  def bmm(a, b, dtype, dev, batch, m, k, n), do: impl().bmm(a, b, dtype, dev, batch, m, k, n)
  def add(a, b, dtype, dev, m, n), do: impl().add(a, b, dtype, dev, m, n)
  def subtract(a, b, dtype, dev, m, n), do: impl().subtract(a, b, dtype, dev, m, n)
  def multiply(a, b, dtype, dev, m, n), do: impl().multiply(a, b, dtype, dev, m, n)
  def divide(a, b, dtype, dev, m, n), do: impl().divide(a, b, dtype, dev, m, n)
  def exp(a, dtype, dev, m, n), do: impl().exp(a, dtype, dev, m, n)
  def softmax_axis(a, dtype, dev, m, n, axis), do: impl().softmax_axis(a, dtype, dev, m, n, axis)
  def sum(a, dtype, dev, m, n), do: impl().sum(a, dtype, dev, m, n)
  def transpose(a, dtype, dev, m, n), do: impl().transpose(a, dtype, dev, m, n)
  def astype(a, from, to, dev, n), do: impl().astype(a, from, to, dev, n)
  def reshape(a, dtype, dev, shape), do: impl().reshape(a, dtype, dev, shape)
  def svd(a, dev, n), do: impl().svd(a, dev, n)
  def attn(q, k, v, scale, dev, b, h, s, d), do: impl().attn(q, k, v, scale, dev, b, h, s, d)
  def tiny_loop(a, b, dev, iters, n), do: impl().tiny_loop(a, b, dev, iters, n)
end
