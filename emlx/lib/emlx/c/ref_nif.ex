defmodule EMLX.C.RefNIF do
  @moduledoc false
  @on_load :load_nifs

  def load_nifs do
    path =
      System.get_env("EMLX_C_REF_NIF_PATH") ||
        Path.join(:code.priv_dir(:emlx) |> to_string(), "libemlx_ref")

    :erlang.load_nif(String.to_charlist(path), 0)
  end

  def device_check, do: :erlang.nif_error(:nif_not_loaded)
  def copy(_bin, _dev, _n, _dtype), do: :erlang.nif_error(:nif_not_loaded)
  def matmul(_a, _b, _dtype, _dev, _m, _k, _n), do: :erlang.nif_error(:nif_not_loaded)
  def bmm(_a, _b, _dtype, _dev, _batch, _m, _k, _n), do: :erlang.nif_error(:nif_not_loaded)
  def add(_a, _b, _dtype, _dev, _m, _n), do: :erlang.nif_error(:nif_not_loaded)
  def subtract(_a, _b, _dtype, _dev, _m, _n), do: :erlang.nif_error(:nif_not_loaded)
  def multiply(_a, _b, _dtype, _dev, _m, _n), do: :erlang.nif_error(:nif_not_loaded)
  def divide(_a, _b, _dtype, _dev, _m, _n), do: :erlang.nif_error(:nif_not_loaded)
  def exp(_a, _dtype, _dev, _m, _n), do: :erlang.nif_error(:nif_not_loaded)
  def softmax_axis(_a, _dtype, _dev, _m, _n, _axis), do: :erlang.nif_error(:nif_not_loaded)
  def sum(_a, _dtype, _dev, _m, _n), do: :erlang.nif_error(:nif_not_loaded)
  def transpose(_a, _dtype, _dev, _m, _n), do: :erlang.nif_error(:nif_not_loaded)
  def astype(_a, _from, _to, _dev, _n), do: :erlang.nif_error(:nif_not_loaded)
  def reshape(_a, _dtype, _dev, _shape), do: :erlang.nif_error(:nif_not_loaded)
  def svd(_a, _dev, _n), do: :erlang.nif_error(:nif_not_loaded)
  def attn(_q, _k, _v, _scale, _dev, _b, _h, _s, _d), do: :erlang.nif_error(:nif_not_loaded)
  def tiny_loop(_a, _b, _dev, _iters, _n), do: :erlang.nif_error(:nif_not_loaded)
end
