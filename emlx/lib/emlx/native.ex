defmodule EMLX.Native do
  @moduledoc """
  Shared utilities for the EMLX native compiler layer.
  """

  @doc """
  Maps an `Nx.Type.t()` to the corresponding MLX type atom understood by the
  C++ NIFs. Sub-byte integer types (`{:u, 2}`, `{:u, 4}`, `{:s, 2}`, `{:s, 4}`)
  are widened to the smallest MLX integer type that can hold them.
  """
  @spec to_mlx_type(Nx.Type.t()) :: atom()
  def to_mlx_type({:u, 2}), do: :uint8
  def to_mlx_type({:u, 4}), do: :uint8
  def to_mlx_type({:u, 8}), do: :uint8
  def to_mlx_type({:u, 16}), do: :uint16
  def to_mlx_type({:u, 32}), do: :uint32
  def to_mlx_type({:u, 64}), do: :uint64
  def to_mlx_type({:s, 2}), do: :int8
  def to_mlx_type({:s, 4}), do: :int8
  def to_mlx_type({:s, 8}), do: :int8
  def to_mlx_type({:s, 16}), do: :int16
  def to_mlx_type({:s, 32}), do: :int32
  def to_mlx_type({:s, 64}), do: :int64
  def to_mlx_type({:f, 8}), do: :float16
  def to_mlx_type({:f, 16}), do: :float16
  def to_mlx_type({:f, 32}), do: :float32
  def to_mlx_type({:f, 64}), do: :float32
  def to_mlx_type({:bf, 16}), do: :bfloat16
  def to_mlx_type({:c, 64}), do: :complex64
  def to_mlx_type({:c, 128}), do: :complex64
  def to_mlx_type(:bool), do: :bool
end
