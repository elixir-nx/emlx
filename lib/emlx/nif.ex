defmodule EMLX.NIF do
  @moduledoc """
  Elixir bindings for MLX array operations.
  """

  for {name, arity} <- EMLX.__mlx_functions__() do
    args = Macro.generate_arguments(arity, __MODULE__)

    def unquote(name)(unquote_splicing(args)) do
      :erlang.nif_error(:nif_not_loaded)
    end
  end

  @on_load :load_nifs
  def load_nifs do
    path = :filename.join(:code.priv_dir(:emlx), ~c"libemlx")
    :erlang.load_nif(path, 0)
  end

  def to_blob(_tensor) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def to_blob(_tensor, _limit) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def memory_info do
    :erlang.nif_error(:nif_not_loaded)
  end

  def clear_cache do
    :erlang.nif_error(:nif_not_loaded)
  end

  def reset_peak_memory do
    :erlang.nif_error(:nif_not_loaded)
  end

  def set_memory_limit(_limit) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def set_cache_limit(_limit) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def tensor_data_ptr(_tensor) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def array_from_ptr(_addr, _shape, _dtype, _byte_size, _deleter) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def tensor_to_shm(_tensor, _permissions) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def array_from_shm(_name, _shape, _dtype, _byte_size) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def shm_unlink_handle(_name) do
    :erlang.nif_error(:nif_not_loaded)
  end
end
