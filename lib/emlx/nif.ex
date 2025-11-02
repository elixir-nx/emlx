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

  # Compilation functions
  def compile(_args, _tag) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def call_compiled_cpu(_compiled_fun, _args) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def call_compiled_gpu(_compiled_fun, _args) do
    :erlang.nif_error(:nif_not_loaded)
  end

  # nif_call callback
  def nif_call_evaluated(_resource, _result) do
    :erlang.nif_error(:nif_not_loaded)
  end
end
