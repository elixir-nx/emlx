defmodule EMLX.Backend do
  @behaviour Nx.Backend

  alias Nx.Tensor, as: T
  alias EMLX.Backend, as: Backend

  defstruct [:ref, :shape, :type, :data]

  @impl true
  def init(opts) do
    Keyword.validate!(opts, device: :cpu)
  end

  @doc """
  Converts from an Nx tensor to an MLX array.
  """
  def from_nx(%T{data: %Backend{ref: device_ref}}), do: device_ref
  def from_nx(%T{} = tensor), do: Nx.backend_transfer(tensor, Backend) |> from_nx()

  @doc """
  Converts an MLX array back to an Nx tensor.
  """
  def to_nx({device, ref} = device_ref, %T{type: type, shape: shape} = t)
      when is_atom(device) and is_reference(ref) do
    # Get the MLX array's type
    mlx_type = EMLX.scalar_type(device_ref)

    # Convert if needed (similar to the torch byte conversion)
    array =
      if needs_type_conversion?(type, mlx_type) do
        EMLX.to_type(device_ref, nx_type_to_mlx(type))
      else
        device_ref
      end

    %T{t | data: %Backend{ref: check_shape_and_type!(array, shape, type)}}
  end

  @impl true
  def inspect(%T{} = tensor, inspect_opts) do
    limit = if inspect_opts.limit == :infinity, do: :infinity, else: inspect_opts.limit + 1

    tensor
    |> to_binary(min(limit, Nx.size(tensor)))
    |> then(&Nx.Backend.inspect(tensor, &1, inspect_opts))
    |> maybe_add_signature(tensor)
  end

  @impl true
  def to_binary(tensor, limit) do
    blob = EMLX.to_blob(from_nx(tensor), limit)
    # IO.inspect(blob, label: "Raw blob from MLX")
    # IO.inspect(byte_size(blob), label: "Blob size in bytes")
    # IO.inspect(tensor.type, label: "Tensor type")
    # IO.inspect(limit, label: "Requested limit")

    # result =
    #   case tensor.type do
    #     {:u, 32} ->
    #       converted = for <<x::64-native <- blob>>, do: <<x::32-native>>, into: <<>>
    #       IO.inspect(converted, label: "Converted u32 blob")
    #       converted

    #     {:u, 16} ->
    #       converted = for <<x::32-native <- blob>>, do: <<x::16-native>>, into: <<>>
    #       IO.inspect(converted, label: "Converted u16 blob")
    #       converted

    #     _ ->
    #       IO.inspect(blob, label: "Unchanged blob")
    #       blob
    #   end

    # IO.inspect(byte_size(result), label: "Final result size in bytes")
    # result
  end

  @impl true
  def from_binary(%T{type: type, shape: shape} = out, binary, backend_options) do
    binary
    |> maybe_pad_binary(type)
    |> EMLX.from_blob(
      shape,
      nx_type_to_mlx(type),
      device_option(backend_options)
    )
    |> to_nx(out)
  end

  defp maybe_pad_binary(bin, {:u, size}) when size in [16, 32] do
    double_size = size * 2
    for <<x::native-size(size) <- bin>>, into: <<>>, do: <<x::native-size(double_size)>>
  end

  defp maybe_pad_binary(bin, {:u, size}) when size in [2, 4] do
    for <<x::native-size(size) <- bin>>, into: <<>>, do: <<x::native-8>>
  end

  defp maybe_pad_binary(bin, {:s, size}) when size in [2, 4] do
    for <<x::native-signed-size(size) <- bin>>, into: <<>>, do: <<x::native-signed-8>>
  end

  defp maybe_pad_binary(bin, _), do: bin

  defp maybe_add_signature(result, %T{data: %Backend{ref: _}}) do
    Inspect.Algebra.concat([
      "EMLX.Backend",
      Inspect.Algebra.line(),
      result
    ])
  end

  # Helper functions
  defp needs_type_conversion?({:u, 8}, :bool), do: true
  defp needs_type_conversion?(_, _), do: false

  defp nx_type_to_mlx({:u, 8}), do: :uint8
  defp nx_type_to_mlx({:u, 16}), do: :uint16
  defp nx_type_to_mlx({:u, 32}), do: :uint32
  defp nx_type_to_mlx({:u, 64}), do: :uint64
  defp nx_type_to_mlx({:s, 8}), do: :int8
  defp nx_type_to_mlx({:s, 16}), do: :int16
  defp nx_type_to_mlx({:s, 32}), do: :int32
  defp nx_type_to_mlx({:s, 64}), do: :int64
  defp nx_type_to_mlx({:f, 16}), do: :float16
  defp nx_type_to_mlx({:f, 32}), do: :float32
  defp nx_type_to_mlx(:bool), do: :bool

  defp check_shape_and_type!(device_ref, expected_shape, expected_type) do
    actual_shape = EMLX.shape(device_ref)
    actual_type = EMLX.scalar_type(device_ref) |> mlx_type_to_nx()

    if actual_shape != expected_shape do
      raise ArgumentError, """
      Shape mismatch in MLX array conversion:
      Expected shape: #{inspect(expected_shape)}
      Got shape: #{inspect(actual_shape)}
      """
    end

    case {actual_type, expected_type} do
      {{:s, 8}, {:s, qint}} when qint in [2, 4] ->
        :ok

      {{:u, 8}, {:u, qint}} when qint in [2, 4] ->
        :ok

      {{:s, 32}, {:u, 16}} ->
        :ok

      {{:s, 64}, {:u, 32}} ->
        :ok

      {{:s, 64}, {:u, 64}} ->
        :ok

      {{:u, 8}, {:u, 32}} ->
        :ok

      _ when actual_type != expected_type ->
        raise "type mismatch in EMLX: expected #{inspect(expected_type)}, got: #{inspect(actual_type)}. " <>
                "Please report this bug"

      _ ->
        :ok
    end

    device_ref
  end

  defp mlx_type_to_nx(:uint8), do: {:u, 8}
  defp mlx_type_to_nx(:uint16), do: {:u, 16}
  defp mlx_type_to_nx(:uint32), do: {:u, 32}
  defp mlx_type_to_nx(:uint64), do: {:u, 64}
  defp mlx_type_to_nx(:int8), do: {:s, 8}
  defp mlx_type_to_nx(:int16), do: {:s, 16}
  defp mlx_type_to_nx(:int32), do: {:s, 32}
  defp mlx_type_to_nx(:int64), do: {:s, 64}
  defp mlx_type_to_nx(:float16), do: {:f, 16}
  defp mlx_type_to_nx(:float32), do: {:f, 32}
  defp mlx_type_to_nx(:bool), do: :bool

  @impl true
  def constant(
        %T{shape: shape, names: names, type: type} = out,
        scalar,
        backend_options
      )
      when scalar in [:infinity, :neg_infinity, :nan] do
    t = apply(Nx.Constants, scalar, [type, [backend: {Backend, backend_options}]])
    Nx.broadcast(t, shape, names: names)
  end

  @impl true
  def constant(%T{shape: {}, type: type} = out, scalar, backend_options) do
    scalar
    |> constant_serialize_scalar()
    |> EMLX.scalar_tensor(nx_type_to_mlx(type), device_option(backend_options))
    |> to_nx(out)
  end

  # FIXME: Use `full` like torchx
  # def constant(%T{shape: shape, type: type} = out, scalar, _backend_options) do
  #   shape_list = Tuple.to_list(shape)

  #   scalar
  #   |> constant_serialize_scalar()
  # end

  @impl true
  def sum(%T{} = out, %T{} = t, opts) do
    axes = opts[:axes] || []
    keep_axes = opts[:keep_axes] || false

    # Calculate the expected output shape based on the input shape and axes
    result =
      t
      |> from_nx()
      |> EMLX.sum(axes, keep_axes)

    # Get the actual shape after summation
    actual_shape = EMLX.shape(result)

    # Create a new output tensor with the correct shape
    %{out | shape: actual_shape}
    |> then(&to_nx(result, &1))
  end

  # Helper function to handle different scalar types
  defp constant_serialize_scalar(scalar) when is_number(scalar), do: scalar
  defp constant_serialize_scalar(%Complex{} = c), do: Complex.abs(c)

  defp device_option(nil), do: :cpu
  defp device_option(backend_opts), do: backend_opts[:device] || :cpu
end
