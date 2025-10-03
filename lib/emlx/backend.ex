defmodule EMLX.Backend do
  @behaviour Nx.Backend

  alias Nx.Tensor, as: T
  alias EMLX.Backend, as: Backend

  require Logger

  defstruct [:ref, :shape, :type, :data]

  @impl true
  def init(opts) do
    Keyword.validate!(opts, device: :cpu)
  end

  @doc """
  Converts from an Nx tensor to an MLX array.
  """
  def from_nx(%T{data: %Backend{ref: device_ref}}) do
    device_ref
  end

  def from_nx(%T{} = other_backend), do: Nx.backend_transfer(other_backend, Backend) |> from_nx()

  @doc """
  Converts an MLX array to an Nx tensor.
  """
  def to_nx({device, ref} = device_ref)
      when is_atom(device) and is_reference(ref) do
    # Get the MLX array's type
    mlx_type = EMLX.scalar_type(device_ref)
    shape = EMLX.shape(device_ref)

    to_nx(device_ref, Nx.template(shape, to_nx_type(mlx_type)))
  end

  @doc """
  Converts an MLX array back to an Nx tensor with type and shape assertions.
  """
  def to_nx({device, ref} = device_ref, %T{type: type, shape: shape} = t)
      when is_atom(device) and is_reference(ref) do
    # Get the MLX array's type

    mlx_type = EMLX.scalar_type(device_ref)

    # Convert if needed (similar to the torch byte conversion)
    array =
      if needs_type_conversion?(type, mlx_type) do
        EMLX.astype(device_ref, to_mlx_type(type))
      else
        device_ref
      end

    %T{
      t
      | data: %Backend{ref: check_shape_and_type!(array, shape, type), shape: shape, type: type}
    }
  end

  @impl true
  def backend_copy(%T{type: type, shape: shape} = tensor, backend, opts) do
    Nx.from_binary(to_binary(tensor, Nx.size(tensor)), type, backend: {backend, opts})
    |> Nx.reshape(shape)
  end

  @impl true
  def backend_transfer(tensor, backend, opts) do
    new_tensor = backend_copy(tensor, backend, opts)
    backend_deallocate(tensor)
    new_tensor
  end

  @impl true
  def backend_deallocate(%T{data: %Backend{ref: ref}}) do
    EMLX.deallocate(ref)
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
    EMLX.to_blob(from_nx(tensor), limit)
    |> maybe_modify_binary(to_nx_type(to_mlx_type(tensor.type)), tensor.type)
  end

  @impl true
  def from_binary(%T{type: type, shape: shape} = out, binary, backend_options) do
    binary
    |> maybe_modify_binary(type, to_nx_type(to_mlx_type(type)))
    |> EMLX.from_blob(
      shape,
      to_mlx_type(type),
      device_option(backend_options)
    )
    |> to_nx(out)
  end

  defp maybe_modify_binary(binary, type, type), do: binary

  defp maybe_modify_binary(binary, {:f, 8}, {:f, 16}) do
    for <<byte::8 <- binary>>, into: <<>> do
      case read_f8(<<byte::8>>) do
        :nan ->
          Nx.Type.nan_binary({:f, 16})

        :infinity ->
          Nx.Type.infinity_binary({:f, 16})

        :neg_infinity ->
          Nx.Type.neg_infinity_binary({:f, 16})

        number ->
          <<number::float-native-size(16)>>
      end
    end
  end

  defp maybe_modify_binary(binary, {:f, 16}, {:f, 8}) do
    for <<float::16 <- binary>>, into: <<>> do
      case <<float::16-native>> do
        <<float::float-native-16>> -> write_finite_f8(float)
        <<0xFC00::16-native>> -> write_non_finite(:neg_infinity, 8)
        <<0x7C00::16-native>> -> write_non_finite(:infinity, 8)
        _ -> write_non_finite(:nan, 8)
      end
    end
  end

  defp maybe_modify_binary(binary, {:f, 64}, {:f, 32}) do
    for <<float::64 <- binary>>, into: <<>> do
      case <<float::64>> do
        <<float::float-native-64>> -> <<float::float-native-size(32)>>
        <<0xFFF0000000000000::64-native>> -> write_non_finite(:neg_infinity, 32)
        <<0x7FF0000000000000::64-native>> -> write_non_finite(:infinity, 32)
        _ -> write_non_finite(:nan, 32)
      end
    end
  end

  defp maybe_modify_binary(binary, {:f, 32}, {:f, 64}) do
    for <<float::32 <- binary>>, into: <<>> do
      case <<float::32>> do
        <<float::float-native-32>> -> <<float::float-native-size(64)>>
        <<0xFF800000::32-native>> -> write_non_finite(:neg_infinity, 64)
        <<0x7F800000::32-native>> -> write_non_finite(:infinity, 64)
        _ -> write_non_finite(:nan, 64)
      end
    end
  end

  defp maybe_modify_binary(binary, {:u, size}, {:u, 8}) when size in [2, 4] do
    for <<bits::integer-native-size(size) <- binary>>, into: <<>> do
      <<bits::integer-native-size(8)>>
    end
  end

  defp maybe_modify_binary(binary, {:u, 8}, {:u, size}) when size in [2, 4] do
    for <<bits::integer-native-size(8) <- binary>>, into: <<>> do
      <<bits::integer-native-size(size)>>
    end
  end

  defp read_f8(<<0xFC::8-native>>), do: :neg_infinity
  defp read_f8(<<0x7C::8-native>>), do: :infinity
  defp read_f8(<<_sign::1, 31::5, mantissa::2>>) when mantissa != 0, do: :nan

  defp read_f8(<<sign::1, exp::5, mantissa::2>>) do
    float = :math.pow(2, exp - 15) * (1 + mantissa / 4)

    case sign do
      0 -> float
      _ -> -float
    end
  end

  if System.endianness() == :little do
    def write_finite_f8(x) do
      binary_part(<<x::float-native-16>>, 0, 1)
    end
  else
    def write_finite_f8(x) do
      binary_part(<<x::float-native-16>>, 1, 1)
    end
  end

  for size <- [8, 16, 32, 64] do
    def write_non_finite(data, unquote(size)) do
      case data do
        :infinity -> unquote(Nx.Type.infinity_binary({:f, size}))
        :neg_infinity -> unquote(Nx.Type.neg_infinity_binary({:f, size}))
        :nan -> unquote(Nx.Type.nan_binary({:f, size}))
      end
    end
  end

  @impl true
  def slice(
        out,
        %T{shape: input_shape} = t,
        start_indices,
        lengths,
        strides
      ) do
    t
    |> from_nx()
    |> mlx_slice(input_shape, start_indices, lengths, strides)
    |> to_nx(out)
  end

  defp mlx_slice(t, input_shape, start_indices, lengths, strides) do
    starts =
      start_indices
      |> Enum.zip(lengths)
      |> Enum.with_index(fn {start, len}, axis ->
        min(to_number(start), elem(input_shape, axis) - len)
      end)

    stops = Enum.zip_with(starts, lengths, &(&1 + &2))

    EMLX.slice(t, starts, stops, strides)
  end

  @impl true
  def squeeze(out, tensor, axes) do
    tensor
    |> from_nx()
    |> EMLX.squeeze(axes)
    |> to_nx(out)
  end

  @impl true
  def transpose(out, tensor, axes) do
    tensor
    |> from_nx()
    |> EMLX.transpose(axes)
    |> to_nx(out)
  end

  @impl true
  def reverse(out, tensor, axes) do
    shape = Tuple.to_list(tensor.shape)

    {starts_stops, strides} =
      shape
      |> Enum.with_index()
      |> Enum.map(fn {dim, idx} ->
        if idx in axes do
          # For reversed axes: start from end, stop at -1, stride backwards
          {{dim - 1, -dim - 1}, -1}
        else
          # For normal axes: start at 0, go to dim, stride forward
          {{0, dim}, 1}
        end
      end)
      |> Enum.unzip()

    {starts, stops} = Enum.unzip(starts_stops)

    tensor
    |> from_nx()
    |> EMLX.slice(starts, stops, strides)
    |> to_nx(out)
  end

  @impl true
  def pad(out, tensor, pad_value, input_config) do
    {axes, low_pad_size, high_pad_size} =
      input_config
      |> Enum.with_index()
      |> Enum.reduce({[], [], []}, fn
        {{low, high, _}, i}, {axes, lows, highs} ->
          {[i | axes], [max(low, 0) | lows], [max(high, 0) | highs]}
      end)

    {_device, pad_value_mx} = from_nx(pad_value)

    interior_padding = Enum.map(input_config, fn {_low, _high, interior} -> interior end)

    tensor
    |> from_nx()
    |> interior_padding_mlx(pad_value_mx, interior_padding)
    |> slice_negative_padding(input_config)
    |> EMLX.pad(axes, low_pad_size, high_pad_size, pad_value_mx)
    |> to_nx(out)
  end

  defp interior_padding(tensor, value, padding_config) do
    new_shape = Tuple.insert_at(EMLX.shape(tensor), tuple_size(EMLX.shape(tensor)), 1)
    tensor = EMLX.reshape(tensor, new_shape)

    {final_tensor, _} =
      Enum.reduce(padding_config, {tensor, 0}, fn interior, {acc, axis_index} ->
        new_tensor = apply_interior_padding(acc, value, axis_index, interior, EMLX.shape(acc))
        {new_tensor, axis_index + 1}
      end)

    final_tensor
    |> EMLX.squeeze([-1])
  end

  defp apply_interior_padding(tensor, _value, _axis_index, 0, _shape) do
    tensor
  end

  defp apply_interior_padding(tensor, value, axis_index, interior_padding, shape) do
    rank = tuple_size(shape)
    next_axis = axis_index + 1
    axis_size = elem(shape, axis_index)
    next_axis_size = elem(shape, next_axis)

    lows = [0]
    highs = [next_axis_size * interior_padding]
    axes = [next_axis]

    padded_tensor = EMLX.pad(tensor, axes, lows, highs, value)

    new_axis_size = axis_size + axis_size * interior_padding

    new_shape =
      shape
      |> put_elem(axis_index, new_axis_size)
      |> put_elem(axis_index + 1, next_axis_size)

    lengths =
      new_shape
      |> Tuple.to_list()
      |> List.replace_at(axis_index, new_axis_size - interior_padding)

    starts = List.duplicate(0, rank)
    strides = List.duplicate(1, rank)

    padded_tensor
    |> EMLX.reshape(new_shape)
    |> mlx_slice(new_shape, starts, lengths, strides)
  end

  defp slice_negative_padding(t_mx, input_config) do
    if Enum.any?(input_config, fn {pre, post, _} -> pre < 0 or post < 0 end) do
      shape = EMLX.shape(t_mx)

      {starts, stops} =
        input_config
        |> Enum.with_index(fn {pre, post, _inner}, axis ->
          start =
            if pre < 0 do
              -pre
            else
              0
            end

          axis_size = elem(shape, axis)

          stop =
            if post < 0 do
              axis_size + post
            else
              axis_size
            end

          {start, stop}
        end)
        |> Enum.unzip()

      strides = List.duplicate(1, tuple_size(shape))
      EMLX.slice(t_mx, starts, stops, strides)
    else
      t_mx
    end
  end

  @impl true
  def bitcast(out, tensor) do
    tensor
    |> from_nx()
    |> EMLX.view(to_mlx_type(out.type))
    |> to_nx(out)
  end

  if Application.compile_env(:emlx, :add_backend_on_inspect, true) do
    defp maybe_add_signature(result, %T{data: %Backend{ref: {device, ref}}}) do
      ~c"#Ref<" ++ rest = :erlang.ref_to_list(ref)

      Inspect.Algebra.concat([
        "EMLX.Backend<#{device}, ",
        List.to_string(rest),
        Inspect.Algebra.line(),
        result
      ])
    end
  else
    defp maybe_add_signature(result, _), do: result
  end

  # Helper functions
  defp needs_type_conversion?({:u, 8}, :bool), do: true
  defp needs_type_conversion?(_, _), do: false

  defp to_mlx_type({:u, 2}), do: :uint8
  defp to_mlx_type({:u, 4}), do: :uint8
  defp to_mlx_type({:u, 8}), do: :uint8
  defp to_mlx_type({:u, 16}), do: :uint16
  defp to_mlx_type({:u, 32}), do: :uint32
  defp to_mlx_type({:u, 64}), do: :uint64
  defp to_mlx_type({:s, 2}), do: :int8
  defp to_mlx_type({:s, 4}), do: :int8
  defp to_mlx_type({:s, 8}), do: :int8
  defp to_mlx_type({:s, 16}), do: :int16
  defp to_mlx_type({:s, 32}), do: :int32
  defp to_mlx_type({:s, 64}), do: :int64
  defp to_mlx_type({:f, 8}), do: :float16
  defp to_mlx_type({:f, 16}), do: :float16
  defp to_mlx_type({:f, 32}), do: :float32
  defp to_mlx_type({:f, 64}), do: :float32
  defp to_mlx_type({:bf, 16}), do: :bfloat16
  defp to_mlx_type({:c, 64}), do: :complex64
  defp to_mlx_type({:c, 128}), do: :complex64
  defp to_mlx_type(:bool), do: :bool

  defp to_nx_type(:uint8), do: {:u, 8}
  defp to_nx_type(:uint16), do: {:u, 16}
  defp to_nx_type(:uint32), do: {:u, 32}
  defp to_nx_type(:uint64), do: {:u, 64}
  defp to_nx_type(:int8), do: {:s, 8}
  defp to_nx_type(:int16), do: {:s, 16}
  defp to_nx_type(:int32), do: {:s, 32}
  defp to_nx_type(:int64), do: {:s, 64}
  defp to_nx_type(:float16), do: {:f, 16}
  defp to_nx_type(:float32), do: {:f, 32}
  defp to_nx_type(:bfloat16), do: {:bf, 16}
  defp to_nx_type(:complex64), do: {:c, 64}
  defp to_nx_type(:bool), do: :bool

  defp to_number(n) when is_number(n), do: n
  defp to_number(%T{} = t), do: t |> from_nx() |> EMLX.item()

  defp check_shape_and_type!(device_ref, expected_shape, expected_type) do
    actual_shape = EMLX.shape(device_ref)
    actual_type = EMLX.scalar_type(device_ref) |> to_nx_type()

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

      {{:f, 16}, {:f, 8}} ->
        :ok

      {{:f, 32}, {:f, 64}} ->
        :ok

      {{:c, 64}, {:c, 128}} ->
        :ok

      _ when actual_type != expected_type ->
        raise "type mismatch in EMLX: expected #{inspect(expected_type)}, got: #{inspect(actual_type)}. " <>
                "Please report this bug"

      _ ->
        :ok
    end

    device_ref
  end

  @impl true
  def constant(
        %T{shape: shape, names: names, type: type},
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
    |> EMLX.scalar_tensor(to_mlx_type(type), device_option(backend_options))
    |> to_nx(out)
  end

  def constant(%T{shape: shape, type: type} = out, scalar, backend_options) do
    scalar
    |> constant_serialize_scalar()
    |> EMLX.full(shape, to_mlx_type(type), device_option(backend_options))
    |> to_nx(out)
  end

  @impl true
  def iota(%{shape: {}} = out, nil, backend_options) do
    constant(out, 0, backend_options)
  end

  def iota(%T{shape: shape, type: type} = out, nil, backend_options) do
    EMLX.arange(
      0,
      Nx.size(shape),
      1,
      Nx.Type.integer?(type),
      device_option(backend_options)
    )
    |> EMLX.astype(to_mlx_type(type))
    |> EMLX.reshape(shape)
    |> to_nx(out)
  end

  @impl true
  def iota(%T{shape: {n}, type: type} = out, 0, backend_options) do
    EMLX.arange(0, n, 1, Nx.Type.integer?(type), device_option(backend_options))
    |> EMLX.astype(to_mlx_type(type))
    |> to_nx(out)
  end

  def iota(%T{shape: shape, type: type} = out, axis, backend_options) do
    # gets the size of iota
    dim = elem(shape, axis)

    # build the iota in one dimension
    aten =
      EMLX.arange(0, dim, 1, Nx.Type.integer?(type), device_option(backend_options))
      |> EMLX.astype(to_mlx_type(type))

    # reshape the tensor above to be have shape where everything is 1, except for dim
    reshape = Tuple.duplicate(1, Nx.rank(shape)) |> put_elem(axis, dim)
    aten = EMLX.reshape(aten, reshape)

    # Now broadcast the tensor using the original shape
    EMLX.broadcast_to(aten, shape) |> to_nx(out)
  end

  # Aggregation (axes)
  ops = [:all, :any, :sum, :product]

  for op <- ops do
    @impl true
    def unquote(op)(out, tensor, opts) do
      axes = opts[:axes] || []
      keep_axes = opts[:keep_axes] || false

      # Calculate the expected output shape based on the input shape and axes
      result =
        tensor
        |> from_nx()
        |> EMLX.unquote(op)(axes, keep_axes)
        |> EMLX.astype(to_mlx_type(out.type))

      # Get the actual shape after summation
      actual_shape = EMLX.shape(result)
      # FIXME: MLX returns whatever the original type is, but Nx expects u8 -> u32
      # scalar_type = EMLX.scalar_type(result)

      # Create a new output tensor with the correct shape
      %{out | shape: actual_shape}
      |> then(&to_nx(result, &1))
    end
  end

  # Aggregation (axis)
  ops = [:argmax, :argmin]

  for op <- ops do
    @impl true
    def unquote(op)(out, tensor, opts) do
      axis = opts[:axis]
      keep_axis = opts[:keep_axis] == true

      if Application.get_env(:emlx, :warn_unsupported_option, true) and opts[:tie_break] == :high do
        Logger.warning(
          "Nx.Backend.#{unquote(op)}/3 with tie_break: :high is not supported in EMLX"
        )
      end

      t_mx = from_nx(tensor)

      result =
        if axis do
          EMLX.unquote(op)(t_mx, axis, keep_axis)
        else
          EMLX.unquote(op)(t_mx, keep_axis)
        end

      result
      |> EMLX.astype(to_mlx_type(out.type))
      |> to_nx(out)
    end
  end

  ops = [:cumulative_sum, :cumulative_product, :cumulative_max, :cumulative_min]

  for op <- ops do
    @impl true
    def unquote(op)(out, tensor, opts) do
      axis = opts[:axis] || 0
      reverse = opts[:reverse] || false

      # Calculate the expected output shape based on the input shape and axes
      inclusive = true

      result =
        tensor
        |> from_nx()
        |> EMLX.unquote(op)(axis, reverse, inclusive)
        |> EMLX.astype(to_mlx_type(out.type))

      # Get the actual shape after summation
      actual_shape = EMLX.shape(result)
      # FIXME: MLX returns whatever the original type is, but Nx expects u8 -> u32
      # scalar_type = EMLX.scalar_type(result)

      # Create a new output tensor with the correct shape
      %{out | shape: actual_shape}
      |> then(&to_nx(result, &1))
    end
  end

  @impl true
  def stack(out, tensors, axis) do
    tensors
    |> Enum.map(&from_nx/1)
    |> EMLX.stack(axis)
    |> to_nx(out)
  end

  @impl true
  def concatenate(out, tensors, axis) do
    tensors
    |> Enum.map(&from_nx/1)
    |> EMLX.concatenate(axis)
    |> to_nx(out)
  end

  @impl true
  def put_slice(out, input, start_indices_unbounded, slice) do
    input_mx = from_nx(input)

    slice_shape_list = Tuple.to_list(slice.shape)

    zip_indices_input = [Tuple.to_list(input.shape), start_indices_unbounded, slice_shape_list]

    {start_indices, stop_indices} =
      Enum.zip_with(zip_indices_input, fn [dim_size, idx, len] ->
        idx = Nx.to_number(idx)
        start = min(max(idx, 0), dim_size - len)
        {start, start + len}
      end)
      |> Enum.unzip()

    slice_mx = slice |> from_nx() |> EMLX.astype(to_mlx_type(out.type))

    input_mx
    |> EMLX.astype(to_mlx_type(out.type))
    |> EMLX.slice_update(slice_mx, start_indices, stop_indices)
    |> to_nx(out)
  end

  @impl true
  def select(out, pred, on_true, on_false) do
    on_true = Nx.as_type(on_true, Nx.type(out))
    on_false = Nx.as_type(on_false, Nx.type(out))
    on_true_torch = from_nx(on_true)
    on_false_torch = from_nx(on_false)

    # Use logical_not to convert any tensor to a boolean tensor
    # because of that, we have to swap true/false tensor
    pred
    |> from_nx()
    |> EMLX.logical_not()
    |> EMLX.where(on_false_torch, on_true_torch)
    |> to_nx(out)
  end

  @impl true
  def take_along_axis(out, tensor, idx, opts) do
    axis = opts[:axis]

    tensor
    |> from_nx()
    |> EMLX.take_along_axis(from_nx(idx), axis)
    |> to_nx(out)
  end

  @impl true
  def take(out, tensor, indices, opts) do
    axis = opts[:axis]

    tensor
    |> from_nx()
    |> EMLX.take(from_nx(indices), axis)
    |> to_nx(out)
  end

  @impl true
  def eye(%T{shape: shape, type: type} = out, backend_options) do
    rank = tuple_size(shape)
    m = elem(shape, rank - 2)
    n = elem(shape, rank - 1)

    EMLX.eye(m, n, to_mlx_type(type), device_option(backend_options))
    |> EMLX.broadcast_to(shape)
    |> to_nx(out)
  end

  defp move_channels_last(list) do
    [batch, channels | spatial] = list
    List.flatten([batch, spatial, channels])
  end

  @impl true
  def conv(%T{type: {:c, _}}, input, kernel, opts) do
    # MLX doesn't support complex inputs,
    # so we rely on the fact that a convolution is effectively
    # a sliding dot product. We can then decompose the dot product
    # of a complex-valued pair of tensors into dot products involving
    # their real and imaginary components.

    # For example, given a tensor v = [a1+b1i c1+d1i] and a
    # kernel k = [a2+b2i c2+d2i], we can expand the dot product into
    # (a1+b1i)(a2+b2i) + (c1+d1i)(c2+d2i)
    # = (a1a2 - b1b2) + (a1b2+a2b1)i + (c1c2 - d1d2) + (c1d2+c2d1)i
    # = (a1a2 + c1c2) - (b1b2 + d1d2) + i[(a1b2 + c1d2) + (a2b1  + c2d1)]
    # = ([a1 c1].[a2 c2] - [b1 d1].[b2 d2]) + i([a1 c1].[b2 d2] + [a2 c2].[b1 d1])
    # = (real(v).real(k) - imag(v).imag(k)) + i(real(v).imag(k) + imag(v).real(k))

    # With the result above, we can turn z = conv(t, k) where either t or k are complex
    # into:
    # real_part = conv(real(t), real(k)) - conv(imag(t), imag(k))
    # imag_part = conv(real(t), imag(k)) + conv(imag(t), real(k))
    # z = complex(real_part, imag_part)

    real_input = Nx.real(input)
    imag_input = Nx.imag(input)
    real_kernel = Nx.real(kernel)
    imag_kernel = Nx.imag(kernel)

    real_part =
      Nx.subtract(
        Nx.conv(real_input, real_kernel, opts),
        Nx.conv(imag_input, imag_kernel, opts)
      )

    imag_part =
      Nx.add(
        Nx.conv(real_input, imag_kernel, opts),
        Nx.conv(imag_input, real_kernel, opts)
      )

    Nx.complex(real_part, imag_part)
  end

  def conv(out, input, kernel, opts) do
    input_permutation = opts[:input_permutation]
    kernel_permutation = opts[:kernel_permutation]
    output_permutation = opts[:output_permutation]
    strides = opts[:strides]
    padding = opts[:padding]
    input_dilation = opts[:input_dilation]
    kernel_dilation = opts[:kernel_dilation]
    feature_group_count = opts[:feature_group_size]
    batch_group_size = opts[:batch_group_size]

    if batch_group_size != 1 do
      raise "MLX doesn't support batch group size"
    end

    permute_channels_last =
      Enum.to_list(0..(tuple_size(input.shape) - 1)) |> move_channels_last()

    input_mx =
      from_nx(input)
      |> EMLX.astype(to_mlx_type(out.type))
      |> EMLX.transpose(input_permutation)
      |> EMLX.transpose(permute_channels_last)

    permute_channels_last =
      Enum.to_list(0..(tuple_size(kernel.shape) - 1)) |> move_channels_last()

    kernel_mx =
      from_nx(kernel)
      |> EMLX.astype(to_mlx_type(out.type))
      |> EMLX.transpose(kernel_permutation)
      |> EMLX.transpose(permute_channels_last)

    {padding_low, padding_high} = Enum.unzip(padding)

    [batch | spatial_and_channels] = Enum.to_list(0..(tuple_size(out.shape) - 1))

    {channels, spatial} = List.pop_at(spatial_and_channels, -1)

    permute_channels_first = [batch, channels | spatial]

    # The permutation that Nx.Shape expects is actually the reverse permutation
    # for the given config
    output_permutation =
      output_permutation
      |> Enum.with_index()
      |> Enum.sort()
      |> Enum.map(&elem(&1, 1))

    input_mx
    |> EMLX.conv_general(
      kernel_mx,
      strides,
      padding_low,
      padding_high,
      kernel_dilation,
      input_dilation,
      feature_group_count
    )
    |> EMLX.transpose(permute_channels_first)
    |> EMLX.transpose(output_permutation)
    |> to_nx(out)
  end

  defp dot_spec_to_einsum_spec(
         left_shape,
         right_shape,
         left_contract_axes,
         left_batch_axes,
         right_contract_axes,
         right_batch_axes
       ) do
    possible_labels = Enum.map(?a..?z, &<<&1>>)
    {left_labels, possible_labels} = Enum.split(possible_labels, tuple_size(left_shape))
    {right_labels, possible_labels} = Enum.split(possible_labels, tuple_size(right_shape))

    if possible_labels == [] and length(right_labels) < tuple_size(right_shape) do
      raise "Not enough labels to generate einsum specification"
    end

    # Assign the same label to batch axes and to contraction axes
    right_labels =
      Enum.zip_reduce(
        left_batch_axes ++ left_contract_axes,
        right_batch_axes ++ right_contract_axes,
        right_labels,
        fn l, r, right_labels ->
          List.replace_at(right_labels, r, Enum.fetch!(left_labels, l))
        end
      )

    # Collect output labels based on batch and non-contracting, non-batching axes

    # Add batch axes
    output_labels = Enum.map(left_batch_axes, fn l -> Enum.fetch!(left_labels, l) end)

    # Add non-contracting, non-batching axes from left

    left_contract_and_batch = left_batch_axes ++ left_contract_axes

    output_labels =
      Enum.reduce(0..(tuple_size(left_shape) - 1), output_labels, fn axis, output_labels ->
        if axis not in left_contract_and_batch do
          [output_labels, Enum.fetch!(left_labels, axis)]
        else
          output_labels
        end
      end)

    # Add non-contracting, non-batching axes from right

    right_contract_and_batch = right_batch_axes ++ right_contract_axes

    output_labels =
      Enum.reduce(0..(tuple_size(right_shape) - 1), output_labels, fn axis, output_labels ->
        if axis not in right_contract_and_batch do
          [output_labels, Enum.fetch!(right_labels, axis)]
        else
          output_labels
        end
      end)

    IO.iodata_to_binary([left_labels, ",", right_labels, "->", output_labels])
  end

  @impl true
  def dot(
        %T{type: out_type} = out,
        %T{type: left_type} = left,
        left_axes,
        # MLX doesn't support batched axes
        left_batched_axes,
        %T{type: right_type} = right,
        right_axes,
        right_batched_axes
      ) do
    left_mx = from_nx(left)
    right_mx = from_nx(right)

    computation_out_type =
      if Nx.Type.integer?(out_type), do: Nx.Type.to_floating(out_type), else: out_type

    if left_batched_axes != [] or right_batched_axes != [] do
      einsum_spec =
        dot_spec_to_einsum_spec(
          left.shape,
          right.shape,
          left_axes,
          left_batched_axes,
          right_axes,
          right_batched_axes
        )

      EMLX.einsum(
        to_typed_ref(left_mx, left_type, computation_out_type),
        to_typed_ref(right_mx, right_type, computation_out_type),
        einsum_spec
      )
      |> to_typed_ref(computation_out_type, out_type)
      |> to_nx(out)
    else
      EMLX.tensordot(
        to_typed_ref(left_mx, left_type, computation_out_type),
        to_typed_ref(right_mx, right_type, computation_out_type),
        left_axes,
        right_axes
      )
      |> to_typed_ref(computation_out_type, out_type)
      |> to_nx(out)
    end
  end

  # Unary Ops

  ops =
    [
      :abs,
      :ceil,
      :floor,
      :negate,
      :round,
      :sign,
      :real,
      :imag,
      :is_nan,
      :logical_not,
      :bitwise_not
    ] ++
      [
        :sigmoid,
        :asin,
        :asinh,
        :acos,
        :acosh,
        :atan,
        :atanh,
        :cos,
        :cosh,
        :erf,
        :erf_inv,
        :exp,
        :expm1,
        :log,
        :log1p,
        :rsqrt,
        :sin,
        :sinh,
        :sqrt,
        :tan,
        :tanh
      ]

  for op <- ops do
    @impl true
    def unquote(op)(out, tensor) do
      EMLX.unquote(op)(from_nx(tensor)) |> to_nx(out)
    end
  end

  @impl true
  def conjugate(out, tensor) do
    tensor
    |> from_nx()
    |> EMLX.astype(to_mlx_type(out.type))
    |> EMLX.conjugate()
    |> to_nx(out)
  end

  @impl true
  def is_infinity(out, %T{type: {:c, _}} = tensor) do
    t_mx = from_nx(tensor)

    imag_mx = t_mx |> EMLX.imag() |> EMLX.is_infinity()
    real_mx = t_mx |> EMLX.real() |> EMLX.is_infinity()

    real_mx
    |> EMLX.logical_or(imag_mx)
    |> to_nx(out)
  end

  def is_infinity(out, tensor) do
    tensor
    |> from_nx()
    |> EMLX.is_infinity()
    |> to_nx(out)
  end

  @impl true
  def cbrt(_out, tensor) do
    Nx.pow(tensor, 1 / 3)
  end

  @impl true
  def erfc(out, tensor) do
    t = from_nx(tensor)

    out_type = to_mlx_type(out.type)
    {dev, _} = erf = EMLX.erf(t) |> EMLX.astype(out_type)

    EMLX.scalar_tensor(1, out_type, dev)
    |> EMLX.subtract(erf)
    |> to_nx(out)
  end

  # Binary Ops

  ops = [:add, :subtract, :multiply, :pow, :left_shift]

  for op <- ops do
    @impl true
    def unquote(op)(out, l, r) do
      {left, right} = maybe_upcast(l, r)
      {left_mx, right_mx} = maybe_broadcast_bin_args(out.shape, left, right)
      result = EMLX.unquote(op)(left_mx, right_mx)

      result
      |> EMLX.astype(to_mlx_type(out.type))
      |> to_nx(out)
    end
  end

  # FFT Ops
  @impl true
  def fft(out, tensor, opts) do
    length = opts[:length]
    axis = opts[:axis] || -1

    tensor
    |> from_nx()
    |> EMLX.fft(length, axis)
    |> to_nx(out)
  end

  @impl true
  def ifft(out, tensor, opts) do
    length = opts[:length]
    axis = opts[:axis] || -1

    tensor
    |> from_nx()
    |> EMLX.ifft(length, axis)
    |> to_nx(out)
  end

  @impl true
  def fft2(out, tensor, opts) do
    lengths = opts[:lengths]
    axes = opts[:axes] || [-2, -1]

    tensor
    |> from_nx()
    |> EMLX.fft2(lengths, axes)
    |> to_nx(out)
  end

  @impl true
  def ifft2(out, tensor, opts) do
    lengths = opts[:lengths]
    axes = opts[:axes] || [-2, -1]

    tensor
    |> from_nx()
    |> EMLX.ifft2(lengths, axes)
    |> to_nx(out)
  end

  @impl true
  def all_close(out, a, b, opts) do
    atol = opts[:atol] || 1.0e-4
    rtol = opts[:rtol] || 1.0e-8
    equal_nan = opts[:equal_nan] == true

    EMLX.allclose(from_nx(a), from_nx(b), atol, rtol, equal_nan)
    |> to_nx(out)
  end

  ops =
    [:divide, :quotient, :atan2] ++
      [:right_shift, :logical_and, :logical_or, :logical_xor] ++
      [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] ++
      [:bitwise_and, :bitwise_or, :bitwise_xor]

  for op <- ops do
    @impl true
    def unquote(op)(out, l, r) do
      {left, right} = maybe_upcast(l, r)
      {left_mx, right_mx} = maybe_broadcast_bin_args(out.shape, left, right)

      EMLX.unquote(op)(left_mx, right_mx)
      |> EMLX.astype(to_mlx_type(out.type))
      |> to_nx(out)
    end
  end

  @impl true
  def remainder(out, l, r) do
    {left, right} = maybe_upcast(l, r)
    {left_mx, right_mx} = maybe_broadcast_bin_args(out.shape, left, right)

    {device, _} =
      rem_mx =
      EMLX.remainder(left_mx, right_mx)
      |> EMLX.astype(to_mlx_type(out.type))

    zero = EMLX.scalar_tensor(0, to_mlx_type(left.type), device)

    left_mx
    |> EMLX.less(zero)
    |> EMLX.where(EMLX.subtract(rem_mx, right_mx), rem_mx)
    |> to_nx(out)
  end

  @impl true
  def min(out, l, r) do
    {left, right} = maybe_upcast(l, r)
    {left_mx, right_mx} = maybe_broadcast_bin_args(out.shape, left, right)

    EMLX.minimum(left_mx, right_mx)
    |> EMLX.astype(to_mlx_type(out.type))
    |> to_nx(out)
  end

  @impl true
  def max(out, l, r) do
    {left, right} = maybe_upcast(l, r)
    {left_mx, right_mx} = maybe_broadcast_bin_args(out.shape, left, right)

    EMLX.maximum(left_mx, right_mx)
    |> EMLX.astype(to_mlx_type(out.type))
    |> to_nx(out)
  end

  @impl true
  def clip(out, tensor, min, max) do
    tensor
    |> from_nx()
    |> EMLX.clip(from_nx(min), from_nx(max))
    |> to_nx(out)
  end

  @impl true
  def sort(out, tensor, opts) do
    axis = opts[:axis]
    asc? = opts[:direction] == :asc

    t = tensor |> from_nx() |> EMLX.sort(axis)

    if asc? do
      to_nx(t, out)
    else
      t
      |> to_nx(out)
      |> Nx.reverse(axes: [axis])
    end
  end

  @impl true
  def argsort(out, tensor, opts) do
    axis = opts[:axis]
    asc? = opts[:direction] == :asc

    if asc? do
      tensor
      |> from_nx()
      |> EMLX.argsort(axis)
      |> EMLX.astype(to_mlx_type(out.type))
      |> to_nx(out)
    else
      tensor
      |> from_nx()
      |> EMLX.negate()
      |> EMLX.argsort(axis)
      |> EMLX.astype(to_mlx_type(out.type))
      |> to_nx(out)
    end
  end

  defp maybe_upcast(%T{type: t} = left, %T{type: t} = right),
    do: {left, right}

  defp maybe_upcast(left, right) do
    type = Nx.Type.merge(left.type, right.type)
    {Nx.as_type(left, type), Nx.as_type(right, type)}
  end

  defp maybe_broadcast_bin_args(out_shape, l, r) do
    l_mx =
      case l.shape do
        ^out_shape ->
          from_nx(l)

        _ ->
          l |> from_nx() |> EMLX.broadcast_to(out_shape)
      end

    r_mx =
      case r.shape do
        ^out_shape -> from_nx(r)
        _ -> r |> from_nx() |> EMLX.broadcast_to(out_shape)
      end

    {l_mx, r_mx}
  end

  @impl true
  def reshape(%T{shape: shape} = out, %T{} = t),
    do: EMLX.reshape(from_nx(t), shape) |> to_nx(out)

  @impl true
  def broadcast(out, %T{} = t, shape, axes) do
    t
    |> maybe_reshape(shape, axes)
    |> from_nx()
    |> EMLX.broadcast_to(shape)
    |> to_nx(out)
  end

  defp maybe_reshape(%T{shape: {}} = t, target_shape, _axes) do
    shape = 1 |> List.duplicate(tuple_size(target_shape)) |> List.to_tuple()
    Nx.reshape(t, shape)
  end

  defp maybe_reshape(%T{shape: shape} = t, target_shape, axes) do
    base_broadcast_shape = 1 |> List.duplicate(tuple_size(target_shape)) |> List.to_tuple()

    new_shape =
      shape
      |> Tuple.to_list()
      |> Enum.zip(axes)
      |> Enum.reduce(base_broadcast_shape, fn {dim_size, target_axis}, shape_acc ->
        shape_acc
        |> Tuple.delete_at(target_axis)
        |> Tuple.insert_at(target_axis, dim_size)
      end)

    Nx.reshape(t, new_shape)
  end

  @impl true
  def as_type(%T{type: type} = out, %T{type: from_type} = t) do
    t = from_nx(t)

    t
    |> EMLX.astype(to_mlx_type(type))
    |> replace_non_finites_for_integer_cast(t, from_type, type)
    |> to_nx(out)
  end

  defp replace_non_finites_for_integer_cast(
         out,
         tensor,
         {from_type, _},
         {:s, _} = to_type
       )
       when from_type in [:f, :bf, :c] do
    # TODO: figure out if this is a bug in MLX (this function shouldn't be necessary, but the mapping for s16 is broken)
    {device, _} = out

    zero = EMLX.scalar_tensor(0, to_mlx_type(to_type), device)
    out = EMLX.where(EMLX.is_nan(tensor), zero, out)

    max_scalar =
      Nx.Constants.max_finite(to_type, backend: {EMLX.Backend, device: device}) |> from_nx()

    min_scalar =
      Nx.Constants.min_finite(to_type, backend: {EMLX.Backend, device: device}) |> from_nx()

    out =
      EMLX.is_infinity(tensor)
      |> EMLX.logical_and(EMLX.greater(tensor, zero))
      |> EMLX.where(
        max_scalar,
        out
      )

    EMLX.is_infinity(tensor)
    |> EMLX.logical_and(EMLX.less(tensor, zero))
    |> EMLX.where(
      min_scalar,
      out
    )
  end

  defp replace_non_finites_for_integer_cast(out, _, _, _), do: out

  @impl true
  def reduce_max(out, tensor, opts) do
    axes = opts[:axes] || Nx.axes(tensor)
    keep_axes = opts[:keep_axes]

    tensor
    |> from_nx()
    |> EMLX.max(axes, keep_axes)
    |> to_nx(out)
  end

  @impl true
  def reduce_min(out, tensor, opts) do
    axes = opts[:axes] || Nx.axes(tensor)
    keep_axes = opts[:keep_axes]

    tensor
    |> from_nx()
    |> EMLX.min(axes, keep_axes)
    |> to_nx(out)
  end

  for op <- [:sum, :product, :max, :min] do
    @impl true
    def unquote(:"window_#{op}")(out, tensor, window_shape, opts) do
      window_op(unquote(op), out, tensor, window_shape, opts)
    end
  end

  defp window_op(op, out, tensor, window_shape, opts) do
    # TODO: window dilations can be implemented after we support internal padding
    # in Nx.pad (we should have pad_internal as a shared defp)
    tensor_rank = tuple_size(tensor.shape)

    axes =
      0..(tuple_size(window_shape) - 1)
      |> Enum.to_list()
      |> Enum.map(fn axis ->
        tensor_rank + axis
      end)

    {low_pad, high_pad} = Enum.unzip(opts[:padding])
    {device, _} = t_mx = from_nx(tensor)

    {_device, pad_mx} =
      case op do
        :sum ->
          EMLX.scalar_tensor(0, to_mlx_type(out.type), device)

        :product ->
          EMLX.scalar_tensor(1, to_mlx_type(out.type), device)

        :max ->
          Nx.Constants.min(tensor.type, backend: {EMLX.Backend, device: device}) |> from_nx()

        :min ->
          Nx.Constants.max(tensor.type, backend: {EMLX.Backend, device: device}) |> from_nx()
      end

    padded_mx = EMLX.pad(t_mx, Nx.axes(tensor), low_pad, high_pad, pad_mx)

    padded_mx
    |> sliding_window_view(EMLX.shape(padded_mx), window_shape, opts[:strides])
    |> then(&apply(EMLX, op, [&1, axes, false]))
    |> to_nx(out)
  end

  defp sliding_window_view(t, tensor_shape, window_shape, opt_strides) do
    strides = EMLX.strides(t)

    strides = strides ++ strides
    window_shape_list = Tuple.to_list(window_shape)

    shape_trimmed =
      Enum.zip_with(
        [Tuple.to_list(tensor_shape), window_shape_list],
        fn [current, dim] ->
          current - dim + 1
        end
      )

    stops = shape_trimmed ++ window_shape_list
    stride_shape = List.to_tuple(stops)

    starts = List.duplicate(0, tuple_size(stride_shape))

    slice_strides = opt_strides ++ List.duplicate(1, tuple_size(window_shape))

    t
    |> EMLX.as_strided(stride_shape, strides, 0)
    |> EMLX.slice(starts, stops, slice_strides)
  end

  @impl true
  def window_scatter_min(out, tensor, source, init_value, window_dims_tuple, opts) do
    window_scatter_function(
      &Nx.argmin(&1, axis: -1, tie_break: :high),
      out,
      tensor,
      source,
      init_value,
      window_dims_tuple,
      opts
    )
  end

  @impl true
  def window_scatter_max(out, tensor, source, init_value, window_dims_tuple, opts) do
    window_scatter_function(
      &Nx.argmax(&1, axis: -1),
      out,
      tensor,
      source,
      init_value,
      window_dims_tuple,
      opts
    )
  end

  defp window_scatter_function(function, out, tensor, source, init_value, window_dims_tuple, opts) do
    # TODO: support window dilations
    unfold_flat = fn tensor ->
      {device, _} = t_mx = from_nx(tensor)
      {_, pad_mx} = EMLX.scalar_tensor(0, EMLX.scalar_type(t_mx), device)

      {low_pad, high_pad} = Enum.unzip(opts[:padding])

      padded_mx = EMLX.pad(t_mx, Nx.axes(tensor), low_pad, high_pad, pad_mx)

      unfolded_mx =
        sliding_window_view(
          padded_mx,
          EMLX.shape(padded_mx),
          window_dims_tuple,
          opts[:strides]
        )

      unfolded_shape = EMLX.shape(unfolded_mx)
      unfolded = to_nx(unfolded_mx)

      {to_keep, to_flatten} =
        unfolded_shape
        |> Tuple.to_list()
        |> Enum.split(-tuple_size(window_dims_tuple))

      flat_shape =
        to_keep
        |> List.to_tuple()
        |> then(&Tuple.insert_at(&1, tuple_size(&1), Enum.product(to_flatten)))

      Nx.reshape(unfolded, flat_shape)
    end

    arg_idx =
      tensor
      |> then(unfold_flat)
      |> then(function)

    indices_to_flatten =
      tensor
      |> Nx.axes()
      |> Enum.map(fn axis ->
        tensor
        |> Nx.shape()
        |> Nx.iota(axis: axis, backend: EMLX.Backend)
        |> then(unfold_flat)
        |> Nx.take_along_axis(Nx.new_axis(arg_idx, -1), axis: -1)
      end)
      |> Nx.concatenate(axis: -1)

    num_axes = tuple_size(out.shape)
    num_rows = div(Nx.size(indices_to_flatten), num_axes)
    indices = Nx.reshape(indices_to_flatten, {num_rows, num_axes})

    flat_source = Nx.flatten(source)

    init_value
    |> Nx.backend_transfer(EMLX.Backend)
    |> Nx.broadcast(out.shape)
    |> Nx.indexed_add(indices, flat_source)
    |> Nx.as_type(out.type)
    |> Nx.rename(out.names)
  end

  @impl true
  def to_batched(out, tensor, opts) do
    leftover = opts[:leftover]

    batch_size = elem(out.shape, 0)
    axis_size = elem(tensor.shape, 0)

    remainder = rem(axis_size, batch_size)
    num_full_batches = div(axis_size, batch_size)

    range =
      if remainder != 0 and leftover == :repeat do
        0..num_full_batches
      else
        0..(num_full_batches - 1)
      end

    Stream.map(range, fn
      ^num_full_batches ->
        Nx.concatenate([
          Nx.slice_along_axis(tensor, num_full_batches * batch_size, remainder),
          Nx.slice_along_axis(tensor, 0, batch_size - remainder)
        ])

      i ->
        start_idx = i * batch_size
        Nx.slice_along_axis(tensor, start_idx, batch_size)
    end)
  end

  @impl true
  def gather(out, tensor, indices, opts) do
    axes = opts[:axes]

    num_axes = Nx.axis_size(indices, -1)

    slice_sizes =
      Enum.map(Nx.axes(tensor), fn axis ->
        if axis in axes do
          1
        else
          Nx.axis_size(tensor, axis)
        end
      end)

    indices_list =
      Enum.map(0..(num_axes - 1), fn entry ->
        {_device, ref} =
          indices
          |> Nx.slice_along_axis(entry, 1, axis: -1)
          |> Nx.squeeze(axes: [-1])
          |> from_nx()

        ref
      end)

    tensor
    |> from_nx()
    |> EMLX.gather(indices_list, axes, slice_sizes)
    |> EMLX.reshape(out.shape)
    |> to_nx(out)
  end

  @impl true
  def indexed_add(out, target, indices, updates, opts) do
    indexed_op(:scatter_add, out, target, indices, updates, opts)
  end

  @impl true
  def indexed_put(out, target, indices, updates, opts) do
    indexed_op(:scatter, out, target, indices, updates, opts)
  end

  defp indexed_op(nif_op, out, target, indices, updates, opts) do
    axes = opts[:axes] || Nx.axes(target)
    num_axes = Nx.axis_size(indices, -1)

    indices_list =
      Enum.map(0..(num_axes - 1), fn entry ->
        {_device, ref} =
          indices
          |> Nx.slice_along_axis(entry, 1, axis: -1)
          |> Nx.squeeze(axes: [-1])
          |> from_nx()

        ref
      end)

    insert_index =
      axes
      |> Enum.scan(&(&1 - &2))
      |> Enum.find_index(&(&1 > 1))
      |> then(&(&1 || num_axes))

    [num_updates | updates_inner_shape] = Tuple.to_list(updates.shape)

    updates_shape =
      [num_updates | List.duplicate(1, num_axes)]
      |> List.insert_at(insert_index + 1, updates_inner_shape)
      |> List.flatten()
      |> List.to_tuple()

    updates_mx = from_nx(updates) |> EMLX.reshape(updates_shape)

    target
    |> from_nx()
    |> EMLX.astype(to_mlx_type(out.type))
    |> then(
      &apply(EMLX, nif_op, [
        &1,
        indices_list,
        EMLX.astype(updates_mx, to_mlx_type(out.type)),
        axes
      ])
    )
    |> to_nx(out)
  end

  @impl true
  def triangular_solve(out, a, b, opts) do
    if Nx.Type.complex?(out.type) do
      raise "complex numbers not supported yet"
    end

    a_mx =
      case opts[:transform_a] do
        :none ->
          to_typed_ref(from_nx(a), a.type, {:f, 32})

        :transpose ->
          a |> from_nx() |> to_typed_ref(a.type, {:f, 32}) |> EMLX.transpose([-2, -1])
      end

    b_mx = to_typed_ref(from_nx(b), b.type, {:f, 32})

    upper = !opts[:lower]

    a_inv_mx = EMLX.tri_inv(a_mx, upper)

    out_mx =
      if opts[:left_side] do
        # Solve AX = B -> X = tri_inv(A)@B
        {batch_axes, [_m, n]} = Nx.axes(EMLX.shape(a_inv_mx)) |> Enum.split(-2)

        {b_batch_axes, b_contract_axes} =
          if Nx.rank(b) == 1 do
            {[], [0]}
          else
            case Nx.axes(b) |> Enum.split(length(batch_axes)) do
              {batch, [x]} ->
                {batch, [x]}

              {batch, [m, _n]} ->
                {batch, [m]}
            end
          end

        EMLX.einsum(
          a_inv_mx,
          b_mx,
          dot_spec_to_einsum_spec(
            EMLX.shape(a_inv_mx),
            EMLX.shape(b_mx),
            [n],
            batch_axes,
            b_contract_axes,
            b_batch_axes
          )
        )
      else
        # Solve XA = B -> X = B@tri_inv(A)
        {batch_axes, [m, _n]} = Nx.axes(EMLX.shape(a_inv_mx)) |> Enum.split(-2)

        {b_batch_axes, b_contract_axes} =
          if Nx.rank(b) == 1 do
            {[], [0]}
          else
            {batch, [_m, n]} = Nx.axes(b) |> Enum.split(-2)
            {batch, [n]}
          end

        EMLX.einsum(
          b_mx,
          a_inv_mx,
          dot_spec_to_einsum_spec(
            EMLX.shape(b_mx),
            EMLX.shape(a_inv_mx),
            b_contract_axes,
            b_batch_axes,
            [m],
            batch_axes
          )
        )
      end

    out_mx
    |> EMLX.astype(to_mlx_type(out.type))
    |> to_nx(out)
  end

  for {op, arity} <- [
        reduce: 5,
        window_reduce: 6,
        population_count: 2,
        count_leading_zeros: 2
      ] do
    args = List.duplicate(Macro.var(:_, __MODULE__), arity)
    @impl true
    def unquote(op)(unquote_splicing(args)) do
      raise "#{unquote(op)} not supported in EMLX"
    end
  end

  for {op, arity} <- [
        lu: 3,
        to_pointer: 2,
        from_pointer: 5
      ] do
    @impl true
    args = List.duplicate(Macro.var(:_, __MODULE__), arity)

    def unquote(op)(unquote_splicing(args)) do
      raise "Nx.Backend.#{unquote(op)}/#{unquote(arity)} not implemented yet in EMLX"
    end
  end

  # Helper function to handle different scalar types
  defp constant_serialize_scalar(scalar) when is_number(scalar), do: scalar
  defp constant_serialize_scalar(%Complex{} = c), do: {c.re, c.im}

  defp to_typed_ref(tensor, expected_type, expected_type),
    do: tensor

  defp to_typed_ref(tensor, _ref_type, expected_type),
    do: EMLX.astype(tensor, to_mlx_type(expected_type))

  defp device_option(nil), do: :cpu
  defp device_option(backend_opts), do: backend_opts[:device] || :cpu
end
