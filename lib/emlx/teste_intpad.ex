defmodule EMLX.TesteIntpad do

  # General solution for interior padding in EMLX.

  # This function implements interior padding by iterating through each axis
  # that has interior padding and applying the padding strategy without using transpose.

  # The strategy is:
  # 1. Create a new axis at the end of the tensor
  # 2. For each axis with interior padding, pad on the next axis
  # 3. Reshape to distribute the padding
  # 4. Squeeze to get the final result

  def intpad(tensor, value, padding_config) do

      tensor = Nx.new_axis(tensor, -1)
    {final_tensor, _} = Enum.reduce(padding_config, {tensor, 0}, fn {_low, _high, interior}, {acc, axis_index} ->
      new_tensor = apply_interior_padding(acc, value, axis_index, interior, acc.shape)
      {new_tensor, axis_index + 1}
    end)
    final_tensor
    |> Nx.squeeze(axes: [-1])
  end

  defp apply_interior_padding(tensor, value, axis_index, 0, shape) do
    tensor
  end

  defp apply_interior_padding(tensor, value, axis_index, interior_padding, shape) do
    rank = tuple_size(shape)
    next_axis = axis_index + 1
    axis_size = elem(shape, axis_index)
    next_axis_size = elem(shape, next_axis)

    # Create padding config: pad on next axis
    padding_config =
      {0, 0, 0}
      |> List.duplicate(rank)
      |> List.replace_at(next_axis, {0, next_axis_size * interior_padding, 0})
    # Apply padding and reshape
    padded_tensor = Nx.pad(tensor, value, padding_config)

    new_axis_size = axis_size + axis_size * interior_padding

    new_shape =
      shape
          |> put_elem(axis_index, new_axis_size)
          |> put_elem(axis_index + 1, next_axis_size)

    padded_tensor
    |> Nx.reshape(new_shape)
    |> Nx.slice_along_axis(0, new_axis_size - interior_padding, axis: axis_index)
  end
end
