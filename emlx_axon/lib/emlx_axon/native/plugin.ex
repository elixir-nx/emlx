defmodule EMLXAxon.Native.Plugin do
  @moduledoc false

  @wire_version 1

  @doc false
  def traced?(tensors) do
    Enum.any?(List.wrap(tensors), &match?(%Nx.Tensor{data: %Nx.Defn.Expr{}}, &1))
  end

  @doc false
  def metadata(fallback, plugin, callback, operands, callback_attrs, outputs, opts \\ []) do
    schema_version = Keyword.get(opts, :schema_version, 1)
    attr_schema_version = Keyword.get(opts, :attr_schema_version, 1)
    templates = Nx.Defn.Composite.flatten_list([outputs])

    encoded_templates =
      Enum.flat_map(templates, fn %Nx.Tensor{type: type, shape: shape} ->
        [EMLX.Native.to_mlx_type(type), tuple_size(shape) | Tuple.to_list(shape)]
      end)

    encoded_attrs = encode_attrs(callback_attrs)

    attrs =
      [
        @wire_version,
        plugin,
        callback,
        schema_version,
        attr_schema_version,
        length(templates)
      ] ++ encoded_templates ++ [length(encoded_attrs) | encoded_attrs]

    Nx.Defn.Expr.metadata(fallback, %{
      __EMLX__: %{op: :plugin, operands: operands, attrs: attrs}
    })
  end

  @doc false
  def call(plugin, callback, operands, callback_attrs, output_templates) do
    [{device, _} | _] = backend_refs = Enum.map(operands, &EMLX.Backend.from_nx/1)

    unless Enum.all?(backend_refs, &(elem(&1, 0) == device)) do
      raise ArgumentError, "plugin operands must use one EMLX device"
    end

    {worker, effective_device} = EMLX.resolve_worker(device)

    refs =
      EMLX.NIF.call_plugin(
        worker,
        plugin,
        callback,
        Enum.map(backend_refs, &elem(&1, 1)),
        encode_attrs(callback_attrs),
        effective_device
      )
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    templates =
      if is_list(output_templates) do
        Enum.flat_map(output_templates, &Nx.Defn.Composite.flatten_list([&1]))
      else
        Nx.Defn.Composite.flatten_list([output_templates])
      end

    if length(refs) != length(templates) do
      raise EMLX.NIFError,
            "plugin callback #{plugin}/#{callback} returned #{length(refs)} outputs, " <>
              "expected #{length(templates)}"
    end

    Enum.zip_with(refs, templates, fn ref, template ->
      EMLX.Backend.to_nx({effective_device, ref}, template)
    end)
  end

  # Floats ride the int64 attr channel as IEEE-754 bit patterns. Integers are
  # left alone so dims/offsets stay raw (1 vs 1.0 is intentional).
  @doc false
  def encode_attrs(attrs), do: Enum.map(attrs, &encode_attr/1)

  defp encode_attr(value) when is_float(value) do
    <<bits::signed-native-64>> = <<value::float-native-64>>
    bits
  end

  defp encode_attr(value), do: value
end
