defmodule EMLXAxon.Native.Plugin do
  @moduledoc false

  alias EMLX.Native.Expr, as: NativeExpr

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

    attrs =
      [
        @wire_version,
        plugin,
        callback,
        schema_version,
        attr_schema_version,
        length(templates)
      ] ++ encoded_templates ++ [length(callback_attrs) | callback_attrs]

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
        callback_attrs,
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

  @doc false
  def f64_bits(value), do: NativeExpr.f64_bits(value)
end
