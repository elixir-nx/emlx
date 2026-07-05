defmodule EMLX.Debug do
  @moduledoc false

  @detect_non_finites Application.compile_env(:emlx, :detect_non_finites, false)

  @doc false
  defmacro assert_no_nan_inf!(tensor_ref, op) do
    if @detect_non_finites do
      quote do
        has_nan = unquote(tensor_ref) |> EMLX.is_nan() |> EMLX.any([], false)
        has_inf = unquote(tensor_ref) |> EMLX.is_infinity() |> EMLX.any([], false)
        EMLX.eval(has_nan)
        EMLX.eval(has_inf)

        if EMLX.item(has_nan) == 1 or EMLX.item(has_inf) == 1 do
          raise ArgumentError,
                "#{unquote(op)} produced NaN or Inf. Disable :detect_non_finites for production."
        end
      end
    else
      quote do: nil
    end
  end
end
