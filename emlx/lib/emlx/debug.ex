defmodule EMLX.Debug do
  @moduledoc """
  Shared compile-time debug-assertion flags/macros.

  A macro defined here expands to its assertion body when the backing flag
  is `true` at compile time, or to `nil` when `false` — leaving no trace in
  the importing module's BEAM opcodes (no call instruction, no atom
  reference). Development-only; forces extra `EMLX.eval` syncs and breaks
  MLX lazy-graph fusion when enabled.

  Centralized here (rather than redefined per module) so every caller reads
  the same `Application.compile_env/3` value — one source of truth for
  whether a given build has the check compiled in.

  Enable via `config/dev.exs`:

      config :emlx, detect_non_finites: true

  After flipping a flag, run `mix compile --force` (module attributes are
  baked in at compile time, not read at runtime).
  """

  @detect_non_finites Application.compile_env(:emlx, :detect_non_finites, false)

  @doc """
  Checks a raw MLX ref for NaN or Inf. Forces two eval syncs — development only.

  No-op (expands to `nil`) when `:detect_non_finites` is off.
  """
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
