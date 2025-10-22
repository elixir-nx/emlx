defmodule EMLX.Nx.DoctestTest do
  use ExUnit.Case, async: true

  setup do
    Nx.default_backend(EMLX.Backend)
    :ok
  end

  @os_specific_rounding_error (case(:os.type()) do
                                 {:unix, :darwin} ->
                                   []

                                 {:unix, _} ->
                                   [
                                     # x86_64 and aarch64
                                     atanh: 1,
                                     # aarch64
                                     ifft: 2
                                   ]

                                 _ ->
                                   []
                               end)

  @rounding_error [
    exp: 1,
    erf: 1,
    erfc: 1,
    expm1: 1,
    atan: 1,
    sigmoid: 1,
    round: 1,
    asinh: 1,
    asin: 1,
    tan: 1,
    cos: 1,
    standard_deviation: 2,
    cosh: 1,
    log10: 1,
    acos: 1,
    covariance: 3,
    # These fail because we're using different representation types
    atan2: 2,
    as_type: 2,
    from_binary: 3
  ]

  @to_be_fixed [
    :moduledoc,
    # MLX sorts NaNs lowest, Nx sorts them highest
    argsort: 2
  ]

  @not_supported [
    reduce: 4,
    window_reduce: 5,
    population_count: 1,
    count_leading_zeros: 1,
    sort: 2,
    # We do not support the same ordering for NaNs as Nx
    argmin: 2,
    argmax: 2
  ]

  doctest Nx,
    except: @rounding_error ++ @os_specific_rounding_error ++ @not_supported ++ @to_be_fixed
end
